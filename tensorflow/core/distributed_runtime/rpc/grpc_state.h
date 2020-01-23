/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_

#include <utility>

#include "grpcpp/generic/generic_stub.h"
#include "grpcpp/grpcpp.h"

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/notification.h"

namespace tensorflow {

/** \class RPCState
 *
 *  \param Response: typename, class;
 *         For example, TensorResponse.
 *
 *  \brief Start a grpc request and get response from server.
 *
 */
// Object allocated per active RPC.
// Manage the state of a single asynchronous RPC request.  If `max_retries`
// is greater than 0, the request will be retried for any transient failures
// as long as the overall deadline has not elapsed.
template <class Response>
class RPCState : public GrpcClientCQTag {
 public:

  /** \brief RPCState Constructor, indirectly start a grpc call from the client
   *         side with request message and get the response message from the
   *         server side.
   *
   *  \param stub: ::grpc::GenericStub* ;
   *         Generic stubs provide a type-unsafe interface to call gRPC methods
   *         by name.
   *
   *  \param cq: ::grpc::CompletionQueue* ;
   *         A thin wrapper around grpc_completion_queue.
   *         Completion Queues enable notification of the completion of
   *         asynchronous actions.
   *
   *  \param method: const ::grpc::string& ;
   *         type alias of std::string.
   *
   *  \param request: const protobuf::Message& ;
   *         class of protocol message.
   *         For example, message GetStatusRequest.
   *
   *  \param response: Response*
   *         Response: typename, class; For example, TensorResponse.
   *
   *  \param done: StatusCallback ;
   *         A lambda function.
   *
   *  \param call_opts: CallOptions* ;
   *         Options passed to grpc interface calls in call_options.h
   *
   *  \param threadpool: thread::ThreadPool* ;
   *         a pool of CPU threads.
   *
   *  \param max_retries: int32, default is 0;
   *         max of retries of grpc. the request will be retried for any
   *         transient failures as long as the overall deadline has not elapsed.
   */
  // Default behavior is to set fail_fast = False and handle timeouts manually.
  RPCState(::grpc::GenericStub* stub, ::grpc::CompletionQueue* cq,
           const ::grpc::string& method, const protobuf::Message& request,
           Response* response, StatusCallback done, CallOptions* call_opts,
           thread::ThreadPool* threadpool, int32 max_retries = 0)
      : RPCState(stub, cq, method, request, response, std::move(done),
                 call_opts, threadpool, /*fail_fast=*/false,
                 /*timeout_in_ms=*/0, max_retries) {}

  template <typename Request>
  RPCState(::grpc::GenericStub* stub, ::grpc::CompletionQueue* cq,
           const ::grpc::string& method, const Request& request,
           Response* response, StatusCallback done, CallOptions* call_opts,
           thread::ThreadPool* threadpool, bool fail_fast, int64 timeout_in_ms,
           int32 max_retries)
      : call_opts_(call_opts),
        threadpool_(threadpool),
        done_(std::move(done)),
        cq_(cq),
        stub_(stub),
        method_(method),
        max_retries_(max_retries),
        timeout_in_ms_(timeout_in_ms),
        fail_fast_(fail_fast) {
    response_ = response;
    ::grpc::Status s = GrpcMaybeUnparseProto(request, &request_buf_);
    if (!s.ok()) {
      LOG(ERROR) << "GrpcMaybeUnparseProto returned with non-ok status: "
                 << s.error_message();
      // Skip retry logic if we fail to parse our request.
      done_(FromGrpcStatus(s));
      delete this;
      return;
    }
    StartCall();
  }

  /** \brief The client starts a grpc call with request message and get response
   *         message from server.
   */
  void StartCall() {
    /// A ::grpc::ClientContext allows the person implementing a service client
    /// to:
    /// - Add custom metadata key-value pairs that will propagated to the server
    ///   side.
    /// - Control call settings such as compression and authentication.
    /// - Initial and trailing metadata coming from the server.
    /// - Get performance metrics (ie, census).
    context_.reset(new ::grpc::ClientContext());
    context_->set_fail_fast(fail_fast_);

    if (timeout_in_ms_ > 0) {
      context_->set_deadline(
          gpr_time_from_millis(timeout_in_ms_, GPR_TIMESPAN));
    }
    /// CallOptions* call_opts_;
    /// Options passed to grpc interface calls in call_options.h
    if (call_opts_) {
      call_opts_->SetCancelCallback([this]() { context_->TryCancel(); });
    }

    VLOG(2) << "Starting call: " << method_;

    call_ = std::move(
        /// stub_: ::grpc::GenericStub* ;
        ///
        /// \fn ::grpc::GenericStub::PrepareUnaryCall
        /// \brief Setup a unary call to a named method method using context,
        ///        and don't start it.
        ///
        /// params:
        /// - context_.get(): grpc::ClientContext * context;
        /// - method_: const grpc::string & method;
        /// - request_buf_: const grpc::ByteBuffer & request;
        /// - cq_: grpc::CompletionQueue * cq;
        /// Return std::unique_ptr<::grpc::GenericClientAsyncResponseReader>
        ///        GenericClientAsyncResponseReader is an alias of
        ///          ClientAsyncResponseReader< ByteBuffer >
        stub_->PrepareUnaryCall(context_.get(), method_, request_buf_, cq_));

    /// start the grpc call.
    call_->StartCall();

    /// Request to receive the server's response msg and final status for the
    /// call, and to notify tag on this call's completion queue when finished.
    ///
    /// This function will return when either:
    ///
    /// when the server's response message and status have been received.
    /// when the server has returned a non-OK status (no message expected in this case).
    /// when the call failed for some reason and the library generated a non-OK status.
    ///
    /// template<class R>
    /// Finish(R* msg, Status* status, void* tag)
    ///
    /// Parameters
    /// [in]	this: tag	Tag identifying this request.
    /// [out]	status_: status	To be updated with the operation status.
    /// [out]	response_buf_: msg	To be filled in with the server's response message.
    call_->Finish(&response_buf_, &status_, this);
  }

  void OnCompleted(bool ok) override {
    if (call_opts_) {
      call_opts_->ClearCancelCallback();
    }
    Status s = FromGrpcStatus(status_);
    if (s.ok() && !ok) {
      // Since this function is only being used for processing the response
      // to Finish for client-side unary calls, ok should never be false
      s.Update(errors::Internal("unexpected ok value at rpc completion"));
    }

    if (s.ok()) {
      if (threadpool_) {
        // Run parse and callback in another thread, returning this
        // one to service more RPCs.
        threadpool_->Schedule([this]() { ParseAndCallDone(); });
      } else {
        ParseAndCallDone();
      }
      return;
    }

    VLOG(1) << method_ << " returned with non-ok status: " << s
            << " Retries: " << num_retries_ << " Max: " << max_retries_ << "\n"
            << context_->debug_error_string();
    // Retry if we have any attempts left
    if (++num_retries_ <= max_retries_ &&
        (errors::IsUnavailable(s) || errors::IsUnknown(s))) {
      response_buf_.Clear();
      VLOG(1) << "Retrying call for " << method_ << "Retry: " << num_retries_
              << " of " << max_retries_;
      StartCall();
    } else {
      // Attach additional GRPC error information if any to the final status
      s = Status(s.code(),
                 strings::StrCat(s.error_message(),
                                 "\nAdditional GRPC error information:\n",
                                 context_->debug_error_string()));
      done_(s);
      delete this;
    }
  }

  void ParseAndCallDone() {
    Status s;
    if (!GrpcMaybeParseProto(&response_buf_, response_)) {
      s.Update(errors::Internal("could not parse rpc response"));
    }
    done_(s);
    delete this;
  }

 private:
  CallOptions* call_opts_;
  std::unique_ptr<::grpc::ClientContext> context_;
  thread::ThreadPool* threadpool_;
  std::unique_ptr<::grpc::GenericClientAsyncResponseReader> call_;
  Response* response_;
  ::grpc::ByteBuffer request_buf_;
  ::grpc::ByteBuffer response_buf_;
  ::grpc::Status status_;
  StatusCallback done_;
  int64 timeout_in_ms_;

  size_t num_retries_ = 0;
  size_t max_retries_;

  ::grpc::CompletionQueue* cq_;
  ::grpc::GenericStub* stub_;
  ::grpc::string method_;
  bool fail_fast_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_
