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
  // 1.
  // GrpcClientCQTag 是什么
  // tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h:29:
  // class GrpcClientCQTag

  // 2.
  // gRPC 的教程
  // https://grpc.io/docs/languages/cpp/basics/
  // https://grpc.io/docs/languages/cpp/async/
  //
  // 2.1 *** define the reqeust and response ***
  //

  // 2.2 *** define the service ***
  // (tf_1_13_1_gdb) wxf@seir19:~/tf1_13_1/baseline/tensorflow$ grep -nwr "service.*{" --include="*.proto"
  // tensorflow/compiler/xla/rpc/xla_service.proto:49:service XlaService {
  // tensorflow/core/protobuf/worker_service.proto:38:service WorkerService {
  // tensorflow/core/protobuf/eager_service.proto:167:service EagerService {
  // tensorflow/core/protobuf/master_service.proto:87:service MasterService {
  // tensorflow/core/profiler/profiler_analysis.proto:64:service ProfileAnalysis {
  // tensorflow/core/profiler/profiler_service.proto:10:service ProfilerService {
  // tensorflow/core/debug/debug_service.proto:85:service EventListener {
  // tensorflow/contrib/rpc/python/kernel_tests/test_example.proto:12:service TestCaseService {
  // tensorflow/contrib/verbs/verbs_service.proto:65:service VerbsService {

  // 2.3
  // server service
  // class WorkerService final
  // tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h:101:
  //


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
  RPCState(
    ::grpc::GenericStub* stub,
    ::grpc::CompletionQueue* cq,
    const ::grpc::string& method,
    const protobuf::Message& request, // e.g., type = /* real type = const tensorflow::GetStatusRequest * */
    Response* response,
    StatusCallback done,
    CallOptions* call_opts,
    thread::ThreadPool* threadpool,
    int32 max_retries = 0)
      : RPCState(
          stub,
          cq,
          method,
          request,
          response,
          std::move(done),
          call_opts,
          threadpool,
          /*fail_fast=*/false,
          /*timeout_in_ms=*/0,
          max_retries) {}
  // 1.
  // callstack
  //
  // 1.1
  // tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc
  //
  // Utility method for issuing a generic asynchronous request. The
  // given callback, `done`, will be called when the RPC completes.
  //
  // void IssueRequest(const protobuf::Message* request,
  //                   protobuf::Message* response, const ::grpc::string& method,
  //                   StatusCallback done, CallOptions* call_opts = nullptr,
  //                   int max_retries = kMaxWorkerRpcRetries) {
  //   new RPCState<protobuf::Message>(&stub_, cq_, method, *request, response,
  //                                   std::move(done), call_opts,
  //                                   callback_threadpool_, max_retries);
  // }

  // 1.2
  //
  // tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc
  //
  // void GetStatusAsync(const GetStatusRequest* request,
  //                     GetStatusResponse* response,
  //                     StatusCallback done) override {
  //   IssueRequest(request, response, getstatus_, std::move(done));
  // }

  // 1.3
  //
  //
  // void NewRemoteDevices(Env* env, WorkerCacheInterface* worker_cache,
  //                       const string& worker_name, NewRemoteDevicesDone done) {
  //   ...
  //   wi->GetStatusAsync(&call->req, &call->resp, cb);
  // }

  // 1.4
  // tensorflow::DeviceFinder::Start() at master.cc:249
  //
  // void Start() {
  //   ...
  //   for (size_t i = 0; i < targets_.size(); ++i) {
  //     NewRemoteDevices(env_->env, worker_cache_, targets_[i],
  //                      std::bind(&ME::WhenFound, this, i, _1, _2));
  //   }
  // }

  // 1.5
  //
  // tensorflow/core/distributed_runtime/master.cc
  //
  // class DeviceFinder {
  //  public:
  //   static Status GetRemoteDevices(
  //       const protobuf::RepeatedPtrField<string>& device_filters, MasterEnv* env,
  //       WorkerCacheInterface* worker_cache,
  //       std::vector<std::unique_ptr<Device>>* out_remote) {
  //     DeviceFinder finder(device_filters, env, worker_cache);
  //     finder.Start(); // <== ✅
  //     TF_RETURN_IF_ERROR(finder.Wait());
  //     finder.GetRemoteDevices(env->local_devices, out_remote);
  //     return Status::OK();
  //   }

  // 1.6
  //
  // tensorflow/core/distributed_runtime/master.cc
  //
  // void Master::CreateSession(const CreateSessionRequest* req,
  //                            CreateSessionResponse* resp, MyClosure done) {
  //   SchedClosure([this, req, resp, done]() {
  //     ...
  //     worker_cache = env_->worker_cache;
  //     // Ping all the workers and build the list of devices that the
  //     // session will use.
  //     status =
  //         DeviceFinder::GetRemoteDevices(req->config().device_filters(), env_,
  //                                        worker_cache, remote_devices.get());
  //      ...
  //   }
  // }

  template <typename Request>
  RPCState(
    ::grpc::GenericStub* stub,
    ::grpc::CompletionQueue* cq,
    const ::grpc::string& method,
    const Request& request,
    Response* response,
    StatusCallback done,
    CallOptions* call_opts,
    thread::ThreadPool* threadpool,
    bool fail_fast,
    int64 timeout_in_ms,
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

    // =======================================================================
    StartCall();
    // =======================================================================
  }

  // The client starts a grpc call with request message and get response
  // message from server.
  // Q: who is server?
  // ...
  void StartCall() {
    context_.reset(new ::grpc::ClientContext());
    // 1.
    // context_ 是干嘛的?
    // ::grpc::ClientContext allows the person implementing a service client
    // to:
    // - Add custom metadata key-value pairs that will propagated to the server
    //   side.
    // - Control call settings such as compression and authentication.
    // - Initial and trailing metadata coming from the server.
    // - Get performance metrics (ie, census).

    context_->set_fail_fast(fail_fast_);

    if (timeout_in_ms_ > 0) {
      context_->set_deadline(
          gpr_time_from_millis(timeout_in_ms_, GPR_TIMESPAN));
    }

    if (call_opts_) {
      // 1.
      // call_opts_: CallOptions*
      // Options passed to grpc interface calls in call_options.h

      call_opts_->SetCancelCallback([this]() { context_->TryCancel(); });
    }

    VLOG(2) << "Starting call: " << method_;
    // 1.
    // 2020-09-29 16:52:09.558755: I
    // ./tensorflow/core/distributed_runtime/rpc/grpc_state.h:90]
    // Starting call: /tensorflow.WorkerService/GetStatus

    call_ = std::move(
        stub_->PrepareUnaryCall(context_.get(),
                                method_,
                                request_buf_,
                                cq_));
        // 1.
        // stub_: ::grpc::GenericStub* ;
        //
        // \fn ::grpc::GenericStub::PrepareUnaryCall
        // \brief Setup a unary call to a named method method using context,
        //        and don't start it.
        //
        // params:
        // - context_.get(): grpc::ClientContext * context;
        // - method_: const grpc::string & method;
        // - request_buf_: const grpc::ByteBuffer & request;
        // - cq_: grpc::CompletionQueue * cq;
        //
        // Return std::unique_ptr<::grpc::GenericClientAsyncResponseReader>
        //        GenericClientAsyncResponseReader is an alias of
        //          ClientAsyncResponseReader< ByteBuffer >

        // 2.
        // call_: std::unique_ptr<::grpc::GenericClientAsyncResponseReader>

        // 3.
        // context_
        // context_.get(): grpc::ClientContext * context;


    // start the grpc call.
    call_->StartCall();

    // Request to receive the server's response msg and final status for the
    // call, and to notify tag on this call's completion queue when finished.
    //
    // This function will return when either:
    //
    // when the server's response message and status have been received.
    // when the server has returned a non-OK status (no message expected in this case).
    // when the call failed for some reason and the library generated a non-OK status.
    //
    // template<class R>
    // Finish(R* msg, Status* status, void* tag)
    //
    // Parameters
    // [in]	this: tag	Tag identifying this request.
    // [out]	status_: status	To be updated with the operation status.
    // [out]	response_buf_: msg	To be filled in with the server's response message.
    call_->Finish(&response_buf_, &status_, this);
    // 1.
    // response_buf_: ::grpc::ByteBuffer
    // request_buf_ : ::grpc::ByteBuffer

    // 2.
    // Async server implementation tutorial:
    // https://grpc.io/docs/languages/cpp/async/
  }
  // note:
  // 下面是到 ParseAndCallDone (本文件内)

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
        // 进入
        // Run parse and callback in another thread, returning this
        // one to service more RPCs.
        // ==========================================================
        threadpool_->Schedule([this]() { ParseAndCallDone(); });
        // ==========================================================
      } else {
        // 未进入!
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
      // 1.
      // 指针呐, 所以就这么当了 output 了.
      // 我猜指针
      // tensorflow/core/distributed_runtime/rpc/grpc_util.cc:89:
      // bool GrpcMaybeParseProto(::grpc::ByteBuffer* src, protobuf::Message* dst)


      s.Update(errors::Internal("could not parse rpc response"));
    }
    done_(s);
    delete this;
  }

 private:
  CallOptions* call_opts_;
  // 1.
  // class CallOptions
  // tensorflow/core/distributed_runtime/call_options.h:34:
  //

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
  // 1.
  //
  // https://www.gresearch.co.uk/article/lessons-learnt-from-writing-asynchronous-streaming-grpc-services-in-c/

  ::grpc::GenericStub* stub_;
  ::grpc::string method_;
  bool fail_fast_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_STATE_H_
