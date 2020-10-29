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

// file notes

// 如何看代码?
// 一个文件一个文件看懂.

// 1.
// 这是最最底层了哦
//
// #0  tensorflow::GrpcRemoteWorker::GrpcRemoteWorker (this=0x7f1d0801bd30, channel=std::shared_ptr<grpc::Channel> (use count 4, weak count 1) = {...}, completion_queue=0x55ab5e705db0, callback_threadpool=0x55ab5e707870, logger=0x55ab5e705cd0) at tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc:69
// #1  0x00007f21daddbb19 in tensorflow::NewGrpcRemoteWorker (channel=std::shared_ptr<grpc::Channel> (empty) = {...}, completion_queue=0x55ab5e705db0, callback_threadpool=0x55ab5e707870, logger=0x55ab5e705cd0) at tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc:314
// #2  0x00007f21dadce250 in tensorflow::(anonymous namespace)::GrpcWorkerCache::CreateWorker (this=0x55ab5e705c60, target="/job:ps/replica:0/task:0") at tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc:80
// #3  0x00007f21dae28566 in tensorflow::NewRemoteDevices(tensorflow::Env*, tensorflow::WorkerCacheInterface*, std::string const&, std::function<void (tensorflow::Status const&, std::vector<tensorflow::Device*, std::allocator<tensorflow::Device*> >*)>) (env=0x55ab5b737820, worker_cache=0x55ab5e705c60, worker_name="/job:ps/replica:0/task:0", done=...) at tensorflow/core/distributed_runtime/remote_device.cc:59
// #4  0x00007f21d69d736f in tensorflow::DeviceFinder::Start (this=0x7f1d11ffa6e0) at tensorflow/core/distributed_runtime/master.cc:249
// #5  0x00007f21d69d5d9f in tensorflow::DeviceFinder::GetRemoteDevices (device_filters=..., env=0x55ab5cfc9b68, worker_cache=0x55ab5e705c60, out_remote=0x7f1d080014f0) at tensorflow/core/distributed_runtime/master.cc:140
// #6  0x00007f21d69d895c in tensorflow::Master::<lambda()>::operator()(void) const (__closure=0x55ab5e9b56c0) at tensorflow/core/distributed_runtime/master.cc:444
// #7  0x00007f21d69df4d8 in std::_Function_handler<void(), tensorflow::Master::CreateSession(const tensorflow::CreateSessionRequest*, tensorflow::CreateSessionResponse*, tensorflow::Master::MyClosure)::<lambda()> >::_M_invoke(const std::_Any_data &) (__functor=...) at /usr/include/c++/7/bits/std_function.h:316

// 2.
// master 会找所有 cluster 内的 devices
//


#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.h"

#include <utility>

#include "grpcpp/generic/generic_stub.h"
#include "grpcpp/grpcpp.h"

#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/transport_options.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

const int kMaxWorkerRpcRetries = 10;

class GrpcRemoteWorker : public WorkerInterface {
// 1.
// tutorial: 看 client 的代码!!! 这里应该是 client !!!
// https://grpc.io/docs/languages/cpp/async/
// https://grpc.io/docs/languages/cpp/basics/#starting-the-server

 public:
  explicit GrpcRemoteWorker(SharedGrpcChannelPtr channel,
                            ::grpc::CompletionQueue* completion_queue,
                            thread::ThreadPool* callback_threadpool,
                            WorkerCacheLogger* logger)
      : channel_(std::move(channel)),
        stub_(channel_),
        cq_(completion_queue),
        callback_threadpool_(callback_threadpool),
        getstatus_(Method(GrpcWorkerMethod::kGetStatus)),
        // 1.
        // getstatus_("/tensorflow.WorkerService/GetStatus")

        createworkersession_(Method(GrpcWorkerMethod::kCreateWorkerSession)),
        // 2.
        //

        deleteworkersession_(Method(GrpcWorkerMethod::kDeleteWorkerSession)),
        registergraph_(Method(GrpcWorkerMethod::kRegisterGraph)),
        deregistergraph_(Method(GrpcWorkerMethod::kDeregisterGraph)),
        rungraph_(Method(GrpcWorkerMethod::kRunGraph)),
        cleanupgraph_(Method(GrpcWorkerMethod::kCleanupGraph)),
        cleanupall_(Method(GrpcWorkerMethod::kCleanupAll)),
        recvtensor_(Method(GrpcWorkerMethod::kRecvTensor)),
        recvbuf_(Method(GrpcWorkerMethod::kRecvBuf)),
        logging_(Method(GrpcWorkerMethod::kLogging)),
        tracing_(Method(GrpcWorkerMethod::kTracing)),
        completegroup_(Method(GrpcWorkerMethod::kCompleteGroup)),
        instancesource_(Method(GrpcWorkerMethod::kCompleteInstance)),
        getstepsequence_(Method(GrpcWorkerMethod::kGetStepSequence)),
        logger_(logger) {}

  ~GrpcRemoteWorker() override {}

  /** \brief GetStatusAsync mainly get the device attributes for managing remote
   *         devices by issuing a grpc call from the client side with request
   *         message and get the response message fromt the server side.
   *
   *  \param request: const GetStatusRequest* ;
   *         message GetStatusRequest contains nothing.
   *
   *  \param response: GetStatusResponse* ;
   *         message GetStatusResponse contains DeviceAttributes.
   *
   *  \param done: StatusCallback;
   *         A lambda function.
   *
   *  \remark No return value.
   */
  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response,
                      StatusCallback done) override {
    IssueRequest(request, response, getstatus_, std::move(done));
    // 1.
    // 这个谁接 Request 呢?
    // ...
  }

  void CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request,
                                CreateWorkerSessionResponse* response,
                                StatusCallback done) override {
    IssueRequest(request, response, createworkersession_, std::move(done));
  }

  void DeleteWorkerSessionAsync(CallOptions* call_opts,
                                const DeleteWorkerSessionRequest* request,
                                DeleteWorkerSessionResponse* response,
                                StatusCallback done) override {
    IssueRequest(request, response, deleteworkersession_, std::move(done),
                 call_opts);
  }

  void RegisterGraphAsync(const RegisterGraphRequest* request,
                          RegisterGraphResponse* response,
                          StatusCallback done) override {
    IssueRequest(request, response, registergraph_, std::move(done));
  }

  void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                            DeregisterGraphResponse* response,
                            StatusCallback done) override {
    IssueRequest(request, response, deregistergraph_, std::move(done));
  }

  void RunGraphAsync(CallOptions* call_opts, const RunGraphRequest* request,
                     RunGraphResponse* response, StatusCallback done) override {
    IssueRequest(request, response, rungraph_, std::move(done), call_opts);
  }
  void RunGraphAsync(CallOptions* call_opts, RunGraphRequestWrapper* request,
                     MutableRunGraphResponseWrapper* response,
                     StatusCallback done) override {
    IssueRequest(&request->ToProto(), get_proto_from_wrapper(response),
                 rungraph_, std::move(done), call_opts);
  }

  void CleanupGraphAsync(const CleanupGraphRequest* request,
                         CleanupGraphResponse* response,
                         StatusCallback done) override {
    IssueRequest(request, response, cleanupgraph_, std::move(done));
  }

  void CleanupAllAsync(const CleanupAllRequest* request,
                       CleanupAllResponse* response,
                       StatusCallback done) override {
    IssueRequest(request, response, cleanupall_, std::move(done));
  }

  void RecvBufAsync(CallOptions* call_opts, const RecvBufRequest* request,
                    RecvBufResponse* response, StatusCallback done) override {
    int64 start_usec = Env::Default()->NowMicros();
    // Type-specialized logging for this method.
    bool logging_active = logger_->LoggingActive() || VLOG_IS_ON(2);
    StatusCallback wrapper_done;
    const StatusCallback* cb_to_use;
    if (!logging_active) {
      cb_to_use = &done;  // No additional work to do, so just use done directly
    } else {
      wrapper_done = [this, request, response, done, start_usec](Status s) {
        if (logger_->LoggingActive()) {
          int64 end_usec = Env::Default()->NowMicros();
          int64 step_id = request->step_id();
          RecvBufRespExtra extra;
          response->transport_options().UnpackTo(&extra);
          int64 num_bytes = 0;
          for (const auto& chunk : extra.tensor_content()) {
            num_bytes += chunk.size();
          }
          int64 send_start_usec = start_usec;
          // Prefer start time reported by the sender, if available.
          if (response->send_start_micros()) {
            send_start_usec = std::max(
                start_usec, static_cast<int64>(response->send_start_micros()));
            send_start_usec = std::min(send_start_usec, end_usec - 1);
          }
          const string& key = request->buf_rendezvous_key();
          logger_->RecordDataTransfer(
              step_id, send_start_usec, end_usec, key, request->src_device(),
              request->dst_device(), num_bytes, "", "RecvBuf");
        }
        VLOG(2) << "done callback, req: " << request->DebugString()
                << " response " << response->DebugString();
        done(s);
      };
      cb_to_use = &wrapper_done;
    }

    IssueRequest(request, response, recvbuf_, *cb_to_use, call_opts);
  }

  void CompleteGroupAsync(CallOptions* call_opts,
                          const CompleteGroupRequest* request,
                          CompleteGroupResponse* response,
                          StatusCallback done) override {
    IssueRequest(request, response, completegroup_, std::move(done), call_opts);
  }

  void CompleteInstanceAsync(CallOptions* call_opts,
                             const CompleteInstanceRequest* request,
                             CompleteInstanceResponse* response,
                             StatusCallback done) override {
    IssueRequest(request, response, instancesource_, std::move(done),
                 call_opts);
  }


  void GetStepSequenceAsync(
    const GetStepSequenceRequest* request,
    GetStepSequenceResponse* response,
    StatusCallback done) override {

    IssueRequest(request, response, getstepsequence_, std::move(done));

  }

  // =======================================================================

  void RecvTensorAsync(
    CallOptions* call_opts,
    const RecvTensorRequest* request,
    TensorResponse* response,
    StatusCallback done) override {

    VLOG(1) << "RecvTensorAsync req: " << request->DebugString();

    int64 start_usec = Env::Default()->NowMicros();

    // Type-specialized logging for this method.
    bool logging_active = logger_->LoggingActive() || VLOG_IS_ON(2);

    StatusCallback wrapper_done;
    const StatusCallback* cb_to_use;

    if (!logging_active) {
      cb_to_use = &done;  // No additional work to do, so just use done directly
    } else {
      wrapper_done = [this, request, response, done, start_usec](Status s) {
        if (logger_->LoggingActive()) {
          int64 end_usec = Env::Default()->NowMicros();
          int64 step_id = request->step_id();
          int64 bytes = response->tensor().TotalBytes();
          int64 send_start_usec = start_usec;
          // If a send start time was reported by the other side, use
          // that instead.  Maybe we should mark the display if we're using
          // our local time instead of the remote start time?
          if (response->metadata().send_start_micros()) {
            // send_start_micros is the timestamp taken when the
            // remote machine began to send the RecvTensor response.
            // Due to clock skew between source and dest machines, it
            // is possible that send_start_micros can be larger than
            // end_usec or less than start_usec.
            //
            // To respect causality, we enforce the invariants that
            // the RecvTensor response can not have been sent before
            // the RecvTensor request, and must have been sent before
            // it was received.
            send_start_usec = std::max(
                start_usec,
                static_cast<int64>(response->metadata().send_start_micros()));
            send_start_usec = std::min(send_start_usec, end_usec - 1);
          }

          const string& key = request->rendezvous_key();

          std::vector<string> key_parts = str_util::Split(key, ';');
          if (key_parts.size() != 5) {
            LOG(WARNING) << "Bad key: " << key;
          } else {
            logger_->RecordRecvTensor(step_id, send_start_usec, end_usec,
                                      key_parts[3],  // tensor name
                                      key_parts[0],  // src_device
                                      key_parts[2],  // dst_device
                                      bytes);
          }
        }

        VLOG(2) << "done callback, req: " << request->DebugString()
                << " response " << response->metadata().DebugString();
        done(s);
      };

      cb_to_use = &wrapper_done;
    }

    IssueRequest(request, response, recvtensor_, *cb_to_use, call_opts);
    // 1.
    // ps 的调用栈主要是这个函数.
    //
    // Thread #247 [python] 62873 [core: 12] (Suspended : Breakpoint)
    // 	tensorflow::RPCState<tensorflow::TensorResponse>::RPCState<google::protobuf::Message>() at grpc_state.h:65 0x7f0fbaddc456
    // 	tensorflow::RPCState<tensorflow::TensorResponse>::RPCState() at grpc_state.h:48 0x7f0fbaddc00a
    // 	tensorflow::GrpcRemoteWorker::IssueRequest() at grpc_remote_worker.cc:276 0x7f0fbaddba3a
    //**tensorflow::GrpcRemoteWorker::RecvTensorAsync() at grpc_remote_worker.cc:246 0x7f0fbaddb680 ⭕️ ✅
    // 	tensorflow::(anonymous namespace)::RpcRecvTensorCall::StartRTCall at rpc_rendezvous_mgr.cc:151 0x7f0fbae23279
    // 	tensorflow::(anonymous namespace)::RpcRecvTensorCall::Start at rpc_rendezvous_mgr.cc:102 0x7f0fbae22e8d
    // 	tensorflow::(anonymous namespace)::RpcRemoteRendezvous::RecvFromRemoteAsync at rpc_rendezvous_mgr.cc:272 0x7f0fbae23ce0
    // 	tensorflow::BaseRemoteRendezvous::RecvAsync() at base_rendezvous_mgr.cc:330 0x7f0fbae5cc1b
    // 	tensorflow::RecvOp::ComputeAsync() at sendrecv_ops.cc:183 0x7f0fbe617379
    // 	tensorflow::Device::ComputeAsync() at device.h:95 0x7f0fbae27798
    // 	<...more frames...>

  }

  void LoggingAsync(const LoggingRequest* request, LoggingResponse* response,
                    StatusCallback done) override {
    IssueRequest(request, response, logging_, done);
  }

  void TracingAsync(const TracingRequest* request, TracingResponse* response,
                    StatusCallback done) override {
    IssueRequest(request, response, tracing_, done);
  }

 private:

  // ===========================================
  // Send request utility function
  // ===========================================

  // Utility method for issuing a generic asynchronous request. The
  // given callback, `done`, will be called when the RPC completes.
  void IssueRequest(
    const protobuf::Message* request,
    protobuf::Message* response,
    const ::grpc::string& method,
    StatusCallback done,
    CallOptions* call_opts = nullptr,
    int max_retries = kMaxWorkerRpcRetries) {
  // 1.
  // protobuf::Message

    new RPCState<protobuf::Message>(
      &stub_,
      cq_,
      method,
      *request,
      response,
      std::move(done),
      call_opts,
      callback_threadpool_,
      max_retries);
    // 1.
    // RPCState 是什么?
    // tensorflow/core/distributed_runtime/rpc/grpc_state.h:39:
    // class RPCState : public GrpcClientCQTag
    //

    // 2.
    // GrpcClientCQTag 是什么
    // tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h:29:
    // class GrpcClientCQTag
  }

  /** \brief Indirectly start a grpc call from the client
   *         side with request message and get the response message fromt the
   *         server side.
   *
   *  \param[in] request: const protobuf::Message* ;
   *         class of protocol message.
   *         For example, message GetStatusRequest.
   *
   *  \param[out] response: TensorResponse* ;
   *         class TensorResponse is defined in tensor_coding.h. It efficiently
   *         decodes the incoming data into Tensor contents as well as
   *         associated metadata.
   *
   *  \param[in] method: const ::grpc::string& ;
   *
   *  \param[in] done: StatusCallback;
   *
   *  \param[in] call_opts: CallOptions*, default is nullptr;
   *
   *  \param[in] max_retries: int, default is kMaxWorkerRpcRetries;
   *
   *  \remark Return no value.
   */
  void IssueRequest(
    const protobuf::Message* request,
    TensorResponse* response,
    const ::grpc::string& method,
    // 1.
    // string method 都在如下内
    // tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.cc

    StatusCallback done,
    CallOptions* call_opts = nullptr,
    int max_retries = kMaxWorkerRpcRetries) {

    new RPCState<TensorResponse>(
      &stub_,
      cq_,
      method,
      *request,
      response,
      std::move(done),
      call_opts,
      callback_threadpool_,
      max_retries);
  }
  // 1.
  // Q.不过我也不知道是谁 handle 了这些 Request.
  //
  // A.
  // see https://owenzhu.github.io/2017/12/19/291217/
  //
  // HandleRPCsLoop
  //
  // Status GrpcServer::Start() {
  //   mutex_lock l(mu_);
  //   switch (state_) {
  //     case NEW: {
  //       master_thread_.reset(
  //           env_->StartThread(ThreadOptions(), "TF_master_service",
  //                             [this] { master_service_->HandleRPCsLoop(); }));
  //       worker_thread_.reset(
  //           env_->StartThread(ThreadOptions(), "TF_worker_service",
  //                             [this] { worker_service_->HandleRPCsLoop(); }));
  //   ...
  // }

  // Helper function for initializing the RpcMethod objects below.
  const char* Method(GrpcWorkerMethod id) {
    return GrpcWorkerMethodName(id);
    // 1.
    // what actually returned!
    //
    // tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.cc
  }

  SharedGrpcChannelPtr channel_;
  ::grpc::GenericStub stub_;
  // 1.
  // ::grpc::GenericStub 究竟是什么?
  // 示意图:
  // https://www.geeksforgeeks.org/remote-procedure-call-rpc-in-operating-system/
  //
  // 描述:
  // Stub: The function of the stub is to provide transparency to the programmer-written application code.
  // On the client side, the stub handles the interface between the client’s local procedure call and the run-time system, marshaling and unmarshaling data, invoking the RPC run-time protocol, and if requested, carrying out some of the binding steps.
  // On the server side, the stub provides a similar interface between the run-time system and the local manager procedures that are executed by the server.

  // 2.
  // 阅读资料:
  // https://en.wikipedia.org/wiki/Stub_(distributed_computing)

  // 3.
  // what is a stub?
  // A stub in distributed computing is a piece of code that
  // converts parameters passed between client and server during
  // a remote procedure call (RPC). The main idea of an RPC is to
  // allow a local computer (client) to remotely call procedures
  // on a different computer (server).

  ::grpc::CompletionQueue* cq_;
  // 1.
  // cq_ : ::grpc::CompletionQueue* 是干嘛的?
  // call CompletionQueue::Next to wait for operations to complete. If a tag appears, it indicates that the corresponding operation is complete.

  thread::ThreadPool* callback_threadpool_;

  const ::grpc::string getstatus_;
  const ::grpc::string createworkersession_;
  // 1.
  // 值是 string , 在下面
  // tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.cc
  // "/tensorflow.WorkerService/GetStatus"

  const ::grpc::string deleteworkersession_;
  const ::grpc::string registergraph_;
  const ::grpc::string deregistergraph_;
  const ::grpc::string rungraph_;
  const ::grpc::string cleanupgraph_;
  const ::grpc::string cleanupall_;
  const ::grpc::string recvtensor_;
  const ::grpc::string recvbuf_;
  const ::grpc::string logging_;
  const ::grpc::string tracing_;
  const ::grpc::string completegroup_;
  const ::grpc::string instancesource_;
  const ::grpc::string getstepsequence_;

  // Support for logging.
  WorkerCacheLogger* logger_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcRemoteWorker);
}; // end of class GrpcRemoteWorker


/** \brief Create a remote worker and initialize their grpc services.
 *
 *  \param channel: SharedGrpcChannelPtr ;
 *         std::shared_ptr<::grpc::Channel> ; a channel pointer to the target.
 *
 *  \param completion_queue: ::grpc::CompletionQueue* ;
 *         Completion Queues enable notification of the completion of
 *         asynchronous actions.
 *
 *  \param callback_threadpool: thread::ThreadPool* ;
 *         CPU threads.
 *
 *  \param logger: WorkerCacheLogger ;
 *         WorkerCacheLogger is a thread-safe utility for use by a WorkerCache
 *         to optionally log some selected RPC activity.  A single instance
 *         should be owned by a WorkerCache, for use by its RemoteWorker
 *         instances.
 *
 *  \return WorkerInterface* ;
 */
WorkerInterface* NewGrpcRemoteWorker(SharedGrpcChannelPtr channel,
                                     ::grpc::CompletionQueue* completion_queue,
                                     thread::ThreadPool* callback_threadpool,
                                     WorkerCacheLogger* logger) {
  return new GrpcRemoteWorker(std::move(channel),
                              completion_queue,
                              callback_threadpool,
                              logger);
}

}  // namespace tensorflow
