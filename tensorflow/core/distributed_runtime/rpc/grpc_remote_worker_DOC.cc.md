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

这个文件构建 channel to the target worker.
Next, issue grpc call to call remote function.


* remote worker 是调用的接口, 因为 grpc channel 已经构建好了.















#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.h"

#include <utility>

#include "grpcpp/generic/generic_stub.h"
#include "grpcpp/grpcpp.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
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
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

class GrpcRemoteWorker : public WorkerInterface {
 public:


  调用实例:
  return new GrpcRemoteWorker(std::move(channel), completion_queue,
                              callback_threadpool, logger, target);



  explicit GrpcRemoteWorker(SharedGrpcChannelPtr channel,
                            ::grpc::CompletionQueue* completion_queue,
                            thread::ThreadPool* callback_threadpool,
                            WorkerCacheLogger* logger, const string& target)
      : channel_(std::move(channel)),
        stub_(channel_),
        cq_(completion_queue),
        callback_threadpool_(callback_threadpool),
        getstatus_(Method(GrpcWorkerMethod::kGetStatus)),
        createworkersession_(Method(GrpcWorkerMethod::kCreateWorkerSession)),
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
        markrecvfinished_(Method(GrpcWorkerMethod::kMarkRecvFinished)),
        logger_(logger),
        target_(target) {}

  ~GrpcRemoteWorker() override {}

  void GetStatusAsync(CallOptions* call_opts, const GetStatusRequest* request,
                      GetStatusResponse* response, bool fail_fast,
                      StatusCallback done) override {
    IssueRequest(request, response, getstatus_, std::move(done), call_opts,
                 fail_fast);
  }

* GetStatusAsync callstack:

 0# tensorflow::GrpcRemoteWorker::GetStatusAsync(tensorflow::CallOptions*, tensorflow::GetStatusRequest const*, tensorflow::GetStatusResponse*, bool, std::function<void (tensorflow::Status const&)>) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::CollectiveRemoteAccessDistributed::CheckPeerHealth(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, long, std::function<void (tensorflow::Status const&)> const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# TFE_CollectiveOpsCheckPeerHealth in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 3# pybind11::cpp_function::initialize<pybind11_init__pywrap_tfe(pybind11::module_&)::{lambda(pybind11::handle const&, char const*, long)#89}, void, pybind11::handle const&, char const*, long, pybind11::name, pybind11::scope, pybind11::sibling>(pybind11_init__pywrap_tfe(pybind11::module_&)::{lambda(pybind11::handle const&, char const*, long)#89}&&, void (*)(pybind11::handle const&, char const*, long), pybind11::name const&, pybind11::scope const&, pybind11::sibling const&)::{lambda(pybind11::detail::function_call&)#3}::_FUN(pybind11::detail::function_call) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tfe.so
 4# pybind11::cpp_function::dispatcher(_object*, _object*, _object*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tfe.so
 5# PyCFunction_Call at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:772
 6# _PyObject_MakeTpCall at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:159
 7# _PyEval_EvalFrameDefault at /tmp/build/80754af9/python_1599203911753/work/Python/ceval.c:3469
 8# _PyEval_EvalCodeWithName at /tmp/build/80754af9/python_1599203911753/work/Python/ceval.c:4308
 9# method_vectorcall at /tmp/build/80754af9/python_1599203911753/work/Objects/classobject.c:60
10# _PyEval_EvalFrameDefault.cold.2790 at /tmp/build/80754af9/python_1599203911753/work/Python/ceval.c:3515
11# _PyFunction_Vectorcall at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:410
12# method_vectorcall at /tmp/build/80754af9/python_1599203911753/work/Objects/classobject.c:67
13# PyVectorcall_Call at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:200
14# _PyEval_EvalFrameDefault at /tmp/build/80754af9/python_1599203911753/work/Python/ceval.c:3559
15# _PyFunction_Vectorcall at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:410
16# _PyEval_EvalFrameDefault.cold.2790 at /tmp/build/80754af9/python_1599203911753/work/Python/ceval.c:3486
17# _PyFunction_Vectorcall at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:410
18# _PyEval_EvalFrameDefault.cold.2790 at /tmp/build/80754af9/python_1599203911753/work/Python/ceval.c:3486
19# _PyFunction_Vectorcall at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:410
20# method_vectorcall at /tmp/build/80754af9/python_1599203911753/work/Objects/classobject.c:67
21# PyVectorcall_Call at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:200
22# t_bootstrap at /tmp/build/80754af9/python_1599203911753/work/Modules/_threadmodule.c:1003
23# pythread_wrapper at /tmp/build/80754af9/python_1599203911753/work/Python/thread_pthread.h:234
24# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
25# clone in /lib/x86_64-linux-gnu/libc.so.6















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
    int64_t start_usec = Env::Default()->NowMicros();
    // Type-specialized logging for this method.
    bool logging_active = logger_->LoggingActive() || VLOG_IS_ON(2);

    auto callback = [this, request, response, done, start_usec,
                     logging_active](Status s) {
      if (logging_active) {
        if (logger_->LoggingActive()) {
          int64_t end_usec = Env::Default()->NowMicros();
          int64_t step_id = request->step_id();
          RecvBufRespExtra extra;
          response->transport_options().UnpackTo(&extra);
          int64_t num_bytes = 0;
          for (const auto& chunk : extra.tensor_content()) {
            num_bytes += chunk.size();
          }
          int64_t send_start_usec = start_usec;
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
      }

      // Note done() can delete this worker object, so we need to call done()
      // last.
      if (response->require_ack()) {
        IssueMarkRecvFinishedRequest(request->request_id());
      }
      done(s);
    };

    IssueRequest(request, response, recvbuf_, callback, call_opts);
  }

  void CompleteGroupAsync(CallOptions* call_opts,
                          const CompleteGroupRequest* request,
                          CompleteGroupResponse* response,
                          StatusCallback done) override {
    IssueRequest(request, response, completegroup_, std::move(done), call_opts,
                 /*fail_fast=*/false);
  }

  void CompleteInstanceAsync(CallOptions* call_opts,
                             const CompleteInstanceRequest* request,
                             CompleteInstanceResponse* response,
                             StatusCallback done) override {
    IssueRequest(request, response, instancesource_, std::move(done),
                 call_opts);
  }

  void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                            GetStepSequenceResponse* response,
                            StatusCallback done) override {
    IssueRequest(request, response, getstepsequence_, std::move(done));
  }

  void RecvTensorAsync(CallOptions* call_opts, const RecvTensorRequest* request,
                       TensorResponse* response, StatusCallback done) override {
    VLOG(1) << "RecvTensorAsync req: " << request->DebugString();
    int64_t start_usec = Env::Default()->NowMicros();
    // Type-specialized logging for this method.
    bool logging_active = logger_->LoggingActive() || VLOG_IS_ON(2);

    auto callback = [this, request, response, done, start_usec,
                     logging_active](Status s) {
      if (logging_active) {
        if (logger_->LoggingActive()) {
          int64_t end_usec = Env::Default()->NowMicros();
          int64_t step_id = request->step_id();
          int64_t bytes = response->tensor().TotalBytes();
          int64_t send_start_usec = start_usec;
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
      }

      // Note done() can delete this worker object, so we need to call done()
      // last.
      if (response->metadata().require_ack()) {
        IssueMarkRecvFinishedRequest(request->request_id());
      }
      done(s);
    };

    IssueRequest(request, response, recvtensor_, callback, call_opts);
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
  // Utility method for issuing a generic asynchronous request. The
  // given callback, `done`, will be called when the RPC completes.
  void IssueRequest(const protobuf::Message* request,
                    protobuf::Message* response, const ::grpc::string& method,
                    StatusCallback done, CallOptions* call_opts = nullptr,
                    bool fail_fast = true) {
    new RPCState<protobuf::Message>(
        &stub_, cq_, method, *request, response, std::move(done), call_opts,
        callback_threadpool_, MaxRetries(), fail_fast, &target_);
  }

  void IssueRequest(const protobuf::Message* request, TensorResponse* response,
                    const ::grpc::string& method, StatusCallback done,
                    CallOptions* call_opts = nullptr) {
    new RPCState<TensorResponse>(&stub_, cq_, method, *request, response,
                                 std::move(done), call_opts,
                                 callback_threadpool_, MaxRetries(),
                                 /*fail_fast=*/true, &target_);
  }

  void IssueMarkRecvFinishedRequest(int64_t request_id) {
    VLOG(2) << "Send MarkRecvFinishedRequest for request " << request_id;
    MarkRecvFinishedRequest request;
    request.set_request_id(request_id);

    MarkRecvFinishedResponse* response = new MarkRecvFinishedResponse();
    auto done = [response](Status status) { delete response; };
    IssueRequest(&request, response, markrecvfinished_, done);
  }

  // Helper function for initializing the RpcMethod objects below.
  const char* Method(GrpcWorkerMethod id) { return GrpcWorkerMethodName(id); }

  // Helper function for configuring max GRPC retries. Defaults to 0 (no
  // retries).
  const int64_t MaxRetries() {
    int64_t max_retries = -1;
    TF_CHECK_OK(ReadInt64FromEnvVar("GRPC_MAX_RETRIES", 0, &max_retries));
    return max_retries;
  }

  SharedGrpcChannelPtr channel_;
  ::grpc::GenericStub stub_;
  ::grpc::CompletionQueue* cq_;
  thread::ThreadPool* callback_threadpool_;

  const ::grpc::string getstatus_;
  const ::grpc::string createworkersession_;
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
  const ::grpc::string markrecvfinished_;

  // Support for logging.
  WorkerCacheLogger* logger_;
  const string target_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcRemoteWorker);
};




WorkerInterface* NewGrpcRemoteWorker(SharedGrpcChannelPtr channel,
                                     ::grpc::CompletionQueue* completion_queue,
                                     thread::ThreadPool* callback_threadpool,
                                     WorkerCacheLogger* logger,
                                     const string& target) {
  return new GrpcRemoteWorker(std::move(channel), completion_queue,
                              callback_threadpool, logger, target);
}

调用栈:

 0# tensorflow::GrpcWorkerMethodName(tensorflow::GrpcWorkerMethod) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::GrpcRemoteWorker::Method(tensorflow::GrpcWorkerMethod) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::GrpcRemoteWorker::GrpcRemoteWorker(std::shared_ptr<grpc_impl::Channel>, grpc_impl::CompletionQueue*, tensorflow::thread::ThreadPool*, tensorflow::WorkerCacheLogger*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so

 3# tensorflow::NewGrpcRemoteWorker(std::shared_ptr<grpc_impl::Channel>, grpc_impl::CompletionQueue*, tensorflow::thread::ThreadPool*, tensorflow::WorkerCacheLogger*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so

 4# tensorflow::(anonymous namespace)::GrpcWorkerCache::GetOrCreateWorker(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 5# tensorflow::CancellableCall::CancellableCall(tensorflow::CancellationManager*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tensorflow::WorkerCacheInterface*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 6# tensorflow::CollectiveParamResolverDistributed::CompleteGroupDistributed(tensorflow::DeviceAttributes const&, tensorflow::CollGroupParams*, tensorflow::CancellationManager*, std::function<void (tensorflow::Status const&)> const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 7# tensorflow::CollectiveParamResolverDistributed::CompleteParamsAsync(tensorflow::DeviceAttributes const&, tensorflow::CollectiveParams*, tensorflow::CancellationManager*, std::function<void (tensorflow::Status const&)> const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 8# tensorflow::BaseCollectiveExecutor::CompleteParamsAsync(tensorflow::DeviceAttributes const&, tensorflow::CollectiveParams*, tensorflow::CancellationManager*, std::function<void (tensorflow::Status const&)>) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 9# tensorflow::(anonymous namespace)::CollectiveOpV2Kernel::Run(tensorflow::OpKernelContext*, tensorflow::CollectiveParams*, std::function<void ()>)::{lambda()#1}::operator()() const in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
10# tensorflow::UnboundedWorkQueue::PooledThreadFunc() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
11# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
12# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
13# clone in /lib/x86_64-linux-gnu/libc.so.6


* SharedGrpcChannelPtr channel
    typedef std::shared_ptr<::grpc::Channel> tensorflow::SharedGrpcChannelPtr
    tensorflow/core/distributed_runtime/rpc/grpc_util.h

* ::grpc::CompletionQueue
    A thin wrapper around grpc_completion_queue (see src/core/lib/surface/completion_queue.h).
    See C++ Performance Notes for notes on best practices for high performance servers.

* new GrpcRemoteWorker 在本文件内, 文件的上面.




}  // namespace tensorflow
