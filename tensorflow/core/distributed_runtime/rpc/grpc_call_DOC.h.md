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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CALL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CALL_H_

#include "grpcpp/completion_queue.h"
#include "grpcpp/impl/service_type.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/async_stream.h"
#include "grpcpp/support/async_unary_call.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

#include "tensorflow/core/util/write_log.h"
#include <boost/stacktrace.hpp>
#define BOOST_STACKTRACE_USE_ADDR2LINE

namespace tensorflow {

// CALL STRUCTURES
// ===============
//
// Each pending (incoming) request corresponds to a call object that
// encapsulates the state of the call. Templates and
// pointers-to-member functions are used to avoid boilerplate and
// redundant closure creation. The class hierarchy is as follows:
//
// * `UntypedCall<Service>`: The base class represents a call that
//   could be associated with any of the methods on a service of type
//   `Service`. Also defines a `Tag` nested class that can be used as
//   the tag in a `grpc::CompletionQueue`.  Each class that
//   instantiates `Service` should have a completion queue polling
//   loop that knows about `UntypedCall<Service>::Tag` objects, and
//   invokes their `OnCompleted()` method to continue processing.
//
// * `Call<Service, GrpcService, Req, Resp>`: This class extends
//   `UntypedCall<Service>` and is additionally parameterized by the
//   gRPC-generated asynchronous service class, and the request and
//   response message types. It defines the state associated with a
//   call (whose type depends on the message types), and stores a
//   pointer to a `Service::HandleFoo()` handler method. Each
//   `Service::HandleFoo()` method knows about the corresponding
//   `Call` type, in order to access its state, and invoke its
//   `SendResponse()` method.
//
// The lifecycle of a call object is as follows.
//
// 1. A `Service` creates a `Call` for a particular method and
//    enqueues it in its completion queue (via an
//    `UntypedCall<Service>::Tag`).
//
// 2. When the tag is returned from `cq_->Next()`, the
//    `UntypedCall::RequestReceived()` method is invoked and takes
//    ownership of the call object. This indirectly invokes the
//    appropriate handler method on `Service`.
//
// 3. After the response has been written (perhaps in another thread),
//    the `Call::SendResponse()` method is invoked. It transfers
//    ownership of the call object back to the completion queue (via
//    an `UntypedCall::Tag`).
//
// 4. When the response has been sent, the tag is returned from
//    `cq_->Next()`, and the call object is deleted.
//


GrpcCallTag

 0# tensorflow::GrpcCallTag<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread>::~GrpcCallTag() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::Call<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread, tensorflow::grpc::WorkerService::AsyncService, tensorflow::CompleteGroupRequest, tensorflow::CompleteGroupResponse>::~Call() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::core::RefCounted::Unref() const in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 3# tensorflow::(anonymous namespace)::GrpcWorkerServiceThread::HandleRPCsLoop() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 4# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 5# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 6# clone in /lib/x86_64-linux-gnu/libc.so.6


template <class Service>
class GrpcCallTag {
 public:
  virtual ~GrpcCallTag() {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
  }

  // Calls the callback associated with this tag.
  virtual void OnCompleted(Service* service, bool ok) = 0;
};









UntypedCall

 0# tensorflow::UntypedCall<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread>::Tag::Tag(tensorflow::UntypedCall<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread>*, tensorflow::UntypedCall<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread>::Tag::Callback) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::(anonymous namespace)::GrpcWorkerServiceThread::HandleRPCsLoop() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 3# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 4# clone in /lib/x86_64-linux-gnu/libc.so.6


// Represents a pending request with unknown message types.
template <class Service>
class UntypedCall : public core::RefCounted {
 public:
  virtual ~UntypedCall() {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
  }

  // The implementation of this method should use `service` to handle
  // an incoming request, and (perhaps asynchronously) send the
  // response.
  //
  // One reference on `this` is transferred to the callee, and the
  // callee is responsible for releasing it (typically via
  // `Call::SendResponse()`).
  //
  // `ok` is true if the request was received in a "regular event",
  // otherwise false.
  virtual void RequestReceived(Service* service, bool ok) = 0;

RequestReceived:
 0# tensorflow::Call<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread, tensorflow::grpc::WorkerService::AsyncService, tensorflow::CompleteGroupRequest, tensorflow::CompleteGroupResponse>::RequestReceived(tensorflow::(anonymous namespace)::GrpcWorkerServiceThread*, bool) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::(anonymous namespace)::GrpcWorkerServiceThread::HandleRPCsLoop() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 3# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 4# clone in /lib/x86_64-linux-gnu/libc.so.6



  // This method will be called either (i) when the server is notified
  // that the request has been canceled, or (ii) when the request completes
  // normally. The implementation should distinguish these cases by querying
  // the `grpc::ServerContext` associated with the request.
  virtual void RequestCancelled(Service* service, bool ok) = 0;

 0# tensorflow::Call<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread, tensorflow::grpc::WorkerService::AsyncService, tensorflow::CompleteGroupRequest, tensorflow::CompleteGroupResponse>::RequestCancelled(tensorflow::(anonymous namespace)::GrpcWorkerServiceThread*, bool) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::(anonymous namespace)::GrpcWorkerServiceThread::HandleRPCsLoop() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 3# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 4# clone in /lib/x86_64-linux-gnu/libc.so.6


  // Associates a tag in a `::grpc::CompletionQueue` with a callback
  // for an incoming RPC.  An active Tag owns a reference on the corresponding
  // Call object.
  class Tag : public GrpcCallTag<Service> {
   public:
    // One enum value per supported callback.
    enum Callback { kRequestReceived, kResponseSent, kCancelled };

    Tag(UntypedCall* call, Callback cb) : call_(call), callback_(cb) {
      write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    }

 0# tensorflow::UntypedCall<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread>::Tag::Tag(tensorflow::UntypedCall<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread>*, tensorflow::UntypedCall<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread>::Tag::Callback) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::Call<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread, tensorflow::grpc::WorkerService::AsyncService, tensorflow::RecvBufRequest, tensorflow::RecvBufResponse>::EnqueueRequestForMethod(tensorflow::grpc::WorkerService::AsyncService*, grpc_impl::ServerCompletionQueue*, int, void (tensorflow::(anonymous namespace)::GrpcWorkerServiceThread::*)(tensorflow::Call<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread, tensorflow::grpc::WorkerService::AsyncService, tensorflow::RecvBufRequest, tensorflow::RecvBufResponse>*), bool) [clone .constprop.765] in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::(anonymous namespace)::GrpcWorkerServiceThread::HandleRPCsLoop() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 3# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 4# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 5# clone in /lib/x86_64-linux-gnu/libc.so.6





    // Calls the callback associated with this tag.
    //
    // The callback takes ownership of `this->call_`.
    void OnCompleted(Service* service, bool ok) override {
      write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
      switch (callback_) {
        case kRequestReceived:
          call_->RequestReceived(service, ok);
          break;
        case kResponseSent:
          // No special handling needed apart from the Unref below.
          break;
        case kCancelled:
          call_->RequestCancelled(service, ok);
          break;
      }
      call_->Unref();  // Ref acquired when tag handed to grpc.
    }

 0# tensorflow::RPCState<google::protobuf::Message>::OnCompleted(bool) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# std::_Function_handler<void (), tensorflow::GrpcWorkerEnv::GrpcWorkerCacheThread::GrpcWorkerCacheThread()::{lambda()#1}>::_M_invoke(std::_Any_data const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 3# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 4# clone in /lib/x86_64-linux-gnu/libc.so.6





   private:
    UntypedCall* const call_;  // `this` owns one reference.
    Callback callback_;
  };
};

// Represents a pending call with known request and response message
// types, and a known request-handling method.
template <class Service, class GrpcService, class RequestMessage,
          class ResponseMessage>
class Call : public UntypedCall<Service> {
 public:
  // Represents the generic signature of a generated
  // `GrpcService::RequestFoo()` method, where `Foo` is the name of an
  // RPC method.
  using EnqueueFunction = void (GrpcService::*)(
      ::grpc::ServerContext*, RequestMessage*,
      ::grpc::ServerAsyncResponseWriter<ResponseMessage>*,
      ::grpc::CompletionQueue*, ::grpc::ServerCompletionQueue*, void*);

  // Represents the generic signature of a `Service::HandleFoo()`
  // method, where `Foo` is the name of an RPC method.
  using HandleRequestFunction = void (Service::*)(
      Call<Service, GrpcService, RequestMessage, ResponseMessage>*);

  Call(HandleRequestFunction handle_request_function)
      : handle_request_function_(handle_request_function), responder_(&ctx_) {
        write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
      }

Call:
 0# tensorflow::Call<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread, tensorflow::grpc::WorkerService::AsyncService, tensorflow::RecvBufRequest, tensorflow::RecvBufResponse>::EnqueueRequestForMethod(tensorflow::grpc::WorkerService::AsyncService*, grpc_impl::ServerCompletionQueue*, int, void (tensorflow::(anonymous namespace)::GrpcWorkerServiceThread::*)(tensorflow::Call<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread, tensorflow::grpc::WorkerService::AsyncService, tensorflow::RecvBufRequest, tensorflow::RecvBufResponse>*), bool) [clone .constprop.765] in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::(anonymous namespace)::GrpcWorkerServiceThread::HandleRPCsLoop() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 3# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 4# clone in /lib/x86_64-linux-gnu/libc.so.6


  virtual ~Call() {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
  }

  void RequestReceived(Service* service, bool ok) override {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    if (ok) {
      this->Ref();
      (service->*handle_request_function_)(this);
    }
  }

RequestReceived:
 0# tensorflow::Call<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread, tensorflow::grpc::WorkerService::AsyncService, tensorflow::GetStatusRequest, tensorflow::GetStatusResponse>::RequestReceived(tensorflow::(anonymous namespace)::GrpcWorkerServiceThread*, bool) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::(anonymous namespace)::GrpcWorkerServiceThread::HandleRPCsLoop() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 3# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 4# clone in /lib/x86_64-linux-gnu/libc.so.6



  void SendResponse(::grpc::Status status) {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    this->Ref();  // Ref for grpc; released in Tag callback.
    responder_.Finish(response, status, &response_sent_tag_);
    this->Unref();
  }

  void RequestCancelled(Service* service, bool ok) override {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    if (ctx_.IsCancelled()) {
      mutex_lock l(mu_);
      if (cancel_callback_) {
        cancel_callback_();
      }
    }
  }

RequestCancelled:
 0# tensorflow::Call<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread, tensorflow::grpc::WorkerService::AsyncService, tensorflow::CompleteGroupRequest, tensorflow::CompleteGroupResponse>::RequestCancelled(tensorflow::(anonymous namespace)::GrpcWorkerServiceThread*, bool) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::(anonymous namespace)::GrpcWorkerServiceThread::HandleRPCsLoop() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 3# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 4# clone in /lib/x86_64-linux-gnu/libc.so.6




  // Registers `callback` as the function that should be called if and when this
  // call is canceled by the client.
  void SetCancelCallback(std::function<void()> callback) {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    mutex_lock l(mu_);
    cancel_callback_ = std::move(callback);
  }

 0# tensorflow::CallOptions::SetCancelCallback(std::function<void ()>) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::RPCState<google::protobuf::Message>::StartCall() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::GrpcRemoteWorker::IssueRequest(google::protobuf::Message const*, google::protobuf::Message*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::function<void (tensorflow::Status const&)>, tensorflow::CallOptions*, bool) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 3# tensorflow::GrpcRemoteWorker::CompleteGroupAsync(tensorflow::CallOptions*, tensorflow::CompleteGroupRequest const*, tensorflow::CompleteGroupResponse*, std::function<void (tensorflow::Status const&)>) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 4# tensorflow::(anonymous namespace)::CompleteGroupCall::IssueCall(std::function<void (tensorflow::Status const&)> const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 5# tensorflow::CancellableCall::Start(std::function<void (tensorflow::Status const&)> const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 6# tensorflow::CollectiveParamResolverDistributed::CompleteGroupDistributed(tensorflow::DeviceAttributes const&, tensorflow::CollGroupParams*, tensorflow::CancellationManager*, std::function<void (tensorflow::Status const&)> const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 7# tensorflow::CollectiveParamResolverDistributed::CompleteParamsAsync(tensorflow::DeviceAttributes const&, tensorflow::CollectiveParams*, tensorflow::CancellationManager*, std::function<void (tensorflow::Status const&)> const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 8# tensorflow::BaseCollectiveExecutor::CompleteParamsAsync(tensorflow::DeviceAttributes const&, tensorflow::CollectiveParams*, tensorflow::CancellationManager*, std::function<void (tensorflow::Status const&)>) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 9# tensorflow::(anonymous namespace)::CollectiveOpV2Kernel::Run(tensorflow::OpKernelContext*, tensorflow::CollectiveParams*, std::function<void ()>)::{lambda()#1}::operator()() const in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
10# tensorflow::UnboundedWorkQueue::PooledThreadFunc() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
11# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
12# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
13# clone in /lib/x86_64-linux-gnu/libc.so.6


Another one:

 0# tensorflow::CallOptions::SetCancelCallback(std::function<void ()>) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::RPCState<google::protobuf::Message>::StartCall() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::GrpcRemoteWorker::IssueRequest(google::protobuf::Message const*, google::protobuf::Message*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::function<void (tensorflow::Status const&)>, tensorflow::CallOptions*, bool) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 3# tensorflow::GrpcRemoteWorker::GetStatusAsync(tensorflow::CallOptions*, tensorflow::GetStatusRequest const*, tensorflow::GetStatusResponse*, bool, std::function<void (tensorflow::Status const&)>) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 4# tensorflow::CollectiveRemoteAccessDistributed::CheckPeerHealth(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, long, std::function<void (tensorflow::Status const&)> const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 5# TFE_CollectiveOpsCheckPeerHealth in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 6# pybind11::cpp_function::initialize<pybind11_init__pywrap_tfe(pybind11::module_&)::{lambda(pybind11::handle const&, char const*, long)#89}, void, pybind11::handle const&, char const*, long, pybind11::name, pybind11::scope, pybind11::sibling>(pybind11_init__pywrap_tfe(pybind11::module_&)::{lambda(pybind11::handle const&, char const*, long)#89}&&, void (*)(pybind11::handle const&, char const*, long), pybind11::name const&, pybind11::scope const&, pybind11::sibling const&)::{lambda(pybind11::detail::function_call&)#3}::_FUN(pybind11::detail::function_call) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tfe.so
 7# pybind11::cpp_function::dispatcher(_object*, _object*, _object*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tfe.so
 8# PyCFunction_Call at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:772
 9# _PyObject_MakeTpCall at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:159
10# _PyEval_EvalFrameDefault at /tmp/build/80754af9/python_1599203911753/work/Python/ceval.c:3469
11# _PyEval_EvalCodeWithName at /tmp/build/80754af9/python_1599203911753/work/Python/ceval.c:4308
12# method_vectorcall at /tmp/build/80754af9/python_1599203911753/work/Objects/classobject.c:60
13# _PyEval_EvalFrameDefault.cold.2790 at /tmp/build/80754af9/python_1599203911753/work/Python/ceval.c:3515
14# _PyFunction_Vectorcall at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:410
15# method_vectorcall at /tmp/build/80754af9/python_1599203911753/work/Objects/classobject.c:67
16# PyVectorcall_Call at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:200
17# _PyEval_EvalFrameDefault at /tmp/build/80754af9/python_1599203911753/work/Python/ceval.c:3559
18# _PyFunction_Vectorcall at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:410
19# _PyEval_EvalFrameDefault.cold.2790 at /tmp/build/80754af9/python_1599203911753/work/Python/ceval.c:3486
20# _PyFunction_Vectorcall at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:410
21# _PyEval_EvalFrameDefault.cold.2790 at /tmp/build/80754af9/python_1599203911753/work/Python/ceval.c:3486
22# _PyFunction_Vectorcall at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:410
23# method_vectorcall at /tmp/build/80754af9/python_1599203911753/work/Objects/classobject.c:67
24# PyVectorcall_Call at /tmp/build/80754af9/python_1599203911753/work/Objects/call.c:200
25# t_bootstrap at /tmp/build/80754af9/python_1599203911753/work/Modules/_threadmodule.c:1003
26# pythread_wrapper at /tmp/build/80754af9/python_1599203911753/work/Python/thread_pthread.h:234
27# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
28# clone in /lib/x86_64-linux-gnu/libc.so.6





  // Clears any cancellation callback that has been registered for this call.
  void ClearCancelCallback() {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    mutex_lock l(mu_);
    cancel_callback_ = nullptr;
  }

 0# tensorflow::CallOptions::ClearCancelCallback() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::RPCState<google::protobuf::Message>::OnCompleted(bool) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# std::_Function_handler<void (), tensorflow::GrpcWorkerEnv::GrpcWorkerCacheThread::GrpcWorkerCacheThread()::{lambda()#1}>::_M_invoke(std::_Any_data const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 3# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 4# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 5# clone in /lib/x86_64-linux-gnu/libc.so.6




EnqueueRequest:

 0# tensorflow::Call<tensorflow::GrpcMasterService, tensorflow::grpc::MasterService::AsyncService, tensorflow::CreateSessionRequest, tensorflow::CreateSessionResponse>::EnqueueRequest(tensorflow::grpc::MasterService::AsyncService*, grpc_impl::ServerCompletionQueue*, void (tensorflow::grpc::MasterService::AsyncService::*)(grpc_impl::ServerContext*, tensorflow::CreateSessionRequest*, grpc_impl::ServerAsyncResponseWriter<tensorflow::CreateSessionResponse>*, grpc_impl::CompletionQueue*, grpc_impl::ServerCompletionQueue*, void*), void (tensorflow::GrpcMasterService::*)(tensorflow::Call<tensorflow::GrpcMasterService, tensorflow::grpc::MasterService::AsyncService, tensorflow::CreateSessionRequest, tensorflow::CreateSessionResponse>*), bool) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::GrpcMasterService::HandleRPCsLoop() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 3# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 4# clone in /lib/x86_64-linux-gnu/libc.so.6

  // Enqueues a new request for the given service on the given
  // completion queue, using the given `enqueue_function`.
  //
  // The request will be handled with the given
  // `handle_request_function`.
  static void EnqueueRequest(GrpcService* grpc_service,
                             ::grpc::ServerCompletionQueue* cq,
                             EnqueueFunction enqueue_function,
                             HandleRequestFunction handle_request_function,
                             bool supports_cancel) {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    auto call = new Call<Service, GrpcService, RequestMessage, ResponseMessage>(
        handle_request_function);
    if (supports_cancel) {
      call->RegisterCancellationHandler();
    }

    // Initial ref for call handed to grpc; released in Tag callback.
    (grpc_service->*enqueue_function)(&call->ctx_, &call->request,
                                      &call->responder_, cq, cq,
                                      &call->request_received_tag_);
  }




EnqueueRequestForMethod:

 0# tensorflow::Call<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread, tensorflow::grpc::WorkerService::AsyncService, tensorflow::GetStatusRequest, tensorflow::GetStatusResponse>::EnqueueRequestForMethod(tensorflow::grpc::WorkerService::AsyncService*, grpc_impl::ServerCompletionQueue*, int, void (tensorflow::(anonymous namespace)::GrpcWorkerServiceThread::*)(tensorflow::Call<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread, tensorflow::grpc::WorkerService::AsyncService, tensorflow::GetStatusRequest, tensorflow::GetStatusResponse>*), bool) [clone .constprop.743] in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::(anonymous namespace)::GrpcWorkerServiceThread::HandleRPCsLoop() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 3# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 4# clone in /lib/x86_64-linux-gnu/libc.so.6

  // Enqueues a new request for the given service on the given
  // completion queue, using the given `method_id`.
  //
  // The request will be handled with the given
  // `handle_request_function`.
  static void EnqueueRequestForMethod(
      GrpcService* grpc_service, ::grpc::ServerCompletionQueue* cq,
      int method_id, HandleRequestFunction handle_request_function,
      bool supports_cancel) {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    auto call = new Call<Service, GrpcService, RequestMessage, ResponseMessage>(
        handle_request_function);
    if (supports_cancel) {
      call->RegisterCancellationHandler();
    }

    // Initial ref for call handed to grpc; released in Tag callback.
    grpc_service->RequestAsyncUnary(method_id, &call->ctx_, &call->request,
                                    &call->responder_, cq, cq,
                                    &call->request_received_tag_);
  }

  RequestMessage request;
  ResponseMessage response;


  const std::multimap<::grpc::string_ref, ::grpc::string_ref>& client_metadata()
      const {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    return ctx_.client_metadata();
  }

 private:
  // Creates a completion queue tag for handling cancellation by the client.
  // NOTE: This method must be called before this call is enqueued on a
  // completion queue.
  void RegisterCancellationHandler() {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    this->Ref();  // Ref for grpc; released in Tag callback.
    ctx_.AsyncNotifyWhenDone(&cancelled_tag_);
  }

  HandleRequestFunction handle_request_function_;
  ::grpc::ServerContext ctx_;
  ::grpc::ServerAsyncResponseWriter<ResponseMessage> responder_;

  // Used as void* completion markers from grpc to indicate different
  // events of interest for a Call.
  typedef typename UntypedCall<Service>::Tag Tag;
  Tag request_received_tag_{this, Tag::kRequestReceived};
  Tag response_sent_tag_{this, Tag::kResponseSent};
  Tag cancelled_tag_{this, Tag::kCancelled};

  mutex mu_;
  std::function<void()> cancel_callback_ TF_GUARDED_BY(mu_);
};

// Lifetime of a server-side bidirectional streaming call:
// - The call is created in the static EnqueueRequest method. It transfers
//   ownership to the kCallOpen tag pushed onto the completion queue.
// - If kCallOpen completes successfully, a read is requested and the
//   kRequestReceived tag takes ownership of the call. If kCallOpen fails,
//   e.g. server is shutdown, no further requests are pushed and the call is
//   destroyed (at the end of Tag::OnCompleted).
// - When the first request is received, we Ref() the call and invoke the
//   handler method thereby transferring ownership to the handler method.
//   The handler is responsible for calling SendResponse() or Finish() on this
//   call.
//   - If the handler calls Finish(), e.g. the request was invalid, Finish()
//     transfers ownership from the handler to the kServerFinished tag that
//     it pushes on the completion queue. The ownership is transferred because
//     the ref count is not incremented before putting the tag on the queue.
//   - If the handler calls SendResponse(), SendResponse() transfers ownership
//     to the kResponseSent tag.
// - When kResponseSent completes, we request a new read, which owns the call
//   now.
// - When the next request is received, it is handled the same way as the first
//   request.
//
// Because we request a read only after the write is sent, we can safely reuse
// the same request and response messages for the whole call.
template <class Service>
class ServerUntypedBidirectionalStreamingCall : public core::RefCounted {
 public:
  virtual void RequestReceived(Service* service) = 0;

  // Enqueues a request on the completion queue to read the next request.
  virtual void CallOpen() = 0;

  virtual void RequestRead() = 0;

  // Associates a tag in a `::grpc::CompletionQueue` with a callback.
  // An active Tag owns a reference on the corresponding Call object.
  class Tag : public GrpcCallTag<Service> {
   public:
    // One enum value per supported callback.
    enum class TagType {
      kCallOpen,
      kRequestReceived,
      kResponseSent,
      kServerFinished,
    };

    Tag(ServerUntypedBidirectionalStreamingCall* call, TagType cb)
        : call_(call), callback_(cb) {
          write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
        }






OnCompleted:

 0# tensorflow::GrpcMaybeUnparseProto(google::protobuf::Message const&, grpc::ByteBuffer*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::GrpcRemoteWorker::IssueRequest(google::protobuf::Message const*, google::protobuf::Message*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::function<void (tensorflow::Status const&)>, tensorflow::CallOptions*, bool) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::GrpcRemoteWorker::CompleteInstanceAsync(tensorflow::CallOptions*, tensorflow::CompleteInstanceRequest const*, tensorflow::CompleteInstanceResponse*, std::function<void (tensorflow::Status const&)>) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 3# tensorflow::(anonymous namespace)::CompleteInstanceCall::IssueCall(std::function<void (tensorflow::Status const&)> const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 4# tensorflow::CancellableCall::Start(std::function<void (tensorflow::Status const&)> const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 5# tensorflow::CollectiveParamResolverDistributed::CompleteInstanceDistributed(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tensorflow::CollectiveParams*, tensorflow::CancellationManager*, std::function<void (tensorflow::Status const&)> const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 6# std::_Function_handler<void (tensorflow::Status const&), tensorflow::CollectiveParamResolverDistributed::CompleteParamsAsync(tensorflow::DeviceAttributes const&, tensorflow::CollectiveParams*, tensorflow::CancellationManager*, std::function<void (tensorflow::Status const&)> const&)::{lambda(tensorflow::Status)#2}>::_M_invoke(std::_Any_data const&, tensorflow::Status const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 7# tensorflow::CollectiveParamResolverLocal::CompleteGroupLocal(tensorflow::DeviceAttributes const&, tensorflow::CollGroupParams*, tensorflow::CancellationManager*, std::function<void (tensorflow::Status const&)>) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 8# std::_Function_handler<void (tensorflow::Status const&), tensorflow::CollectiveParamResolverDistributed::CompleteGroupDistributed(tensorflow::DeviceAttributes const&, tensorflow::CollGroupParams*, tensorflow::CancellationManager*, std::function<void (tensorflow::Status const&)> const&)::{lambda(tensorflow::Status const&)#3}>::_M_invoke(std::_Any_data const&, tensorflow::Status const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so


 9# std::_Function_handler<void (), tensorflow::RPCState<google::protobuf::Message>::OnCompleted(bool)::{lambda()#2}>::_M_invoke(std::_Any_data const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so


10# Eigen::ThreadPoolTempl<tensorflow::thread::EigenEnvironment>::WorkerLoop(int) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
11# std::_Function_handler<void (), tensorflow::thread::EigenEnvironment::CreateThread(std::function<void ()>)::{lambda()#1}>::_M_invoke(std::_Any_data const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
12# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
13# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
14# clone in /lib/x86_64-linux-gnu/libc.so.6


    // Calls the callback associated with this tag and Unrefs this->call_.
    void OnCompleted(Service* service, bool ok) override {
      write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
      switch (callback_) {
        case TagType::kCallOpen:
          // Non-ok value indicates that the server has been shutdown before we
          // received a message for this call type. We do nothing to let this
          // call object be destroyed and avoid enqueuing request for another
          // call.
          if (ok) {
            call_->CallOpen();
          }
          break;
        case TagType::kRequestReceived:
          // Non-ok value from completion queue here means that we will not
          // receive any more messages from the client, e.g. the client called
          // WritesDone. There is nothing we need to do in this case. The call
          // will be Unref'ed and deleted. If the client wants to open a new
          // call, we have already enqueued a request for a new call in CallOpen
          // above.
          if (ok) {
            call_->RequestReceived(service);
          }
          break;
        case TagType::kResponseSent:
          if (ok) {
            // The obvious place to request a read would be at the end of
            // RequestReceived(). Unfortunately, this can result in multiple
            // outstanding write requests in the completion queue. This is
            // currently not supported by gRPC, which requires at most one
            // outstanding write request in the completion queue.
            // Requesting a read here, in ResponseSent, works because at
            // this point, the completion queue has no write requests
            // (kResponseSent happens when a write completes).
            // This might be synchronizing the processing more than strictly
            // necessary, but is probably fine because, AFAICT from gRPC docs,
            // the write request completes as soon as it can be written to
            // outgoing buffer.
            call_->RequestRead();
          }
          // ok == false means that the response is not going on the wire
          // because the call is already dead (i.e., canceled, deadline
          // expired, other side dropped the channel, etc). Since the call is
          // dead, there is nothing for us to do, we just let the call be
          // deleted.
          break;
        case TagType::kServerFinished:
          // Whether our finish request is successful or not (whether it went
          // on the wire towards the client), there is nothing for us to do.
          // In the current implementation, there can be no read or write
          // requests in the completion queue (see the comment in kResponseSent)
          // above. Even if there were pending requests, they would complete
          // with a non-ok status, we would not do anything, and let the call be
          // deleted.
          break;
      }
      call_->Unref();  // Ref acquired when tag was handed to grpc.
    }

   private:
    ServerUntypedBidirectionalStreamingCall* const
        call_;  // `this` owns one reference.
    TagType callback_;
  };
};

// Represents a pending call with known request and response message
// types, and a known request-handling method.
// Common usage pattern is to have a single thread waiting on events from
// completion queue and calling Tag::OnCompleted(), which invokes methods
// on this.
// This implementation assumes that the server will generate a single response
// message for each request message. More precisely, this class expects that
// each time it invokes handle_request_function_, the service implementation
// will either call SendResponse or Finish exactly once.
// Not thread-safe.
template <class Service, class GrpcService, class RequestMessage,
          class ResponseMessage>
class ServerBidirectionalStreamingCall
    : public ServerUntypedBidirectionalStreamingCall<Service> {
 public:
  // Represents the generic signature of a generated
  // `GrpcService::RequestFoo()` method, where `Foo` is the name of an
  // RPC method.
  using EnqueueFunction = void (GrpcService::*)(
      ::grpc::ServerContext*,
      ::grpc::ServerAsyncReaderWriter<ResponseMessage, RequestMessage>*,
      ::grpc::CompletionQueue*, ::grpc::ServerCompletionQueue*, void*);

  // Represents the generic signature of a `Service::HandleFoo()`
  // method, where `Foo` is the name of an RPC method.
  using HandleRequestFunction = void (Service::*)(
      ServerBidirectionalStreamingCall<Service, GrpcService, RequestMessage,
                                       ResponseMessage>*);

  ServerBidirectionalStreamingCall(
      HandleRequestFunction handle_request_function, GrpcService* grpc_service,
      ::grpc::ServerCompletionQueue* cq, EnqueueFunction enqueue_function)
      : handle_request_function_(handle_request_function),
        stream_(&ctx_),
        grpc_service_(grpc_service),
        cq_(cq),
        enqueue_function_(enqueue_function) {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    VLOG(3) << "Creating ServerBidirectionalStreamingCall " << this;
  }

ServerBidirectionalStreamingCall:
 0# tensorflow::ServerBidirectionalStreamingCall<tensorflow::eager::GrpcEagerServiceImpl, tensorflow::eager::EagerService::WithAsyncMethod_CreateContext<tensorflow::eager::EagerService::WithAsyncMethod_UpdateContext<tensorflow::eager::EagerService::WithAsyncMethod_Enqueue<tensorflow::eager::EagerService::WithAsyncMethod_StreamingEnqueue<tensorflow::eager::EagerService::WithAsyncMethod_WaitQueueDone<tensorflow::eager::EagerService::WithAsyncMethod_RunComponentFunction<tensorflow::eager::EagerService::WithAsyncMethod_KeepAlive<tensorflow::eager::EagerService::WithAsyncMethod_CloseContext<tensorflow::eager::EagerService::Service> > > > > > > >, tensorflow::eager::EnqueueRequest, tensorflow::eager::EnqueueResponse>::EnqueueRequest(tensorflow::eager::EagerService::WithAsyncMethod_CreateContext<tensorflow::eager::EagerService::WithAsyncMethod_UpdateContext<tensorflow::eager::EagerService::WithAsyncMethod_Enqueue<tensorflow::eager::EagerService::WithAsyncMethod_StreamingEnqueue<tensorflow::eager::EagerService::WithAsyncMethod_WaitQueueDone<tensorflow::eager::EagerService::WithAsyncMethod_RunComponentFunction<tensorflow::eager::EagerService::WithAsyncMethod_KeepAlive<tensorflow::eager::EagerService::WithAsyncMethod_CloseContext<tensorflow::eager::EagerService::Service> > > > > > > >*, grpc_impl::ServerCompletionQueue*, void (tensorflow::eager::EagerService::WithAsyncMethod_CreateContext<tensorflow::eager::EagerService::WithAsyncMethod_UpdateContext<tensorflow::eager::EagerService::WithAsyncMethod_Enqueue<tensorflow::eager::EagerService::WithAsyncMethod_StreamingEnqueue<tensorflow::eager::EagerService::WithAsyncMethod_WaitQueueDone<tensorflow::eager::EagerService::WithAsyncMethod_RunComponentFunction<tensorflow::eager::EagerService::WithAsyncMethod_KeepAlive<tensorflow::eager::EagerService::WithAsyncMethod_CloseContext<tensorflow::eager::EagerService::Service> > > > > > > >::*)(grpc_impl::ServerContext*, grpc_impl::ServerAsyncReaderWriter<tensorflow::eager::EnqueueResponse, tensorflow::eager::EnqueueRequest>*, grpc_impl::CompletionQueue*, grpc_impl::ServerCompletionQueue*, void*), void (tensorflow::eager::GrpcEagerServiceImpl::*)(tensorflow::ServerBidirectionalStreamingCall<tensorflow::eager::GrpcEagerServiceImpl, tensorflow::eager::EagerService::WithAsyncMethod_CreateContext<tensorflow::eager::EagerService::WithAsyncMethod_UpdateContext<tensorflow::eager::EagerService::WithAsyncMethod_Enqueue<tensorflow::eager::EagerService::WithAsyncMethod_StreamingEnqueue<tensorflow::eager::EagerService::WithAsyncMethod_WaitQueueDone<tensorflow::eager::EagerService::WithAsyncMethod_RunComponentFunction<tensorflow::eager::EagerService::WithAsyncMethod_KeepAlive<tensorflow::eager::EagerService::WithAsyncMethod_CloseContext<tensorflow::eager::EagerService::Service> > > > > > > >, tensorflow::eager::EnqueueRequest, tensorflow::eager::EnqueueResponse>*)) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::eager::GrpcEagerServiceImpl::HandleRPCsLoop() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 3# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 4# clone in /lib/x86_64-linux-gnu/libc.so.6

 0# tensorflow::ServerBidirectionalStreamingCall<tensorflow::eager::GrpcEagerServiceImpl, tensorflow::eager::EagerService::WithAsyncMethod_CreateContext<tensorflow::eager::EagerService::WithAsyncMethod_UpdateContext<tensorflow::eager::EagerService::WithAsyncMethod_Enqueue<tensorflow::eager::EagerService::WithAsyncMethod_StreamingEnqueue<tensorflow::eager::EagerService::WithAsyncMethod_WaitQueueDone<tensorflow::eager::EagerService::WithAsyncMethod_RunComponentFunction<tensorflow::eager::EagerService::WithAsyncMethod_KeepAlive<tensorflow::eager::EagerService::WithAsyncMethod_CloseContext<tensorflow::eager::EagerService::Service> > > > > > > >, tensorflow::eager::EnqueueRequest, tensorflow::eager::EnqueueResponse>::EnqueueRequest(tensorflow::eager::EagerService::WithAsyncMethod_CreateContext<tensorflow::eager::EagerService::WithAsyncMethod_UpdateContext<tensorflow::eager::EagerService::WithAsyncMethod_Enqueue<tensorflow::eager::EagerService::WithAsyncMethod_StreamingEnqueue<tensorflow::eager::EagerService::WithAsyncMethod_WaitQueueDone<tensorflow::eager::EagerService::WithAsyncMethod_RunComponentFunction<tensorflow::eager::EagerService::WithAsyncMethod_KeepAlive<tensorflow::eager::EagerService::WithAsyncMethod_CloseContext<tensorflow::eager::EagerService::Service> > > > > > > >*, grpc_impl::ServerCompletionQueue*, void (tensorflow::eager::EagerService::WithAsyncMethod_CreateContext<tensorflow::eager::EagerService::WithAsyncMethod_UpdateContext<tensorflow::eager::EagerService::WithAsyncMethod_Enqueue<tensorflow::eager::EagerService::WithAsyncMethod_StreamingEnqueue<tensorflow::eager::EagerService::WithAsyncMethod_WaitQueueDone<tensorflow::eager::EagerService::WithAsyncMethod_RunComponentFunction<tensorflow::eager::EagerService::WithAsyncMethod_KeepAlive<tensorflow::eager::EagerService::WithAsyncMethod_CloseContext<tensorflow::eager::EagerService::Service> > > > > > > >::*)(grpc_impl::ServerContext*, grpc_impl::ServerAsyncReaderWriter<tensorflow::eager::EnqueueResponse, tensorflow::eager::EnqueueRequest>*, grpc_impl::CompletionQueue*, grpc_impl::ServerCompletionQueue*, void*), void (tensorflow::eager::GrpcEagerServiceImpl::*)(tensorflow::ServerBidirectionalStreamingCall<tensorflow::eager::GrpcEagerServiceImpl, tensorflow::eager::EagerService::WithAsyncMethod_CreateContext<tensorflow::eager::EagerService::WithAsyncMethod_UpdateContext<tensorflow::eager::EagerService::WithAsyncMethod_Enqueue<tensorflow::eager::EagerService::WithAsyncMethod_StreamingEnqueue<tensorflow::eager::EagerService::WithAsyncMethod_WaitQueueDone<tensorflow::eager::EagerService::WithAsyncMethod_RunComponentFunction<tensorflow::eager::EagerService::WithAsyncMethod_KeepAlive<tensorflow::eager::EagerService::WithAsyncMethod_CloseContext<tensorflow::eager::EagerService::Service> > > > > > > >, tensorflow::eager::EnqueueRequest, tensorflow::eager::EnqueueResponse>*)) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::eager::GrpcEagerServiceImpl::HandleRPCsLoop() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 3# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 4# clone in /lib/x86_64-linux-gnu/libc.so.6


  ~ServerBidirectionalStreamingCall() override {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    VLOG(3) << "Destroying ServerBidirectionalStreamingCall " << this;
  }

  void CallOpen() override {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    // Let gRPC know that we can accept another call.
    ServerBidirectionalStreamingCall<
        Service, GrpcService, RequestMessage,
        ResponseMessage>::EnqueueRequest(grpc_service_, cq_, enqueue_function_,
                                         handle_request_function_);
    RequestRead();
  }

  void RequestRead() override {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    this->Ref();
    request_.Clear();
    stream_.Read(&request_, &request_received_tag_);
  }

  void RequestReceived(Service* service) override {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    this->Ref();
    // Request handling should result in a call to SendResponse or Finish.
    (service->*handle_request_function_)(this);
  }

RequestReceived:
 0# tensorflow::Call<tensorflow::(anonymous namespace)::GrpcWorkerServiceThread, tensorflow::grpc::WorkerService::AsyncService, tensorflow::CompleteGroupRequest, tensorflow::CompleteGroupResponse>::RequestReceived(tensorflow::(anonymous namespace)::GrpcWorkerServiceThread*, bool) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::(anonymous namespace)::GrpcWorkerServiceThread::HandleRPCsLoop() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 3# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 4# clone in /lib/x86_64-linux-gnu/libc.so.6







  void SendResponse() {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    // Transferring ownership of this to the response_sent_tag_.
    stream_.Write(response_, &response_sent_tag_);
    // stream_.Write does not save references to response_. We are free to muck
    // around with it as soon as Write returns.
    // We clear the response_ to prepare it for the next response.
    response_.Clear();
  }

  void Finish(::grpc::Status status) {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    // Transferring ownership of this to the server_finished_tag_.
    stream_.Finish(status, &server_finished_tag_);
  }








EnqueueRequest:
 0# tensorflow::ServerBidirectionalStreamingCall<tensorflow::eager::GrpcEagerServiceImpl, tensorflow::eager::EagerService::WithAsyncMethod_CreateContext<tensorflow::eager::EagerService::WithAsyncMethod_UpdateContext<tensorflow::eager::EagerService::WithAsyncMethod_Enqueue<tensorflow::eager::EagerService::WithAsyncMethod_StreamingEnqueue<tensorflow::eager::EagerService::WithAsyncMethod_WaitQueueDone<tensorflow::eager::EagerService::WithAsyncMethod_RunComponentFunction<tensorflow::eager::EagerService::WithAsyncMethod_KeepAlive<tensorflow::eager::EagerService::WithAsyncMethod_CloseContext<tensorflow::eager::EagerService::Service> > > > > > > >, tensorflow::eager::EnqueueRequest, tensorflow::eager::EnqueueResponse>::EnqueueRequest(tensorflow::eager::EagerService::WithAsyncMethod_CreateContext<tensorflow::eager::EagerService::WithAsyncMethod_UpdateContext<tensorflow::eager::EagerService::WithAsyncMethod_Enqueue<tensorflow::eager::EagerService::WithAsyncMethod_StreamingEnqueue<tensorflow::eager::EagerService::WithAsyncMethod_WaitQueueDone<tensorflow::eager::EagerService::WithAsyncMethod_RunComponentFunction<tensorflow::eager::EagerService::WithAsyncMethod_KeepAlive<tensorflow::eager::EagerService::WithAsyncMethod_CloseContext<tensorflow::eager::EagerService::Service> > > > > > > >*, grpc_impl::ServerCompletionQueue*, void (tensorflow::eager::EagerService::WithAsyncMethod_CreateContext<tensorflow::eager::EagerService::WithAsyncMethod_UpdateContext<tensorflow::eager::EagerService::WithAsyncMethod_Enqueue<tensorflow::eager::EagerService::WithAsyncMethod_StreamingEnqueue<tensorflow::eager::EagerService::WithAsyncMethod_WaitQueueDone<tensorflow::eager::EagerService::WithAsyncMethod_RunComponentFunction<tensorflow::eager::EagerService::WithAsyncMethod_KeepAlive<tensorflow::eager::EagerService::WithAsyncMethod_CloseContext<tensorflow::eager::EagerService::Service> > > > > > > >::*)(grpc_impl::ServerContext*, grpc_impl::ServerAsyncReaderWriter<tensorflow::eager::EnqueueResponse, tensorflow::eager::EnqueueRequest>*, grpc_impl::CompletionQueue*, grpc_impl::ServerCompletionQueue*, void*), void (tensorflow::eager::GrpcEagerServiceImpl::*)(tensorflow::ServerBidirectionalStreamingCall<tensorflow::eager::GrpcEagerServiceImpl, tensorflow::eager::EagerService::WithAsyncMethod_CreateContext<tensorflow::eager::EagerService::WithAsyncMethod_UpdateContext<tensorflow::eager::EagerService::WithAsyncMethod_Enqueue<tensorflow::eager::EagerService::WithAsyncMethod_StreamingEnqueue<tensorflow::eager::EagerService::WithAsyncMethod_WaitQueueDone<tensorflow::eager::EagerService::WithAsyncMethod_RunComponentFunction<tensorflow::eager::EagerService::WithAsyncMethod_KeepAlive<tensorflow::eager::EagerService::WithAsyncMethod_CloseContext<tensorflow::eager::EagerService::Service> > > > > > > >, tensorflow::eager::EnqueueRequest, tensorflow::eager::EnqueueResponse>*)) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::eager::GrpcEagerServiceImpl::HandleRPCsLoop() in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::(anonymous namespace)::PThread::ThreadFn(void*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/../libtensorflow_framework.so.2
 3# start_thread in /lib/x86_64-linux-gnu/libpthread.so.0
 4# clone in /lib/x86_64-linux-gnu/libc.so.6

  // Enqueues a new request for the given service on the given
  // completion queue, using the given `enqueue_function`.
  //
  // The request will be handled by the given `handle_request_function`.
  static void EnqueueRequest(GrpcService* grpc_service,
                             ::grpc::ServerCompletionQueue* cq,
                             EnqueueFunction enqueue_function,
                             HandleRequestFunction handle_request_function) {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    auto call =
        new ServerBidirectionalStreamingCall<Service, GrpcService,
                                             RequestMessage, ResponseMessage>(
            handle_request_function, grpc_service, cq, enqueue_function);

    // Initial ref for call handed to grpc; released in Tag callback.
    (grpc_service->*enqueue_function)(&call->ctx_, &call->stream_, cq, cq,
                                      &call->call_open_tag_);
  }

  const RequestMessage& request() const {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    return request_;
  }
  ResponseMessage* mutable_response() {
    write_log(boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
    return &response_;
  }

 private:
  // Request and response messages are reused for each request/response exchange
  // between the client and the server.
  RequestMessage request_;
  ResponseMessage response_;
  ::grpc::ServerContext ctx_;

  HandleRequestFunction handle_request_function_;
  ::grpc::ServerAsyncReaderWriter<ResponseMessage, RequestMessage> stream_;

  // Used as void* completion markers from grpc to indicate different
  // events of interest for a ServerBidirectionalStreamingCall.
  typedef typename ServerUntypedBidirectionalStreamingCall<Service>::Tag Tag;
  // At most one tag of each kind may be given to gRPC at any one time.
  // Beyond semantic sanity, this is needed to ensure proper ref counting
  // of this call object.
  Tag call_open_tag_{this, Tag::TagType::kCallOpen};
  Tag request_received_tag_{this, Tag::TagType::kRequestReceived};
  Tag response_sent_tag_{this, Tag::TagType::kResponseSent};
  Tag server_finished_tag_{this, Tag::TagType::kServerFinished};

  // These fields are used only to spawn another instance of this to accept
  // more streaming calls.
  GrpcService* grpc_service_;
  ::grpc::ServerCompletionQueue* cq_;
  EnqueueFunction enqueue_function_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CALL_H_
