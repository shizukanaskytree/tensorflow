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

#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

#include "grpcpp/grpcpp.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/server_builder.h"

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

// Represents a pending request with unknown message types.
template <class Service>
class UntypedCall : public core::RefCounted {
 public:
  virtual ~UntypedCall() {}

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

  // This method will be called either (i) when the server is notified
  // that the request has been canceled, or (ii) when the request completes
  // normally. The implementation should distinguish these cases by querying
  // the `grpc::ServerContext` associated with the request.
  virtual void RequestCancelled(Service* service, bool ok) = 0;

  /** \class UntypedCall::Tag
   *
   *  \brief
   */
  // Associates a tag in a `::grpc::CompletionQueue` with a callback
  // for an incoming RPC.  An active Tag owns a reference on the corresponding
  // Call object.
  class Tag {
   public:
    // One enum value per supported callback.
    enum Callback { kRequestReceived, kResponseSent, kCancelled };

    Tag(UntypedCall* call, Callback cb) : call_(call), callback_(cb) {}

    /** \brief Once server receive a request from the client, OnCompleted
     *         indirectly invokes the grpc implementation functions of server
     *         side.
     *
     *  \param[in] service: Service* ;
     *         Service is typename, for example, class GrpcMasterService.
     *         The grpc server side handler functions are defined in class
     *         GrpcMasterService.
     *
     *  \param[in] ok: bool ;
     *         true if it is a successful event, false otherwise.
     */
    // Calls the callback associated with this tag.
    //
    // The callback takes ownership of `this->call_`.
    void OnCompleted(Service* service, bool ok) {
      switch (callback_) {
        case kRequestReceived:
          /// When the request is received, it is going to handle it.
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

   private:
    UntypedCall* const call_;  // `this` owns one reference.
    Callback callback_;
  };
};

/** \class Call
 *
 *  \brief Define call objects.
 *
 *  \param Service: typename, class;
 *         For example, GrpcMasterService, which defines rpc service function
 *         handlers, e.g., CreateSessionHandler, and HandleRPCsLoop.
 *
 *  \param GrpcService: typename, class;
 *         For example, grpc::MasterService::AsyncService, Represents an
 *         abstract class definding asynchronous service that handles incoming
 *         RPCs with a polling loop, i.e. HandleRPCsLoop.
 *
 *  \param RequestMessage: typename, class;
 *         For example, message CreateSessionRequest, which is defined in
 *         master.proto. It is used in master_service.proto.
 *         rpc CreateSession(CreateSessionRequest) returns (CreateSessionResponse);
 *
 *  \param ResponseMessage: typename, class;
 *         For example, message CreateSessionResponse, which is defined in
 *         master.proto. It is used in master_service.proto.
 *         rpc CreateSession(CreateSessionRequest) returns (CreateSessionResponse);
 *
 */
// Represents a pending call with known request and response message
// types, and a known request-handling method.
template <class Service, class GrpcService, class RequestMessage,
          class ResponseMessage>
class Call : public UntypedCall<Service> {
 public:

  /** \brief Define the signature of auto-generated functions of
   *         **a simple grpc service without stream**.
   *
   *  \param GrpcService::*;
   *         Template of function name: Request______.
   *
   *  \param ::grpc::ServerContext* context;
   *
   *  \param RequestMessage* ;
   *         For example, message CreateSessionRequest.
   *
   *  \param ::grpc::ServerAsyncResponseWriter<ResponseMessage>*
   *
   *  \param ::grpc::CompletionQueue* new_call_cq;
   *
   *  \param ::grpc::ServerCompletionQueue*;
   *         notification completion queue;
   *
   *  \param void* tag;
   *
   *  \return void
   *
   *  \details Example of signature of an grpc hello world example:
   *           https://gist.github.com/shizukanaskytree/a057861909d682dbd897f3bd7a86f091
   *   - class StubInterface (都包含如下格式的)
   *       - GetFeature
   *       - Async·GetFeature
   *       - PrepareAsync·GetFeature
   *       - class experimental_async_interface
   *           - 函数名和 rpc service 的名字是一样的 overloadded functions
   *       - private: (都包含如下格式的)
   *           - Async·GetFeature·Raw
   *           - PrepareAsync·GetFeature·Raw
   *   - class Stub final : public StubInterface  (都包含如下格式的)
   *       - GetFeature
   *       - Async·GetFeature
   *       - PrepareAsync·GetFeature
   *       - class experimental_async final :
   *           - 函数名和 rpc service 的名字是一样的 overloadded functions
   *       - private: 函数格式如下
   *           - Async·GetFeature·Raw
   *           - PrepareAsync·GetFeature·Raw
   *           - private member values 比如
   *               - rpcmethod_·GetFeature·_
   *   - class Service : public ::grpc::Service
   *       - 虚函数，函数名和 rpc service 的名字是一样.
   *   - class WithAsyncMethod_·GetFeature : public BaseClass
   *       - 其他的类名模板部分和 rpc service 的名字是一样
   *   - class ExperimentalWithCallbackMethod_·GetFeature : public BaseClass
   *       - 其他的类名模板部分和 rpc service 的名字是一样
   *   - class WithGenericMethod_·GetFeature : public BaseClass
   *       - 其他的类名模板部分和 rpc service 的名字是一样
   *   - class WithRawMethod_·GetFeature : public BaseClass
   *       - 其他的类名模板部分和 rpc service 的名字是一样
   *   - class ExperimentalWithRawCallbackMethod_GetFeature : public BaseClass
   *       - 其他的类名模板部分和 rpc service 的名字是一样
   *   - class WithStreamedUnaryMethod_GetFeature : public BaseClass
   *       - 只有那个和 stream 毫无关系的 GetFeature
   *       - A simple RPC.
   *           - `  rpc ListFeatures(Rectangle) returns (stream Feature) {}`
   *
   */
  // Represents the generic signature of a generated
  // `GrpcService::RequestFoo()` method, where `Foo` is the name of an
  // RPC method.
  using EnqueueFunction = void (GrpcService::*)(
      ::grpc::ServerContext*, RequestMessage*,
      ::grpc::ServerAsyncResponseWriter<ResponseMessage>*,
      ::grpc::CompletionQueue*, ::grpc::ServerCompletionQueue*, void*);


  /** \brief Signature of grpc handler functions. e.g., CreateSessionHandler in
   *         grpc_master_service.cc.
   *
   *  \param Service::*
   *
   *  \param Call<Service, GrpcService, RequestMessage, ResponseMessage>*
   *         A pointer to Call object.
   *
   *  \param Service: typename, class;
   *         For example, GrpcMasterService, which defines rpc service function
   *         handlers, e.g., CreateSessionHandler, and HandleRPCsLoop.
   *
   *  \param GrpcService: typename, class;
   *         For example, grpc::MasterService::AsyncService, Represents an
   *         abstract class definding asynchronous service that handles incoming
   *         RPCs with a polling loop, i.e. HandleRPCsLoop.
   *
   *  \param RequestMessage: typename, class;
   *         For example, message CreateSessionRequest, which is defined in
   *         master.proto. It is used in master_service.proto.
   *         rpc CreateSession(CreateSessionRequest) returns (CreateSessionResponse);
   *
   *  \param ResponseMessage: typename, class;
   *         For example, message CreateSessionResponse, which is defined in
   *         master.proto. It is used in master_service.proto.
   *         rpc CreateSession(CreateSessionRequest) returns (CreateSessionResponse);
   *
   */
  // Represents the generic signature of a `Service::HandleFoo()`
  // method, where `Foo` is the name of an RPC method.
  using HandleRequestFunction = void (Service::*)(
      Call<Service, GrpcService, RequestMessage, ResponseMessage>*);

  Call(HandleRequestFunction handle_request_function)
      : handle_request_function_(handle_request_function), responder_(&ctx_) {}

  virtual ~Call() {}

  /** \brief Once client sends a request, server is going to handle it via this
   *         function.
   *
   *  \param[in] service: Service*
   *         Service is a typename, for example, GrpcMasterService. Master will
   *         provide service functions for client's request.
   *
   *  \param[in] ok: bool
   *         true if it is a successful event, false otherwise.
   *
   *  \details RequestReceived is the implementation of base class UntypedCall.
   *           The implementation of this method should use `service` to handle
   *           an incoming request, and (perhaps asynchronously) send the
   *           response.
   *           One reference on `this` is transferred to the callee, and the
   *           callee is responsible for releasing it (typically via
   *           `Call::SendResponse()`).
   *           `ok` is true if the request was received in a "regular event",
   *           otherwise false.
   */
  void RequestReceived(Service* service, bool ok) override {
    if (ok) {
      this->Ref();
      /// For example,
      /// service is GrpcMasterService;
      /// handle_request_function_ is CreateSessionHandler;
      /// So, it will redirect to grpc_master_service.cc,
      /// CreateSessionHandler(
      ///   MasterCall<CreateSessionRequest, CreateSessionResponse>* call){...}
      (service->*handle_request_function_)(this);
    }
  }

  /** \brief grpc Server sends Response back to the client.
   *
   *  \param status[in]: ::grpc::Status
   */
  void SendResponse(::grpc::Status status) {
    this->Ref();  // Ref for grpc; released in Tag callback.

    /// \fn Finish
    /// responder_: ::grpc::ServerAsyncResponseWriter<ResponseMessage>
    ///
    /// \details
    /// Request to receive the server's response \a msg and final \a status for
    /// the call, and to notify \a tag on this call's completion queue when
    /// finished.
    ///
    /// \note tag
    /// call CompletionQueue::Next to wait for operations to complete.
    /// If a "tag" appears, it indicates that the corresponding operation is
    /// complete.
    ///
    /// This function will return when either:
    /// - when the server's response message and status have been received by
    ///   the client.
    /// - when the server has returned a non-OK status (no message expected in
    ///   this case).
    /// - when the call failed for some reason and the library generated a
    ///   non-OK status.
    ///
    /// \param[in] tag Tag identifying this request.
    /// \param[out] status To be updated with the operation status.
    /// \param[out] msg To be filled in with the server's response message.
    ///
    /// \details
    /// Server also must make sure the client has got the response from itself.
    /// Once client get the response for sure, the tag will be returned from the
    /// completion queue to indicate that rpc is finished.
    responder_.Finish(response, status, &response_sent_tag_);

    this->Unref();
  }

  void RequestCancelled(Service* service, bool ok) override {
    if (ctx_.IsCancelled()) {
      mutex_lock l(mu_);
      if (cancel_callback_) {
        cancel_callback_();
      }
    }
  }

  // Registers `callback` as the function that should be called if and when this
  // call is canceled by the client.
  void SetCancelCallback(std::function<void()> callback) {
    mutex_lock l(mu_);
    cancel_callback_ = std::move(callback);
  }

  // Clears any cancellation callback that has been registered for this call.
  void ClearCancelCallback() {
    mutex_lock l(mu_);
    cancel_callback_ = nullptr;
  }

  /** \brief Invoke the request service function via grpc API
   *         ::grpc::Service::RequestAsyncUnary.
   *
   *  \param grpc_service: GrpcService*, typename, class;
   *         For example, grpc::MasterService::AsyncService,
   *         represents an abstract class definding asynchronous service. //that
   *         //handles incoming RPCs with a polling loop, i.e. HandleRPCsLoop.
   *
   *  \param cq: ::grpc::ServerCompletionQueue*;
   *         A specific type of completion queue used by the processing of
   *         notifications by servers.
   *         A derived class of ::grpc::CompletionQueue, which provides functions
   *         like reading from the queue.
   *         Instantiated by ::grpc::ServerBuilder.
   *
   *  \param enqueue_function: EnqueueFunction;
   *         This function is mainly called by this wrapper function, which in
   *         turn further call Master::CreateSession to handle the request and
   *         response.
   *         For example,
   *         tensorflow::grpc::MasterService::AsyncService::RequestCreateSession
   *         implements the auto-generated grpc service function by proto file.
   *         in tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h
   *
   *  \param handle_request_function;
   *         Signature of grpc handler functions, e.g., CreateSessionHandler in
   *         grpc_master_service.cc.
   *         grpc server side handler function.
   *         Define the signature of auto-generated functions of
   *         **a simple grpc service without stream**.
   *         For example, GrpcMasterService::CreateSessionHandler,
   *         ExtendSessionHandler, PartialRunSetupHandler, RunStepHandler...
   *         It further calls class Master::CreateSession to handle session
   *         creation.
   *
   *  \param supports_cancel: bool;
   */
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
    /// New a Call object.
    auto call = new Call<Service, GrpcService, RequestMessage, ResponseMessage>(
        handle_request_function);
    if (supports_cancel) {
      call->RegisterCancellationHandler();
    }

    /// Call the grpc auto-generated function according to the signature.
    // Initial ref for call handed to grpc; released in Tag callback.
    (grpc_service->*enqueue_function)(&call->ctx_, &call->request,
                                      &call->responder_, cq, cq,
                                      &call->request_received_tag_);
  }

  /**
   *
   *
   *
   *
   *
   *
   *
   *
   */
  // Enqueues a new request for the given service on the given
  // completion queue, using the given `method_id`.
  //
  // The request will be handled with the given
  // `handle_request_function`.
  static void EnqueueRequestForMethod(
      GrpcService* grpc_service, ::grpc::ServerCompletionQueue* cq,
      int method_id, HandleRequestFunction handle_request_function,
      bool supports_cancel) {
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
    return ctx_.client_metadata();
  }

 private:
  // Creates a completion queue tag for handling cancellation by the client.
  // NOTE: This method must be called before this call is enqueued on a
  // completion queue.
  void RegisterCancellationHandler() {
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
  std::function<void()> cancel_callback_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_CALL_H_
