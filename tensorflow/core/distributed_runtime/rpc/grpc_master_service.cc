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

// GrpcMasterService implements the RPC service MasterSerivce.
//
// A GrpcMasterService maintains the state of live graph computation
// sessions, each session orchestrates both local and remote devices
// to carry out the graph computation.
//
// A GrpcMasterService knows ahead of time local devices available as
// client devices.
//
// A GrpcMasterService discovers remote devices in the background and
// keeps track of statistics of those remote devices.
//
// Each session analyzes the graph, places nodes across available
// devices, and ultimately drives the graph computation by initiating
// RunGraph on workers.
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service.h"

#include "grpcpp/alarm.h"
#include "grpcpp/server_builder.h"

#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_call.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow {

class GrpcMasterService : public AsyncServiceInterface {
 public:
  GrpcMasterService(Master* master, const ConfigProto& default_session_config,
                    ::grpc::ServerBuilder* builder)
      : master_impl_(master),
        is_shutdown_(false),
        default_session_config_(default_session_config) {
    builder->RegisterService(&master_service_);
    cq_ = builder->AddCompletionQueue();
  }

  ~GrpcMasterService() override { delete shutdown_alarm_; }

  void Shutdown() override {
    bool did_shutdown = false;
    {
      mutex_lock l(mu_);
      if (!is_shutdown_) {
        LOG(INFO) << "Shutting down GrpcMasterService.";
        is_shutdown_ = true;
        did_shutdown = true;
      }
    }
    if (did_shutdown) {
      // NOTE(mrry): This enqueues a special event (with a null tag)
      // that causes the completion queue to be shut down on the
      // polling thread.
      shutdown_alarm_ =
          new ::grpc::Alarm(cq_.get(), gpr_now(GPR_CLOCK_MONOTONIC), nullptr);
    }
  }

/** \brief Indirectly call, for example, for CreateSession:
 *         Call<GrpcMasterService, grpc::MasterService::AsyncService           \
 *              CreateSessionRequest, CreateSessionResponse>::                 \
 *         EnqueueRequest(&master_service_, cq_.get(),                         \
 *                   &grpc::MasterService::AsyncService::RequestCreateSession, \
 *                   &GrpcMasterService::CreateSessionHandler,                 \
 *                   (supports_cancel));
 *
 *  \param method: function name;
 *         For example, CreateSession
 *
 *  \param supports_cancel: bool;
 *
 *  \fn EnqueueRequest
 *  \param master_service_: grpc::MasterService::AsyncService;
 *         Implement ::grpc::Service auto-generated request interface functions,
 *         Request##method. e.g.,
 *         RequestCreateSession, RequestExtendSession, RequestPartialRunSetup...
 *
 *  \param cq_: std::unique_ptr<::grpc::ServerCompletionQueue>
 *         A specific type of completion queue used by the processing of
 *         notifications by servers.
 *
 *  \param grpc::MasterService::AsyncService::Request##method
 *         This function is mainly called by this wrapper function, which in
 *         turn further call Master::CreateSession to handle the request and
 *         response.
 *         For example,
 *         tensorflow::grpc::MasterService::AsyncService::RequestCreateSession
 *         implements the auto-generated grpc service function by proto file.
 *         in tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h
 *
 *  \param GrpcMasterService::method##Handler
 *         grpc server side handler function.
 *         For example, GrpcMasterService::CreateSessionHandler,
 *         ExtendSessionHandler, PartialRunSetupHandler, RunStepHandler...
 *         It further calls class Master::CreateSession to handle session
 *         creation.
 *
 *  \details Take `ENQUEUE_REQUEST(CreateSession, true);` as an example.
 *           method is CreateSession; supports_cancel is true;
 *           Call::EnqueueRequest is defined in grpc_call.h.
 *           Call::EnqueueRequest indirectly invokes the grpc auto-generated
 *           grpc service function via grpc signature of
 *           grpc::MasterService::AsyncService::Request##method defined in
 *           class tensorflow::grpc::MasterService::AsyncService . in
 *           tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h.
 */
// This macro creates a new request for the given RPC method name
// (e.g., `ENQUEUE_REQUEST(RunStep);`), and enqueues it on
// `this->cq_`.
//
// This macro is invoked one or more times for each RPC method to
// ensure that there are sufficient completion queue entries to
// handle incoming requests without blocking.
//
// The implementation of the request handler for each RPC method
// must ensure that it calls ENQUEUE_REQUEST() for that RPC method,
// to keep accepting new requests.
#define ENQUEUE_REQUEST(method, supports_cancel)                              \
  do {                                                                        \
    mutex_lock l(mu_);                                                        \
    if (!is_shutdown_) {                                                      \
      Call<GrpcMasterService, grpc::MasterService::AsyncService,              \
           method##Request, method##Response>::                               \
          EnqueueRequest(&master_service_, cq_.get(),                         \
                         &grpc::MasterService::AsyncService::Request##method, \
                         &GrpcMasterService::method##Handler,                 \
                         (supports_cancel));                                  \
    }                                                                         \
  } while (0)

  /** \brief Server starts to serve RPC request in a new thread.
   *
   *  \remark No params and no return.
   *
   *  \details This macro is invoked one or more times for each RPC method to
   *           ensure that there are sufficient completion queue entries to
   *           handle incoming requests without blocking.
   *
   *  \todo Why should these macro be called one or more times for completion
   *        queue?
   */
  void HandleRPCsLoop() override {
    /// \todo Q. What is the diff between ENQUEUE_REQUEST for one time and
    ///       multiple times?
    ENQUEUE_REQUEST(CreateSession, true);
    ENQUEUE_REQUEST(ExtendSession, false);
    for (int i = 0; i < 100; ++i) {
      ENQUEUE_REQUEST(PartialRunSetup, false);
      ENQUEUE_REQUEST(RunStep, true);
    }
    ENQUEUE_REQUEST(CloseSession, false);
    ENQUEUE_REQUEST(ListDevices, false);
    ENQUEUE_REQUEST(Reset, false);
    ENQUEUE_REQUEST(MakeCallable, false);
    for (int i = 0; i < 100; ++i) {
      ENQUEUE_REQUEST(RunCallable, true);
    }
    ENQUEUE_REQUEST(ReleaseCallable, false);

    /// the tag uniquely identifying the request
    void* tag;
    bool ok;

    /// \fn Next
    ///
    /// \brief Read from the queue, blocking until an event(tag) is available or
    ///        the queue is shutting down. Next is blocking here for client side
    ///        request.
    ///
    /// \param tag: void* ;
    ///        [out] Updated to point to the read event's tag.
    ///
    /// \param ok: bool ;
    ///        [out] true if read a successful event, false otherwise.
    ///
    /// \details
    /// Next grpc API:
    /// https://grpc.github.io/grpc/cpp/classgrpc__impl_1_1_completion_queue.html#aed4c03e1d101c102ef289c2c472aa933
    /// Next returns true if got an event, false if the queue is fully drained
    /// and shut down.
    ///
    /// cq_: std::unique_ptr<::grpc::ServerCompletionQueue>
    /// the completion queue "cq" used for asynchronous communication.
    /// Tutorial: https://grpc.io/docs/tutorials/async/helloasync-cpp/
    while (cq_->Next(&tag, &ok)) {
      UntypedCall<GrpcMasterService>::Tag* callback_tag =
          static_cast<UntypedCall<GrpcMasterService>::Tag*>(tag);
      if (callback_tag) {
        /// Once received request from the client,
        /// OnCompleted indirectly invokes the grpc implementation functions.
        callback_tag->OnCompleted(this, ok);
      } else {
        // NOTE(mrry): A null `callback_tag` indicates that this is
        // the shutdown alarm.
        cq_->Shutdown();
      }
    }
  }

 private:
  /// class Master implements the server side grpc service function.
  Master* master_impl_ = nullptr;  // Not owned.
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  grpc::MasterService::AsyncService master_service_;

  mutex mu_;
  bool is_shutdown_ GUARDED_BY(mu_);
  const ConfigProto default_session_config_;
  ::grpc::Alarm* shutdown_alarm_ = nullptr;

  template <class RequestMessage, class ResponseMessage>
  using MasterCall = Call<GrpcMasterService, grpc::MasterService::AsyncService,
                          RequestMessage, ResponseMessage>;

  /** \brief Server side grpc handler function to create a session. Then, master
   *         server delegates it to Worker by requesting a grpc.
   *
   *  \param[in] call: MasterCall<CreateSessionRequest, CreateSessionResponse>*
   *         MasterCall is a type alias of a class "Call" defined in
   *         tensorflow/core/distributed_runtime/rpc/grpc_call.h.
   *         \todo direction is not clear, in or in,out?
   *
   *  \details
   *  - typename CreateSessionRequest;
   *         message CreateSessionRequest includes
   *         1. GraphDef: NodeDef, versionDef, FunctionDefLibrary;
   *         2. ConfigProto, including Session configuration parameters.
   *         3. The target string used from the client's perspective.
   *            e.g., "grpc://localhost:2223"
   *
   *  - typename CreateSessionResponse;
   *         message CreateSessionResponse includes
   *         1. session_handle string, and
   *         2. graph version for the subsequent call to ExtendSession.
   *
   *  \remark No return.
   */
  // RPC handler for creating a session.
  void CreateSessionHandler(
      MasterCall<CreateSessionRequest, CreateSessionResponse>* call) {
    CreateSessionRequest* rewritten_req = new CreateSessionRequest;
    rewritten_req->mutable_config()->MergeFrom(default_session_config_);
    /// request is an instance of CreateSessionRequest, template typename in
    /// class Call is RequestMessage. CreateSessionRequest is a proto message.
    /// \fn MergeFrom
    ///  Merge the fields from the given message into this message.
    ///  Singular fields will be overwritten, if specified in from, except for
    ///  embedded messages which will be merged.
    ///  Repeated fields will be concatenated. The given message must be of the
    ///  same type as this message same class).
    rewritten_req->MergeFrom(call->request);
    
    /// master_impl_ is Master*. Master::CreateSession is in master.cc.
    /// master_impl_->CreateSession will create a session in another thread.
    /// The callback lambda function is called at the very end of CreateSession
    /// to make sure client has received the server's response message.
    master_impl_->CreateSession(rewritten_req, &call->response,
                                [call, rewritten_req](const Status& status) {
                                  call->SendResponse(ToGrpcStatus(status));
                                  delete rewritten_req;
                                });
    /// This macro is invoked one or more times for each RPC method to
    /// ensure that there are sufficient completion queue entries to
    /// handle incoming requests without blocking.
    ENQUEUE_REQUEST(CreateSession, true);
  }

  // RPC handler for extending a session.
  void ExtendSessionHandler(
      MasterCall<ExtendSessionRequest, ExtendSessionResponse>* call) {
    master_impl_->ExtendSession(&call->request, &call->response,
                                [call](const Status& status) {
                                  call->SendResponse(ToGrpcStatus(status));
                                });
    ENQUEUE_REQUEST(ExtendSession, false);
  }

  // RPC handler for setting up a partial run call.
  void PartialRunSetupHandler(
      MasterCall<PartialRunSetupRequest, PartialRunSetupResponse>* call) {
    master_impl_->PartialRunSetup(&call->request, &call->response,
                                  [call](const Status& status) {
                                    call->SendResponse(ToGrpcStatus(status));
                                  });
    ENQUEUE_REQUEST(PartialRunSetup, false);
  }

  /** \brief Client send a request to master to run one step via a new thread.
   *
   *  \param call: MasterCall<RunStepRequest, RunStepResponse>*
   *
   *  \param RunStepRequest
   *
   *  \param RunStepResponse
   *
   *  \return
   */
  // RPC handler for running one step in a session.
  void RunStepHandler(MasterCall<RunStepRequest, RunStepResponse>* call) {
    auto* trace = TraceRpc("RunStep/Server", call->client_metadata());
    CallOptions* call_opts = new CallOptions;
    if (call->request.options().timeout_in_ms() > 0) {
      call_opts->SetTimeout(call->request.options().timeout_in_ms());
    } else {
      call_opts->SetTimeout(default_session_config_.operation_timeout_in_ms());
    }
    RunStepRequestWrapper* wrapped_request =
        new ProtoRunStepRequest(&call->request);
    MutableRunStepResponseWrapper* wrapped_response =
        new NonOwnedProtoRunStepResponse(&call->response);
    call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
    master_impl_->RunStep(
        call_opts, wrapped_request, wrapped_response,
        [call, call_opts, wrapped_request, wrapped_response,
         trace](const Status& status) {
          call->ClearCancelCallback();
          delete call_opts;
          delete wrapped_request;
          delete trace;
          if (call->request.store_errors_in_response_body() && !status.ok()) {
            call->response.set_status_code(status.code());
            call->response.set_status_error_message(status.error_message());
            call->SendResponse(ToGrpcStatus(Status::OK()));
          } else {
            call->SendResponse(ToGrpcStatus(status));
          }
        });
    ENQUEUE_REQUEST(RunStep, true);
  }

  // RPC handler for deleting a session.
  void CloseSessionHandler(
      MasterCall<CloseSessionRequest, CloseSessionResponse>* call) {
    master_impl_->CloseSession(&call->request, &call->response,
                               [call](const Status& status) {
                                 call->SendResponse(ToGrpcStatus(status));
                               });
    ENQUEUE_REQUEST(CloseSession, false);
  }

  // RPC handler for listing devices.
  void ListDevicesHandler(
      MasterCall<ListDevicesRequest, ListDevicesResponse>* call) {
    master_impl_->ListDevices(&call->request, &call->response,
                              [call](const Status& status) {
                                call->SendResponse(ToGrpcStatus(status));
                              });
    ENQUEUE_REQUEST(ListDevices, false);
  }

  // RPC handler for resetting all sessions.
  void ResetHandler(MasterCall<ResetRequest, ResetResponse>* call) {
    master_impl_->Reset(&call->request, &call->response,
                        [call](const Status& status) {
                          call->SendResponse(ToGrpcStatus(status));
                        });
    ENQUEUE_REQUEST(Reset, false);
  }

  // RPC handler for making a callable.
  void MakeCallableHandler(
      MasterCall<MakeCallableRequest, MakeCallableResponse>* call) {
    master_impl_->MakeCallable(&call->request, &call->response,
                               [call](const Status& status) {
                                 call->SendResponse(ToGrpcStatus(status));
                               });
    ENQUEUE_REQUEST(MakeCallable, false);
  }

  // RPC handler for running a callable.
  void RunCallableHandler(
      MasterCall<RunCallableRequest, RunCallableResponse>* call) {
    auto* trace = TraceRpc("RunCallable/Server", call->client_metadata());
    CallOptions* call_opts = new CallOptions;
    // The timeout may be overridden by a non-zero timeout in the
    // callable's `RunOptions`; this overriding will happen inside the
    // `MasterSession` implementation.
    call_opts->SetTimeout(default_session_config_.operation_timeout_in_ms());
    call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
    master_impl_->RunCallable(call_opts, &call->request, &call->response,
                              [call, call_opts, trace](const Status& status) {
                                call->ClearCancelCallback();
                                delete call_opts;
                                delete trace;
                                call->SendResponse(ToGrpcStatus(status));
                              });
    ENQUEUE_REQUEST(RunCallable, false);
  }

  // RPC handler for making a callable.
  void ReleaseCallableHandler(
      MasterCall<ReleaseCallableRequest, ReleaseCallableResponse>* call) {
    master_impl_->ReleaseCallable(&call->request, &call->response,
                                  [call](const Status& status) {
                                    call->SendResponse(ToGrpcStatus(status));
                                  });
    ENQUEUE_REQUEST(ReleaseCallable, false);
  }

#undef ENQUEUE_REQUEST

  // Start tracing, including the ID attached to the RPC.
  tracing::ScopedActivity* TraceRpc(
      StringPiece name,
      const std::multimap<::grpc::string_ref, ::grpc::string_ref>& metadata) {
    StringPiece id;
    auto it = metadata.find(GrpcIdKey());
    if (it != metadata.end()) {
      id = StringPiece(it->second.data(), it->second.size());
    }
    return new tracing::ScopedActivity(name, id);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcMasterService);
};

AsyncServiceInterface* NewGrpcMasterService(
    Master* master, const ConfigProto& default_session_config,
    ::grpc::ServerBuilder* builder) {
  return new GrpcMasterService(master, default_session_config, builder);
}

}  // end namespace tensorflow
