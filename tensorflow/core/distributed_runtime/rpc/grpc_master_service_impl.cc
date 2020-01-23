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

#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"

#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/channel_interface.h"
#include "grpcpp/impl/codegen/client_unary_call.h"
#include "grpcpp/impl/codegen/method_handler_impl.h"
#include "grpcpp/impl/codegen/rpc_service_method.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/sync_stream.h"

namespace tensorflow {

namespace grpc {

static const char* grpcMasterService_method_names[] = {
    "/tensorflow.MasterService/CreateSession",
    "/tensorflow.MasterService/ExtendSession",
    "/tensorflow.MasterService/PartialRunSetup",
    "/tensorflow.MasterService/RunStep",
    "/tensorflow.MasterService/CloseSession",
    "/tensorflow.MasterService/ListDevices",
    "/tensorflow.MasterService/Reset",
    "/tensorflow.MasterService/MakeCallable",
    "/tensorflow.MasterService/RunCallable",
    "/tensorflow.MasterService/ReleaseCallable",
};

/** \brief MasterService::NewStub.
 *
 *  \param[in] channel: const std::shared_ptr< ::grpc::ChannelInterface>& ;
 *
 *  \param[in] options: const ::grpc::StubOptions&,
 *         its default is ::grpc::StubOptions()
 *         Its default is set in declaration of NewStub.
 *
 *  \return std::unique_ptr<MasterService::Stub>
 */
std::unique_ptr<MasterService::Stub> MasterService::NewStub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel,
    const ::grpc::StubOptions& options) {
  std::unique_ptr<MasterService::Stub> stub(new MasterService::Stub(channel));
  return stub;
}

/** \brief MasterService::Stub constructor.
 *
 *  \param channel: const std::shared_ptr< ::grpc::ChannelInterface>& ;
 *         ::grpc::ChannelInterface is Codegen interface for grpc::Channel.
 */
MasterService::Stub::Stub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel)
    : channel_(channel),

      /// \brief rpcmethod_CreateSession_ is an object of
      ///        grpc::internal::RpcMethod. Here, we initialize this instance by
      ///        calling RpcMethod constructor.
      ///
      /// \param[in] grpcMasterService_method_names[0]: const char *name
      ///
      /// \param[in] ::grpc::internal::RpcMethod::NORMAL_RPC: RpcType type
      ///        RpcType { NORMAL_RPC = 0, CLIENT_STREAMING,
      ///                  SERVER_STREAMING, BIDI_STREAMING }
      ///
      /// \param[in] channel: const std::shared_ptr< ChannelInterface > &channel
      ///        ChannelInterface is Codegen interface for grpc::Channel, which
      ///        represents a connection to an endpoint.
      rpcmethod_CreateSession_(grpcMasterService_method_names[0],
                               ::grpc::internal::RpcMethod::NORMAL_RPC,
                               channel),
      rpcmethod_ExtendSession_(grpcMasterService_method_names[1],
                               ::grpc::internal::RpcMethod::NORMAL_RPC,
                               channel),
      rpcmethod_PartialRunSetup_(grpcMasterService_method_names[2],
                                 ::grpc::internal::RpcMethod::NORMAL_RPC,
                                 channel),
      rpcmethod_RunStep_(grpcMasterService_method_names[3],
                         ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_CloseSession_(grpcMasterService_method_names[4],
                              ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_ListDevices_(grpcMasterService_method_names[5],
                             ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_Reset_(grpcMasterService_method_names[6],
                       ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_MakeCallable_(grpcMasterService_method_names[7],
                              ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_RunCallable_(grpcMasterService_method_names[8],
                             ::grpc::internal::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_ReleaseCallable_(grpcMasterService_method_names[9],
                                 ::grpc::internal::RpcMethod::NORMAL_RPC,
                                 channel) {}

/** \brief grpc stub of creating a session. This function is called by a client.
 *         Delegate it to worker grpc.
 *
 *  \param context: ::grpc::ClientContext* ;
 *  A ClientContext allows the person implementing a service client to:
 *  - Add custom metadata key-value pairs that will propagated to the server
 *    side.
 *  - Control call settings such as compression and authentication.
 *  - Initial and trailing metadata coming from the server.
 *  - Get performance metrics (ie, census).
 *
 *  \param request: const CreateSessionRequest& ;
 *         CreateSessionRequest message includes 1. GraphDef 2. ConfigProto
 *         3. target string.
 *
 *  \param response: CreateSessionResponse* ;
 *         CreateSessionResponse message includes 1. session_handle string
 *         2. graph_version.
 *
 *  \return ::grpc::Status
 */
::grpc::Status MasterService::Stub::CreateSession(
    ::grpc::ClientContext* context, const CreateSessionRequest& request,
    CreateSessionResponse* response) {

  /// \note
  /// \fn BlockingUnaryCall
  ///
  /// \brief a wrapper that performs a blocking unary call. A unary call means
  ///        a request by a client and a response by a server.
  ///
  /// \param channel_.get(): ChannelInterface * channel ;
  ///        ChannelInterface is Codegen interface for grpc::Channel, which
  ///        represents a connection to an endpoint.
  ///
  /// \param rpcmethod_CreateSession_: const RpcMethod & method ;
  ///        Descriptor of an RPC method.
  ///
  /// \param context: ClientContext * context ;
  ///
  /// \param request: const InputMessage & request ; InputMessage is a typename.
  ///
  /// \param response: OutputMessage * result ; OutputMessage is a typename.
  ///
  /// \return ::grpc::Status
  ///
  /// \note The response must be returned, then the next response can be sent.

  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_CreateSession_, context, request, response);
}

::grpc::Status MasterService::Stub::ExtendSession(
    ::grpc::ClientContext* context, const ExtendSessionRequest& request,
    ExtendSessionResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_ExtendSession_, context, request, response);
}

::grpc::Status MasterService::Stub::PartialRunSetup(
    ::grpc::ClientContext* context, const PartialRunSetupRequest& request,
    PartialRunSetupResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_PartialRunSetup_, context, request, response);
}

::grpc::Status MasterService::Stub::RunStep(::grpc::ClientContext* context,
                                            const RunStepRequest& request,
                                            RunStepResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(channel_.get(), rpcmethod_RunStep_,
                                             context, request, response);
}

::grpc::Status MasterService::Stub::CloseSession(
    ::grpc::ClientContext* context, const CloseSessionRequest& request,
    CloseSessionResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_CloseSession_, context, request, response);
}

::grpc::Status MasterService::Stub::ListDevices(
    ::grpc::ClientContext* context, const ListDevicesRequest& request,
    ListDevicesResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_ListDevices_, context, request, response);
}

::grpc::Status MasterService::Stub::Reset(::grpc::ClientContext* context,
                                          const ResetRequest& request,
                                          ResetResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(channel_.get(), rpcmethod_Reset_,
                                             context, request, response);
}

::grpc::Status MasterService::Stub::MakeCallable(
    ::grpc::ClientContext* context, const MakeCallableRequest& request,
    MakeCallableResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_MakeCallable_, context, request, response);
}

::grpc::Status MasterService::Stub::RunCallable(
    ::grpc::ClientContext* context, const RunCallableRequest& request,
    RunCallableResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_RunCallable_, context, request, response);
}

::grpc::Status MasterService::Stub::ReleaseCallable(
    ::grpc::ClientContext* context, const ReleaseCallableRequest& request,
    ReleaseCallableResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_ReleaseCallable_, context, request, response);
}

MasterService::AsyncService::AsyncService() {
  int method_len = sizeof(grpcMasterService_method_names) /
                    sizeof(grpcMasterService_method_names[0]);
  for (int i = 0; i < method_len; ++i) {
    /// void grpc::Service::AddMethod(internal::RpcServiceMethod * method)
    AddMethod(new ::grpc::internal::RpcServiceMethod(
        grpcMasterService_method_names[i],
        ::grpc::internal::RpcMethod::NORMAL_RPC, nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

MasterService::AsyncService::~AsyncService() {}

}  // namespace grpc

}  // namespace tensorflow
