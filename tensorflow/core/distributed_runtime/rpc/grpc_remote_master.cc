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
/** \file grpc_remote_master.cc
 *
 *  \brief This file is a wrapper of grpc_master_service_impl.cc .
 *         It indirectly calls those grpc stubs, which are called by clients,
 *         from grpc_master_service_impl.cc .
 */
#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_master.h"

#include <utility>

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/master_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow {

// GrpcRemoteMaster is an implementation of the MasterInterface
// that uses gRPC to talk to the Master service.
class GrpcRemoteMaster : public MasterInterface {
  using MasterServiceStub = grpc::MasterService::Stub;

 public:
  /** \brief GrpcRemoteMaster Constructor, initialize grpc stub/service that a
   *         client can use to send a request to the server.
   *
   *  \param[in] client_channel: const SharedGrpcChannelPtr&
   *         SharedGrpcChannelPtr is std::shared_ptr<::grpc::Channel>, a pointer
   *         of channel represents a connection to the target.
   *
   *  \details A client sends a request to the server using the stub. So, here
   *           Remote Master creates a MasterService::Stub .
   */
  explicit GrpcRemoteMaster(const SharedGrpcChannelPtr& client_channel)
      : stub_(grpc::MasterService::NewStub(client_channel)) {}

  ~GrpcRemoteMaster() override {}

  /** \brief A grpc stub/interface to call CreateSession by a client.
   *         This the request that client sends. This function is blocking until
   *         server sends response back to client itself.
   *
   *  \param[in] call_options: CallOptions* ;
   *         CallOptions is options for the grpc call.
   *
   *  \param[in] request: const CreateSessionRequest* ;
   *         message CreateSessionRequest includes
   *         1. GraphDef: NodeDef, versionDef, FunctionDefLibrary;
   *         2. ConfigProto, including Session configuration parameters.
   *         3. The target string used from the client's perspective.
   *            e.g., "grpc://localhost:2223"
   *
   *  \param[in,out] response: CreateSessionResponse* ;
   *         message CreateSessionResponse includes
   *         1. session_handle string, and 2. graph version for the subsequent
   *         call to ExtendSession.
   *
   *  \return Status
   */
  Status CreateSession(CallOptions* call_options,
                       const CreateSessionRequest* request,
                       CreateSessionResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::CreateSession);
  }

  Status ExtendSession(CallOptions* call_options,
                       const ExtendSessionRequest* request,
                       ExtendSessionResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::ExtendSession);
  }

  Status PartialRunSetup(CallOptions* call_options,
                         const PartialRunSetupRequest* request,
                         PartialRunSetupResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::PartialRunSetup);
  }

  /** \brief
   *
   */
  Status RunStep(CallOptions* call_options, RunStepRequestWrapper* request,
                 MutableRunStepResponseWrapper* response) override {
    return CallWithRetry(call_options, &request->ToProto(),
                         get_proto_from_wrapper(response),
                         &MasterServiceStub::RunStep, "RunStep/Client");
  }

  Status CloseSession(CallOptions* call_options,
                      const CloseSessionRequest* request,
                      CloseSessionResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::CloseSession);
  }

  Status ListDevices(CallOptions* call_options,
                     const ListDevicesRequest* request,
                     ListDevicesResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::ListDevices);
  }

  Status Reset(CallOptions* call_options, const ResetRequest* request,
               ResetResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::Reset);
  }

  Status MakeCallable(CallOptions* call_options,
                      const MakeCallableRequest* request,
                      MakeCallableResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::MakeCallable);
  }
  Status RunCallable(CallOptions* call_options,
                     const RunCallableRequest* request,
                     RunCallableResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::RunCallable);
  }
  Status ReleaseCallable(CallOptions* call_options,
                         const ReleaseCallableRequest* request,
                         ReleaseCallableResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::ReleaseCallable);
  }

 private:
  // Start tracing, attaching a unique ID to both the trace and the RPC.
  tracing::ScopedActivity* NewTraceRpc(StringPiece name,
                                       ::grpc::ClientContext* ctx) {
    string trace_id = strings::StrCat(tracing::GetUniqueArg());
    ctx->AddMetadata(GrpcIdKey(), trace_id);
    return new tracing::ScopedActivity(name, trace_id);
  }

  /** \brief A template grpc function to handle grpc calls and its return status
   *
   *  \param Request: typename ; e.g.,
   *         - CreateSessionRequest
   *         - ExtendSessionRequest
   *         - RunStepRequestWrapper etc.
   *
   *  \param Response: typename ; e.g.,
   *         - CreateSessionResponse
   *         - ExtendSessionResponse
   *         - MutableRunStepResponseWrapper etc.
   *
   *  \param[in] call_options: CallOptions* ;
   *         CallOptions is options for the grpc call.
   *
   *  \param[in] request: const Request* ;
   *         e.g., a variable of type of CreateSessionRequest.
   *
   *  \param[in,out] response: Response* ;
   *         e.g. a variable of type of CreateSessionResponse.
   *
   *  \param[in] ::grpc::Status (MasterServiceStub::*pfunc)(
   *                                                    ::grpc::ClientContext*,
   *                                                    const Request&,
   *                                                    Response*)
   *         is a lambda function. Its signature is a MasterService::Stub member
   *         function.
   *
   *  \param trace_string: string, default value is {} ;
   *
   *  \return Status
   */
  template <typename Request, typename Response>
  Status CallWithRetry(CallOptions* call_options, const Request* request,
                       Response* response,
                       ::grpc::Status (MasterServiceStub::*pfunc)(
                           ::grpc::ClientContext*, const Request&, Response*),
                       string trace_string = {}) {
    int64 timeout_in_ms = call_options->GetTimeout();
    int64 expired_time_micros = Env::Default()->NowMicros();
    if (timeout_in_ms > 0) {
      expired_time_micros += (timeout_in_ms / 1000.);
    }
    Status s;
    /// retrying the RPC but not exceeds kMaxRetries = 10.
    for (int num_retries = 0;; ++num_retries) {
      ::grpc::ClientContext ctx;
      std::unique_ptr<tracing::ScopedActivity> trace;
      if (!trace_string.empty()) {
        trace.reset(NewTraceRpc(trace_string, &ctx));
      }
      ctx.set_fail_fast(false);
      if (timeout_in_ms > 0) {
        // We do not modify the timeout here to match legacy behavior. However,
        // this could violate the contract of tensorflow::Session. If we retry
        // an RPC just before the deadline is exceeded, we will still set the
        // timeout to the original value. This leads to the overall timeout
        // being double what was expected.
        // TODO(b/117162170): investigate fixing this behavior for legacy and
        // gRPC RPC layers.
        ctx.set_deadline(gpr_time_from_millis(timeout_in_ms, GPR_TIMESPAN));
      }

      /// \note
      ///  This is the wildcard grpc function requested by a client.
      ///  This function is blocking until server sends response back to the
      ///  client.
      ///  Its implementation is in
      ///  tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.cc
      ///  After invoking the stub of grpc, checking its return status.
      s = FromGrpcStatus((stub_.get()->*pfunc)(&ctx, *request, response));
      if (!errors::IsUnavailable(s)) {
        /// return if grpc is successful.
        return s;
      }
      // TODO(b/117162170): we may want to make this configurable.
      constexpr int kMaxRetries = 10;
      LOG(WARNING) << "RPC failed with status = \"" << s
                   << "\" and grpc_error_string = \""
                   << ctx.debug_error_string() << "\", maybe retrying the RPC";
      if (num_retries >= kMaxRetries) {
        LOG(WARNING) << "Too many retries, returning last status: " << s;
        return s;
      }
      const int64 now_micros = Env::Default()->NowMicros();
      const int64 deadline_with_backoff_micros =
          now_micros + ComputeBackoffMicroseconds(num_retries);
      // Wait for a short period of time before retrying the RPC.  If our
      // backoff would put us past the RPC deadline, we truncate it to ensure
      // our RPC starts before the deadline.
      const auto backoff_until =
          (timeout_in_ms <= 0 ||
           expired_time_micros > deadline_with_backoff_micros)
              ? deadline_with_backoff_micros
              : expired_time_micros;
      Env::Default()->SleepForMicroseconds(backoff_until - now_micros);
      if (Env::Default()->NowMicros() > expired_time_micros &&
          timeout_in_ms > 0) {
        // If timeout_in_ms is set, exit the retry loop on timeout.
        return errors::DeadlineExceeded(ctx.debug_error_string());
      }
    }
  }

  std::unique_ptr<MasterServiceStub> stub_;
};

/** \brief Create a new GrpcRemoteMaster for clients to call stubs about Session
 *         interface, 1. create session 2. extend session 3. run step etc.
 *
 *  \param[in] channel: const SharedGrpcChannelPtr& ;
 *         SharedGrpcChannelPtr is std::shared_ptr<::grpc::Channel>, a pointer
 *         of channel represents a connection to the target.
 *
 *  \return MasterInterface* ;
 *          Master side grpc session 1.creation 2. extend 3. run step etc.
 *          interface.
 *
 *  \details MasterService::Stub :
 *           MasterService implements stub for clients to call defined in
 *           master_service.proto interface.
 */
MasterInterface* NewGrpcMaster(const SharedGrpcChannelPtr& channel) {
  return new GrpcRemoteMaster(channel);
}

}  // namespace tensorflow
