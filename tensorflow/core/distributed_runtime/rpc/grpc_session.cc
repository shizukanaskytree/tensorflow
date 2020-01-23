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

#include "tensorflow/core/distributed_runtime/rpc/grpc_session.h"

#include <unordered_map>

#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/local_master.h"
#include "tensorflow/core/distributed_runtime/master_interface.h"
#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_master.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow {

const char* const kSchemePrefix = "grpc://";
const size_t kSchemePrefixLength = strlen(kSchemePrefix);

/** \brief GrpcSession Constructor
 *
 *  \param options: const SessionOptions& ;
 *
 *  \note Constructor doesn't do any processing. It only store the
 *        SessionOptions.
 */
GrpcSession::GrpcSession(const SessionOptions& options)
    : options_(options), current_graph_version_(-1) {}

GrpcSession::~GrpcSession() {}

/** \brief Create a GrpcSession subject to SessionOptions.
 *
 *  \param[in] options: SessionOptions& ;
 *         SessionOptions is used to provide 1. environment used; 2. target
 *         used to perform all computations according to host:port. 3. A bunch
 *         of Session configuration parameters.
 *
 *  \param[out] out_session: std::unique_ptr<GrpcSession>* ;
 *         GrpcSession is the return value.
 *         GrpcSession implements the interface from class Session.
 *         1. Create; 2. Extend; 3. Close; 4. Run; etc.
 *
 *  \return Status.
 *
 *  \details
 *   Finally, it set one attributes of the GrpcSession.
 *      - std::unique_ptr<MasterInterface> master_;
 *   GrpcSession provides grpc stubs via GrpcSession::master_ , a
 *   MasterInterface.
 *   GrpcSession is a derived class of Session interface, including
 *   1. Create 2. Extend 3. Close 4. LocalDeviceManager
 *   5. MakeCallable 6. RunCallable 7. ReleaseCallable.
 */
/* static */
Status GrpcSession::Create(const SessionOptions& options,
                           std::unique_ptr<GrpcSession>* out_session) {
  /// A temp GrpcSession unique pointer for the return value.
  std::unique_ptr<GrpcSession> session(new GrpcSession(options));
  /// Abstract interface for communicating with the TensorFlow Master service.
  std::unique_ptr<MasterInterface> master;
  // For testing, we enable the client to disable the use of the local
  // master registry, so that the RPC stack is exercised.
  if (!options.config.rpc_options().use_rpc_for_inprocess_master()) {
    /// Get LocalMaster if the ConfigProto option specifies
    /// use_rpc_for_inprocess_master option, which can avoid grpc stack.
    /// This option is primarily for used testing the RPC stack.
    /// Local master registry has local master inside, so ret is nullptr.
    master = LocalMaster::Lookup(options.target);
  }
  if (!master) {
    /// When master from local is nullptr, so it tries to new a remote master.
    /// SharedGrpcChannelPtr is a type alias of std::shared_ptr<::grpc::Channel>
    /// Channels represent a connection to an endpoint.
    SharedGrpcChannelPtr master_channel;
    TF_RETURN_IF_ERROR(
        NewHostPortGrpcChannel(options.target.substr(kSchemePrefixLength),
                               &options.config.rpc_options(), &master_channel));
    /// This is the main purpose of GrpcSession:
    /// Create a master handler to handle client grpc request.
    master.reset(NewGrpcMaster(master_channel));
  }
  session->SetRemoteMaster(std::move(master));
  *out_session = std::move(session);
  return Status::OK();
}

namespace {

/** \brief
 *
 *  \param[in,out] gdef: GraphDef* ;
 *
 */
// Re-encodes constant represented in tensor proto into
// tensor_content, which is slightly better (less copies and lower peak
// memory usage) when used with rpc subsystems.
void ReEncodeConsts(GraphDef* gdef) {
  for (NodeDef& ndef : *(gdef->mutable_node())) {
    if (ndef.op() == "Const") {
      /// TensorProto is a message protobuf defining representing a tensor.
      TensorProto* proto = nullptr;
      /// mutable_attr: Returns a pointer to the mutable attr object that stores
      /// the field's value.
      for (auto& attr : *ndef.mutable_attr()) {
        if (attr.first == "value") {
          proto = attr.second.mutable_tensor();
        }
      }
      if (proto != nullptr && proto->tensor_content().empty() &&
          proto->ByteSizeLong() > 64) {
        // If the constant is encoded with repeated proto fields and
        // it is moderate large, we re-encode it in tensor_content as
        // a Cord. This is mildly helpful for reducing the peak memory
        // usage on the server side where GraphDef/NodeDef are copied
        // quite often.
        /// \note dtype: DataType
        /// \note class Tensor
        /// Tensor constructor creates an empty Tensor of the given data
        /// type with 1-dimensional, 0-element Tensor.
        Tensor parsed(proto->dtype());
        /// FromProto function parses `*proto` and initializes the tensor.
        if (parsed.FromProto(*proto)) {
          /// \brief
          /// By taking advantage of AsProtoTensorContent from class Tensor to
          /// encode content from proto of the gdef in a compact form.
          ///
          /// AsProtoTensorContent encodes the content in
          /// `proto.tensor_content()` in a compact form.
          parsed.AsProtoTensorContent(proto);
        }
      }
    }
  }
}
}  // namespace

/** \brief To extract return value, session handler string and graph version,
 *         from message CreateSessionResponse resp.
 *
 *  \param[in] handle: string
 *        - set handle_, returned by the master grpc to identify this session.
 *
 *  \param[in] graph_version: int64
 *        - The initial version number for the graph, to be used in the next
 *          call to ExtendSession.
 *
 *  \remark No return value.
 */
void GrpcSession::SetHandleAndGraphVersion(string handle, int64 graph_version) {
  mutex_lock l(mu_);
  handle_ = std::move(handle);
  current_graph_version_ = graph_version;
}

Status GrpcSession::Handle(string* out_handle) {
  mutex_lock l(mu_);
  if (handle_.empty()) {
    return errors::InvalidArgument("A session is not created yet....");
  }
  *out_handle = handle_;
  return Status::OK();
}

/** \brief grpc stub of creating a session called by a client.
 *
 *  \param[in] call_options: CallOptions* ;
 *        - Options passed to grpc interface calls.
 *
 *  \param[in] graph: const GraphDef& ;
 *        - GraphDef message represents the graph of operations, including
 *         NodeDef, VersionDef, and FunctionDefLibrary.
 *
 *  \return Status
 *
 *  \details
 *   The functionality of this function is to set
 *   - GrpcSession::handle_ , to identify remote master session.
 *   - Grpc::current_graph_version_
 */
Status GrpcSession::CreateImpl(CallOptions* call_options,
                               const GraphDef& graph) {
  {
    mutex_lock l(mu_);
    if (!handle_.empty()) {
      return errors::InvalidArgument("A session is alive.");
    }
  }
  CreateSessionRequest req;
  *req.mutable_config() = options_.config;
  *req.mutable_graph_def() = graph;
  req.set_target(options_.target);
  ReEncodeConsts(req.mutable_graph_def());
  CreateSessionResponse resp;

  /// master_: std::unique_ptr<MasterInterface>
  /// \note master_ real type is class GrpcRemoteMaster, a derived class
  ///       implements MasterInterface.
  /// grpc session use this master interface/stub to invoke session related
  /// functions, like create a session, extend a session, run a step.
  Status s = master_->CreateSession(call_options, &req, &resp);
  if (s.ok()) {
    /// extract return value, remote master session handler string and graph
    /// version, from message CreateSessionResponse resp.
    SetHandleAndGraphVersion(resp.session_handle(), resp.graph_version());
  }
  return s;
}

/** \brief This is the real grpc stub of creating a session by a client, and
 *         called by a client.
 *
 *  \param graph[in]: const GraphDef& ;
 *
 *  \details
 */
Status GrpcSession::Create(const GraphDef& graph) {
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  return CreateImpl(&call_options, graph);
}

/** \brief A wrapper to call CreateImpl.
 *
 *  \param[in] run_options: const RunOptions& ;
 *         - RunOptions message is Options for a single Run() call.
 *
 *  \param[in] graph: const GraphDef& ;
 *         - GraphDef message represents the graph of operations, including
 *         NodeDef, VersionDef, and FunctionDefLibrary.
 *
 *  \details
 *   Finally, it set two attributes of the GrpcSession.
 *   - GrpcSession::handle_: string
 *     - handle_ returned by the master to identify this session.
 *   - GrpcSession::current_graph_version_: int64
 *     - The current version of the graph.
 *
 *  \return Status
 */
Status GrpcSession::Create(const RunOptions& run_options,
                           const GraphDef& graph) {
  CallOptions call_options;
  call_options.SetTimeout(run_options.timeout_in_ms());
  return CreateImpl(&call_options, graph);
}

/** \brief Extend the graph by adding more nodes to its original graph.
 *
 *  \param[in] call_options: CallOptions*
 *         CallOptions are options passed to interface calls. This class
 *         provides portable functionality across different RPC systems on top
 *         of platform-specific mechanisms (for client and server contexts,
 *         cancellation, etc.).
 *
 *  \param[in] graph: const GraphDef&
 *         GraphDef message represents the graph of operations, including
 *         NodeDef, VersionDef, and FunctionDefLibrary.
 *
 *  \return Status
 *
 *  \details This function only changes GrpcSession::current_graph_version_
 *
 *  \todo Q. Why is enough to only get current_graph_version_ by this function?
 */
Status GrpcSession::ExtendImpl(CallOptions* call_options,
                               const GraphDef& graph) {
  bool handle_is_empty;
  {
    mutex_lock l(mu_);
    /// handle_ is returned by the master to identify this session.
    handle_is_empty = handle_.empty();
  }
  if (handle_is_empty) {
    // Session was unitialized, so simply initialize the session with 'graph'.
    /// When I do gdb, handle_ is empty even the client has called the
    /// GrpcSession::Create
    /// The result is to set
    /// - GrpcSession::handle_ , to identify this session.
    /// - Grpc::current_graph_version_
    return Create(graph);
  }
  mutex_lock l(mu_);
  ExtendSessionRequest req;
  /// select the remote master session handler string, i.e., to specify which
  /// master session in the server side to use.
  req.set_session_handle(handle_);
  *req.mutable_graph_def() = graph;
  req.set_current_graph_version(current_graph_version_);
  ExtendSessionResponse resp;
  /// master_: std::unique_ptr<MasterInterface>
  /// \note master_ real type is class GrpcRemoteMaster, a derived class
  ///       implements MasterInterface.
  Status s = master_->ExtendSession(call_options, &req, &resp);
  if (s.ok()) {
    current_graph_version_ = resp.new_graph_version();
  }
  return s;
}

/** \brief A wrapper of Extend function, which further call ExtendImpl.
 *
 *  \param[in] graph: const GraphDef&
 *         GraphDef message represents the graph of operations, including
 *         NodeDef, VersionDef, and FunctionDefLibrary.
 *
 *  \return Status
 */
Status GrpcSession::Extend(const GraphDef& graph) {
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  return ExtendImpl(&call_options, graph);
}

Status GrpcSession::Extend(const RunOptions& run_options,
                           const GraphDef& graph) {
  CallOptions call_options;
  call_options.SetTimeout(run_options.timeout_in_ms());
  return ExtendImpl(&call_options, graph);
}

 /** \brief Client sends a request to run the graph.
  *
  *  \param run_options: const RunOptions& ;
  *         Options for a single Run() call.
  *
  *  \param inputs: const std::vector<std::pair<string,Tensor>>& ;
  *  \todo  Let data be stored close to GPU and we don't need to send tensor by
  *         grpc.
  *
  *  \param output_tensor_names: const std::vector<string>& ;
  *
  *  \param target_node_names: const std::vector<string>& ;
  *
  *  \param outputs: std::vector<Tensor>* ;
  *         class Tensor is defined in tensor.h; Tensor represents n-dimensional
  *         array of values.
  *
  *  \param run_metadata: RunMetadata* ;
  *         Metadata output (i.e., non-Tensor) for a single Run() call.
  *
  *  \param prun_handle: const string& ;
  *         It is "".
  *
  *  \return Status ;
  */
Status GrpcSession::RunHelper(
    const RunOptions& run_options,
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& output_tensor_names,
    const std::vector<string>& target_node_names, std::vector<Tensor>* outputs,
    RunMetadata* run_metadata, const string& prun_handle) {
  // Convert to proto
  std::unique_ptr<MutableRunStepRequestWrapper> req(
      master_->CreateRunStepRequest());
  std::unique_ptr<MutableRunStepResponseWrapper> resp(
      master_->CreateRunStepResponse());

  *req->mutable_options() = run_options;

  if (run_options.timeout_in_ms() == 0) {
    req->mutable_options()->set_timeout_in_ms(
        options_.config.operation_timeout_in_ms());
  }

  if (!prun_handle.empty()) {
    req->set_partial_run_handle(prun_handle);
  }

  for (const auto& it : inputs) {
    req->add_feed(it.first, it.second);
  }

  // Support long error messages by storing the error code in the response body.
  req->set_store_errors_in_response_body(true);

  // Build an index from fetch tensor name to first index in
  // output_tensor_names.
  std::unordered_map<string, int> output_name_to_offset;
  for (int i = 0; i < output_tensor_names.size(); ++i) {
    const string& name = output_tensor_names[i];
    if (output_name_to_offset.insert(std::make_pair(name, i)).second) {
      req->add_fetch(name);
    }
  }
  for (const string& target : target_node_names) {
    req->add_target(target);
  }

  CallOptions call_options;
  call_options.SetTimeout(req->options().timeout_in_ms());
  /// RunProto invokes the grpc call
  TF_RETURN_IF_ERROR(RunProto(&call_options, req.get(), resp.get()));

  // Look for an extended error returned in the response body.
  if (resp->status_code() != error::Code::OK) {
    return Status(resp->status_code(), resp->status_error_message());
  }

  if (!output_tensor_names.empty()) {
    outputs->resize(output_tensor_names.size());
  }

  // Convert response back to Tensors in the correct order.
  for (size_t i = 0; i < resp->num_tensors(); ++i) {
    auto fetch_it = output_name_to_offset.find(resp->tensor_name(i));
    if (fetch_it == output_name_to_offset.end()) {
      return errors::Internal("Received response for unrequested fetch: ",
                              resp->tensor_name(i));
    }

    Tensor output;
    TF_RETURN_IF_ERROR(resp->TensorValue(i, &output));
    (*outputs)[fetch_it->second] = output;
  }
  // In the unlikely event that output_tensor_names contains duplicates, fill in
  // the duplicate values.
  if (output_name_to_offset.size() != output_tensor_names.size()) {
    for (int i = 0; i < output_tensor_names.size(); ++i) {
      const string& name = output_tensor_names[i];
      int offset = output_name_to_offset[name];
      if (offset != i) {
        (*outputs)[i] = (*outputs)[offset];
      }
    }
  }

  if (run_metadata) {
    run_metadata->Swap(resp->mutable_metadata());
  }

  return Status::OK();
}

/** \brief Client side calls Run.
 *
 *  \param run_options: const RunOptions& ;
 *         Options for a single Run() call.
 *
 *  \param inputs: const std::vector<std::pair<string,Tensor>>& ;
 *  \todo  Let data be stored close to GPU and we don't need to send tensor by
 *         grpc.
 *
 *  \param output_tensor_names: const std::vector<string>& ;
 *
 *  \param target_node_names: const std::vector<string>& ;
 *
 *  \param outputs: std::vector<Tensor>* ;
 *         class Tensor is defined in tensor.h; Tensor represents n-dimensional
 *         array of values.
 *
 *  \param run_metadata: RunMetadata* ;
 *         Metadata output (i.e., non-Tensor) for a single Run() call.
 *
 *  \return Status ;
 */
Status GrpcSession::Run(const RunOptions& run_options,
                        const std::vector<std::pair<string, Tensor>>& inputs,
                        const std::vector<string>& output_tensor_names,
                        const std::vector<string>& target_node_names,
                        std::vector<Tensor>* outputs,
                        RunMetadata* run_metadata) {
  return RunHelper(run_options, inputs, output_tensor_names, target_node_names,
                   outputs, run_metadata, /* prun_handle */ "");
}

Status GrpcSession::Run(const std::vector<std::pair<string, Tensor>>& inputs,
                        const std::vector<string>& output_tensor_names,
                        const std::vector<string>& target_node_names,
                        std::vector<Tensor>* outputs) {
  RunOptions run_options;
  run_options.set_timeout_in_ms(options_.config.operation_timeout_in_ms());
  return Run(run_options, inputs, output_tensor_names, target_node_names,
             outputs, nullptr);
}

/** \brief
 *
 *  \param call_options: CallOptions* ;
 *
 *  \param req: MutableRunStepRequestWrapper* ;
 *         Abstract interface for a mutable RunStepRequest message.
 *         The real type is MutableProtoRunStepRequest.
 *         This wrapper class should be used for RunStep requests between a
 *         client and master in different address spaces.
 *
 *  \param resp: MutableRunStepResponseWrapper* ;
 *         Abstract interface for a mutable RunStepResponse message.
 *
 */
Status GrpcSession::RunProto(CallOptions* call_options,
                             MutableRunStepRequestWrapper* req,
                             MutableRunStepResponseWrapper* resp) {
  string handle;
  TF_RETURN_IF_ERROR(Handle(&handle));
  req->set_session_handle(handle);
  return master_->RunStep(call_options, req, resp);
}

Status GrpcSession::PRunSetup(const std::vector<string>& input_names,
                              const std::vector<string>& output_names,
                              const std::vector<string>& target_nodes,
                              string* handle) {
  // Convert to proto
  PartialRunSetupRequest req;
  PartialRunSetupResponse resp;
  CallOptions call_options;
  TF_RETURN_IF_ERROR(Handle(req.mutable_session_handle()));
  for (const string& feed : input_names) {
    req.add_feed(feed);
  }
  for (const string& fetch : output_names) {
    req.add_fetch(fetch);
  }
  for (const string& target : target_nodes) {
    req.add_target(target);
  }
  req.set_request_id(GetUniqueRequestId());
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  TF_RETURN_IF_ERROR(master_->PartialRunSetup(&call_options, &req, &resp));
  *handle = resp.partial_run_handle();
  return Status::OK();
}

Status GrpcSession::PRun(const string& handle,
                         const std::vector<std::pair<string, Tensor>>& inputs,
                         const std::vector<string>& output_names,
                         std::vector<Tensor>* outputs) {
  RunOptions run_options;
  run_options.set_timeout_in_ms(options_.config.operation_timeout_in_ms());
  return RunHelper(run_options, inputs, output_names, /* targets */ {}, outputs,
                   /* run_metadata */ nullptr, handle);
}

Status GrpcSession::Close() {
  CloseSessionRequest req;
  {
    mutex_lock l(mu_);
    if (handle_.empty()) {
      return Status::OK();
    }
    req.set_session_handle(handle_);
    handle_.clear();
  }
  CloseSessionResponse resp;
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  return master_->CloseSession(&call_options, &req, &resp);
}

Status GrpcSession::ListDevices(std::vector<DeviceAttributes>* response) {
  ListDevicesRequest req;
  {
    mutex_lock l(mu_);
    req.set_session_handle(handle_);
  }
  if (req.session_handle().empty()) {
    LOG(WARNING) << "GrpcSession::ListDevices will initialize the session with "
                    "an empty graph and other defaults because the session has "
                    "not yet been created.";
    GraphDef graph_def;
    TF_RETURN_IF_ERROR(Create(graph_def));
    {
      mutex_lock l(mu_);
      req.set_session_handle(handle_);
    }
  }
  ListDevicesResponse resp;
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  Status s = master_->ListDevices(&call_options, &req, &resp);
  if (!s.ok()) {
    LOG(ERROR) << "Could not list devices: " << s;
    return s;
  }

  response->clear();
  response->reserve(resp.local_device_size() + resp.remote_device_size());
  for (const auto& device_attr : resp.local_device()) {
    response->emplace_back(device_attr);
  }
  for (const auto& device_attr : resp.remote_device()) {
    response->emplace_back(device_attr);
  }
  return Status::OK();
}

/** \brief Fill in or set the master field of the client grpc Session,
*          GrpcSession.
 *         MasterInterface master is responsible for serving as a handler of
 *         all grpc stubs called by client side.
 *         MasterInterface master will redirect to GrpcRemoteMaster as derived
 *         class polymorphism.
 *
 *  \param master: std::unique_ptr<MasterInterface>
 *         MasterInterface defines grpc stub functions interfaces related to
 *         Session interface in master_interface.h, like 1. CreateSession,
 *         2. ExtendSession 3. RunStep etc.
 *         It is a base class for GrpcRemoteMaster. So, actuall, master will
 *         redirect to member methods in the class GrpcRemoteMaster in
 *         grpc_remote_master.cc .
 *
 *  \remark No return.
 */
void GrpcSession::SetRemoteMaster(std::unique_ptr<MasterInterface> master) {
  master_ = std::move(master);
}

// Static method.
Status GrpcSession::Reset(const SessionOptions& options,
                          const std::vector<string>& containers) {
  SharedGrpcChannelPtr master_channel;
  TF_RETURN_IF_ERROR(
      NewHostPortGrpcChannel(options.target.substr(kSchemePrefixLength),
                             /*rpc_options=*/nullptr, &master_channel));
  auto master = NewGrpcMaster(master_channel);
  ResetRequest req;
  for (const auto& c : containers) req.add_container(c);
  ResetResponse resp;
  CallOptions call_options;
  call_options.SetTimeout(options.config.operation_timeout_in_ms());
  Status ret = master->Reset(&call_options, &req, &resp);
  delete master;
  return ret;
}

Status GrpcSession::MakeCallable(const CallableOptions& callable_options,
                                 CallableHandle* out_handle) {
  MakeCallableRequest req;
  TF_RETURN_IF_ERROR(Handle(req.mutable_session_handle()));
  *req.mutable_options() = callable_options;
  req.set_request_id(GetUniqueRequestId());
  MakeCallableResponse resp;
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  TF_RETURN_IF_ERROR(master_->MakeCallable(&call_options, &req, &resp));
  *out_handle = resp.handle();
  return Status::OK();
}

Status GrpcSession::RunCallable(CallableHandle handle,
                                const std::vector<Tensor>& feed_tensors,
                                std::vector<Tensor>* fetch_tensors,
                                RunMetadata* run_metadata) {
  RunCallableRequest req;
  TF_RETURN_IF_ERROR(Handle(req.mutable_session_handle()));
  req.set_handle(handle);
  req.set_request_id(GetUniqueRequestId());
  for (const Tensor& feed : feed_tensors) {
    feed.AsProtoTensorContent(req.mutable_feed()->Add());
  }

  RunCallableResponse resp;
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  TF_RETURN_IF_ERROR(master_->RunCallable(&call_options, &req, &resp));
  for (const TensorProto& fetch : resp.fetch()) {
    Tensor fetch_tensor;
    if (!fetch_tensor.FromProto(cpu_allocator(), fetch)) {
      return errors::Internal(
          "Could not parse fetched tensor data in response from master.");
    }
    fetch_tensors->push_back(std::move(fetch_tensor));
  }
  return Status::OK();
}

Status GrpcSession::ReleaseCallable(CallableHandle handle) {
  ReleaseCallableRequest req;
  TF_RETURN_IF_ERROR(Handle(req.mutable_session_handle()));
  req.set_handle(handle);
  ReleaseCallableResponse resp;
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  return master_->ReleaseCallable(&call_options, &req, &resp);
}

class GrpcSessionFactory : public SessionFactory {
 public:
  bool AcceptsOptions(const SessionOptions& options) override {
    return str_util::StartsWith(options.target, kSchemePrefix);
  }

  /** \brief Create a new GrpcSession subject to Session options.
   *
   *  \param[in] options: SessionOptions& ;
   *         - SessionOptions is used to provide 1. environment used; 2. target
   *         used to perform all computations according to host:port. 3. A bunch
   *         of Session configuration parameters.
   *
   *  \param[out] out_session: Session** ;
   *         - The real type is GrpcSession. Session is an interface class
   *         providing Session 1. Create; 2. Extend; 3. Close; 4. Run; etc.
   *
   *  \return Status
   */
  Status NewSession(const SessionOptions& options,
                    Session** out_session) override {
    std::unique_ptr<GrpcSession> session;
    /// Call the **static** function GrpcSession::Create(options, &session)
    TF_RETURN_IF_ERROR(GrpcSession::Create(options, &session));
    *out_session = session.release();
    return Status::OK();
  }

  // Invokes the session specific static method to reset containers.
  Status Reset(const SessionOptions& options,
               const std::vector<string>& containers) override {
    return GrpcSession::Reset(options, containers);
  }
};

class GrpcSessionRegistrar {
 public:
  GrpcSessionRegistrar() {
    SessionFactory::Register("GRPC_SESSION", new GrpcSessionFactory());
  }
};
static GrpcSessionRegistrar registrar;

}  // namespace tensorflow
