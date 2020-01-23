/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DIRECT_SESSION_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DIRECT_SESSION_H_

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/common_runtime/debugger_state_interface.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/graph_execution_state.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/session_state.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

class CostModel;
// 1.
// class CostModel 数据结构
//

class DebugGateway;
class Device;
class DirectSessionFactory;

class DirectSession : public Session {
 public:
  typedef std::function<void(Session*)> CloseCallback;

  // Takes ownership of 'device_mgr'.
  // 'factory' is used to unregister the DirectSession with 'factory' when its
  // closed. This ensures that Reset requests from the 'factory' don't get sent
  // to sessions that are already closed.
  DirectSession(
    const SessionOptions& options,
    const DeviceMgr* device_mgr,
    DirectSessionFactory* factory);

  ~DirectSession() override;


  typedef std::vector<std::pair<string, Tensor>> NamedTensorList;
  // NamedTensorList 类型用途:
  // Session::Run 的 inputs 输入 arguments

  typedef std::unordered_map<StringPiece, Node*, StringPieceHasher> NameNodeMap;
  // NameNodeMap 类型的用途

  ::tensorflow::Status Create(const GraphDef& graph) override;

  ::tensorflow::Status Extend(const GraphDef& graph) override;

  ::tensorflow::Status Run(const NamedTensorList& inputs,
                           const std::vector<string>& output_names,
                           const std::vector<string>& target_nodes,
                           std::vector<Tensor>* outputs) override;

  // NOTE: Experimental and subject to change.
  ::tensorflow::Status Run(const ::tensorflow::RunOptions& run_options,
                           const NamedTensorList& inputs,
                           const std::vector<string>& output_names,
                           const std::vector<string>& target_nodes,
                           std::vector<Tensor>* outputs,
                           RunMetadata* run_metadata) override;

  // NOTE: PRunSetup and PRun are added to support partial execution. This
  // feature is experimental and subject to change.
  ::tensorflow::Status PRunSetup(const std::vector<string>& input_names,
                                 const std::vector<string>& output_names,
                                 const std::vector<string>& target_nodes,
                                 string* handle) override;

  ::tensorflow::Status PRun(const string& handle, const NamedTensorList& inputs,
                            const std::vector<string>& output_names,
                            std::vector<Tensor>* outputs) override;

  // Reset clears 'containers' from the device_mgr of the DirectSession.
  // If 'containers' is empty, then Reset clears the default container.
  ::tensorflow::Status Reset(const std::vector<string>& containers);

  ::tensorflow::Status ListDevices(
      std::vector<DeviceAttributes>* response) override;

  ::tensorflow::Status Close() override;

  ::tensorflow::Status LocalDeviceManager(const DeviceMgr** output) override {
    *output = device_mgr_.get();
    return ::tensorflow::Status::OK();
  }

  void ExportCostModels(CostModelManager::CostModelMap* cost_models) {
    cost_model_manager_.ExportCostModels(cost_models);
  }

  ::tensorflow::Status MakeCallable(const CallableOptions& callable_options,
                                    CallableHandle* out_handle) override;

  ::tensorflow::Status RunCallable(CallableHandle handle,
                                   const std::vector<Tensor>& feed_tensors,
                                   std::vector<Tensor>* fetch_tensors,
                                   RunMetadata* run_metadata) override;

  ::tensorflow::Status ReleaseCallable(CallableHandle handle) override;


 ////////////////////////////////////////////////////////////////////////
 private:
 ////////////////////////////////////////////////////////////////////////

  // For access to collective_graph_key_.
  friend class DirectSessionCollectiveTest;

  // We create one executor and its dependent library runtime for
  // every partition.
  // 1.
  // QQQ. PerPartitionExecutorsAndLib 什么时候被构造的，初始化的？
  // AAA.
  struct PerPartitionExecutorsAndLib {
    Graph* graph = nullptr;                  // not owned.
    Device* device = nullptr;                // not owned.
    FunctionLibraryRuntime* flib = nullptr;  // not owned.
    std::unique_ptr<Executor> executor;
  };
  // 1.
  // struct PerPartitionExecutorsAndLib 数据结构
  // - graph: Graph*
  // - device: Device*
  // - flib: FunctionLibraryRuntime*
  // - executor: std::unique_ptr<Executor>

  // 2.
  // class FunctionLibraryRuntime 数据结构
  // tensorflow/core/framework/function.h
  // - struct InstantiateOptions
  //  - 概述:
  //    Instantiate a function with the given "attrs".
  //  - target: string
  //  - is_multi_device_function: bool, default : false
  //  - input_devices: std::vector<string>
  //  - output_devices: std::vector<string>
  //  - overlay_lib: const FunctionLibraryDefinition*
  //  - state_handle: string
  //  - executor_type : string
  //  - create_kernels_eagerly: bool, default: false
  //  - config_proto: ConfigProto
  //  - optimize_graph_fn:
  //      std::function<Status(std::vector<string> /*ret_node_names*/,
  //                           std::vector<string> /*keep_node_names*/,
  //                           FunctionLibraryDefinition*, const DeviceSet&,
  //                           Device* /*cpu_device*/, std::unique_ptr<Graph>*)>
  //  - graph_collector: GraphCollector*
  // - typedef uint64 Handle
  // - struct Options
  //  - step_id: int64
  //  - rendezvous: Rendezvous*
  //  - cancellation_manager: CancellationManager*
  //  - collective_executor: CollectiveExecutor*
  //  - step_container: ScopedStepContainer*
  //  - stats_collector: StepStatsCollectorInterface
  //  - runner: std::function<void(std::function<void()>)>*
  //  - remote_execution: bool
  //  - source_device: string
  //  - args_alloc_attrs: std::vector<AllocatorAttributes>
  //  - rets_alloc_attrs: std::vector<AllocatorAttributes>
  //  - create_rendezvous: bool
  //  - allow_dead_tensors: bool
  // - DoneCallback
  //
  // 核心函数
  // - CreateKernel

  // 3. class Executor 数据结构
  // tensorflow/core/common_runtime/executor.h
  // 继承类: ExecutorImpl
  //
  // * struct Args
  //    - step_id: int64
  //    - rendezvous: Rendezvous*
  //    - stats_collector: StepStatsCollectorInterface*
  //    - call_frame: CallFrameInterface*
  //    - cancellation_manager: CancellationManager*
  //    - session_state: SessionState*
  //    - session_handle: string
  //    - tensor_store: TensorStore*
  //    - step_container: ScopedStepContainer*
  //    - collective_executor: CollectiveExecutor*
  //    - sync_on_finish: bool
  // * typedef std::function<void()> Closure;
  // * typedef std::function<void(Closure)> Runner;
  // - runner: Runner

  ////////////////////////////////////////////////////////////////////////


  struct ExecutorsAndKeys {

    ExecutorsAndKeys() : step_count(0) {}

    std::atomic_int_fast64_t step_count;

    std::unique_ptr<Graph> graph;

    NameNodeMap name_to_node;

    // ------------------------------------------------------------------------
    // 最重要
    std::vector<PerPartitionExecutorsAndLib> items;
    // ------------------------------------------------------------------------

    std::unordered_map<string, size_t> input_name_to_index;

    std::unordered_map<string, string> input_name_to_rendezvous_key;

    std::unordered_map<string, size_t> output_name_to_index;

    std::unordered_map<string, string> output_name_to_rendezvous_key;

    DataTypeVector input_types;
    DataTypeVector output_types;

    // message type
    CallableOptions callable_options;
    int64 collective_graph_key = BuildGraphOptions::kNoCollectiveGraphKey;
  };
  // 1.
  // struct ExecutorsAndKeys 数据结构
  // 综述:An ExecutorsAndKeys is created for a given set of feeds/fetches.
  // 变量说明:
  //   ================================================
  // - items: std::vector<PerPartitionExecutorsAndLib>  # 最重要
  //   ================================================
  // - step_count: std::atomic_int_fast64_t
  //    - the number of times this graph is executed.
  // - graph: std::unique_ptr<Graph>
  //    - the entire graph being executed.
  // - name_to_node: NameNodeMap
  //    - maps node name to node.
  //      We keep 'graph' and 'name_to_node' only in the case of partial runs.
  //      Each item in 'items' is the executor for a partition of the graph
  //      bundled with its dependent library runtime.
  // - input_name_to_index: std::unordered_map<string, size_t>
  // - input_name_to_rendezvous_key: std::unordered_map<string, string>
  // - output_name_to_index: std::unordered_map<string, size_t>
  // - output_name_to_rendezvous_key: std::unordered_map<string, string>
  // - input_types: DataTypeVector
  //    - the rendezvous keys for the feeds
  // - output_types: DataTypeVector
  //    - rendezvous keys for the fetches.
  // - callable_options: CallableOptions
  // - collective_graph_key: int64, default: BuildGraphOptions::kNoCollectiveGraphKey

  // 2.
  // message CallableOptions 数据结构
  // tensorflow/core/protobuf/config.proto:559
  //
  // 功能和目的:
  // Defines a subgraph in another `GraphDef` as a set of feed points and nodes
  // to be fetched or executed.
  //
  // - feed: repeated string
  //    - Tensors to be fed in the callable. Each feed is the name of a tensor.
  // - fetch: repeated string
  //    - Fetches. A list of tensor names. The caller of the callable expects a
  //      tensor to be returned for each fetch[i] (see RunStepResponse.tensor). The
  //      order of specified fetches does not change the execution order.
  // - target: repeated string
  //    - Target Nodes. A list of node names. The named nodes will be run by the
  //      callable but their outputs will not be returned.
  // - run_options: RunOptions
  //    - Options that will be applied to each run.
  // - tensor_connection: repeated TensorConnection
  //    - Tensors to be connected in the callable. Each TensorConnection denotes
  //      a pair of tensors in the graph, between which an edge will be created
  //      in the callable.
  // - feed_devices: map<string, string>
  // - fetch_devices: map<string, string>
  // - fetch_skip_sync: bool

  // A FunctionInfo object is created for every unique set of feeds/fetches.
  // This info could be folded into the ExecutorsAndKeys object but we would
  // like to maintain a deletion order in which the OpKernels (owned by the
  // executor) should be destroyed first, followed by the resources in the
  // device and then followed by the function stuff.
  // TODO(rohanj): Consolidate function library definitions so that we can
  // instantiate only one ProcFLR and lib_def and make this just a member
  // variable and not a vector.
  // 'flib_def' is the function library used.
  // 'proc_flr' is the collection of FunctionLibraryRuntime objects, one per
  // device.
  struct FunctionInfo {
    std::unique_ptr<FunctionLibraryDefinition> flib_def;
    std::unique_ptr<ProcessFunctionLibraryRuntime> proc_flr;
  };
  // 1.
  // FunctionInfo 数据结构
  // tensorflow/core/common_runtime/direct_session.h:192:  struct FunctionInfo
  // - flib_def: std::unique_ptr<FunctionLibraryDefinition>
  // - proc_flr: std::unique_ptr<ProcessFunctionLibraryRuntime>
  //
  // 概述:
  // A FunctionInfo object is created for every unique set of feeds/fetches.
  // This info could be folded into the ExecutorsAndKeys object but we would
  // like to maintain a deletion order in which the OpKernels (owned by the
  // executor) should be destroyed first, followed by the resources in the
  // device and then followed by the function stuff.
  // TODO(rohanj): Consolidate function library definitions so that we can
  // instantiate only one ProcFLR and lib_def and make this just a member
  // variable and not a vector.
  // 'flib_def' is the function library used.
  // 'proc_flr' is the collection of FunctionLibraryRuntime objects, one per
  // device.

  ///////////////////////////////////////////////////////////////////////////

  // For each live partial execution, the session maintains a RunState.
  // 'status' is the current status of this partial execution. 'executor_done'
  // is "notified" when all executors are done. 'pending_inputs' are the set
  // of pending feeds and 'pending_outputs' are the set of pending fetches.
  struct RunState {
    mutex mu_;
    Status status GUARDED_BY(mu_);

    IntraProcessRendezvous* rendez = nullptr;

    std::unique_ptr<CollectiveExecutor::Handle> collective_executor;

    std::unique_ptr<StepStatsCollector> collector;

    Notification executors_done;

    std::unordered_map<string, bool> pending_inputs;   // true if fed
    std::unordered_map<string, bool> pending_outputs;  // true if fetched

    TensorStore tensor_store;

    // tensorflow/core/framework/resource_mgr.h:87:
    // class ScopedStepContainer
    ScopedStepContainer step_container;

    // 为什么要 devices ?
    RunState(int64 step_id, const std::vector<Device*>* devices);

    RunState(const std::vector<string>& pending_input_names,
             const std::vector<string>& pending_output_names,
             int64 step_id,
             const std::vector<Device*>* devices);

    // Returns true if all pending inputs and outputs have been completed.
    bool PendingDone() const;

    ~RunState();
  };  // RunState END
  // 1.
  // struct RunState 数据结构
  // tensorflow/core/common_runtime/direct_session.h
  //
  // 概述
  // For each live partial execution, the session maintains a RunState.
  // 'status' is the current status of this partial execution. 'executor_done'
  // is "notified" when all executors are done. 'pending_inputs' are the set
  // of pending feeds and 'pending_outputs' are the set of pending fetches.
  //
  // - mu_: mutex
  // - rendez: IntraProcessRendezvous*
  // - collective_executor: std::unique_ptr<CollectiveExecutor::Handle>
  // - collector: std::unique_ptr<StepStatsCollector>
  // - executors_done: Notification
  // - pending_inputs: std::unordered_map<string, bool>
  //   true if fed
  // - pending_outputs: std::unordered_map<string, bool>
  //   true if fetched
  // - tensor_store: TensorStore
  // - step_container: ScopedStepContainer

  ///////////////////////////////////////////////////////////////////////////


  ///////////////////////////////////////////////////////////////////////////

  struct RunStateArgs {
    RunStateArgs(const DebugOptions& options) : debug_options(options) {}

    bool is_partial_run = false;
    string handle;
    std::unique_ptr<Graph> graph;
    const DebugOptions& debug_options;
    int64 collective_graph_key = BuildGraphOptions::kNoCollectiveGraphKey;
  };
  // 1.
  // struct RunStateArgs 数据结构
  // tensorflow/core/common_runtime/direct_session.h
  // - is_partial_run: bool, default_value : false
  // - handle : string
  // - graph: std::unique_ptr<Graph>
  // - debug_options: const DebugOptions&
  // - collective_graph_key: int64, default_value: BuildGraphOptions::kNoCollectiveGraphKey

  ///////////////////////////////////////////////////////////////////////////


  // Initializes the base execution state given the 'graph',
  // if not already initialized.
  Status MaybeInitializeExecutionState(const GraphDef& graph,
                                       bool* out_already_initialized)
      EXCLUSIVE_LOCKS_REQUIRED(graph_state_lock_);

  // Retrieves an already existing set of executors to run 'inputs' and
  // 'outputs', or creates and caches them for future use.
  ::tensorflow::Status GetOrCreateExecutors(
      gtl::ArraySlice<string> inputs,
      gtl::ArraySlice<string> outputs,
      gtl::ArraySlice<string> target_nodes,
      ExecutorsAndKeys** executors_and_keys,
      RunStateArgs* run_state_args);

  // Creates a set of executors to run the subgraph defined by
  // `callable_options`.
  ::tensorflow::Status CreateExecutors(
      const CallableOptions& callable_options,
      std::unique_ptr<ExecutorsAndKeys>* out_executors_and_keys,
      std::unique_ptr<FunctionInfo>* out_func_info,
      RunStateArgs* run_state_args);

  // Creates several graphs given the existing graph_def_ and the
  // input feeds and fetches, given 'devices'. The graphs share a common
  // function library 'flib_def'.
  ::tensorflow::Status CreateGraphs(
      const BuildGraphOptions& options,
      std::unordered_map<string, std::unique_ptr<Graph>>* outputs,
      std::unique_ptr<FunctionLibraryDefinition>* flib_def,
      RunStateArgs* run_state_args, DataTypeVector* input_types,
      DataTypeVector* output_types, int64* collective_graph_key);

  ::tensorflow::Status RunInternal(int64 step_id, const RunOptions& run_options,
                                   CallFrameInterface* call_frame,
                                   ExecutorsAndKeys* executors_and_keys,
                                   RunMetadata* run_metadata);

  // Returns whether inter-op execution uses a global pool or the input
  // `run_options` requests being run on inter_op_thread_pool = 0 in case
  // multiple pools are configured.
  bool ShouldUseRunHandlerPool(const RunOptions& run_options) const;

  ::tensorflow::Status ExtendLocked(const GraphDef& graph)
      EXCLUSIVE_LOCKS_REQUIRED(graph_state_lock_);

  ::tensorflow::Status ResourceHandleToInputTensor(
      const Tensor& resource_tensor, Tensor* retrieved_tensor);

  // Feeds more inputs to the executors, triggering further execution.
  ::tensorflow::Status SendPRunInputs(
      const std::vector<std::pair<string, Tensor>>& inputs,
      const ExecutorsAndKeys* executors_and_keys,
      IntraProcessRendezvous* rendez);

  // Fetches more outputs from the executors. It waits until the output
  // tensors are computed.
  ::tensorflow::Status RecvPRunOutputs(
      const std::vector<string>& output_names,
      const ExecutorsAndKeys* executors_and_keys, RunState* run_state,
      std::vector<Tensor>* outputs);

  // Check if the specified fetches can be computed from the feeds
  // that we have already provided.
  ::tensorflow::Status CheckFetch(
      const std::vector<std::pair<string, Tensor>>& feeds,
      const std::vector<string>& fetches,
      const ExecutorsAndKeys* executors_and_keys, const RunState* run_state);


  // Use the appropriate WaitForNotification function based on whether
  // operation_timeout_in_ms is greater than 0.
  //
  // If the timeout expires, the `cm->StartCancel()` will be called.
  ::tensorflow::Status WaitForNotification(Notification* n,
                                           int64 timeout_in_ms);

  void WaitForNotification(RunState* run_state, CancellationManager* cm,
                           int64 timeout_in_ms);

  ::tensorflow::Status CheckNotClosed() {
    mutex_lock l(closed_lock_);
    if (closed_) return errors::Cancelled("Session has been closed.");
    return ::tensorflow::Status::OK();
  }

  ::tensorflow::Status CheckGraphCreated(const char* method) {
    mutex_lock l(graph_state_lock_);
    if (!graph_created_) {
      return errors::InvalidArgument(
          "Session was not created with a graph before ", method, "!");
    }
    return ::tensorflow::Status::OK();
  }

  ::tensorflow::Status CreateDebuggerState(
      const CallableOptions& options, int64 global_step,
      int64 session_run_index, int64 executor_step_index,
      std::unique_ptr<DebuggerStateInterface>* debugger_state);

  ::tensorflow::Status DecorateAndPublishGraphForDebug(
      const DebugOptions& debug_options, Graph* graph, Device* device);

  const SessionOptions options_;
  // 1.
  // options_ : SessionOptions 变量说明:
  // SessionOptions 数据结构
  // tensorflow/core/public/session_options.h:28:struct SessionOptions
  // - config: ConfigProto
  //   sess.run Configuration options.
  // - target: string
  // - env: Env*

  // Device structures.
  const std::unique_ptr<const DeviceMgr> device_mgr_;
  std::vector<Device*> devices_;  // not owned
  DeviceSet device_set_;

  // Unique session identifier.
  string session_handle_;
  // session_handle_ 变量说明:
  // Unique session identifier.
  // session_handle_: string

  mutex graph_state_lock_;
  bool graph_created_ GUARDED_BY(graph_state_lock_) = false;

  // The thread-pools to use for running ops, with a bool indicating if the pool
  // is owned.
  std::vector<std::pair<thread::ThreadPool*, bool>> thread_pools_;

  Status init_error_;  // Set to an error if construction failed.

  // If true, blocks until device has finished all queued operations in a step.
  bool sync_on_finish_ = true;
  // Schedules 'c' for execution on pool.
  void SchedClosure(thread::ThreadPool* pool, std::function<void()> c);

  std::vector<std::unique_ptr<FunctionInfo>> functions_
      GUARDED_BY(executor_lock_);

  mutex executor_lock_;  // protects executors_


  // 最重要
  // Holds mappings from signature to the executors that process
  // it. The reason for a level of indirection around mapped_type is
  // to guarantee address stability.
  // The map value is a shared_ptr since multiple map keys can point to the
  // same ExecutorsAndKey object.
  // -----------------------------------------------------------------------
  std::unordered_map<string, std::shared_ptr<ExecutorsAndKeys>> executors_
      GUARDED_BY(executor_lock_);
  // -----------------------------------------------------------------------
  // 1.
  // string 是 key, 规则是
  //   key = inputs,->outputs,/target_nodes,/is_partial_run/debug_tensor_watches_summary
  // 比如说:
  // p key
  // $7 = "->out:0//0/"

  // 2.
  // struct ExecutorsAndKeys 数据结构
  // 综述:An ExecutorsAndKeys is created for a given set of feeds/fetches.
  // tensorflow/core/common_runtime/direct_session.h
  // 变量说明:
  //   ================================================
  // - items: std::vector<PerPartitionExecutorsAndLib>  # 最重要
  //   ================================================
  // - step_count: std::atomic_int_fast64_t
  //    - the number of times this graph is executed.
  // - graph: std::unique_ptr<Graph>  # 1. QQQ. 何时被赋值的?
  //    - the entire graph being executed.
  // - name_to_node: NameNodeMap
  //    - maps node name to node.
  //      We keep 'graph' and 'name_to_node' only in the case of partial runs.
  //      Each item in 'items' is the executor for a partition of the graph
  //      bundled with its dependent library runtime.
  // - input_name_to_index: std::unordered_map<string, size_t>
  // - input_name_to_rendezvous_key: std::unordered_map<string, string>
  // - output_name_to_index: std::unordered_map<string, size_t>
  // - output_name_to_rendezvous_key: std::unordered_map<string, string>
  // - input_types: DataTypeVector
  //    - the rendezvous keys for the feeds
  // - output_types: DataTypeVector
  //    - rendezvous keys for the fetches.
  // - callable_options: CallableOptions
  // - collective_graph_key: int64, default: BuildGraphOptions::kNoCollectiveGraphKey


  class RunCallableCallFrame;

  struct Callable {
    std::shared_ptr<ExecutorsAndKeys> executors_and_keys;
    std::shared_ptr<FunctionInfo> function_info;
    ~Callable();
  };
  // 1.
  // DirectSession:: struct Callable 数据结构
  // - executors_and_keys: std::shared_ptr<ExecutorsAndKeys>
  // - function_info: std::shared_ptr<FunctionInfo>

  mutex callables_lock_;
  int64 next_callable_handle_ GUARDED_BY(callables_lock_) = 0;
  std::unordered_map<int64, Callable> callables_ GUARDED_BY(callables_lock_);

  // Holds mappings from handle to partial run state.
  std::unordered_map<string, std::unique_ptr<RunState>> partial_runs_
      GUARDED_BY(executor_lock_);

  // =======================================================================
  // This holds all the tensors that are currently alive in the session.
  //  The session state remembers the tensors we choose to keep across
  // multiple run calls.
  SessionState session_state_;
  // =======================================================================

  DirectSessionFactory* const factory_;  // not owned
  CancellationManager* cancellation_manager_;
  std::unique_ptr<CollectiveExecutorMgrInterface> collective_executor_mgr_;

  // Map of placed stateful nodes, i.e. nodes for which is_stateful()
  // is true, such as "params" and "queue" nodes.  Once placed these
  // nodes can not be moved to a different device.  Maps node names to
  // device names.
  std::unordered_map<string, string> stateful_placements_
      GUARDED_BY(graph_state_lock_);

  // Execution_state; used when placing the entire graph.
  // IMPT, DirectSession 的核心数据结构
  // =======================================================================
  std::unique_ptr<GraphExecutionState> execution_state_
      GUARDED_BY(graph_state_lock_);
  // =======================================================================
  // 1.
  // class GraphExecutionState 数据结构整理:
  // - graph_: Graph*
  //    - The dataflow graph owned by this object.
  // - rewrite_metadata_: std::unique_ptr<subgraph::RewriteGraphMetadata>
  //    - `rewrite_metadata_` is only set for GraphExecutionState objects created by `MakeForPrunedGraph()`.
  // - flib_def_: std::unique_ptr<FunctionLibraryDefinition>
  //    - 'flib_def_' is initialized from the initial graph def's library, and may be updated by a graph optimization pass.
  // - node_name_to_cost_id_map_: NodeNameToCostIdMap
  //    - Map from name to Node for the full graph in placed_.
  // - session_handle_: string
  //    -  Unique session identifier. Can be empty.
  // - session_options_: const SessionOptions*
  // - device_set_: const DeviceSet*
  // - original_graph_def_: GraphDef
  // - stateful_placements_: std::unordered_map<string, string>
  //    - Map of placed stateful nodes, i.e. nodes for which is_stateful()
  //    - is true, such as "params" and "queue" nodes.  Once placed these
  //    - nodes can not be moved to a different device.  Maps node names to
  //    - device names.
  //
  // 数据结构说明
  // GraphExecutionState is responsible for generating an
  // executable ClientGraph from the original GraphDef that specifies
  // the complete graph and from BuildGraphOptions which specifies
  // input/output nodes.
  //
  // An executable Graph differs from a GraphDef by being Placed,
  // meaning that each Node is assigned to a single Device in the
  // available set.
  //
  // When GraphExecutionState is first constructed it instantiates
  // a full Graph from the provided GraphDef, and places it, using only
  // the static device assignments from the GraphDef.  Nodes without are
  // currently placed in a very naive way.  Since stateful Nodes cannot
  // be moved after initial placement, it is important that stateful
  // Nodes get sensible initial device assignments in the graph
  // definition.
  //
  // Subsequently, GraphExecutionState generates a SimpleClientGraph on
  // demand, which is a sub-graph of the latest placement of the full
  // Graph.  MasterSession uses such a ClientGraph to execute one or
  // more similar client requests.

  // 2.
  // QQQ. 一个 DirectSession 为什么内部只有一个 GraphExecutionState ?
  // AAA.


  // The function library, before any rewrites or optimizations have been
  // performed. In particular, CreateGraphs() may need to modify the function
  // library; it copies and modifies the function library.
  // 比如说 mkl 优化会替换原来的 function
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;

  // true if the Session has been Closed.
  mutex closed_lock_;
  bool closed_ GUARDED_BY(closed_lock_) = false;

  // For generating unique names for this session instance.
  std::atomic<int64> edge_name_counter_ = {0};
  std::atomic<int64> handle_name_counter_ = {0};

  // For generating step ids that are unique across this sessions.
  static std::atomic_int_fast64_t step_id_counter_;

  // Global timeout for all blocking operations in this session.
  const int64 operation_timeout_in_ms_ = 0;

  // Manages all the cost models for the graphs executed in this session.
  CostModelManager cost_model_manager_;

  // For testing collective graph key generation.
  mutex collective_graph_key_lock_;
  int64 collective_graph_key_ GUARDED_BY(collective_graph_key_lock_) = -1;

  TF_DISALLOW_COPY_AND_ASSIGN(DirectSession);

  // EXPERIMENTAL: debugger (tfdbg) related
  friend class DebugGateway;
};
// 1.
// DirectSession 数据结构整理
//  * struct PerPartitionExecutorsAndLib
//  * struct ExecutorsAndKeys
//  * struct FunctionInfo
//  * struct RunState
//  * struct RunStateArgs
//  * struct Callable
//
//  - executors_: std::unordered_map<string, std::shared_ptr<ExecutorsAndKeys>> # 重要
//  - device_mgr_: const std::unique_ptr<const DeviceMgr>
//  - devices_: std::vector<Device*>
//  - device_set_: DeviceSet
//  - session_handle_: string
//  - graph_created_: bool
//  - thread_pools_: std::vector<std::pair<thread::ThreadPool*, bool>>
//  - sync_on_finish_: bool
//  - functions_: std::vector<std::unique_ptr<FunctionInfo>>
//  - next_callable_handle_: int64
//  - callables_: std::unordered_map<int64, Callable>
//  - partial_runs_: std::unordered_map<string, std::unique_ptr<RunState>>
//  - session_state_: SessionState
//  - factory_: DirectSessionFactory* const
//  - cancellation_manager_: CancellationManager*
//  - collective_executor_mgr_: std::unique_ptr<CollectiveExecutorMgrInterface>
//  - stateful_placements_: std::unordered_map<string, string>
//    =======================================================================
//  - execution_state_: std::unique_ptr<GraphExecutionState>  # 重要
//    p execution_state_->graph_->ToGraphDefDebug().DebugString()
//    p low_priority_execution_state_->graph_->ToGraphDefDebug().DebugString()
//    =======================================================================
//  - flib_def_: std::unique_ptr<FunctionLibraryDefinition>
//  - closed_: bool
//  - edge_name_counter_: std::atomic<int64> # 可以用来测试自己的设计
//  - handle_name_counter_: std::atomic<int64>
//  - step_id_counter_: static std::atomic_int_fast64_t
//  - operation_timeout_in_ms_: const int64
//  - cost_model_manager_: CostModelManager
//  - collective_graph_key_: int64

// 2.
// struct ExecutorsAndKeys 数据结构
// 综述:An ExecutorsAndKeys is created for a given set of feeds/fetches.
// 变量说明:
//   ================================================
// - items: std::vector<PerPartitionExecutorsAndLib>  # 最重要
//   ================================================
// - step_count: std::atomic_int_fast64_t
//    - the number of times this graph is executed.
// - graph: std::unique_ptr<Graph>
//    - the entire graph being executed.
// - name_to_node: NameNodeMap
//    - maps node name to node.
//      We keep 'graph' and 'name_to_node' only in the case of partial runs.
//      Each item in 'items' is the executor for a partition of the graph
//      bundled with its dependent library runtime.
// - input_name_to_index: std::unordered_map<string, size_t>
// - input_name_to_rendezvous_key: std::unordered_map<string, string>
// - output_name_to_index: std::unordered_map<string, size_t>
// - output_name_to_rendezvous_key: std::unordered_map<string, string>
// - input_types: DataTypeVector
//    - the rendezvous keys for the feeds
// - output_types: DataTypeVector
//    - rendezvous keys for the fetches.
// - callable_options: CallableOptions
// - collective_graph_key: int64, default: BuildGraphOptions::kNoCollectiveGraphKey

// 3.
// struct PerPartitionExecutorsAndLib 数据结构
// - graph: Graph*
// - device: Device*
// - flib: FunctionLibraryRuntime*
// - executor: std::unique_ptr<Executor>

// 4.
// class FunctionLibraryRuntime 数据结构
// tensorflow/core/framework/function.h
// - struct InstantiateOptions
//  - 概述:
//    Instantiate a function with the given "attrs".
//  - target: string
//  - is_multi_device_function: bool, default : false
//  - input_devices: std::vector<string>
//  - output_devices: std::vector<string>
//  - overlay_lib: const FunctionLibraryDefinition*
//  - state_handle: string
//  - executor_type : string
//  - create_kernels_eagerly: bool, default: false
//  - config_proto: ConfigProto
//  - optimize_graph_fn:
//      std::function<Status(std::vector<string> /*ret_node_names*/,
//                           std::vector<string> /*keep_node_names*/,
//                           FunctionLibraryDefinition*, const DeviceSet&,
//                           Device* /*cpu_device*/, std::unique_ptr<Graph>*)>
//  - graph_collector: GraphCollector*
// - typedef uint64 Handle
// - struct Options
//  - step_id: int64
//  - rendezvous: Rendezvous*
//  - cancellation_manager: CancellationManager*
//  - collective_executor: CollectiveExecutor*
//  - step_container: ScopedStepContainer*
//  - stats_collector: StepStatsCollectorInterface
//  - runner: std::function<void(std::function<void()>)>*
//  - remote_execution: bool
//  - source_device: string
//  - args_alloc_attrs: std::vector<AllocatorAttributes>
//  - rets_alloc_attrs: std::vector<AllocatorAttributes>
//  - create_rendezvous: bool
//  - allow_dead_tensors: bool
// - DoneCallback
//
// 核心函数
// - CreateKernel

// 5.
// class Executor 数据结构
// tensorflow/core/common_runtime/executor.h
// 继承类: ExecutorImpl
//
// * struct Args
//    - step_id: int64
//    - rendezvous: Rendezvous*
//    - stats_collector: StepStatsCollectorInterface*
//    - call_frame: CallFrameInterface*
//    - cancellation_manager: CancellationManager*
//    - session_state: SessionState*
//    - session_handle: string
//    - tensor_store: TensorStore*
//    - step_container: ScopedStepContainer*
//    - collective_executor: CollectiveExecutor*
//    - sync_on_finish: bool
// * typedef std::function<void()> Closure;
// * typedef std::function<void(Closure)> Runner;
// - runner: Runner

// 6.
// message CallableOptions 数据结构
// tensorflow/core/protobuf/config.proto:559
//
// 功能和目的:
// Defines a subgraph in another `GraphDef` as a set of feed points and nodes
// to be fetched or executed.
//
// - feed: repeated string
//    - Tensors to be fed in the callable. Each feed is the name of a tensor.
// - fetch: repeated string
//    - Fetches. A list of tensor names. The caller of the callable expects a
//      tensor to be returned for each fetch[i] (see RunStepResponse.tensor). The
//      order of specified fetches does not change the execution order.
// - target: repeated string
//    - Target Nodes. A list of node names. The named nodes will be run by the
//      callable but their outputs will not be returned.
// - run_options: RunOptions
//    - Options that will be applied to each run.
// - tensor_connection: repeated TensorConnection
//    - Tensors to be connected in the callable. Each TensorConnection denotes
//      a pair of tensors in the graph, between which an edge will be created
//      in the callable.
// - feed_devices: map<string, string>
// - fetch_devices: map<string, string>
// - fetch_skip_sync: bool


// 7.
// class GraphExecutionState 数据结构整理:
// - graph_: Graph*
//    - The dataflow graph owned by this object.
// - rewrite_metadata_: std::unique_ptr<subgraph::RewriteGraphMetadata>
//    - `rewrite_metadata_` is only set for GraphExecutionState objects created by `MakeForPrunedGraph()`.
// - flib_def_: std::unique_ptr<FunctionLibraryDefinition>
//    - 'flib_def_' is initialized from the initial graph def's library, and may be updated by a graph optimization pass.
// - node_name_to_cost_id_map_: NodeNameToCostIdMap
//    - Map from name to Node for the full graph in placed_.
// - session_handle_: string
//    -  Unique session identifier. Can be empty.
// - session_options_: const SessionOptions*
// - device_set_: const DeviceSet*
// - original_graph_def_: GraphDef
// - stateful_placements_: std::unordered_map<string, string>
//    - Map of placed stateful nodes, i.e. nodes for which is_stateful()
//    - is true, such as "params" and "queue" nodes.  Once placed these
//    - nodes can not be moved to a different device.  Maps node names to
//    - device names.
//
// 数据结构说明
// GraphExecutionState is responsible for generating an
// executable ClientGraph from the original GraphDef that specifies
// the complete graph and from BuildGraphOptions which specifies
// input/output nodes.
//
// An executable Graph differs from a GraphDef by being Placed,
// meaning that each Node is assigned to a single Device in the
// available set.
//
// When GraphExecutionState is first constructed it instantiates
// a full Graph from the provided GraphDef, and places it, using only
// the static device assignments from the GraphDef.  Nodes without are
// currently placed in a very naive way.  Since stateful Nodes cannot
// be moved after initial placement, it is important that stateful
// Nodes get sensible initial device assignments in the graph
// definition.
//
// Subsequently, GraphExecutionState generates a SimpleClientGraph on
// demand, which is a sub-graph of the latest placement of the full
// Graph.  MasterSession uses such a ClientGraph to execute one or
// more similar client requests.


}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DIRECT_SESSION_H_
