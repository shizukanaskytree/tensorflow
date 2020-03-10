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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_H_

#include <thread>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/session_state.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#include "tensorflow/core/util/env_var.h"


namespace tensorflow {

// Creates an Executor that computes the given "graph".
//
// If successful, returns the constructed executor in "*executor". Otherwise,
// returns an error status.
//
// "params" provides a set of context for the executor. We expect that
// different context would provide different implementations.
struct LocalExecutorParams {
  Device* device;

  // wxf: Low priority Device
  //Device* low_priority_device;

  // The library runtime support.
  FunctionLibraryRuntime* function_library = nullptr;
  // wxf: low priority
  //FunctionLibraryRuntime* low_priority_function_library = nullptr;

  // create_kernel returns an instance of op kernel based on NodeDef.
  // delete_kernel is called for every kernel used by the executor
  // when the executor is deleted.
  std::function<Status(const NodeDef&, OpKernel**)> create_kernel;
  // wxf: low priority
  //std::function<Status(const NodeDef&, OpKernel**)> create_low_priority_kernel;

  std::function<void(OpKernel*)> delete_kernel;
  // wxf: low priority
  //std::function<void(OpKernel*)> delete_low_priority_kernel;
};

/// Declare class ExecutorImpl
class ExecutorImpl;
//class ExecutorsPoolManager;
//class GraphView;
//class ExecutorState;

// tid_execution_priority_map_ stores tid and its executors' priority
extern std::unordered_map<std::thread::id, int> tid_execution_priority_map_;


class StepStatsCollector;

// Executor runs a graph computation.
// Example:
//   Graph* graph = ...;
//      ... construct graph ...
//   Executor* executor;
//   TF_CHECK_OK(NewSimpleExecutor(my_device, graph, &executor));
//   Rendezvous* rendezvous = NewNaiveRendezvous();
//   TF_CHECK_OK(rendezvous->Send("input", some_input_tensor));
//   TF_CHECK_OK(executor->Run({ExecutorOpts, rendezvous, nullptr}));
//   TF_CHECK_OK(rendezvous->Recv("output", &output_tensor));
//   ... ...
//
// Multiple threads can call Executor::Run concurrently.
class Executor {
 public:
  virtual ~Executor() {}

  // RunAsync() executes the graph computation. "done" is run when the
  // graph computation completes. If any error happens during the
  // computation, "done" is run and the error is passed to "done".
  //
  // RunAsync() is given a few arguments in Args. The caller must
  // ensure objects passed in Args (rendezvous, stats_collector, etc.)
  // are alive at least until done is invoked. All pointers to the
  // argument objects can be nullptr.
  //
  // "step_id" is a process-wide unique identifier for the step being
  // run. Executors on different devices may receive the same step_id
  // in the case that a step runs Ops on more than one device. The
  // step_id is used for tracking resource usage of a given step.
  //
  // RunAsync() uses the given "rendezvous", if not null, as the
  // mechanism to communicate inputs and outputs of the underlying
  // graph computation.
  //
  // RunAsync() calls "stats_collector", if not null, to keep track of
  // stats. This allows us to collect statistics and traces on demand.
  //
  // RunAsync() is provided a "call_frame", if the executor is used
  // for executing a function, is used to pass arguments and return
  // values between the caller and the callee.
  //
  // RunAsync() uses "cancellation_manager", if not nullptr, to
  // register callbacks that should be called if the graph computation
  // is canceled. Note that the callbacks merely unblock any
  // long-running computation, and a canceled step will terminate by
  // returning/calling the DoneCallback as usual.
  //
  // RunAsync() dispatches closures to "runner". Typically, "runner"
  // is backed up by a bounded threadpool.
  struct Args {
    int64 step_id = 0;
    Rendezvous* rendezvous = nullptr;
    StepStatsCollectorInterface* stats_collector = nullptr;
    CallFrameInterface* call_frame = nullptr;
    CancellationManager* cancellation_manager = nullptr;
    SessionState* session_state = nullptr;
    // Unique session identifier. Can be empty.
    string session_handle;
    TensorStore* tensor_store = nullptr;
    ScopedStepContainer* step_container = nullptr;
    CollectiveExecutor* collective_executor = nullptr;

    // If true, calls Sync() on the device.
    bool sync_on_finish = false;

    typedef std::function<void()> Closure;
    typedef std::function<void(Closure)> Runner;
    Runner runner = nullptr;

    // wxf
    // Priority of this ExecutorState instance inherited from DirectSession which
    // creates this ExecutorState instance. 
    // It should be initialized by const Executor::Args& args.
    int executor_state_priority = 0;

    // wxf
    // It will be initialized as either "LPU" or "HPU", BUT NOT empty "". 
    // So, it can be used to indicate which device type the current 
    // DirectSession's Executor/ExecutorImpl/ExecutoState is executing on.
    string device_type_executing_on = "";
  };
  typedef std::function<void(const Status&)> DoneCallback;
  virtual void RunAsync(const Args& args, DoneCallback done) = 0;

  // Synchronous wrapper for RunAsync().
  Status Run(const Args& args) {
    Status ret;
    Notification n;
    RunAsync(args, [&ret, &n](const Status& s) {
      ret = s;
      n.Notify();
    });
    n.WaitForNotification();
    return ret;
  }
};

// --------------------------------------------------------------------

struct EdgeInfo {
  int dst_id;
  int output_slot : 31;
  // true if this is the last info for output_slot in the EdgeInfo list.
  bool is_last : 1;
  int input_slot;
};

// Time the execution of kernels (in CPU cycles).  Used to dynamically identify
// inexpensive kernels which can be dispatched inline.
struct KernelTimer {
  uint64 start_cycles = profile_utils::CpuUtils::GetCurrentClockCycle();

  uint64 ElapsedCycles() {
    return profile_utils::CpuUtils::GetCurrentClockCycle() - start_cycles;
  }
};


struct NodeItem {
  NodeItem() {}

  // A graph node.
  const Node* node = nullptr;

  // The kernel for this node.
  OpKernel* kernel = nullptr;

  bool kernel_is_async : 1;      // True iff kernel->AsAsync() != nullptr
  bool is_merge : 1;             // True iff IsMerge(node)
  bool is_enter : 1;             // True iff IsEnter(node)
  bool is_constant_enter : 1;    // True iff IsEnter(node) and
                                 // node->GetAttr("is_constant") == true.
  bool is_exit : 1;              // True iff IsExit(node)
  bool is_control_trigger : 1;   // True iff IsControlTrigger(node)
  bool is_sink : 1;              // True iff IsSink(node)
  // True iff IsEnter(node) || IsExit(node) || IsNextIteration(node)
  bool is_enter_exit_or_next_iter : 1;

  // Cached values of node->num_inputs() and node->num_outputs(), to
  // avoid levels of indirection.
  int num_inputs;
  int num_outputs;

  // ExecutorImpl::tensors_[input_start] is the 1st positional input
  // for this node.
  int input_start = 0;

  // Number of output edges.
  size_t num_output_edges;

  PendingCounts::Handle pending_id;

  const EdgeInfo* output_edge_list() const { return output_edge_base(); }

  // ith output edge.
  const EdgeInfo& output_edge(int i) const {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, num_output_edges);
    return output_edge_base()[i];
  }

  DataType input_type(int i) const {
    DCHECK_LT(i, num_inputs);
    return static_cast<DataType>(input_type_base()[i]);
  }
  DataType output_type(int i) const {
    DCHECK_LT(i, num_outputs);
    return static_cast<DataType>(output_type_base()[i]);
  }

  // Return array of per-output allocator attributes.
  const AllocatorAttributes* output_attrs() const { return output_attr_base(); }

  // Return array of expected input index from which each output should
  // be forwarded:
  // kNeverForward (-2) for DO NOT FORWARD (must allocate).
  // kNoReservation (-1) for no expected forwarding.
  // 0... for forward from that input.
  const int* forward_from() const { return forward_from_base(); }

 private:
  friend class GraphView;

  // Variable length section starts immediately after *this
  // (uint8 is enough for DataType).
  //   EdgeInfo            out_edges[num_out_edges];
  //   AllocatorAttributes output_attr[num_outputs];
  //   int                 forward_from[num_outputs];
  //   uint8               input_type[num_inputs];
  //   uint8               output_type[num_outputs];

  // Return pointer to variable length section.
  char* var() const {
    return const_cast<char*>(reinterpret_cast<const char*>(this) +
                             sizeof(NodeItem));
  }

  EdgeInfo* output_edge_base() const {
    return reinterpret_cast<EdgeInfo*>(var());
  }
  AllocatorAttributes* output_attr_base() const {
    return reinterpret_cast<AllocatorAttributes*>(var() + sizeof(EdgeInfo) *
                                                              num_output_edges);
  }
  int* forward_from_base() const {
    return reinterpret_cast<int*>(var() + sizeof(EdgeInfo) * num_output_edges +
                                  sizeof(AllocatorAttributes) * num_outputs);
  }
  uint8* input_type_base() const {
    return reinterpret_cast<uint8*>(
        var() + sizeof(EdgeInfo) * num_output_edges +
        sizeof(AllocatorAttributes) * num_outputs + sizeof(int) * num_outputs);
  }
  uint8* output_type_base() const {
    return reinterpret_cast<uint8*>(
        var() + sizeof(EdgeInfo) * num_output_edges +
        sizeof(AllocatorAttributes) * num_outputs + sizeof(int) * num_outputs +
        sizeof(uint8) * num_inputs);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(NodeItem);
};


// Immutable view of a Graph organized for efficient execution.
class GraphView {
 public:
  GraphView() : space_(nullptr) {}
  ~GraphView();

  void Initialize(const Graph* g);
  Status SetAllocAttrs(const Graph* g, const Device* device);
  void SetScopedAllocatorAttrs(const std::vector<const Node*>& sa_nodes);

  NodeItem* node(size_t id) const {
    DCHECK_GE(id, 0);
    DCHECK_LT(id, num_nodes_);
    uint32 offset = node_offsets_[id];
    return ((offset == kuint32max)
                ? nullptr
                : reinterpret_cast<NodeItem*>(space_ + node_offsets_[id]));
  }

 private:
  char* InitializeNode(char* ptr, const Node* n);
  size_t NodeItemBytes(const Node* n);

  int32 num_nodes_ = 0;
  uint32* node_offsets_ = nullptr;  // array of size "graph_.num_node_ids()"
  // node_offsets_[id] holds the byte offset for node w/ "id" in space_

  char* space_;  // NodeItem objects are allocated here

  TF_DISALLOW_COPY_AND_ASSIGN(GraphView);
};


typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
typedef gtl::InlinedVector<DeviceContext*, 4> DeviceContextVec;
typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;


// -----------------------------------------------------------------------------

// ExecutorImpl Manager
// QDD
class ExecutorsPoolManager{
 public:
  ExecutorsPoolManager():high_priority_executors_ref_count_(0),
                         low_priority_executors_ref_count_(0) {}
  ~ExecutorsPoolManager();

  // Do I really need to store all ExecutorImpl?
  void AddExecutorAndPriority(ExecutorImpl** executor);

  // Delete ExecutorImpl when it deconstructs.
  void DeleteExecutor(ExecutorImpl* executor);

  // inquire priority by ExecutorImpl pointer
  int InquirePriorityByExecutorImpl(const ExecutorImpl* executor);

  // The count of ExecutorImpl instances
  std::atomic<int> high_priority_executors_ref_count_;
  std::atomic<int> low_priority_executors_ref_count_;

 private:

  // mutex to protect multiple threads from accessing executor_priority_map_
  mutex add_mu_;
  mutex delete_mu_;
  mutex read_mu_;

  // A map between ExecutorImpl pointer to its execution priority
  std::unordered_map<ExecutorImpl*, int> executor_priority_map_;

  // Define two kind of ExecutorsPool:
  // - HighPriorityExecutorsPool
  // - LowPriorityExecutorsPool
  // If there are only two kinds, we only need a FIFO Queue for each of them.
};

// -----------------------------------------------------------------------------

class ExecutorImpl : public Executor {
 public:
  ExecutorImpl(const LocalExecutorParams& p, std::unique_ptr<const Graph> g)
      : params_(p), graph_(std::move(g)), gview_() {
    CHECK(p.create_kernel != nullptr);
    CHECK(p.delete_kernel != nullptr);
  }

  ~ExecutorImpl() override {
    for (int i = 0; i < graph_->num_node_ids(); i++) {
      NodeItem* item = gview_.node(i);
      if (item != nullptr) {
        params_.delete_kernel(item->kernel);
      }
    }
    for (auto fiter : frame_info_) {
      delete fiter.second;
    }

    // update executors_pool_manager_
    executors_pool_manager_->DeleteExecutor(this);
  }

  Status Initialize();

  // Process all Nodes in the current graph, attempting to infer the
  // memory allocation attributes to be used wherever they may allocate
  // a tensor buffer.
  Status SetAllocAttrs();

  void RunAsync(const Args& args, DoneCallback done) override;

  // ExecutorsPoolManager to manager all ExecutorImpl
  static ExecutorsPoolManager* executors_pool_manager_;

 private:
  friend class ExecutorState;

  struct ControlFlowInfo {
    gtl::FlatSet<string> unique_frame_names;
    std::vector<string> frame_names;
  };

  struct FrameInfo {
    FrameInfo()
        : input_count(0),
          total_inputs(0),
          pending_counts(nullptr),
          nodes(nullptr) {}

    // The total number of inputs to a frame.
    int input_count;

    // The total number of input tensors of a frame.
    // == sum(nodes[*].num_inputs()) where nodes are the nodes in the frame.
    int total_inputs;

    // Used to determine the next place to allocate space in the
    // pending_counts data structure we'll eventually construct
    PendingCounts::Layout pending_counts_layout;

    // Each frame has its own PendingCounts only for the nodes in the frame.
    PendingCounts* pending_counts;  // Owned

    // The nodes in a frame. Used only for debugging.
    std::vector<const Node*>* nodes;  // Owned

    ~FrameInfo() {
      delete pending_counts;
      delete nodes;
    }
  };

  static Status BuildControlFlowInfo(const Graph* graph,
                                     ControlFlowInfo* cf_info);
  void InitializePending(const Graph* graph, const ControlFlowInfo& cf_info);

  FrameInfo* EnsureFrameInfo(const string& fname) {
    auto slot = &frame_info_[fname];
    if (*slot == nullptr) {
      *slot = new FrameInfo;
    }
    return *slot;
  }

  // Owned.
  LocalExecutorParams params_;
  std::unique_ptr<const Graph> graph_;
  GraphView gview_;

  // A cached value of params_
  bool device_record_tensor_accesses_ = false;

  // Root nodes (with no in edges) that should form the initial ready queue
  std::vector<const Node*> root_nodes_;

  // Mapping from frame name to static information about the frame.
  // TODO(yuanbyu): We could cache it along with the graph so to avoid
  // the overhead of constructing it for each executor instance.
  gtl::FlatMap<string, FrameInfo*> frame_info_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorImpl);
}; // end of class ExecutorImpl

// -----------------------------------------------------------------------------

// The state associated with one invocation of ExecutorImpl::Run.
// ExecutorState dispatches nodes when they become ready and keeps
// track of how many predecessors of a node have not done (pending_).
class ExecutorState {
 public:
  ExecutorState(const Executor::Args& args, ExecutorImpl* impl);
  ~ExecutorState();

  void RunAsync(Executor::DoneCallback done);

  // wxf
  // if detect high priority task comes in, freeze the state as catched_.
  // static is used to for all low priority ExecutorStates.
  static bool catched_ ;
  // wxf
  // Has already entered the Finish function by one thread. Other threads
  // must not enter again.
  bool entered_ = false;

 private:
  enum ExecutorStatePriority{
    HIGH = 2,
    LOW = 1, 
    // 0 is reserved.
  };
  // wxf
  // Priority of this ExecutorState instance inherited from DirectSession which
  // creates this ExecutorState instance. 
  // It should be initialized by const Executor::Args& args.
  int executor_state_priority_;

  // wxf
  // Tell the ExecutorState which device type it is running on now.
  // It is used to construct the ExecutorState instances.
  string device_type_executing_on_;

 public: // wxf
  // Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
  // TODO(yuanbyu): A better way to do "has_value"?
  struct Entry {
    Entry() {}
    Entry(const Entry& other)
        : ref(other.ref),
          ref_mu(other.ref_mu),
          has_value(other.has_value),
          val_field_is_set(other.val_field_is_set),
          alloc_attr(other.alloc_attr),
          device_context(other.device_context) {
      if (val_field_is_set) {
        val.Init(*other.val);
      }
    }
    ~Entry() {
      if (val_field_is_set) val.Destroy();
    }

    Entry& operator=(const Entry& other) {
      if (val_field_is_set) {
        val.Destroy();
      }
      ref = other.ref;
      ref_mu = other.ref_mu;
      has_value = other.has_value;
      val_field_is_set = other.val_field_is_set;
      alloc_attr = other.alloc_attr;
      device_context = other.device_context;
      if (val_field_is_set) {
        val.Init(*other.val);
      }
      return *this;
    }

    Entry& operator=(Entry&& other) {
      if (val_field_is_set) {
        val.Destroy();
      }
      ref = other.ref;
      ref_mu = other.ref_mu;
      has_value = other.has_value;
      val_field_is_set = other.val_field_is_set;
      alloc_attr = other.alloc_attr;
      device_context = other.device_context;
      if (val_field_is_set) {
        val.Init(std::move(*other.val));
      }
      return *this;
    }

    // Clears the <val> field.
    void ClearVal() {
      if (val_field_is_set) {
        val.Destroy();
        val_field_is_set = false;
        has_value = false;
      }
    }

    // A tensor value, if val_field_is_set.
    ManualConstructor<Tensor> val;

    Tensor* ref = nullptr;    // A tensor reference.
    mutex* ref_mu = nullptr;  // mutex for *ref if ref is not nullptr.

    // Whether the value exists, either in <val> or <ref>.
    bool has_value = false;

    bool val_field_is_set = false;

    // The attributes of the allocator that creates the tensor.
    AllocatorAttributes alloc_attr;

    // Every entry carries an optional DeviceContext containing
    // Device-specific information about how the Tensor was produced.
    DeviceContext* device_context = nullptr;
  };

 private: // wxf

  // Contains a value for [node->id()] for the device context assigned by the
  // device at the beginning of a step.
  DeviceContextMap device_context_map_;

  struct TaggedNode;
  typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;
  typedef gtl::InlinedVector<Entry, 4> EntryVector;

  struct IterationState {
    explicit IterationState(const PendingCounts* pending_counts,
                            int total_input_tensors)
        : input_tensors(new Entry[total_input_tensors]),
          outstanding_ops(0),
          outstanding_frame_count(0),
          counts_(*pending_counts) {  // Initialize with copy of *pending_counts
    }

    // The state of an iteration.

    // One copy per iteration. For iteration k, i-th node's j-th input is in
    // input_tensors[k][impl_->nodes[i].input_start + j]. An entry is either
    // a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
    //
    // NOTE: No need to protect input_tensors[i] by any locks because it
    // is resized once. Each element of tensors_ is written once by the
    // source node of an edge and is cleared by the destination of the same
    // edge. The latter node is never run concurrently with the former node.
    Entry* input_tensors;

    // The number of outstanding ops for each iteration.
    size_t outstanding_ops;

    // The number of outstanding frames for each iteration.
    int outstanding_frame_count;
    int pending(PendingCounts::Handle h) { return counts_.pending(h); }
    int decrement_pending(PendingCounts::Handle h, int v) {
      return counts_.decrement_pending(h, v);
    }
    // Mark a merge node as live
    // REQUIRES: Node corresponding to "h" is a merge node
    void mark_live(PendingCounts::Handle h) { counts_.mark_live(h); }
    // Mark a node to show that processing has started.
    void mark_started(PendingCounts::Handle h) { counts_.mark_started(h); }
    // Mark a node to show that processing has completed.
    void mark_completed(PendingCounts::Handle h) { counts_.mark_completed(h); }
    PendingCounts::NodeState node_state(PendingCounts::Handle h) {
      return counts_.node_state(h);
    }

    int dead_count(PendingCounts::Handle h) { return counts_.dead_count(h); }
    void increment_dead_count(PendingCounts::Handle h) {
      counts_.increment_dead_count(h);
    }
    void adjust_for_activation(PendingCounts::Handle h, bool increment_dead,
                               int* pending_result, int* dead_result) {
      counts_.adjust_for_activation(h, increment_dead, pending_result,
                                    dead_result);
    }

    ~IterationState() { delete[] input_tensors; }

   private:
    PendingCounts counts_;
  };

  struct FrameState {
    explicit FrameState(const ExecutorImpl* impl, int parallel_iters)
        : executor(impl),
          max_parallel_iterations(parallel_iters),
          num_outstanding_iterations(1) {}

    // A new frame is created for each loop. Execution starts at iteration 0.
    // When a value at iteration 0 passes through a NextIteration node,
    // iteration 1 is created and starts running. Note that iteration 0 may
    // still be running so multiple iterations may run in parallel. The
    // frame maintains the state of iterations in several data structures
    // such as pending_count and input_tensors. When iteration 0 completes,
    // we garbage collect the state of iteration 0.
    //
    // A frame instance is considered "done" and can be garbage collected
    // if all its inputs have entered and all its iterations are "done".
    //
    // A frame manages the live iterations of an iterative computation.
    // Iteration i is considered "done" when there are no outstanding ops,
    // frames at iteration i are done, all recvs for this iteration are
    // completed, and iteration i-1 is done. For iteration 0, we instead
    // wait for there to be no more pending inputs of the frame.
    //
    // Frames and iterations are garbage collected once they are done.
    // The state we need to keep around is highly dependent on the
    // parallelism enabled by the scheduler. We may want to have the
    // scheduler dynamically control the outstanding number of live
    // parallel frames and iterations. To reduce the state space, the
    // scheduler might want to schedule ops in inner frames first and
    // lower iterations first.
    //
    // This frame state is mostly initialized lazily on demand so we
    // don't introduce unnecessary overhead.

    // The executor the frame is in.
    const ExecutorImpl* executor = nullptr;

    // The name of this frame, which is the concatenation of its parent
    // frame name, the iteration of the parent frame when this frame was
    // created, and the value of the attr 'frame_name'.
    string frame_name;

    // The unique id for this frame. Generated by fingerprinting
    // frame_name.
    uint64 frame_id;

    // The iteration id of its parent frame when this frame is created.
    // -1 if there is no parent frame. The frame_name/parent_iter pair
    // uniquely identifies this FrameState.
    int64 parent_iter = -1;

    // The FrameState of its parent frame.
    FrameState* parent_frame = nullptr;

    // The maximum allowed number of parallel iterations.
    const int max_parallel_iterations;

    // The number of inputs this frame is still waiting.
    int num_pending_inputs = 0;

    // The highest iteration number we have reached so far in this frame.
    int64 iteration_count GUARDED_BY(mu) = 0;

    // The number of outstanding iterations.
    int num_outstanding_iterations GUARDED_BY(mu) = 1;

    // The active iteration states of this frame.
    gtl::InlinedVector<IterationState*, 12> iterations;

    // The NextIteration nodes to enter a new iteration. If the number of
    // outstanding iterations reaches the limit, we will defer the start of
    // the next iteration until the number of outstanding iterations falls
    // below the limit.
    std::vector<std::pair<const Node*, Entry>> next_iter_roots GUARDED_BY(mu);

    // The values of the loop invariants for this loop. They are added into
    // this list as they "enter" the frame. When a loop invariant enters,
    // we make it available to all active iterations. When the frame starts
    // a new iteration, we make all the current loop invariants available
    // to the new iteration.
    std::vector<std::pair<const Node*, Entry>> inv_values GUARDED_BY(mu);

    // The list of dead exit nodes for the current highest iteration. We
    // will only "execute" the dead exits of the final iteration.
    std::vector<const Node*> dead_exits GUARDED_BY(mu);

    // Static information specific to this frame.
    PendingCounts* pending_counts = nullptr;
    int total_input_tensors = 0;
    std::vector<const Node*>* nodes = nullptr;

    // Lock ordering: ExecutorState.mu_ < mu;
    // during structured traversal: parent_frame->mu < mu.
    mutex mu;

    void InitializeFrameInfo(const string& enter_name) {
      auto it_frame_info = executor->frame_info_.find(enter_name);
      DCHECK(it_frame_info != executor->frame_info_.end());
      ExecutorImpl::FrameInfo* finfo = it_frame_info->second;
      pending_counts = finfo->pending_counts;
      total_input_tensors = finfo->total_inputs;
      num_pending_inputs = finfo->input_count;
      nodes = finfo->nodes;
    }

    inline IterationState* GetIteration(int64 iter)
        EXCLUSIVE_LOCKS_REQUIRED(mu) {
      size_t index = iter % iterations.size();
      return iterations[index];
    }

    inline void SetIteration(int64 iter, IterationState* state)
        EXCLUSIVE_LOCKS_REQUIRED(mu) {
      size_t index = iter % iterations.size();
      DCHECK(state == nullptr || iterations[index] == nullptr);
      iterations[index] = state;
    }

    // Decrement the outstanding op count and clean up the iterations in the
    // frame. Return true iff the execution of the frame is done.
    inline bool DecrementOutstandingOps(const GraphView* gview, int64 iter,
                                        TaggedNodeSeq* ready) {
      mutex_lock l(mu);
      return DecrementOutstandingOpsLocked(gview, iter, ready);
    }

    // Decrement the outstanding op count and clean up the iterations in the
    // frame. Return true iff the execution of the frame is done.
    inline bool DecrementOutstandingOpsLocked(const GraphView* gview,
                                              int64 iter, TaggedNodeSeq* ready)
        EXCLUSIVE_LOCKS_REQUIRED(mu) {
      IterationState* istate = GetIteration(iter);
      istate->outstanding_ops--;
      if (istate->outstanding_ops != 0) {
        return false;
      } else {
        return CleanupIterations(gview, iter, ready);
      }
    }

    // Returns true if the computation in the frame is completed.
    inline bool IsFrameDone() EXCLUSIVE_LOCKS_REQUIRED(mu) {
      return (num_pending_inputs == 0 && num_outstanding_iterations == 0);
    }

    // Returns true if the iteration of the frame is completed.
    bool IsIterationDone(int64 iter) EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Increments the iteration id. If this is a new iteration, initialize it.
    void IncrementIteration(const GraphView* gview, TaggedNodeSeq* ready)
        EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Activate all the deferred NextIteration nodes in a new iteration.
    void ActivateNexts(const GraphView* gview, int64 iter, TaggedNodeSeq* ready)
        EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Activate all the current loop invariants in a new iteration.
    void ActivateLoopInvs(const GraphView* gview, int64 iter,
                          TaggedNodeSeq* ready) EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Add a new loop invariant and make it available to all active
    // iterations.
    void AddLoopInv(const NodeItem* item, const Entry& value,
                    TaggedNodeSeq* ready) EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Activate the successors of a node. Contents of *outputs are left in an
    // indeterminate state after returning from this method.
    void ActivateNodes(const NodeItem* item, const bool is_dead, int64 iter,
                       EntryVector* outputs, TaggedNodeSeq* ready)
        EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Cleanup iterations of this frame starting from iteration iter.
    bool CleanupIterations(const GraphView* gview, int64 iter,
                           TaggedNodeSeq* ready) EXCLUSIVE_LOCKS_REQUIRED(mu);

    ~FrameState() {
      for (size_t i = 0; i < iterations.size(); ++i) {
        delete iterations[i];
        iterations[i] = nullptr;
      }
    }
  };

  // A tagged node: <frame*, iter, node*>.
  struct TaggedNode {
    const Node* node = nullptr;
    FrameState* input_frame = nullptr;
    int64 input_iter = -1;
    bool is_dead = false;

    TaggedNode(const Node* t_node, FrameState* in_frame, int64 in_iter,
               bool dead) {
      node = t_node;
      input_frame = in_frame;
      input_iter = in_iter;
      is_dead = dead;
    }
  };

  // A drop-in replacement for std::deque<TaggedNode>.  We typically don't
  // have that many nodes in the ready queue, so we just use a vector and
  // don't free up memory from the queue as we consume nodes.
  class TaggedNodeReadyQueue {
   public:
    TaggedNodeReadyQueue() : front_index_(0) {}

    void push_back(TaggedNode node) { ready_.push_back(node); }
    TaggedNode front() const {
      DCHECK_LT(front_index_, ready_.size());
      return ready_[front_index_];
    }
    void pop_front() {
      DCHECK_LT(front_index_, ready_.size());
      front_index_++;
      if ((front_index_ == ready_.size()) || (front_index_ > 16384)) {
        if (front_index_ == ready_.size()) {
          ready_.clear();
        } else {
          // Lots of unused entries at beginning of vector: move everything
          // down to start of vector.
          ready_.erase(ready_.begin(), ready_.begin() + front_index_);
        }
        front_index_ = 0;
      }
    }
    bool empty() const { return ready_.empty(); }
    const TaggedNode* begin() const { return ready_.begin() + front_index_; }
    const TaggedNode* end() const { return ready_.end(); }

    // wxf
    void clear() { ready_.clear(); }

    // wxf
    int size() { return (end()-begin()); }

   private:
    gtl::InlinedVector<TaggedNode, 16> ready_;
    int front_index_;
  };

  struct AsyncState;

  const bool vlog_;  // true if VLOG_IS_ON(1). Used to check vlog cheaply.

  // true if LogMemory::IsEnabled(). Used to check memory enabled cheaply.
  const bool log_memory_;

  int64 step_id_;
  // Not owned.
  Rendezvous* rendezvous_;
  CollectiveExecutor* collective_executor_ = nullptr;
  SessionState* session_state_;
  string session_handle_;
  TensorStore* tensor_store_;
  // Step-local container.
  ScopedStepContainer* step_container_;
  StepStatsCollectorInterface* const stats_collector_;
  const tracing::EventCollector* const event_collector_;
  Context context_;

  // QUESTION: Make it a checkpoint::TensorSliceReaderCacheWrapper
  // instead of a pointer?  (avoids having to delete).
  checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache_;
  CallFrameInterface* call_frame_;
  const ExecutorImpl* impl_;
  CancellationManager* cancellation_manager_;
  Executor::Args::Runner runner_;
  bool sync_on_finish_;
  const bool trace_using_annotations_;

  // Owned.

  // A flag that is set on error after the frame state has been
  // dumped for diagnostic purposes.
  bool dumped_on_error_ = false;

  // The root frame in which the execution of this step is started.
  FrameState* root_frame_;

  // Invoked when the execution finishes.
  Executor::DoneCallback done_cb_;

  std::atomic_int_fast32_t num_outstanding_ops_;

  // Available via OpKernelContext to every OpKernel invocation.
  mutex num_deferred_ops_mu_;
  condition_variable no_deferred_ops_cv_;
  int64 num_deferred_ops_ GUARDED_BY(num_deferred_ops_mu_) = 0;

  mutex mu_;
  Status status_ GUARDED_BY(mu_);

  // Mapping from frame name to outstanding frames. A new frame is created
  // at some iteration of an active frame. So the unique key for the new
  // child frame is composed of the name of the parent frame, the iteration
  // number at which the parent frame is creating the new frame, and the
  // name of the new frame from nodedef.
  gtl::FlatMap<string, FrameState*> outstanding_frames_ GUARDED_BY(mu_);

  // The unique name of a frame.
  inline string MakeFrameName(FrameState* frame, int64 iter_id,
                              const string& name) {
    return strings::StrCat(frame->frame_name, ";", iter_id, ";", name);
  }

  // Find an existing or create a new child frame in the frame 'frame' at
  // iteration 'iter'.
  void FindOrCreateChildFrame(FrameState* frame, int64 iter, const Node* node,
                              FrameState** child);

  // Delete a frame. Called when the frame is done.
  void DeleteFrame(FrameState* frame, TaggedNodeSeq* ready);

  // Cleanup frames and iterations starting from frame/iter. Called when
  // a child frame is done.
  void CleanupFramesIterations(FrameState* frame, int64 iter,
                               TaggedNodeSeq* ready);

  // Process a ready node in current thread.
  void Process(TaggedNode node, int64 scheduled_nsec);

  // Before invoking item->kernel, fills in its "inputs".
  Status PrepareInputs(const NodeItem& item, Entry* first_input,
                       TensorValueVec* inputs,
                       DeviceContextVec* input_device_contexts,
                       AllocatorAttributeVec* input_alloc_attrs,
                       bool* is_input_dead);

  // After item->kernel computation is done, processes its outputs.
  Status ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                        EntryVector* outputs, NodeExecStatsInterface* stats);

  // After processing the outputs, propagates the outputs to their dsts.
  // Contents of *outputs are left in an indeterminate state after
  // returning from this method.
  void PropagateOutputs(const TaggedNode& tagged_node, const NodeItem* item,
                        EntryVector* outputs, TaggedNodeSeq* ready);

  // "node" just finishes. Takes ownership of "stats". Returns true if
  // execution has completed.
  bool NodeDone(const Status& s, const Node* node, const TaggedNodeSeq& ready,
                NodeExecStatsInterface* stats,
                TaggedNodeReadyQueue* inline_ready);


  /////////////////////////////////////////////////////////////
  // wxf: I will add some logic inside this function
  // Schedule all the expensive nodes in 'ready', and put all the inexpensive
  // nodes in 'ready' into 'inline_ready'.
  void ScheduleReady(const TaggedNodeSeq& ready,
                     TaggedNodeReadyQueue* inline_ready);
  /////////////////////////////////////////////////////////////


  // For debugging/logging only.
  inline void MaybeMarkCompleted(FrameState* frame, int64 iter, int64 id);

  // Provide debugging output about an outstanding node in the executor.
  void DumpPendingNodeState(const int node_id, const Entry* input_vector,
                            bool show_nodes_with_no_ready_inputs);
  void DumpActiveNodeState(const int node_id, const Entry* input_vector);

  // Provide debugging output about an outstanding iteration in the executor.
  void DumpIterationState(const FrameState* frame, IterationState* iteration);

  // Provide debugging output of the state of the executor.
  void DumpState();
  const Tensor* GetTensorValueForDump(const Entry& input);

  // Clean up when this executor is done.
  void Finish();
  // Schedule Finish() on a separate thread if it needs to wait for deferred
  // async ops to complete; otherwise run it on the current thread.
  void ScheduleFinish();

  // A standalone routine for this expression so that we can express
  // that we don't want thread safety analysis on this reference (it's
  // safe to do without the lock because the iterations array never
  // resizes and this particular iteration's array element will not
  // be changed out from under us because the iteration is still alive).
  Entry* GetInputTensors(FrameState* input_frame,
                         int64 input_iter) const NO_THREAD_SAFETY_ANALYSIS {
    return input_frame->GetIteration(input_iter)->input_tensors;
  }
};


// State kept alive for executing an asynchronous node in another
// thread.  NOTE: We need to make a copy of p.input,
// p.input_device_contexts, and p.input_alloc_attrs for asynchronous
// kernels because OpKernelContext methods like input_type(i) needs
// the param points to valid input type vector. It's not an issue for
// sync kernels because these vectors are kept on the stack.
struct ExecutorState::AsyncState {
  AsyncState(const OpKernelContext::Params& p, const TaggedNode& _tagged_node,
             const NodeItem* _item, Entry* _first_input,
             NodeExecStatsInterface* _stats)
      : saved_inputs(*p.inputs),
        saved_input_device_contexts(*p.input_device_contexts),
        saved_input_alloc_attrs(*p.input_alloc_attrs),
        params(p),
        tagged_node(_tagged_node),
        item(_item),
        first_input(_first_input),
        // ParamsButClearingEigenGPUDevice does equivalent of
        //   params.eigen_gpu_device = nullptr;
        ctx(ParamsButClearingEigenGPUDevice(&params), item->num_outputs),
        stats(_stats) {
    params.inputs = &saved_inputs;
    params.input_device_contexts = &saved_input_device_contexts;
    params.input_alloc_attrs = &saved_input_alloc_attrs;
  }

  TensorValueVec saved_inputs;
  DeviceContextVec saved_input_device_contexts;
  AllocatorAttributeVec saved_input_alloc_attrs;
  OpKernelContext::Params params;
  TaggedNode tagged_node;
  const NodeItem* item;
  Entry* first_input;
  OpKernelContext ctx;
  NodeExecStatsInterface* stats;

 private:
  OpKernelContext::Params* ParamsButClearingEigenGPUDevice(
      OpKernelContext::Params* p) {
    // Ensure OpKernelContext constructor will make a new eigen GPU device if
    // necessary.
    p->eigen_gpu_device = nullptr;  // Force allocation
    return p;
  }
};

// -----------------------------------------------------------------------------

::tensorflow::Status NewLocalExecutor(const LocalExecutorParams& params,
                                      std::unique_ptr<const Graph> graph,
                                      Executor** executor);

// A class to help run multiple executors in parallel and wait until
// all of them are complete.
//
// ExecutorBarrier deletes itself after the function returned by Get()
// is called.
class ExecutorBarrier {
 public:
  typedef std::function<void(const Status&)> StatusCallback;

  // Create an ExecutorBarrier for 'num' different executors.
  //
  // 'r' is the shared Rendezvous object that is used to communicate
  // state.  If any of the executors experiences an error, the
  // rendezvous object will be aborted exactly once.
  //
  // 'done' is called after the last executor completes, and
  // ExecutorBarrier is deleted.
  ExecutorBarrier(size_t num, Rendezvous* r, StatusCallback done)
      : rendez_(r), done_cb_(done), pending_(num) {}

  ~ExecutorBarrier() {}

  // Returns a closure that Executors must call when they are done
  // computing, passing the status of their execution as an argument.
  StatusCallback Get() {
    return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1);
  }

 private:
  Rendezvous* rendez_ = nullptr;
  StatusCallback done_cb_ = nullptr;

  mutable mutex mu_;
  int pending_ GUARDED_BY(mu_) = 0;
  StatusGroup status_group_ GUARDED_BY(mu_);

  void WhenDone(const Status& s) {
    // wxf
    // filter the tensorflow::error::CANCELLED status as ok.
    Status s_new;
    if (s.code() != tensorflow::error::SESS_RUN_CANCELLED &&
        s.code() != tensorflow::error::OK) {
      s_new.Update(s);
    }

    //~wxf
    
    Rendezvous* error_rendez = nullptr;
    StatusCallback done = nullptr;
    Status status;

    {
      mutex_lock l(mu_);

      // If we are the first error encountered, trigger an abort of the
      // Rendezvous object by this thread only.
      //if (status_group_.ok() && !s.ok()) {
      // wxf
      if (status_group_.ok() && !s_new.ok()) {
        error_rendez = rendez_;
        error_rendez->Ref();
      }

      //status_group_.Update(s);
      // wxf
      status_group_.Update(s_new);

      // If this is the last call to WhenDone, call the final callback
      // below.
      if (--pending_ == 0) {
        CHECK(done_cb_ != nullptr);
        std::swap(done, done_cb_);
        status = status_group_.as_status();
      }
    }

    if (error_rendez != nullptr) {
      error_rendez->StartAbort(
          errors::Aborted("Stopping remaining executors."));
      error_rendez->Unref();
    }

    if (done != nullptr) {
      delete this;
      done(status);
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorBarrier);
};

// A few helpers to facilitate create/delete kernels.

// Creates a kernel based on "ndef" on device "device". The kernel can
// access the functions in the "flib". The caller takes ownership of
// returned "*kernel".
Status CreateNonCachedKernel(Device* device, FunctionLibraryRuntime* flib,
                             const NodeDef& ndef, int graph_def_version,
                             OpKernel** kernel);

// Deletes "kernel" returned by CreateKernel.
void DeleteNonCachedKernel(OpKernel* kernel);

// wxf
// global var to reuse input data
extern std::vector<ExecutorState::Entry*> reuse_entry_inputs;
extern bool matmul01_is_ok;
extern bool matmul02_is_ok;

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_H_
