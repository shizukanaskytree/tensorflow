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

#include "tensorflow/core/common_runtime/executor.h"

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
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
#include "tensorflow/core/platform/env.h"   // 时间
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {
namespace {

// 1-D, 0 element tensor.
static const Tensor* const kEmptyTensor = new Tensor;

bool IsInitializationOp(const Node* node) {
  return node->op_def().allows_uninitialized_input();
}

// Helper routines for collecting step stats.
namespace nodestats {
inline int64 NowInNsec() { return Env::Default()->NowNanos(); }
// 1.
// Env::Default() 函数说明:
// Env* Env::Default()
// tensorflow/core/platform/posix/env.cc

// 2.
// NowNanos() 函数说明:
// virtual uint64 NowNanos() { return envTime->NowNanos(); }
// tensorflow/core/platform/env.h

// 3.
// envTime 变量说明:
// class Env::envTime: EnvTime* , default_value : EnvTime::Default()
// uint64 NowNanos() override
// tensorflow/core/platform/posix/env_time.cc
//
// 3.1
// 概述:
// 返回自 1970 年开始至今的 总 elapsed time (单位:NanoSeconds)

void SetScheduled(NodeExecStatsInterface* stats, int64 micros) {
  if (!stats) return;
  stats->SetScheduled(micros * EnvTime::kMicrosToNanos);
}
// 1.
// kMicrosToNanos 变量说明
// static constexpr uint64 kMicrosToNanos = 1000ULL;
// tensorflow/core/platform/env_time.h

// 1.1
// kMicrosToNanos 变量作用
// 1.1.1
// 把 NanoSeconds 转换为 MicroSeconds
// - now_nanos / EnvTime::kMicrosToNanos

// 1.1.2
// micros * EnvTime::kMicrosToNanos
// 把 MicroSeconds 转换为 NanoSeconds
// - micros * EnvTime::kMicrosToNanos


void SetAllStart(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordExecutorStarted();
}

void SetOpStart(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordComputeStarted();
  // 1.
  // RecordComputeStarted 函数说明:
  // void NodeExecStatsWrapper::RecordComputeStarted()
  // tensorflow/core/common_runtime/step_stats_collector.cc
  //
  // 1.1
  // 概述:
  // 仅仅只是记录调用这个函数被调用时刻的时间于 NodeExecStats::op_start_rel_micros 内

  // 2.
  // class NodeExecStatsInterface 数据结构:
  // tensorflow/core/common_runtime/step_stats_collector.h
  // 纯虚函数
  //
  // 2.1
  // 继承类
  // - class NodeExecStatsWrapper [final]
  //   tensorflow/core/common_runtime/step_stats_collector.h
  // - class SimpleNodeExecStats [final]
  //   tensorflow/core/kernels/data/captured_function.cc
  //
  // 2.2
  // 接口函数:
  // - void Done(const string& device)
  //   Called when the statistics collection for the node has finished. Once this
  //   method is called, the caller should not make assumptions about the validity
  //   of this object.
  // - void RecordExecutorStarted()
  //   Called immediately after this node starts being processed by the executor.
  // - void RecordComputeStarted()
  //   Called immediately before this node's `Compute()` or `ComputeAsync()` method is called.
  // - void RecordComputeEnded()
  //   Called immediately after this node's `Compute()` method returned (or, for asynchronous operations, the callback passed to its `ComputeAsync()` method was called).
  // - void RecordExecutorEnded()
  //   Called immediately after this executor finishes processing this node.
  // - bool TrackAllocations() const
  //   Returns `true` if this object should track memory allocations.
  // - void SetMemory(OpKernelContext* ctx)
  //   Records information about the memory allocated during the execution of this node.
  //   Takes ownership of any `TrackingAllocator` objects stored in `ctx`.
  // - void SetOutput(int slot, const Tensor* tensor)
  //   Records information about the tensor produced by this node at the given output slot.
  // - void SetReferencedTensors(const TensorReferenceVector& tensors)
  //   Records information about the tensors that were accessed during the execution of this node.
  // - void SetScheduled(int64 nanos)
  //   Records the absolute time in nanoseconds at which this node became runnable (i.e. was scheduled for execution).
}

void SetOpEnd(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordComputeEnded();
}

void SetAllEnd(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordExecutorEnded();
}

void SetOutput(NodeExecStatsInterface* stats, int slot, const Tensor* v) {
  if (!stats) return;
  stats->SetOutput(slot, v);
}

void SetMemory(NodeExecStatsInterface* stats, OpKernelContext* ctx) {
  if (!stats) return;
  stats->SetMemory(ctx);
}

void SetReferencedTensors(NodeExecStatsInterface* stats,
                          const TensorReferenceVector& tensors) {
  if (!stats) return;
  stats->SetReferencedTensors(tensors);
}

}  // namespace nodestats

class ExecutorImpl;
class GraphView;

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


// =======================================================================

// QQQ. 我就不明白了，为什么要多此一举画蛇添足地增加一个 和 Node 功能类型的数据结构 NodeItem？
// AAA. 大型工程，多人协作的前提下，要向前兼容，只能构造新的数据结构，不好改以前的。
struct NodeItem {
  NodeItem() {}

  // A graph node.
  const Node* node = nullptr;

  // The kernel for this node.
  OpKernel* kernel = nullptr;
  // OpKernel 数据结构
  // tensorflow/core/framework/op_kernel.h:78:class OpKernel
  // // Initial time (in CPU cycles) we expect an operation to take.  Used to
  // // determine whether an operation should be place in a threadpool.  Operations
  // // start out "expensive".
  // static const uint64 kInitialCostEstimateCycles = 100 * 1000 * 1000;
  // static const uint64 kOpIsExpensiveThresholdCycles = 5000;
  // static const uint64 kCostDecay = 10;
  // - def_ : const std::unique_ptr<const NodeDef>
  // - input_types_: const DataTypeVector
  // - input_memory_types_ : const MemoryTypeVector
  // - output_types_: const DataTypeVector
  // - output_memory_types_: const MemoryTypeVector
  // - graph_def_version_ : const int
  // - is_internal_: const int; // True if this is an internal operation
  // - input_name_map_: NameRangeMap
  // - output_name_map_: NameRangeMap
  //  * NameRangeMap: typedef gtl::FlatMap<StringPiece, std::pair<int, int>, hash<StringPiece>>
  // - expensive_: bool
  // - cost_estimate_: std::atomic_uint_fast64_t


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
  // EdgeInfo 数据结构
  // tensorflow/core/common_runtime/executor.cc
  // - dst_id: int
  // - output_slot: int
  // - is_last: bool
  // - input_slot: int

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

  /// \note 什么意思? 指的是什么?
  ///       对应的都是下面函数的返回值，下面的都是数组，所以是如此计算
  // Variable length section starts immediately after *this
  // (uint8 is enough for DataType).
  //   EdgeInfo            out_edges[num_out_edges];
  //   AllocatorAttributes output_attr[num_outputs];
  //   int                 forward_from[num_outputs];
  //   uint8               input_type[num_inputs];
  //   uint8               output_type[num_outputs];

  /// 下面的我完全不懂是为什么这样算。
  /// 地址 + 数据结构长度(offset)
  /// |NodeItem|EdgeInfo ...多个重复 |AllocatorAttributes|int|uint8|uint8|
  /// this 的起始地址是最左边的竖杠
  // Return pointer to variable length section.
  char* var() const {
    return const_cast<char*>(reinterpret_cast<const char*>(this) +
                             sizeof(NodeItem));
  }

  /// 从起始位置开始后的数据结构 cast 成 EdgeInfo 类型
  EdgeInfo* output_edge_base() const {
    return reinterpret_cast<EdgeInfo*>(var());
  }

  /// 举例来说，如果这个 Node 有四个 Edge, 然后存储完他们后的结构如下
  /// |NodeItem|EdgeInfo EdgeInfo EdgeInfo EdgeInfo|
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

// =======================================================================

typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
// 1.
// TensorValueVec (typedef)数据结构
// typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
// tensorflow/core/common_runtime/executor.cc

// 2.
// struct TensorValue 数据结构
// tensorflow/core/framework/op_kernel.h
// 概述:
// Holds a tensor or tensor reference. For tensor references, we need
// a mutex to prevent concurrent access to the tensor.
//
// - tensor: Tensor*
// - mutex_if_ref: mutex*


typedef gtl::InlinedVector<DeviceContext*, 4> DeviceContextVec;
// 1.
// DeviceContextVec (typedef) 数据结构
// typedef gtl::InlinedVector<DeviceContext*, 4> DeviceContextVec;
// tensorflow/core/common_runtime/executor.cc

// 2.
// class DeviceContext 数据结构
// tensorflow/core/framework/device_base.h
//
// class DeviceContext: public core::RefCounted
// tensorflow/core/framework/device_base.h
// 概述:
// A class that devices can subclass to pass around
// Device-specific context to OpKernels.
//
// 没有成员变量
//
// 接口:
// - stream()
// - MaintainLifetimeOnStream
// - CopyCPUTensorToDevice
// - CopyTensorInSameDevice
// - CopyDeviceTensorToCPU
// - ThenExecute

typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;
// 1.
// AllocatorAttributes 数据结构
// tensorflow/core/framework/allocator.h
//
//  00000000
//  ^^^^^^^^
//  ||||||||
//  |||||||+----+on host
//  ||||||+-----+nic compatible
//  |||||+------+gpu compatible
//  ||||+-------+
//  |||+--------+
//  ||+---------+
//  |+----------+
//  +-----------+

// =======================================================================

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
// 1.
// class GraphView 数据结构
// tensorflow/core/common_runtime/executor.h
// 概述:
// Immutable view of a Graph organized for efficient execution.
//
// - num_nodes_: int32 , default value : 0
// - node_offsets_: uint32* , default value : nullptr
//   array of size "graph_.num_node_ids()"
// - space_: char*
//   NodeItem objects are allocated here

// =======================================================================


// =======================================================================

class ExecutorImpl : public Executor {
 public:
  ExecutorImpl(
    const LocalExecutorParams& p,
    std::unique_ptr<const Graph> g)
      : params_(p),
        graph_(std::move(g)),
        gview_() {
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
  }

  Status Initialize();

  // Process all Nodes in the current graph, attempting to infer the
  // memory allocation attributes to be used wherever they may allocate
  // a tensor buffer.
  Status SetAllocAttrs();

  void RunAsync(const Args& args, DoneCallback done) override;

 private:
  friend class ExecutorState;

  struct ControlFlowInfo {
    // unique_frame_names 的含义:
    // 如果这个 Node 没有 in edges 那么就把
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
  // 1.
  // class ExecutorImpl::FrameInfo 数据结构
  // - input_count: int
  //    The total number of inputs to a frame.
  // - total_inputs:
  //    The total number of input tensors of a frame.
  //    == sum(nodes[*].num_inputs()) where nodes are the nodes in the frame.
  // - pending_counts_layout: PendingCounts::Layout
  //    Used to determine the next place to allocate space in the
  //    pending_counts data structure we'll eventually construct
  // - pending_counts: PendingCounts*
  //    Each frame has its own PendingCounts only for the nodes in the frame.
  // - nodes: std::vector<const Node*>*
  //    The nodes in a frame. Used only for debugging.

  // 2.
  // PendingCounts 数据结构
  // tensorflow/core/common_runtime/pending_counts.h:48:class PendingCounts
  // 概述:
  // PendingCounts is an internal helper class to keep track of pending and
  // dead counts for nodes, for use in the ExecutorState module.  It
  // holds a map from Handles to various counts for that handle.  This
  // information is needed per frame iteration. The amount of memory
  // needed for an iteration is the same across all executions of the
  // iteration. The memory amount and handles are precomputed at startup
  // using a Layout object.

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
};
// class ExecutorImpl : public Executor END.
// 1.
// class ExecutorImpl 数据结构
// class ExecutorImpl : public Executor
// tensorflow/core/common_runtime/executor.cc
// * struct FrameInfo
// * struct ControlFlowInfo
// - params_: LocalExecutorParams
// - graph_: std::unique_ptr<const Graph>
// - gview_: GraphView
// - device_record_tensor_accesses_: bool, default_value: false
// - root_nodes_: std::vector<const Node*>
// - frame_info_: gtl::FlatMap<string, FrameInfo*>


// =======================================================================


// =======================================================================

// Infer memory allocation attributes of a node n's output,
// based on its use node dst.  Note that dst might not be directly
// connected to n by a single edge, but might be a downstream
// consumer of n's output by reference.  *attr is updated with any
// necessary attributes.
Status InferAllocAttr(const Node* n, const Node* dst,
                      const DeviceNameUtils::ParsedName& local_dev_name,
                      AllocatorAttributes* attr);

GraphView::~GraphView() {
  static_assert(std::is_trivially_destructible<AllocatorAttributes>::value,
                "Update code if AllocatorAttributes gains a destructor");
  static_assert(std::is_trivially_destructible<EdgeInfo>::value,
                "Update code if EdgeInfo gains a destructor");
  for (int i = 0; i < num_nodes_; i++) {
    NodeItem* n = node(i);
    if (n != nullptr) {
      n->NodeItem::~NodeItem();
      // Memory for "n" itself is held in space_ & gets cleaned up below
    }
  }
  delete[] node_offsets_;
  delete[] space_;
}

/** \brief
 *
 *  \param [in] n: const Node*
 *
 *  \return The total size of a Node in size_t
 *
 */
size_t GraphView::NodeItemBytes(const Node* n) {
  /// num of output edge != num of output
  const size_t num_output_edges = n->out_edges().size();
  const int num_inputs = n->num_inputs();
  const int num_outputs = n->num_outputs();

  // Compute number of bytes needed for NodeItem and variable length data.
  // We do not subtract sizeof(var) since num_inputs/num_outputs might
  // both be zero.
  const size_t raw_bytes =
      sizeof(NodeItem)                             // Fixed
      /// \note 什么意思? 指的是什么?
      ///       对应的都是 NodeItem private 内函数的返回值，都是数组
      ///       所以是如此计算
      + num_output_edges * sizeof(EdgeInfo)        // output_edges[...]
      + num_outputs * sizeof(AllocatorAttributes)  // output_attr[...]
      + num_outputs * sizeof(int)                  // forward_from[num_outputs]
      + num_inputs * sizeof(uint8)                 // input_type[num_inputs]
      + num_outputs * sizeof(uint8);               // output_type[num_outputs]

  /// NodeItem must be aligned with kItemAlignment
  /// EdgeInfo must be aligned with kItemAlignment
  /// AllocatorAttributes must be aligned with kItemAlignment
  /// NodeItem must be aligned with EdgeInfo
  /// NodeItem must be aligned with AllocatorAttributes
  /// EdgeInfo must be aligned with AllocatorAttributes
  static constexpr size_t kItemAlignment = sizeof(NodeItem*);
  static_assert(kItemAlignment % alignof(NodeItem) == 0,
                "NodeItem must be aligned with kItemAlignment");
  static_assert(kItemAlignment % alignof(EdgeInfo) == 0,
                "EdgeInfo must be aligned with kItemAlignment");
  static_assert(kItemAlignment % alignof(AllocatorAttributes) == 0,
                "AllocatorAttributes must be aligned with kItemAlignment");
  static_assert(sizeof(NodeItem) % alignof(EdgeInfo) == 0,
                "NodeItem must be aligned with EdgeInfo");
  static_assert(sizeof(NodeItem) % alignof(AllocatorAttributes) == 0,
                "NodeItem must be aligned with AllocatorAttributes");
  static_assert(sizeof(EdgeInfo) % alignof(AllocatorAttributes) == 0,
                "EdgeInfo must be aligned with AllocatorAttributes");
  /// kItemAlignment = 8, 感觉是一种向高位补齐的方法
  /// 先算 ((raw_bytes + kItemAlignment - 1) / kItemAlignment) 是小数但是舍去小数部分
  /// 奇怪的是 p kItemAlignment
  /// $2 = 216455366371460479
  /// p sizeof(NodeItem*)
  /// $3 = 8
  const size_t bytes =
      ((raw_bytes + kItemAlignment - 1) / kItemAlignment) * kItemAlignment;
  return bytes;
}

/** \brief 对一个 Node，其起始的某一段地址空间，初始化一系列 EdgeInfo, AllocatorAttributes
 *         forward_from_base, input_type, output_type 这几个参数。
 */
char* GraphView::InitializeNode(char* ptr, const Node* n) {
  const int id = n->id();
  /// kuint32max
  CHECK(node_offsets_[id] == kuint32max);  // Initial value in constructor

  const size_t bytes = NodeItemBytes(n);
  constexpr size_t kItemAlignment = sizeof(NodeItem*);
  CHECK_EQ(reinterpret_cast<uintptr_t>(ptr) % kItemAlignment, 0);
  /// ptr 指向 space_ , char* space_; , NodeItem objects are allocated at space_

  // -----------------------------------------------------------------------
  // 初始化 item 内的各个内容
  NodeItem* item = reinterpret_cast<NodeItem*>(ptr);
  // struct NodeItem 数据结构
  // tensorflow/core/common_runtime/executor.cc:226
  // - node: Node*
  // - kernel: OpKernel* , 进入 executor.cc, 看 NodeItem , 已经完备整理了
  // - kernel_is_async: bool
  // - is_merge: bool
  // - is_enter: bool
  // - is_constant_enter: bool
  // - is_exit: bool
  // - is_control_trigger: bool
  // - is_sink: bool
  // - is_enter_exit_or_next_iter: bool
  // - num_inputs: int              INIT-ED
  // - num_outputs: int             INIT-ED
  // - input_start: int
  // - num_output_edges: size_t     INIT-ED
  // - pending_id: PendingCounts::Handle
  // -----------------------------------------------------------------------

  // We store a 32-bit offset relative to the beginning of space_, so that we
  // only need an array of 32-bit values to map from node id to the NodeItem*,
  // (versus 64 bits on most machines if we just stored an array of NodeItem*
  // pointers). Casting to int64 is needed on 32bit CPU to avoid comparing
  // values as "int" vs "size_t" in CHECK_LE.
  /// ptr - space_ 是 0
  CHECK_LE(static_cast<int64>(ptr - space_), kuint32max);
  const uint32 offset = static_cast<uint32>(ptr - space_);
  node_offsets_[id] = offset;
  /// 偏移一个 NodeItem
  ptr += bytes;

  const size_t num_output_edges = n->out_edges().size();
  const int num_inputs = n->num_inputs();
  const int num_outputs = n->num_outputs();

  /// 构造一个 NodeItem, 然后填充各个 member variable.
  new (item) NodeItem();
  item->num_inputs = num_inputs;
  item->num_outputs = num_outputs;
  item->num_output_edges = num_output_edges;

  // Fill output edges.
  // Keep track of the last EdgeInfo in the EdgeInfo array that references
  // a given output slot.  For all but the last, we need to do a copy of the
  // Tensor when propagating results downstream in the graph, but for the
  // last one, we can just do a move of the Tensor object to propagate it.
  gtl::InlinedVector<EdgeInfo*, 4> last_indices(num_outputs, nullptr);

  /// 其实 dst_edge 是临时变量，用于给 item->output_edge_base()
  /// |NodeItem|EdgeInfo ...| 地址空间是这样的
  /// 后面对 EdgeInfo 开始初始化
  EdgeInfo* dst_edge = item->output_edge_base(); // 地址从 base 开始

  // 对于 node 的 out edges 的每条边
  for (auto e : n->out_edges()) {

    dst_edge->dst_id = e->dst()->id();

    CHECK_LE(e->src_output(), 0x3FFFFFFF);  // Must fit in 31 bits

    dst_edge->output_slot = e->src_output();

    dst_edge->is_last = false;

    const int output_slot = dst_edge->output_slot;

    if (output_slot >= 0) {
      last_indices[output_slot] = dst_edge;
    }

    dst_edge->input_slot = e->dst_input();
    dst_edge++;
  }

  for (EdgeInfo* edge_info : last_indices) {
    if (edge_info != nullptr) {
      edge_info->is_last = true;
    }
  }

  AllocatorAttributes* output_attrs = item->output_attr_base();
  // 开始折腾 output_attr_base 把地址偏移到了 AllocatorAttributes 的位置
  // |NodeItem|EdgeInfo EdgeInfo ...|AllocatorAttributes|
  //                              |
  //                    从 output_attrs base 开始

  for (int i = 0; i < num_outputs; i++) {
    new (&output_attrs[i]) AllocatorAttributes();
  }


  DCHECK_LT(DataType_MAX, 255);  // Must fit in uint8
  uint8* input_types = item->input_type_base();
  // 开始折腾 output_attr_base 把地址偏移到了 input type 的位置
  // |NodeItem|EdgeInfo EdgeInfo ...|AllocatorAttributes ... | uint8 ... |


  for (int i = 0; i < num_inputs; i++) {
    input_types[i] = static_cast<uint8>(n->input_type(i));
    DCHECK_EQ(item->input_type(i), n->input_type(i));
  }

  // Check ScopedAllocatorAttrs and forward_from.  Also assign output_types.
  {
    std::vector<int> forward_input;

    // get node attributes
    Status fwd_status =
        GetNodeAttr(n->attrs(), "_forward_input", &forward_input);

    std::vector<int> scoped_allocator_attrs;

    Status sa_status =
        GetNodeAttr(n->attrs(), "_scoped_allocator", &scoped_allocator_attrs);

    int* forward_from = item->forward_from_base();

    uint8* output_types = item->output_type_base();

    for (int i = 0; i < num_outputs; ++i) {
      output_types[i] = static_cast<uint8>(n->output_type(i));
      DCHECK_EQ(item->output_type(i), n->output_type(i));

      forward_from[i] = OpKernelContext::Params::kNoReservation;

      if (sa_status.ok()) {
        for (int j = 0; j < scoped_allocator_attrs.size(); j += 2) {
          if (scoped_allocator_attrs[j] == i) {
            // This output slot must be explicitly allocated from a
            // ScopedAllocator.
            forward_from[i] = OpKernelContext::Params::kNeverForward;
            DCHECK_EQ(output_attrs[i].scope_id, 0);
            output_attrs[i].scope_id = scoped_allocator_attrs[j + 1];
          }
        }
      }

      if (fwd_status.ok() &&
          forward_from[i] == OpKernelContext::Params::kNoReservation) {
        DCHECK_EQ(forward_input.size() % 2, 0);
        for (int j = 0; j < forward_input.size(); j += 2) {
          if (forward_input[j + 1] == i) {
            DCHECK_EQ(forward_from[i], OpKernelContext::Params::kNoReservation);
            forward_from[i] = forward_input[j];
            break;
          }
        }
      }

    }
  }

  return ptr;
}


void GraphView::Initialize(const Graph* g) {

  CHECK(node_offsets_ == nullptr);

  const int num_nodes = g->num_node_ids();
  // 打印
  // p num_nodes, $5 = 12

  num_nodes_ = num_nodes;

  size_t total_bytes = 0;

  for (const Node* n : g->nodes()) {
    total_bytes += NodeItemBytes(n);
  }

  node_offsets_ = new uint32[num_nodes];
  for (int i = 0; i < num_nodes; i++) {
    /// kuint32max 是 32 bit 都是 1 的数
    node_offsets_[i] = kuint32max;
  }

  /// char* space_; for NodeItem objects, which are allocated at space_
  space_ = new char[total_bytes];  // NodeItem objects are allocated here
  char* ptr = space_;
  for (const Node* n : g->nodes()) {
    ptr = InitializeNode(ptr, n);
  }
  CHECK_EQ(ptr, space_ + total_bytes);
}



void GetMaxPendingCounts(const Node* n, size_t* max_pending,
                         size_t* max_dead_count) {
  const size_t num_in_edges = n->in_edges().size();
  size_t initial_count;
  if (IsMerge(n)) {
    // merge waits all control inputs so we initialize the pending
    // count to be the number of control edges.
    int32 num_control_edges = 0;
    for (const Edge* edge : n->in_edges()) {
      if (edge->IsControlEdge()) {
        num_control_edges++;
      }
    }
    // Use bit 0 to indicate if we are waiting for a ready live data input.
    initial_count = 1 + (num_control_edges << 1);
  } else {
    initial_count = num_in_edges;
  }

  *max_pending = initial_count;
  *max_dead_count = num_in_edges;
}


Status ExecutorImpl::Initialize() {
  gview_.Initialize(graph_.get());

  // Build the information about frames in this subgraph.
  ControlFlowInfo cf_info;
  // struct ControlFlowInfo 数据结构
  // tensorflow/core/common_runtime/executor.cc:450
  // - unique_frame_names: gtl::FlatSet<string>
  // - frame_names: std::vector<string>

  TF_RETURN_IF_ERROR(BuildControlFlowInfo(
    graph_.get(), // input
    &cf_info)); // output
  // 打印: https://gist.github.com/shizukanaskytree/be140cbe64c23c43360938bece040ea6

  // Cache this value so we make this virtual function call once, rather
  // that O(# steps * # nodes per step) times.
  device_record_tensor_accesses_ =
      params_.device->RequiresRecordingAccessedTensors();

  // ExecutorImpl::params_: struct LocalExecutorParams


  for (auto& it : cf_info.unique_frame_names) {
    EnsureFrameInfo(it)->nodes = new std::vector<const Node*>;
  }

  // Preprocess every node in the graph to create an instance of op
  // kernel for each node.
  /// 对每一个 Node, 初始化 NodeItem 内的各个参数
  for (const Node* n : graph_->nodes()) {
    const int id = n->id();

    const string& frame_name = cf_info.frame_names[id];
    FrameInfo* frame_info = EnsureFrameInfo(frame_name);
    // 1.
    // frame_info 变量说明
    // 概述:
    // 这个指针其实是指向 ExecutorImpl::frame_info_ 的一个临时变量指针，下面应该是对其初始化
    // 打印:
    // p *frame_info
    // $16 = {input_count = 0, total_inputs = 0, pending_counts_layout = {next_offset_ = 0}, pending_counts = 0x0, nodes = 0x555d99cf4b00}

    // 2.
    // EnsureFrameInfo 函数说明
    // EnsureFrameInfo 函数构造了 ExecutorImpl::frame_info_[frame_name] 内的 一个 FrameInfo instance

    // 3.
    // class ExecutorImpl::FrameInfo 数据结构
    // - input_count: int
    //    The total number of inputs to a frame.
    // - total_inputs:
    //    The total number of input tensors of a frame.
    //    == sum(nodes[*].num_inputs()) where nodes are the nodes in the frame.
    // - pending_counts_layout: PendingCounts::Layout
    //    Used to determine the next place to allocate space in the
    //    pending_counts data structure we'll eventually construct
    // - pending_counts: PendingCounts*
    //    Each frame has its own PendingCounts only for the nodes in the frame.
    // - nodes: std::vector<const Node*>*
    //    The nodes in a frame. Used only for debugging.

    // 4.
    // PendingCounts 数据结构
    // tensorflow/core/common_runtime/pending_counts.h:48:class PendingCounts
    // 概述:
    // PendingCounts is an internal helper class to keep track of pending and
    // dead counts for nodes, for use in the ExecutorState module.  It
    // holds a map from Handles to various counts for that handle.  This
    // information is needed per frame iteration. The amount of memory
    // needed for an iteration is the same across all executions of the
    // iteration. The memory amount and handles are precomputed at startup
    // using a Layout object.

    // See if this node is a root node, and if so, add to root_nodes_.
    if (n->in_edges().empty()) {
      root_nodes_.push_back(n);
      // SOURCE_ Node 一定是属于 root_nodes_
      // 打印
      // p n->DebugString()
      // $17 = "{name:'_SOURCE' id:0 source}"
    }

    /// 从一段连续的地址空间取出第 id 个 NodeItem，然后下面对 NodeItem 的各项进行初始化
    NodeItem* item = gview_.node(id);
    item->node = n; // 这个时候才初始化 item->node 指针

    item->input_start = frame_info->total_inputs;
    // 打印 p item->input_start 结果为 0
    // ptype frame_info->total_inputs
    // type = int
    // p frame_info->total_inputs
    // $19 = 0

    // 2.
    // 这里 input_start 的意思是连续的 frame_info->total_inputs 的一个计数

    frame_info->total_inputs += n->num_inputs();
    // 你是说，你把所有 node 的 所有 input 都编好

    // 根据 Node definition 创建 kernel, 然后赋值给 item 内的一项参数
    // ------------------------------------------------------------------------
    Status s = params_.create_kernel(n->def(), &item->kernel);
    // ------------------------------------------------------------------------
    // 1.
    // item->kernel 变量说明:
    //
    // p item->kernel, 对于 SOURCE_
    // $20 = (tensorflow::OpKernel *) 0x0

    // 2.
    // item 变量说明
    // item: NodeItem*

    // 3.
    // struct NodeItem 数据结构
    // tensorflow/core/common_runtime/executor.cc:226
    // - node: Node*
    // - kernel: OpKernel* , 进入 executor.cc, 看 NodeItem , 已经完备整理了
    // - kernel_is_async: bool
    // - is_merge: bool
    // - is_enter: bool
    // - is_constant_enter: bool
    // - is_exit: bool
    // - is_control_trigger: bool
    // - is_sink: bool
    // - is_enter_exit_or_next_iter: bool
    // - num_inputs: int              INIT-ED
    // - num_outputs: int             INIT-ED
    // - input_start: int
    // - num_output_edges: size_t     INIT-ED
    // - pending_id: PendingCounts::Handle


    if (!s.ok()) {
      item->kernel = nullptr;
      s = AttachDef(s, *n);
      LOG(ERROR) << "Executor failed to create kernel. " << s;
      return s;
    }

    CHECK(item->kernel);
    item->kernel_is_async = (item->kernel->AsAsync() != nullptr);
    item->is_merge = IsMerge(n);
    item->is_enter = IsEnter(n);
    if (item->is_enter) {
      bool is_constant_enter;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(n->attrs(), "is_constant", &is_constant_enter));
      item->is_constant_enter = is_constant_enter;
    } else {
      item->is_constant_enter = false;
    }
    item->is_exit = IsExit(n);
    item->is_control_trigger = IsControlTrigger(n);
    item->is_sink = IsSink(n);
    item->is_enter_exit_or_next_iter =
        (IsEnter(n) || IsExit(n) || IsNextIteration(n));

    // Compute the maximum values we'll store for this node in the
    // pending counts data structure, and allocate a handle in
    // that frame's pending counts data structure that has enough
    // space to store these maximal count values.
    size_t max_pending, max_dead;
    GetMaxPendingCounts(n, &max_pending, &max_dead);
    item->pending_id =
        frame_info->pending_counts_layout.CreateHandle(max_pending, max_dead);

    // Initialize static information about the frames in the graph.
    frame_info->nodes->push_back(n);
    if (IsEnter(n)) {
      string enter_name;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "frame_name", &enter_name));
      EnsureFrameInfo(enter_name)->input_count++;
    }
  }

  // Initialize PendingCounts only after item->pending_id is initialized for
  // all nodes.
  InitializePending(graph_.get(), cf_info);

  return gview_.SetAllocAttrs(graph_.get(), params_.device);
}

// If a Node has been marked to use a ScopedAllocator x for output i, then
// sc_attr will contain the subsequence (i, x) at an even offset.  This function
// extracts and transfers that ScopedAllocator id to alloc_attr.  For now, we
// only allow one ScopedAllocator use per Node.
bool ExtractScopedAllocatorAttr(const std::vector<int>& sc_attr,
                                int output_index,
                                AllocatorAttributes* alloc_attr) {
  DCHECK_LE(2, sc_attr.size());
  for (int i = 0; i < sc_attr.size(); i += 2) {
    if (sc_attr[i] == output_index) {
      CHECK_EQ(alloc_attr->scope_id, 0);
      alloc_attr->scope_id = sc_attr[i + 1];
      return true;
    }
  }
  return false;
}

void GraphView::SetScopedAllocatorAttrs(
    const std::vector<const Node*>& sa_nodes) {
  for (const Node* sa : sa_nodes) {
    NodeItem* sa_item = node(sa->id());
    AllocatorAttributes* sa_attrs = sa_item->output_attr_base();
    // Control edges out of the ScopedAllocator should be use instances, but may
    // include a few other nodes.
    for (const auto& e : sa->out_edges()) {
      if (!e->IsControlEdge()) {
        continue;
      }
      Node* use_node = e->dst();
      NodeItem* item = node(use_node->id());
      AllocatorAttributes* use_attrs = item->output_attr_base();
      std::vector<int> scoped_allocator_attrs;
      Status s = GetNodeAttr(use_node->attrs(), "_scoped_allocator",
                             &scoped_allocator_attrs);
      if (!s.ok()) {
        VLOG(2) << "Failed to find expected ScopedAllocator attr on "
                << use_node->name();
        continue;
      }
      // There can be more than one output using ScopedAllocation, but this
      // analysis assumes they use the same ScopedAllocator.
      for (const auto& e : use_node->out_edges()) {
        if (!e->IsControlEdge()) {
          AllocatorAttributes attr;
          if (ExtractScopedAllocatorAttr(scoped_allocator_attrs,
                                         e->src_output(), &attr)) {
            // Set the scope_id on this use instance node.
            (use_attrs + e->src_output())->Merge(attr);
            // Propagate the other attributes of this node back to the SA node.
            attr = *(use_attrs + e->src_output());
            attr.scope_id = 0;
            sa_attrs->Merge(attr);
          }
        }
      }
    }
  }
}

Status GraphView::SetAllocAttrs(const Graph* g, const Device* device) {
  Status s;
  DeviceNameUtils::ParsedName local_dev_name = device->parsed_name();

  std::vector<const Node*> scoped_allocator_instances;
  for (const Node* n : g->nodes()) {
    NodeItem* item = node(n->id());
    AllocatorAttributes* attrs = item->output_attr_base();
    if (IsScopedAllocator(n)) {
      scoped_allocator_instances.push_back(n);
    }

    // Examine the out edges of each node looking for special use
    // cases that may affect memory allocation attributes.
    for (const auto& e : n->out_edges()) {
      if (!e->IsControlEdge()) {
        AllocatorAttributes attr;
        s = InferAllocAttr(n, e->dst(), local_dev_name, &attr);
        if (!s.ok()) return s;
        if (attr.value != 0 || attr.scope_id != 0) {
          attrs[e->src_output()].Merge(attr);
        }
      }
    }

    for (int out = 0; out < n->num_outputs(); out++) {
      const OpKernel* op_kernel = item->kernel;
      DCHECK_LT(out, op_kernel->output_memory_types().size());
      bool on_host = op_kernel->output_memory_types()[out] == HOST_MEMORY;
      if (on_host) {
        AllocatorAttributes h;
        h.set_on_host(on_host);
        attrs[out].Merge(h);
      }
    }
  }
  SetScopedAllocatorAttrs(scoped_allocator_instances);
  return s;
}

Status InferAllocAttr(const Node* n, const Node* dst,
                      const DeviceNameUtils::ParsedName& local_dev_name,
                      AllocatorAttributes* attr) {
  Status s;
  // Note that it's possible for *n to be a Recv and *dst to be a Send,
  // so these two cases are not mutually exclusive.
  if (IsRecv(n)) {
    string src_name;
    s = GetNodeAttr(n->attrs(), "send_device", &src_name);
    if (!s.ok()) return s;
    DeviceNameUtils::ParsedName parsed_src_name;
    if (!DeviceNameUtils::ParseFullName(src_name, &parsed_src_name)) {
      s = errors::Internal("Bad send_device attr '", src_name, "' in node ",
                           n->name());
      return s;
    }
    if (!DeviceNameUtils::IsSameAddressSpace(parsed_src_name, local_dev_name)) {
      // Value is going to be the sink of an RPC.
      attr->set_nic_compatible(true);
      VLOG(2) << "node " << n->name() << " is the sink of an RPC in";
    } else if ((local_dev_name.type == "CPU" || n->IsHostRecv()) &&
               parsed_src_name.type != "CPU") {
      // Value is going to be the sink of a local DMA from GPU to CPU (or
      // other types of accelerators).
      attr->set_gpu_compatible(true);
      VLOG(2) << "node " << n->name() << " is the sink of a gpu->cpu copy";
    } else {
      VLOG(2) << "default alloc case local type " << local_dev_name.type
              << " remote type " << parsed_src_name.type;
    }
  }
  if (IsSend(dst)) {
    string dst_name;
    s = GetNodeAttr(dst->attrs(), "recv_device", &dst_name);
    if (!s.ok()) return s;
    DeviceNameUtils::ParsedName parsed_dst_name;
    if (!DeviceNameUtils::ParseFullName(dst_name, &parsed_dst_name)) {
      s = errors::Internal("Bad recv_device attr '", dst_name, "' in node ",
                           n->name());
      return s;
    }
    if (!DeviceNameUtils::IsSameAddressSpace(parsed_dst_name, local_dev_name)) {
      // Value is going to be the source of an RPC.
      attr->set_nic_compatible(true);
      VLOG(2) << "node " << n->name() << " is the source of an RPC out";
    } else if ((local_dev_name.type == "CPU" || dst->IsHostSend()) &&
               parsed_dst_name.type != "CPU") {
      // Value is going to be the source of a local DMA from CPU to GPU (or
      // other types of accelerators).
      // Note that this does not cover the case where the allocation of the
      // output tensor is not generated by the src: n.
      attr->set_gpu_compatible(true);
      VLOG(2) << "node " << n->name() << " is the source of a cpu->gpu copy";
    } else {
      VLOG(2) << "default alloc case local type " << local_dev_name.type
              << " remote type " << parsed_dst_name.type;
    }
  }
  if (n->IsCollective()) {
    // We'll make the sweeping assumption that any collective op is going
    // to be involved in network i/o.
    attr->set_nic_compatible(true);
  }
  return s;
}


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////


// Very long: 919 - 1451
//
// The state associated with one invocation of ExecutorImpl::Run.
// ExecutorState dispatches nodes when they become ready and keeps
// track of how many predecessors of a node have not done (pending_).
class ExecutorState {
 public:
  ExecutorState(const Executor::Args& args, ExecutorImpl* impl);
  ~ExecutorState();

  void RunAsync(Executor::DoneCallback done);

 private:

  // -----------------------------------------------------------------------
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

      // 如果 有值 ，那么初始化值。
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
    // class ManualConstructor 数据结构
    // tensorflow/core/lib/gtl/manual_constructor.h:120:

    Tensor* ref = nullptr;    // A tensor reference.
    // class Tensor 数据结构
    // tensorflow/core/framework/tensor.h:57:

    mutex* ref_mu = nullptr;  // mutex for *ref if ref is not nullptr.

    // Whether the value exists, either in <val> or <ref>.
    bool has_value = false;

    bool val_field_is_set = false;
    // A tensor value, if val_field_is_set

    // The attributes of the allocator that creates the tensor.
    AllocatorAttributes alloc_attr;
    // struct AllocatorAttributes 数据结构
    // tensorflow/core/framework/allocator.h:359:

    // Every entry carries an optional DeviceContext containing
    // Device-specific information about how the Tensor was produced.
    DeviceContext* device_context = nullptr;
    // class DeviceContext : public core::RefCounted 数据结构
    // 继承类
    // tensorflow/core/framework/device_base.h:68
    // tensorflow/compiler/jit/xla_device_context.h:52:class XlaDeviceContext : public DeviceContext {
    // tensorflow/core/common_runtime/sycl/sycl_device_context.h:28:class SYCLDeviceContext : public DeviceContext {
    // tensorflow/core/common_runtime/gpu_device_context.h:29:class GPUDeviceContext : public DeviceContext {
  };
  // 1.
  // struct Entry 数据结构
  // class ExecutorState::struct Entry
  // tensorflow/core/common_runtime/executor.cc
  //
  // 概述:
  // 可以和 Tensor 等价了
  //
  // - val: ManualConstructor<Tensor>
  //   A tensor value, if val_field_is_set.
  // - ref: Tensor*
  //   A tensor reference.
  // - ref_mu: mutex*
  //   mutex for *ref if ref is not nullptr.
  // - has_value: bool
  //   Whether the value exists, either in <val> or <ref>.
  // - val_field_is_set: bool , default value: false
  //   A tensor value if val_field_is_set
  // - alloc_attr: AllocatorAttributes
  //   The attributes of the allocator that creates the tensor.
  // - device_context: DeviceContext* , default value: nullptr
  //   Every entry carries an optional DeviceContext containing
  //   Device-specific information about how the Tensor was produced.

  // 2.
  // class ManualConstructor 数据结构
  // tensorflow/core/lib/gtl/manual_constructor.h



  // Contains a value for [node->id()] for the device context assigned by the
  // device at the beginning of a step.
  DeviceContextMap device_context_map_;
  // 1.
  // DeviceContextMap 数据结构
  // typedef std::vector<DeviceContext*> DeviceContextMap;
  // tensorflow/core/framework/device_base.h:112
  // 概述：
  // Contains a value for [node->id()] for the device context assigned by the
  // device at the beginning of a step.

  // class DeviceContext : public core::RefCounted 数据结构
  // 继承类
  // tensorflow/core/framework/device_base.h
  // - class GPUDeviceContext : public DeviceContext
  //   tensorflow/core/common_runtime/gpu_device_context.h
  // - class XlaDeviceContext : public DeviceContext
  //   tensorflow/compiler/jit/xla_device_context.h
  // - class SYCLDeviceContext : public DeviceContext
  //   tensorflow/core/common_runtime/sycl/sycl_device_context.h

  struct TaggedNode;

  typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;
  // 1.
  // TaggedNodeSeq 数据结构:
  // tensorflow/core/common_runtime/executor.cc
  // typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;
  // Seq : 序列

  // 2.
  // struct TaggedNode 数据结构
  // tensorflow/core/common_runtime/executor.cc
  // node: const Node*, default : nullptr;
  // input_frame: FrameState*, default : nullptr;
  // input_iter: int64, default : -1;
  // is_dead: bool, default : false;

  typedef gtl::InlinedVector<Entry, 4> EntryVector;
  // 1.
  // EntryVector 数据结构
  // tensorflow/core/common_runtime/executor.h
  // typedef gtl::InlinedVector<Entry, 4> EntryVector;

  // 2.
  // struct Entry 数据结构
  // - val: ManualConstructor<Tensor>
  //   A tensor value, if val_field_is_set.
  // - ref: Tensor*
  //   A tensor reference.
  // - ref_mu: mutex*
  //   mutex for *ref if ref is not nullptr.
  // - has_value: bool
  //   Whether the value exists, either in <val> or <ref>.
  // - val_field_is_set: bool , default value: false
  //   A tensor value if val_field_is_set
  // - alloc_attr: AllocatorAttributes
  //   The attributes of the allocator that creates the tensor.
  // - device_context: DeviceContext* , default value: nullptr
  //   Every entry carries an optional DeviceContext containing
  //   Device-specific information about how the Tensor was produced.

  struct IterationState {
    // 构造函数
    explicit IterationState(
      const PendingCounts* pending_counts,
      int total_input_tensors)
        : input_tensors(new Entry[total_input_tensors]),
          outstanding_ops(0),
          // 什么样的 op 是突出的？
          outstanding_frame_count(0),
          counts_(*pending_counts) {  // Initialize with copy of *pending_counts
    }
    // 1.
    // struct IterationState 数据结构:
    // IterationState 维护 FrameState (也就是一个实例化的 whileloop) 里的一次迭代相关的状态变量。
    // 博客:
    // https://blog.csdn.net/zhenhailiu/article/details/81202064

    // 2.
    // A new frame is created when the executor sees an Enter node.
    // A frame is removed when it sees an Exit node.
    // The next iteration of the frame is progressed to when it sees a NextIteration node.

    // 3. In TensorFlow, every op is executed in an ​execution frame.
    // and the control-flow primitives are responsible for creating and managing these execution frames.
    // Intuitively, for each while loop, the TensorFlow runtime sets up an execution frame and runs all the ops belonging to the while loop inside the execution frame.
    // Nested while loops run in nested execution frames. Ops from different execution frames can run in parallel as long as there is no data dependency between them.


    // The state of an iteration.
    // One copy per iteration.
    // For iteration k, i-th(某个) node's j-th(某个) input is in
    // input_tensors[k][impl_->nodes[i].input_start + j].
    //               |               |                |
    //     iteration k               i-th(某个) node's j-th(某个) input
    //
    // An entry is either
    // a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
    //
    // NOTE: No need to protect input_tensors[i] by any locks because it
    // is resized once. Each element of tensors_ is written once by the
    // source node of an edge and is cleared by the destination of the same
    // edge. The latter node is never run concurrently with the former node.
    Entry* input_tensors;
    // class ExecutorState::struct Entry 数据结构
    // tensorflow/core/common_runtime/executor.cc:531

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
    // class PendingCounts 数据结构
    // tensorflow/core/common_runtime/pending_counts.h:48:class PendingCounts
  };
  // struct IterationState 数据结构 END!
  // -----------------------------------------------------------------------


  // -----------------------------------------------------------------------
  struct FrameState {

    // FrameState 构造函数
    explicit FrameState(
      const ExecutorImpl* impl,
      int parallel_iters)
        : executor(impl),
          max_parallel_iterations(parallel_iters),
          num_outstanding_iterations(1) {}

    // A new frame is created for each loop. Execution starts at iteration 0.
    // 一个新的框架被构建为了每个循环。 执行开始于第一个遍历。
    // When a value at iteration 0 passes through a NextIteration node,
    // 当一个数值在迭代 0 穿过 NextIteration 节点时，
    // iteration 1 is created and starts running. Note that iteration 0 may
    // 迭代 1 被构建并开始执行。 注意，迭代 0 任然可以并行地执行多个迭代。
    // still be running so multiple iterations may run in parallel. The
    // frame maintains the state of iterations in several data structures
    // 框架在几个数据结构中保持迭代的状态
    // such as pending_count and input_tensors. When iteration 0 completes,
    // 比如 pending_count 和 input_tensors。当迭代 0 结束，
    // we garbage collect the state of iteration 0.
    // 我们垃圾回收迭代 0 的状态。
    //
    // A frame instance is considered "done" and can be garbage collected
    // 一个框架实例被认为“完成”了可以被垃圾回收
    // if all its inputs have entered and all its iterations are "done".
    // 如果所有的它的输入都已经进入，并且所有的它的迭代都已经“完成”。
    //
    // A frame manages the live iterations of an iterative computation.
    // 一个框架管理迭代计算的实时的迭代。
    // Iteration i is considered "done" when there are no outstanding ops,
    // 迭代 i 被认为"完成"了是在没有突出算子的时候，
    // frames at iteration i are done, all recvs for this iteration are
    // 在迭代 i 的框架完成，所有的这个迭代的接收都完成。
    // completed, and iteration i-1 is done. For iteration 0, we instead
    // 迭代 i-1 完成。对于迭代 0， 我们相反等待这个框架没有等待的输入。
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
    // 好烦，这个又要改了后面。

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
    // pending == waiting
    int num_pending_inputs = 0;

    // The highest iteration number we have reached so far in this frame.
    int64 iteration_count GUARDED_BY(mu) = 0;

    // The number of outstanding iterations.
    int num_outstanding_iterations GUARDED_BY(mu) = 1;

    // 想象这个类就是 while loop , 每个 iteration 都是 while loop 的 iteration
    // The active iteration states of this frame.
    gtl::InlinedVector<IterationState*, 12> iterations;
    // struct IterationState 数据结构
    // tensorflow/core/common_runtime/executor.cc:946:  struct IterationState

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
      // 1.
      // executor 变量说明：
      // executor: const ExecutorImpl*

      // 2.
      // executor->frame_info_ 变量说明:
      // ExecutorImpl::frame_info_ : gtl::FlatMap<string, FrameInfo*>
      //                                          ------
      //                                          frame_name

      // 3.
      // class ExecutorImpl::FrameInfo 数据结构
      // - input_count: int
      //    The total number of inputs to a frame.
      // - total_inputs:
      //    The total number of input tensors of a frame.
      //    == sum(nodes[*].num_inputs()) where nodes are the nodes in the frame.
      // - pending_counts_layout: PendingCounts::Layout
      //    Used to determine the next place to allocate space in the
      //    pending_counts data structure we'll eventually construct
      // - pending_counts: PendingCounts*
      //    Each frame has its own PendingCounts only for the nodes in the frame.
      // - nodes: std::vector<const Node*>*
      //    The nodes in a frame. Used only for debugging.

      DCHECK(it_frame_info != executor->frame_info_.end());

      ExecutorImpl::FrameInfo* finfo = it_frame_info->second;

      pending_counts = finfo->pending_counts;
      // struct ExecutorState::struct FrameState::pending_counts

      total_input_tensors = finfo->total_inputs;

      num_pending_inputs = finfo->input_count;

      nodes = finfo->nodes;
    }


    ////////////////////////////////////////////////////////////////////////
    // ** GetIteration 主要逻辑 **
    // 想象这个类就是 while loop , 每个 iteration 都是 while loop 的 iteration
    // The active iteration states of this frame.
    // gtl::InlinedVector<IterationState*, 12> iterations;
    inline IterationState* GetIteration(int64 iter)
        EXCLUSIVE_LOCKS_REQUIRED(mu) {

      size_t index = iter % iterations.size();

      // 打印 iterations.size() == 1

      // 打印 iter == 0
      // p *(iterations[0])
      // {
      //   input_tensors = 0x5604c69d4d78,
      //   outstanding_ops = 1,
      //   outstanding_frame_count = 0,
      //   counts_ = {
      //     static kMaxCountForPackedCounts = 7,
      //     num_bytes_ = 12,
      //     bytes_ = 0x5604c69d4f80 "\301\205\001\201\001\001\001\001\202\201\001\201\006\177"
      //   }
      // }
      //

      /*

      p (*(iterations[0])).input_tensors
      $9 = (tensorflow::ExecutorState::Entry *) 0x5604c69d4d78

      p *((*(iterations[0])).input_tensors)
      $10={
            val= {
              space_="x\313\064\353\006\177",
              '\000'< repeats 18 times>,
              "\017\000\b\000\000\000",
              < incomplete sequence\ 343>
            }
            ,
            ref=0x0,
            ref_mu=0x0,
            has_value=false,
            val_field_is_set=false,
            alloc_attr= {
              value=0,
              scope_id=0
            }
            ,
            device_context=0x0
          }
      */
      return iterations[index];
    }
    ////////////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////////////
    inline void SetIteration(int64 iter, IterationState* state)
        EXCLUSIVE_LOCKS_REQUIRED(mu) {

      size_t index = iter % iterations.size();

      DCHECK(state == nullptr || iterations[index] == nullptr);

      iterations[index] = state;

    }
    ////////////////////////////////////////////////////////////////////////


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
        // 这个表示这个 iteration 结束了
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
  }; // struct FrameState end!
  // -----------------------------------------------------------------------
  // struct FrameState 数据结构说明:
  // - executor: const ExecutorImpl*
  //    The executor the frame is in.
  // - frame_name: string
  //    The name of this frame, which is the concatenation of its parent
  //    frame name, the iteration of the parent frame when this frame was
  //    created, and the value of the attr 'frame_name'.
  // - frame_id: uint64
  //    The unique id for this frame. Generated by fingerprinting
  //    frame_name.
  // - parent_iter: int64
  //    The iteration id of its parent frame when this frame is created.
  //    -1 if there is no parent frame. The frame_name/parent_iter pair
  //    uniquely identifies this FrameState.
  // - parent_frame: FrameState*
  //    The FrameState of its parent frame.
  // - max_parallel_iterations: const int
  //    The maximum allowed number of parallel iterations.
  // - num_pending_inputs: int, default : 0
  //    The number of inputs this frame is still waiting.
  //    pending == waiting
  // - iteration_count: int64, default : 0
  //    The highest iteration number we have reached so far in this frame.
  // - num_outstanding_iterations: int , default: 1
  //    The number of outstanding iterations.
  // - iterations: gtl::InlinedVector<IteratorState*, 12>
  //    The active iteration states of this frame.
  // - next_iter_roots: std::vector<std::pair<const Node*, Entry>>
  //    The NextIteration nodes to enter a new iteration. If the number of
  //    outstanding iterations reaches the limit, we will defer the start of
  //    the next iteration until the number of outstanding iterations falls
  //    below the limit.
  // - inv_values: std::vector<std::pair<const Node*, Entry>>
  //    The values of the loop invariants for this loop. They are added into
  //    this list as they "enter" the frame. When a loop invariant enters,
  //    we make it available to all active iterations. When the frame starts
  //    a new iteration, we make all the current loop invariants available
  //    to the new iteration.
  // - dead_exits: std::vector<const Node*>
  //    The list of dead exit nodes for the current highest iteration. We
  //    will only "execute" the dead exits of the final iteration.
  // - pending_counts: PendingCounts*
  //    Static information specific to this frame.
  // - total_input_tensors: int
  // - nodes: std::vector<const Node*>


  // -----------------------------------------------------------------------
  // A tagged node: <frame*, iter, node*>.
  struct TaggedNode {
    const Node* node = nullptr;
    FrameState* input_frame = nullptr;
    int64 input_iter = -1;
    bool is_dead = false;

    TaggedNode(
      const Node* t_node,
      FrameState* in_frame,
      int64 in_iter,
      bool dead)
    {
        node = t_node;
        input_frame = in_frame;
        input_iter = in_iter;
        is_dead = dead;
    }

  };
  // 1.
  // struct TaggedNode 数据结构
  // tensorflow/core/common_runtime/executor.cc
  // 概述:
  // A tagged node: <frame*, iter, node*>.
  //
  // node: const Node*, default : nullptr;
  // input_frame: FrameState*, default : nullptr;
  // input_iter: int64, default : -1;
  // is_dead: bool, default : false;

  // -----------------------------------------------------------------------


  // -----------------------------------------------------------------------
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
    // 1.
    // TaggedNodeReadyQueue::pop_front() 原理图
    // https://docs.google.com/document/d/1iUSROf80zgjOKQHXiOtbJX3D90_pqjbkiGZczkvHfPo/edit

    bool empty() const { return ready_.empty(); }

    const TaggedNode* begin() const { return ready_.begin() + front_index_; }

    const TaggedNode* end() const { return ready_.end(); }

   private:
    gtl::InlinedVector<TaggedNode, 16> ready_;
    int front_index_;
  };
  // 1.
  // class TaggedNodeReadyQueue
  // tensorflow/core/common_runtime/executor.cc
  //
  // 概述:
  // A drop-in replacement for std::deque<TaggedNode>.  We typically don't
  // have that many nodes in the ready queue, so we just use a vector and
  // don't free up memory from the queue as we consume nodes.
  //
  // - ready_: gtl::InlinedVector<TaggedNode, 16>
  // - front_index_: int

  // 2.
  // struct TaggedNode 数据结构
  // tensorflow/core/common_runtime/executor.cc
  // 概述:
  // A tagged node: <frame*, iter, node*>.
  //
  // node: const Node*, default : nullptr;
  // input_frame: FrameState*, default : nullptr;
  // input_iter: int64, default : -1;
  // is_dead: bool, default : false;


  // -----------------------------------------------------------------------

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


  /// 通过调用 StepStatsCollector::Save 存下的 nodestats::tracing_fn 信息
  /// tensorflow/core/common_runtime/step_stats_collector.cc
  /// tensorflow/core/common_runtime/step_stats_collector.h
  StepStatsCollectorInterface* const stats_collector_;


  const tracing::EventCollector* const event_collector_;
  Context context_;
  // Context 类型说明
  // tensorflow/core/platform/default/context.h:21:class Context
  // 没啥用的感觉


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

  // Schedule all the expensive nodes in 'ready', and put all the inexpensive
  // nodes in 'ready' into 'inline_ready'.
  void ScheduleReady(const TaggedNodeSeq& ready,
                     TaggedNodeReadyQueue* inline_ready);

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


  // 1. QQQ. GetInputTensors 的原理?

  // A standalone routine for this expression so that we can express
  // that we don't want thread safety analysis on this reference (it's
  // safe to do without the lock because the iterations array never
  // resizes and this particular iteration's array element will not
  // be changed out from under us because the iteration is still alive).
  Entry* GetInputTensors(
    FrameState* input_frame,
    int64 input_iter) const NO_THREAD_SAFETY_ANALYSIS {
    // 1.
    // struct FrameState 数据结构
    // tensorflow/core/common_runtime/executor.cc:673:
    // 数据结构说明:
    // 首先，Frame 对应 while loop
    // FrameState 表示一个 Frame 的实例化状态 (也就是一个实例化的 whileloop) 。
    // 因为 Frame (while) 可以嵌套，嵌套在一个 whileloop 里的 whileloop 可能会被多次实例化，
    // 一次实例化对应一个 FrameState ，也就是说一个 whileloop ，只对应一个 FrameInfo ，
    // 但是可能会实例化多个 FrameState 。FrameState 在图计算的过程中动态地创建，修改和销毁。
    // FrameState 维护的重要状态是 IterationState 数组。

    // 2.
    // - executor: const ExecutorImpl*
    // - frame_name: string
    // - frame_id: uint64
    // - parent_iter: int64
    // - parent_frame: FrameState*
    // - max_parallel_iterations: const int
    // - num_pending_inputs: int
    // - iteration_count: int64
    // - num_outstanding_iterations: int
    // - iterations: gtl::InlinedVector<IterationState*, 12>
    // - next_iter_roots: std::vector<std::pair<const Node*, Entry>>
    // - inv_values: std::vector<std::pair<const Node*, Entry>>
    // - dead_exits: std::vector<const Node*>
    // - pending_counts: PendingCounts*
    // - total_input_tensors: int
    // - nodes: std::vector<const Node*>*
    // - mu: mutex

    return input_frame->GetIteration(input_iter)->input_tensors;
    // 1.
    // FrameState::GetIteration 的函数说明:
    // tensorflow/core/common_runtime/executor.cc:1108:
    // inline IterationState* GetIteration(int64 iter)
    //
    // 铺垫
    // ExecutorState::struct IterationState
    // IterationState 维护FrameState（也就是一个实例化的whileloop）里的一次迭代相关的状态变量。
    //
    // 给我的感觉噢，就是一个 while loop 里面所有的 iteration 都已经实例化且枚举了存在
    // iterations: gtl::InlinedVector<IterationState*, 12> 里面
    // 每次要用了就去取出相应的
  }

}; // class ExecutorState End.
// 1.
// class ExecutorState
//   * struct Entry
//   - device_context_map_: DeviceContextMap
//   * struct TaggedNode
//   * TaggedNodeSeq: typedef gtl::InlinedVector<TaggedNode, 8>
//   * EntryVector: typedef gtl::InlinedVector<Entry, 4>
//   * struct IterationState
//   * struct FrameState
//   * struct TaggedNode
//   * class TaggedNodeReadyQueue
//   * struct AsyncState
//   - vlog_: const bool
//   - log_memory_: const bool
//   - step_id: int64
//   - rendezvous_: Rendezvous*
//   - collective_executor_: CollectiveExecutor*, default: nullptr
//   - session_state_: SessionState*
//   - session_handle_: string
//   - tensor_store_: TensorStore*
//   - step_container_: ScopedStepContainer*
//   - stats_collector_: StepStatsCollectorInterface* const
//   - event_collector_: const tracing::EventCollector* const
//   - context_: Context
//   - slice_reader_cache_: checkpoint::TensorSliceReaderCacheWrapper*
//   - call_frame_: CallFrameInterface*
//   - impl_: onst ExecutorImpl*
//   - cancellation_manager_: CancellationManager*
//   - runner_: Executor::Args::Runner
//   - sync_on_finish_: bool
//   - trace_using_annotations_: const bool
//   - dumped_on_error_: bool, default: false
//   - root_frame_: FrameState*
//   - done_cb_: Executor::DoneCallback
//   - num_outstanding_ops_: std::atomic_int_fast32_t
//   - no_deferred_ops_cv_: condition_variable
//   - num_deferred_ops_: int64
//   - outstanding_frames_: gtl::FlatMap<string, FrameState*>


// 1.
// ExecutorImpl::RunAsync 里面构造了 ExecutorState
ExecutorState::ExecutorState(
  const Executor::Args& args,
  // 1.
  // args 变量说明:
  // Executor::Args args;

  // 2.
  // Executor::Args 数据结构
  //  struct Args {
  //    int64 step_id = 0;
  //    Rendezvous* rendezvous = nullptr;
  //    StepStatsCollectorInterface* stats_collector = nullptr;
  //    CallFrameInterface* call_frame = nullptr;
  //
  //    =======================================================================
  //    CancellationManager* cancellation_manager = nullptr;
  //    =======================================================================
  //
  //    SessionState* session_state = nullptr;
  //
  //    // Unique session identifier. Can be empty.
  //    string session_handle;
  //    TensorStore* tensor_store = nullptr;
  //    ScopedStepContainer* step_container = nullptr;
  //    CollectiveExecutor* collective_executor = nullptr;
  //
  //    // If true, calls Sync() on the device.
  //    bool sync_on_finish = false;
  //
  //    typedef std::function<void()> Closure;
  //    typedef std::function<void(Closure)> Runner;
  //    Runner runner = nullptr;
  //  };

  // 2.
  // class CancellationManager [final] 数据结构
  // tensorflow/core/framework/cancellation.h
  // 成员变量
  // - is_cancelling_: bool
  // - is_cancelled_: std::atomic_bool
  // - mu_: mutex
  // - cancelled_notification_: Notification
  // - next_cancellation_token_: CancellationToken
  // - callbacks_: gtl::FlatMap<CancellationToken, CancelCallback>
  //
  // 部分的接口函数
  // - StartCancel()
  //   Run all callbacks associated with this manager.
  // - IsCancelled()
  //   Returns true iff StartCancel() has been called.
  // - Reset()
  //   Resets the cancellation manager to its original pre-cancelled state.
  // - get_cancellation_token()
  //   Returns a token that must be used in calls to RegisterCallback
  //   and DeregisterCallback.
  // - RegisterCallback
  //   Attempts to register the given callback to be invoked when this
  //   manager is cancelled.
  // - ...

  // 2.1
  // CancellationManager 构造函数说明:
  // CancellationManager::CancellationManager()
  // tensorflow/core/framework/cancellation.cc
  // : is_cancelling_(false),
  //   is_cancelled_(false),
  //   next_cancellation_token_(0) {}
  //
  // - is_cancelling_ : false
  // - is_cancelled_: false
  // - next_cancellation_token_: 0

  ExecutorImpl* impl)

    : vlog_(VLOG_IS_ON(1)),
      log_memory_(LogMemory::IsEnabled()),
      step_id_(args.step_id),
      rendezvous_(args.rendezvous),
      collective_executor_(args.collective_executor),
      session_state_(args.session_state),
      session_handle_(args.session_handle),
      tensor_store_(args.tensor_store),
      step_container_(args.step_container),
      ////////////////////////////////////////////////////////////////////////
      // 这个决定了是否打开 traing 和收集数据
      ////////////////////////////////////////////////////////////////////////
      stats_collector_(args.stats_collector),
      ////////////////////////////////////////////////////////////////////////
      event_collector_(
          tracing::GetEventCollector(tracing::EventCategory::kCompute)),
      context_(ContextKind::kThread),
      slice_reader_cache_(new checkpoint::TensorSliceReaderCacheWrapper),
      call_frame_(args.call_frame),
      // -------------------------------------------------------------
      impl_(impl),
      // -------------------------------------------------------------

      // -------------------------------------------------------------
      cancellation_manager_(args.cancellation_manager),
      // 1.
      // cancellation_manager_ 概述:
      // 提前 abort 这个 step
      // -------------------------------------------------------------

      runner_(args.runner),
      sync_on_finish_(args.sync_on_finish),

      // gpu_device.h:73:  bool TraceUsingAnnotations() const { return true; }
      // GPU 的 kernel 是一定 只要是 tracing 就是 Annotations 这个分支。
      trace_using_annotations_(impl->params_.device->TraceUsingAnnotations()),
      num_outstanding_ops_(0) {

  // We start the entire execution in iteration 0 of the root frame
  // so let us create the root frame and the state for iteration 0.
  // We assume root_frame_->frame_name.empty().
  root_frame_ = new FrameState(impl_, 1);
  // struct FrameState 数据结构说明:
  // - executor: const ExecutorImpl*
  //    The executor the frame is in.
  // - frame_name: string
  //    The name of this frame, which is the concatenation of its parent
  //    frame name, the iteration of the parent frame when this frame was
  //    created, and the value of the attr 'frame_name'.
  // - frame_id: uint64
  //    The unique id for this frame. Generated by fingerprinting
  //    frame_name.
  // - parent_iter: int64
  //    The iteration id of its parent frame when this frame is created.
  //    -1 if there is no parent frame. The frame_name/parent_iter pair
  //    uniquely identifies this FrameState.
  // - parent_frame: FrameState*
  //    The FrameState of its parent frame.
  // - max_parallel_iterations: const int
  //    The maximum allowed number of parallel iterations.
  // - num_pending_inputs: int, default : 0
  //    The number of inputs this frame is still waiting.
  //    pending == waiting
  // - iteration_count: int64, default : 0
  //    The highest iteration number we have reached so far in this frame.
  // - num_outstanding_iterations: int , default: 1
  //    The number of outstanding iterations.
  // - iterations: gtl::InlinedVector<IteratorState*, 12>
  //    The active iteration states of this frame.
  // - next_iter_roots: std::vector<std::pair<const Node*, Entry>>
  //    The NextIteration nodes to enter a new iteration. If the number of
  //    outstanding iterations reaches the limit, we will defer the start of
  //    the next iteration until the number of outstanding iterations falls
  //    below the limit.
  // - inv_values: std::vector<std::pair<const Node*, Entry>>
  //    The values of the loop invariants for this loop. They are added into
  //    this list as they "enter" the frame. When a loop invariant enters,
  //    we make it available to all active iterations. When the frame starts
  //    a new iteration, we make all the current loop invariants available
  //    to the new iteration.
  // - dead_exits: std::vector<const Node*>
  //    The list of dead exit nodes for the current highest iteration. We
  //    will only "execute" the dead exits of the final iteration.
  // - pending_counts: PendingCounts*
  //    Static information specific to this frame.
  // - total_input_tensors: int
  // - nodes: std::vector<const Node*>

  // 2.
  // explicit FrameState(
  //   const ExecutorImpl* impl,
  //   int parallel_iters)
  //     : executor(impl),
  //       max_parallel_iterations(parallel_iters),
  //       num_outstanding_iterations(1) {}

  root_frame_->frame_id = 0;  // must be 0
  root_frame_->InitializeFrameInfo(root_frame_->frame_name);
  // 1.
  // root_frame_->frame_name 应该是 ""

  // 2.
  // InitializeFrameInfo 函数说明
  // tensorflow/core/common_runtime/executor.cc
  // 概述:
  // 用 executor->frame_info_ 的信息初始化了 struct ExecutorState::struct FrameState::
  // - pending_counts
  // - total_input_tensors
  // - num_pending_inputs
  // - nodes

  // 3.
  // EnsureFrameInfo 函数
  // EnsureFrameInfo 函数构造了 ExecutorImpl::frame_info_[frame_name] 内的 一个 FrameInfo instance

  // Initialize iteration 0.
  root_frame_->iterations.resize(root_frame_->max_parallel_iterations);
  // root_frame_->max_parallel_iterations 值为 1
  // 所以，root_frame_->iterations size 为 1

  root_frame_->iterations[0] = new IterationState(
      root_frame_->pending_counts,
      root_frame_->total_input_tensors);

  outstanding_frames_.insert(
                        {
                          root_frame_->frame_name,
                          root_frame_
                        }
                      );
}

ExecutorState::~ExecutorState() {
  for (auto name_frame : outstanding_frames_) {
    delete name_frame.second;
  }
  for (auto it : device_context_map_) {
    // 1.
    // device_context_map_ 变量说明:
    //

    it->Unref();
  }
  delete slice_reader_cache_;
}

/** \brief 这个函数好像没有起什么作用。 frame_name 都是空的。
 *         把图 g 全部遍历一遍。
 *  \param [in] g: const Graph*
 *  \param [out] cf_info: ControlFlowInfo*
 */

// 功能: 初始化 这个 node 的 parent node (称为 parent_frame)
//                        frame_names

// 由于 frame_name 是 "" , 所以下面两项都是 "".
// cf_info->frame_names[out_id] = frame_name;
// cf_info->unique_frame_names.insert(frame_name);
//
Status ExecutorImpl::BuildControlFlowInfo(const Graph* g, // input
                                          ControlFlowInfo* cf_info) // output
{
  const int num_nodes = g->num_node_ids();
  cf_info->frame_names.resize(num_nodes);
  std::vector<Node*> parent_nodes;
  parent_nodes.resize(num_nodes);
  std::vector<bool> visited;
  visited.resize(num_nodes);

  // frame_name = ""
  string frame_name;

  std::deque<Node*> ready;

  // Initialize with the root nodes.
  for (Node* n : g->nodes()) {
    if (n->in_edges().empty()) {
    // 只有 SOURCE node 是没有 in edges 的，其他的都没有进入
      visited[n->id()] = true;
      cf_info->unique_frame_names.insert(frame_name); //frame_name=""
      ready.push_back(n);
    }
  }

  // 起始时，ready 里面只有一个 SOURCE Node.
  while (!ready.empty()) {

    Node* curr_node = ready.front();

    int curr_id = curr_node->id();

    ready.pop_front();

    Node* parent = nullptr;

    // Is Enter?
    // Is Exit ?
    //
    if (IsEnter(curr_node)) {

      // Enter a child frame.
      TF_RETURN_IF_ERROR(
          GetNodeAttr(curr_node->attrs(), "frame_name", &frame_name));
      parent = curr_node;

    } else if (IsExit(curr_node)) {

      // Exit to the parent frame.
      /// 如果这个节点是 Exit 类型，那么这个节点的 parent 节点用 parent 指针指着临时用.
      parent = parent_nodes[curr_id];
      frame_name = cf_info->frame_names[parent->id()];
      parent = parent_nodes[parent->id()];

    } else {

      // NOT Enter or Exit Node
      parent = parent_nodes[curr_id];
      frame_name = cf_info->frame_names[curr_id];

    }

    /// 构建前驱后继的关系，让后继节点能够找到前驱节点的 parent
    /// 一个后继节点来自一个前驱节点，固定且唯一。
    for (const Edge* out_edge : curr_node->out_edges()) {
      // Node (src) x ---- x (dst)
      //                      out Node
      Node* out = out_edge->dst();
      const int out_id = out->id();

      // Add to ready queue if not visited.
      bool is_visited = visited[out_id];
      if (!is_visited) {
        ready.push_back(out);
        visited[out_id] = true;

        // Process the node 'out'.
        cf_info->frame_names[out_id] = frame_name;
        parent_nodes[out_id] = parent;
        cf_info->unique_frame_names.insert(frame_name);
      }
    }
  }

  return Status::OK();
}

void ExecutorImpl::InitializePending(const Graph* graph,
                                     const ControlFlowInfo& cf_info) {
  for (auto& it : cf_info.unique_frame_names) {
    FrameInfo* finfo = EnsureFrameInfo(it);
    PendingCounts* counts = new PendingCounts(finfo->pending_counts_layout);
    DCHECK_EQ(finfo->pending_counts, nullptr);
    finfo->pending_counts = counts;
  }
  for (const Node* n : graph->nodes()) {
    const int id = n->id();
    const string& name = cf_info.frame_names[id];
    size_t max_pending, max_dead;
    GetMaxPendingCounts(n, &max_pending, &max_dead);
    const NodeItem* item = gview_.node(id);
    PendingCounts* counts = EnsureFrameInfo(name)->pending_counts;
    counts->set_initial_count(item->pending_id, max_pending);
  }
}

/** \brief Run the graph computation. The real computation starts.
 *
 *  \param[in] done: Executor::DoneCallback;
 */
void ExecutorState::RunAsync(Executor::DoneCallback done) {
  // 1.
  // done 变量说明
  // 来自 `item.executor->RunAsync(args, barrier->Get());`

  // 2.
  // DoneCallback done 说明
  // tensorflow/core/common_runtime/executor.h
  //
  // typedef std::function<void(const Status&)> DoneCallback;
  //
  // 由 `item.executor->RunAsync(args, barrier->Get());` 的 barrier->Get() 传入

  // 3.1
  // barrier->Get() 函数说明:
  // 概述:
  // Returns a closure that Executors must call when they are done
  // computing, passing the status of their execution as an argument.
  //
  // 定义:
  // executor.h
  // StatusCallback Get() {
  //   return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1);
  // }

  // 3.2
  // StatusCallback 数据结构
  // tensorflow/core/common_runtime/executor.h:150:
  // typedef std::function<void(const Status&)> StatusCallback;

  // 3.3
  // ExecutorBarrier::WhenDone 函数说明:
  // tensorflow/core/common_runtime/executor.h
  // 概述:
  // 更新 Status, 然后如果 !s.ok() 就 StartAbort 否则 就 执行 callback done 函数

  // 4.
  // barrier 变量说明:
  // ExecutorBarrier* barrier

  // 5.
  // class ExecutorBarrier 数据结构
  // tensorflow/core/common_runtime/executor.h
  //
  // 概述:
  // A class to help run multiple executors in parallel and wait until
  // all of them are complete.
  //
  // ExecutorBarrier deletes itself after the function returned by Get()
  // is called.
  //
  // - rendez_: Rendezvous* , default value : nullptr
  // - done_cb_: StatusCallback, default value : nullptr
  // - mu_: mutable mutex
  // - pending_: int
  // - status_group_: StatusGroup
  //
  // 重要接口:
  // - WhenDone(const Status& s)
  // - StatusCallback Get()

  const Graph* graph = impl_->graph_.get();
  TaggedNodeSeq ready;

  // Ask the device to fill in the device context map.
  Device* device = impl_->params_.device;

  // -----------------------------------------------------------------------
  // GPU 有具体的 FillContextMap 实现, CPU 没有。
  //
  // GPU 的实现是:
  // 分配 graph op node 到对应的 物理的 GPU 软件 stream 内
  // 调用栈: https://docs.google.com/document/d/1Paqq-fJbqYIWzt5S2ZRSLGUb18Hb5T0LerpV4ZCcgV0/edit#
  // -----------------------------------------------------------------------
  const Status fill_status =
      device->FillContextMap(graph, &device_context_map_);
  // -----------------------------------------------------------------------

  if (!fill_status.ok()) {
    delete this;
    done(fill_status);
    return;
  }

  // Initialize the ready queue.
  for (const Node* n : impl_->root_nodes_) {
    DCHECK_EQ(n->in_edges().size(), 0);
    // -------------------------------------------------------------
    ready.push_back(TaggedNode{n, root_frame_, 0, false});
    // -------------------------------------------------------------
    // 1.
    // struct TaggedNode 数据结构 | 重要
    // tensorflow/core/common_runtime/executor.cc
    // const Node* node = nullptr;
    // FrameState* input_frame = nullptr;
    // int64 input_iter = -1;
    // bool is_dead = false;
    // 构造:
    // TaggedNode(const Node* t_node, FrameState* in_frame, int64 in_iter, is_dead)

    // 2.
    // input_frame 和 input_iter 很重要，每次都要靠它找到 input tensor
  }
  if (ready.empty()) {
    delete this;
    done(Status::OK());
  } else {
    num_outstanding_ops_ = ready.size();
    root_frame_->iterations[0]->outstanding_ops = ready.size();
    done_cb_ = std::move(done);
    // Schedule to run all the ready ops in thread pool.
    ScheduleReady(ready, nullptr);
  }
}

// State kept alive for executing an asynchronous node in another
// thread.  NOTE: We need to make a copy of p.input,
// p.input_device_contexts, and p.input_alloc_attrs for asynchronous
// kernels because OpKernelContext methods like input_type(i) needs
// the param points to valid input type vector. It's not an issue for
// sync kernels because these vectors are kept on the stack.
struct ExecutorState::AsyncState {

  AsyncState(
    const OpKernelContext::Params& p,
    const TaggedNode& _tagged_node,
    const NodeItem* _item,
    Entry* _first_input,
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
        stats(_stats)
  {

    params.inputs = &saved_inputs;
    // params.inputs 变量说明
    // tensorflow/core/framework/op_kernel.h
    // Inputs to this op kernel.
    // struct Params
    // const gtl::InlinedVector<TensorValue, 4>* inputs = nullptr;

    params.input_device_contexts = &saved_input_device_contexts;
    // params.input_device_contexts 变量说明
    // Device contexts.
    // const gtl::InlinedVector<DeviceContext*, 4>* input_device_contexts =
    //    nullptr;

    params.input_alloc_attrs = &saved_input_alloc_attrs;
    // params.input_alloc_attrs 变量说明
    // const gtl::InlinedVector<AllocatorAttributes, 4>* input_alloc_attrs =
    //     nullptr;
  }

  TensorValueVec saved_inputs;
  DeviceContextVec saved_input_device_contexts;
  AllocatorAttributeVec saved_input_alloc_attrs;
  OpKernelContext::Params params;
  TaggedNode tagged_node;
  const NodeItem* item;
  Entry* first_input;
  // =======================================================================
  OpKernelContext ctx;
  // =======================================================================
  // 最最重要
  // - 包含了 这个 op 的 output_

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
// 1.
// struct ExecutorState::AsyncState 数据结构
// executor.cc
// - saved_inputs: TensorValueVec
// - saved_input_device_contexts: DeviceContextVec
// - saved_input_alloc_attrs: AllocatorAttributeVec
// - params: OpKernelContext::Params
// - tagged_node: TaggedNode
// - item: const NodeItem*
// - first_input: Entry*
// - ctx: OpKernelContext
// - stats: NodeExecStatsInterface*

// Returns true if `item` might be traced by the given trace and event
// collectors. Returns false only if `item` definitely will not be traced.
//
// QQQ. using_annotations 是怎么确定的？
// AAA.
// gpu_device.h:73:bool TraceUsingAnnotations() const { return true; }
// GPU 的 kernel 是一定 只要是 tracing 就是 Annotations 这个分支。
//
bool MightTrace(const NodeItem& item,
                const tracing::EventCollector* event_collector,
                bool using_annotations) {
  // Tracing will only be enabled if either `event_collector` is non null,
  // or `trace_collector` is non-null and enabled for this particular kernel.
  // Although `tracing::ScopedActivity`,
  // `tracing::ScopedAnnotation`, and `tracing::ScopedRegion` check subsets of
  // these properties internally in their constructors, the cost of passing the
  // necessary arguments to them can be significant, so we avoid constructing
  // them in the common case (when we know they will not be used).
  if (event_collector != nullptr) {
    return true;
  }

  /// TraceCollectorImpl 是 TraceCollector 的继承类，TraceCollector 是虚类。
  /// 通常的 tracing 是因为 trace_collector 被构造了，所以会 tracing.
  auto* trace_collector = tracing::GetTraceCollector();

  if (trace_collector) {
    /// GPU Kernel 一定是 using_annotations == true
    if (using_annotations) {
      return trace_collector->IsEnabledForAnnotations();
    } else {
      return trace_collector->IsEnabledForActivities(
          item.kernel->IsExpensive());
    }
  }
  return false;
}



void ExecutorState::Process(
  TaggedNode tagged_node,  // input
  int64 scheduled_nsec) { // input
  // 1.
  // struct TaggedNode 数据结构 | 重要
  // tensorflow/core/common_runtime/executor.cc
  // const Node* node = nullptr;
  // FrameState* input_frame = nullptr;
  // int64 input_iter = -1;
  // bool is_dead = false;

  // 2.
  // Initialize the ready queue.

  WithContext wc(context_);
  // 1.
  // class WithContext 数据结构
  // tensorflow/core/platform/default/context.h

  // 2.
  // context_ 变量说明
  // ExecutorState::context_ : class Context

  // 2.1
  // class Context 数据结构
  // tensorflow/core/platform/default/context.h
  // 1.1 概述
  // Context is a container for request-specific information that should be passed
  // to threads that perform related work. The default constructor should capture
  // all relevant context.

  // 2.1.2
  // enum class ContextKind 数据结构
  // core/platform/context.h
  // - kDefault
  //   Initial state with default (empty) values.
  // - kThread
  //   Initial state inherited from the creating or scheduling thread.

  const GraphView& gview = impl_->gview_;
  // 1.
  // class GraphView 数据结构
  // tensorflow/core/common_runtime/executor.h
  // 概述:
  // Immutable view of a Graph organized for efficient execution.
  //
  // - num_nodes_: int32 , default value : 0
  // - node_offsets_: uint32* , default value : nullptr
  //   array of size "graph_.num_node_ids()"
  // - space_: char*
  //   NodeItem objects are allocated here

  TaggedNodeSeq ready;
  // 1.
  // 作用概述:
  // 这个是用来临时地收集容纳 这个正在处理的节点的后继节点 的。
  // 这个会在 ScheduleReady 内, 逐个判定是否是 expensive or not，
  // 然后分发给 inline_ready 内, 即本线程继续执行某些个 nodes 或者
  // 作为一个 task 继续开始调度起另一个休眠的线程继续执行。

  // 2.
  // TaggedNodeSeq 数据结构:
  // tensorflow/core/common_runtime/executor.cc
  // typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;
  // Seq : 序列

  // 3.
  // struct TaggedNode 数据结构
  // tensorflow/core/common_runtime/executor.cc
  // 概述:
  // A tagged node: <frame*, iter, node*>.
  //
  // node: const Node*, default : nullptr;
  // input_frame: FrameState*, default : nullptr;
  // input_iter: int64, default : -1;
  // is_dead: bool, default : false;



  TaggedNodeReadyQueue inline_ready;
  // 1.
  // inline_ready 的意思是继续留在
  // 这个线程内执行某些节点
  // 这些节点来自后继节点集合: ready queue 的分发：
  // 1) (如上)
  // 2) 开启新线程执行一个 expensive 节点

  // 2.
  // class TaggedNodeReadyQueue
  // tensorflow/core/common_runtime/executor.cc
  //
  // 概述:
  // A drop-in replacement for std::deque<TaggedNode>.  We typically don't
  // have that many nodes in the ready queue, so we just use a vector and
  // don't free up memory from the queue as we consume nodes.
  //
  // - ready_: gtl::InlinedVector<TaggedNode, 16>
  // - front_index_: int

  // 3.
  // struct TaggedNode 数据结构
  // tensorflow/core/common_runtime/executor.cc
  // 概述:
  // A tagged node: <frame*, iter, node*>.
  //
  // node: const Node*, default : nullptr;
  // input_frame: FrameState*, default : nullptr;
  // input_iter: int64, default : -1;
  // is_dead: bool, default : false;


  // Parameters passed to OpKernel::Compute.
  TensorValueVec inputs;
  // 1.
  // TensorValueVec (typedef) 数据结构
  // typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
  // tensorflow/core/common_runtime/executor.cc

  // 2.
  // struct TensorValue 数据结构
  // tensorflow/core/framework/op_kernel.h
  // 概述:
  // Holds a tensor or tensor reference. For tensor references, we need
  // a mutex to prevent concurrent access to the tensor.
  //
  // - tensor: Tensor*
  // - mutex_if_ref: mutex*


  DeviceContextVec input_device_contexts;
  // 1.
  // DeviceContextVec (typedef) 数据结构
  // typedef gtl::InlinedVector<DeviceContext*, 4> DeviceContextVec;
  // tensorflow/core/common_runtime/executor.cc

  // 2.
  // class DeviceContext 数据结构
  // tensorflow/core/framework/device_base.h
  //
  // class DeviceContext: public core::RefCounted
  // tensorflow/core/framework/device_base.h
  // 概述:
  // A class that devices can subclass to pass around
  // Device-specific context to OpKernels.
  //
  // 没有成员变量
  //
  // 接口:
  // - stream()
  // - MaintainLifetimeOnStream
  // - CopyCPUTensorToDevice
  // - CopyTensorInSameDevice
  // - CopyDeviceTensorToCPU
  // - ThenExecute

  AllocatorAttributeVec input_alloc_attrs;
  // 1.
  // AllocatorAttributeVec 数据结构
  // typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

  // 2.
  // struct AllocatorAttributes 数据结构
  // tensorflow/core/framework/allocator.h
  // - uint32 value = 0;
  //   value 是比特的方
  //
  //  00000000
  //  ^^^^^^^^
  //  ||||||||
  //  |||||||+----+ on host ?
  //  ||||||+-----+ nic compatible ?
  //  |||||+------+ gpu compatible ?
  //  ||||+-------+
  //  |||+--------+
  //  ||+---------+
  //  |+----------+
  //  +-----------+

  // 2.1
  // 介绍:
  // A tensorflow Op may need access to different kinds of memory that
  // are not simply a function of the device to which the Op has been
  // assigned.  For example, an Op executing on a GPU may still need
  // to allocate CPU RAM for some purpose.  Internal to the tensorflow
  // runtime we may choose to allocate CPU ram from special regions
  // that have been prepared for higher performance in some use
  // contexts, e.g. doing DMA with particular devices.  For these
  // reasons, the Device interface does not expose just one memory
  // Allocator, but instead provides an accessor that takes a
  // specification of the desired memory attributes in order to select
  // an Allocator.
  //
  // Example use:
  //  // Allocator for ordinary device memory:
  //  Allocator* a = allocator(AllocatorAttributes());
  // ...
  //  // Allocator for CPU RAM, regardless of where Op is executing:
  //  AllocatorAttributes attr;
  //  attr.set_on_host(true);
  //  Allocator* a = allocator(attr);


  OpKernelContext::Params params;
  // 1.
  // class OpKernelContext
  // * typedef std::pair<Allocator*, TrackingAllocator*> WrappedAllocator;
  // =======================================================================
  // - struct Params
  //    * step_id: int64
  //    * op_kernel: OpKernel*
  //    * device: DeviceBase*
  //    * eigen_gpu_device: PerOpGpuDevice*
  //    * track_allocations: bool
  //    * log_memory: bool
  //    * record_tensor_accesses: bool
  //    * output_attr_array: const AllocatorAttributes*
  //    * resource_manager: ResourceMgr*  # 这个和 tensor 是怎么联系起来的？
  //    * step_container: ScopedStepContainer*
  //      Per-step resources accessible by this op kernel invocation should be stored in this container.
  //    * rendezvous: Rendezvous*
  //      Mechanism used by this op kernel invocation to communicate with computations running on other devices.
  //    * collective_executor: CollectiveExecutor*
  //      Mechanism for executing a collective op that needs to coordinate with parallel instances running on other devices.
  //    * session_state: SessionState*
  //      The session state for this op.
  //    * session_handle: string
  //      Unique session identifier. Can be empty.
  //    * tensor_store: TensorStore*  # 留意这个
  //      The tensor store for this op.
  //    * cancellation_manager: CancellationManager*
  //    * inputs: const gtl::InlinedVector<TensorValue, 4>*  # 关注一下
  //    * is_input_dead: bool
  //    * input_alloc_attrs: const gtl::InlinedVector<AllocatorAttributes, 4>*
  //    * input_device_contexts: const gtl::InlinedVector<DeviceContext*, 4>*
  //    * op_device_context: DeviceContext*
  //    * frame_iter: FrameAndIter
  //      Control-flow op supports.
  //    * call_frame: CallFrameInterface*
  //    * function_library: FunctionLibraryRuntime*
  //    * runner: std::function<void(std::function<void()>)>*
  //    * stats_collector: StepStatsCollectorInterface*
  //    * graph_collector: GraphCollector*
  //    * slice_reader_cache: checkpoint::TensorSliceReaderCacheWrapper*
  //    * forward_from_array: const int*
  //    * inc_num_deferred_ops_function: std::function<void()>
  //    * dec_num_deferred_ops_function: std::function<void()>
  // =======================================================================
  // - params_: Params*
  // - status_: Status
  // - wrapped_allocators_: gtl::InlinedVector<WrappedAllocator, 4>
  // - outputs_: gtl::InlinedVector<TensorValue, 4>
  // - referenced_tensors_: ManualConstructor<UniqueTensorReferences>
  // - temp_memory_allocated_: int64
  // - persistent_memory_allocated_: int64
  // - temp_tensor_buffer_and_size_: std::unique_ptr<gtl::InlinedVector<std::pair<const void*, int64>, 2>>
  // - persistent_alloc_ids_: std::unique_ptr<gtl::InlinedVector<int64, 2>>

  params.step_id = step_id_;

  // ------------------------------------------------------------------------
  Device* device = impl_->params_.device;
  params.device = device;
  // ------------------------------------------------------------------------

  params.log_memory = log_memory_;
  params.record_tensor_accesses = impl_->device_record_tensor_accesses_;
  params.rendezvous = rendezvous_;
  params.collective_executor = collective_executor_;
  params.session_state = session_state_;
  params.session_handle = session_handle_;
  params.tensor_store = tensor_store_;

  // -----------------------------------------------------------------------
  params.cancellation_manager = cancellation_manager_;
  // -----------------------------------------------------------------------

  params.call_frame = call_frame_;
  params.function_library = impl_->params_.function_library;
  params.resource_manager = device->resource_manager();
  params.step_container = step_container_;
  params.slice_reader_cache = slice_reader_cache_;

  params.inputs = &inputs;
  // 1.
  // params.inputs 变量说明
  // params.inputs: const gtl::InlinedVector<TensorValue, 4>*
  // 输入的 input tensor

  // ------------------------------------------------------------------------
  params.input_device_contexts = &input_device_contexts;
  // 1.
  // input_device_contexts 变量说明
  // const gtl::InlinedVector<DeviceContext*, 4>* input_device_contexts
  // framework/op_kernel.h

  // 2.
  // class DeviceContext 数据结构
  // framework/device_base.h
  // 虚函数被继承
  // ------------------------------------------------------------------------

  params.input_alloc_attrs = &input_alloc_attrs;
  // 1.
  // params.input_alloc_attrs 变量说明:
  // input_alloc_attrs: const gtl::InlinedVector<AllocatorAttributes, 4>*

  // 2.
  // AllocatorAttributes 数据结构
  // tensorflow/core/framework/allocator.h
  //
  //  00000000
  //  ^^^^^^^^
  //  ||||||||
  //  |||||||+----+on host
  //  ||||||+-----+nic compatible
  //  |||||+------+gpu compatible
  //  ||||+-------+
  //  |||+--------+
  //  ||+---------+
  //  |+----------+
  //  +-----------+

  params.runner = &runner_;

  /// 这个决定了是否打开 traing 和收集数据
  params.stats_collector = stats_collector_;

  params.inc_num_deferred_ops_function = [this]() {
    mutex_lock lock(num_deferred_ops_mu_);
    num_deferred_ops_++;
  };

  params.dec_num_deferred_ops_function = [this]() {
    mutex_lock lock(num_deferred_ops_mu_);
    num_deferred_ops_--;
    if (num_deferred_ops_ == 0) {
      no_deferred_ops_cv_.notify_all();
    }
  };

  Status s;

  NodeExecStatsInterface* stats = nullptr;
  // 1.
  // 说明:
  // stats = nullptr 说明
  // 算是初始化指针而已吧

  // 2.
  // class NodeExecStatsInterface 数据结构
  // tensorflow/core/common_runtime/step_stats_collector.h
  //
  // 概述
  // Statistics collection interface for individual node execution.
  //
  // See `NodeExecStatsWrapper` for a concrete implementation of this interface
  // that interfaces with the `Session` layer.
  //
  // 纯虚函数
  //
  // 1.1
  // 继承类
  // - class NodeExecStatsWrapper [final]
  //   tensorflow/core/common_runtime/step_stats_collector.h
  // - class SimpleNodeExecStats [final]
  //   tensorflow/core/kernels/data/captured_function.cc
  //
  // 1.2
  // 接口函数:
  // - void Done(const string& device)
  //   Called when the statistics collection for the node has finished. Once this
  //   method is called, the caller should not make assumptions about the validity
  //   of this object.
  // - void RecordExecutorStarted()
  //   Called immediately after this node starts being processed by the executor.
  // - void RecordComputeStarted()
  //   Called immediately before this node's `Compute()` or `ComputeAsync()` method is called.
  // - void RecordComputeEnded()
  //   Called immediately after this node's `Compute()` method returned (or, for asynchronous operations, the callback passed to its `ComputeAsync()` method was called).
  // - void RecordExecutorEnded()
  //   Called immediately after this executor finishes processing this node.
  // - bool TrackAllocations() const
  //   Returns `true` if this object should track memory allocations.
  // - void SetMemory(OpKernelContext* ctx)
  //   Records information about the memory allocated during the execution of this node.
  //   Takes ownership of any `TrackingAllocator` objects stored in `ctx`.
  // - void SetOutput(int slot, const Tensor* tensor)
  //   Records information about the tensor produced by this node at the given output slot.
  // - void SetReferencedTensors(const TensorReferenceVector& tensors)
  //   Records information about the tensors that were accessed during the execution of this node.
  // - void SetScheduled(int64 nanos)
  //   Records the absolute time in nanoseconds at which this node became runnable (i.e. was scheduled for execution).


  EntryVector outputs;
  // 1.
  // EntryVector 数据结构
  // tensorflow/core/common_runtime/executor.h
  // typedef gtl::InlinedVector<Entry, 4> EntryVector;

  // 2.
  // struct Entry 数据结构
  // - val: ManualConstructor<Tensor>
  //   A tensor value, if val_field_is_set.
  // - ref: Tensor*
  //   A tensor reference.
  // - ref_mu: mutex*
  //   mutex for *ref if ref is not nullptr.
  // - has_value: bool
  //   Whether the value exists, either in <val> or <ref>.
  // - val_field_is_set: bool , default value: false
  //   A tensor value if val_field_is_set
  // - alloc_attr: AllocatorAttributes
  //   The attributes of the allocator that creates the tensor.
  // - device_context: DeviceContext* , default value: nullptr
  //   Every entry carries an optional DeviceContext containing
  //   Device-specific information about how the Tensor was produced.

  bool completed = false;

  inline_ready.push_back(tagged_node);

  // ***********************************************************************
  // 循环
  while (!inline_ready.empty()) {
  // ***********************************************************************

    tagged_node = inline_ready.front();

    inline_ready.pop_front();
    // 1.
    // inline_ready.pop_front() 原理图
    // TaggedNodeReadyQueue::pop_front() 原理图
    // https://docs.google.com/document/d/1iUSROf80zgjOKQHXiOtbJX3D90_pqjbkiGZczkvHfPo/edit

    const Node* node = tagged_node.node;

    ////////////////////////////////////////////////////////////////////////
    // 困惑和费解的地方
    // 想象，如果有循环，那么确实要区分是第几次循环，每个节点也要额外的记录
    FrameState* input_frame = tagged_node.input_frame;
    const int64 input_iter = tagged_node.input_iter;
    // 评价: node, frame, iter 三个唯一确定一个状态
    ////////////////////////////////////////////////////////////////////////

    // tagged_node: struct TaggedNode
    // int64 input_iter = -1; 是初始值
    const int id = node->id();


    //***********************************************************************
    // 重要！
    // 取出 item 开始执行 op kernel
    const NodeItem& item = *gview.node(id);
    // QQQ. 这个数据结构来自哪里?
    // AAA. 这个数据结构来自于 gview, 来自于 ExecutorState::impl_->gview_
    //***********************************************************************


    // TODO(misard) Replace with a finer-grain enabling flag once we
    // add better optional debugging support.
    if (vlog_ && VLOG_IS_ON(1)) {
      mutex_lock l(input_frame->mu);
      input_frame->GetIteration(input_iter)->mark_started(item.pending_id);
    }

    // Set the device_context for this node id, if it exists.

    // device_context_map_ 变量说明:
    // device_context_map_: DeviceContextMap ;
    // 说明: Contains a value for [node->id()] for the device context assigned by the
    // device at the beginning of a step.

    // DeviceContextMap 数据结构
    // tensorflow/core/framework/device_base.h:112:
    // typedef std::vector<DeviceContext*> DeviceContextMap;

    // class DeviceContext 数据结构
    // tensorflow/core/framework/device_base.h:68
    // class DeviceContext : public core::RefCounted
    if (id < device_context_map_.size()) {
      params.op_device_context = device_context_map_[id];
    }

    params.track_allocations = false;
    stats = nullptr;
    // 1.
    // stats 变量说明:
    // stats: NodeExecStatsInterface*, default_value: nullptr
    // 新的节点重新初始化指针为 nullptr.

    // -----------------------------------------------------------------------
    // Profiler
    // -----------------------------------------------------------------------
    // 打开的话，需要  stats_collector_ == true!!!
    if (stats_collector_ && !tagged_node.is_dead) {
      stats = stats_collector_->CreateNodeExecStats(node);
      // 1.
      // stats_collector_ 变量说明:
      // stats_collector_: StepStatsCollectorInterface* const
      // tensorflow/core/common_runtime/executor.cc

      // 2.
      // class StepStatsCollectorInterface 数据结构
      // 纯虚函数
      // - NodeExecStatsInterface* CreateNodeExecStats(const Node* node)
      //   Creates an instance of `NodeExecStatsInterface` that should be used for
      //   collecting statistics about individual node execution.
      // - string ReportAllocsOnResourceExhausted(const string& err)
      //   Generates a string reporting the currently used memory based
      //   on ResourceExhausted OOM `err` message.
      //   `err` message needs to contain device name and allocator name, e.g.:
      //   "ResourceExhaustedError: OOM when allocating tensor ...
      //   on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc"

      // 2.1
      // class StepStatsCollectorInterface 继承类
      // - class StepStatsCollector
      //   tensorflow/core/common_runtime/step_stats_collector.h
      // - class SimpleStepStatsCollector
      //   tensorflow/core/kernels/data/captured_function.cc

      // Track allocations if and only if we are collecting statistics, and
      // `stats` object is expecting allocations to be tracked.
      params.track_allocations = stats ? stats->TrackAllocations() : false;
      nodestats::SetScheduled(stats, scheduled_nsec);
      nodestats::SetAllStart(stats);
    }


    if (vlog_) {
      VLOG(1) << "Process node: " << id << " step " << params.step_id << " "
              << SummarizeNode(*node) << (tagged_node.is_dead ? " is dead" : "")
              << " device: " << device->name();
    }

    // 用于获取 这个节点的 input tensors
    // =======================================================================
    Entry* input_tensors = GetInputTensors(input_frame, input_iter);
    // =======================================================================
    // 1.
    // Entry 数据结构
    // tensorflow/core/common_runtime/executor.cc
    // Entry 应该是直接和 Tensor 类比了。
    // - val: ManualConstructor<Tensor>, A tensor value, if val_field_is_set.
    // - ref: Tensor*
    // - has_value: bool
    // - val_field_is_set: bool
    // - alloc_attr: AllocatorAttributes
    // - device_context: DeviceContext*

    // 2.
    // GetInputTensors 函数说明
    // ExecutorState::GetInputTensors
    // tensorflow/core/common_runtime/executor.cc:1372:
    // Entry* GetInputTensors(FrameState* input_frame, int64 input_iter)
    // 总结起来说就是 : while loop 的 NO.input_iter-th iteration 的 input_tensors

    // 3.
    // input_frame 变量说明:
    // FrameState* input_frame = tagged_node.input_frame;
    // const int64 input_iter = tagged_node.input_iter;
    /*
    p *input_frame
    {
      executor=0x5604be126820,
      frame_name="",
      frame_id=0,
      parent_iter=-1,
      parent_frame=0x0,
      max_parallel_iterations=1,
      num_pending_inputs=0,
      iteration_count=0,
      num_outstanding_iterations=1,
      iterations= {
      }
      ,
      next_iter_roots=std::vector of length 0,
      capacity 0,
      inv_values=std::vector of length 0,
      capacity 0,
      dead_exits=std::vector of length 0,
      capacity 0,
      pending_counts=0x5604be16c680,
      total_input_tensors=7,
      nodes=0x5604be125e70,
      mu= {
        mu_= {
          space= {
            0x0,
            0x0
          }
        }
      }
    }

    p input_iter
    $21 = 0
    */

    // 4.
    // p *input_frame
    // $20 =
    // p input_iter
    // $21 = 0

    // 5.
    // 关于从 input_frame 里面得到的 tensor 都是临时存入的
    // ExecutorState::FrameState::ActivateNodes 里面的关键几步可以证明这些 tensor 都是临时变量
    // Entry* input_tensors = iter_state->input_tensors;
    // if (dst_need_input) {
    //   input_tensors[dst_loc] = (*outputs)[src_slot];
    // }

    // 用于获取这个节点的 第一个 input tensor
    // =======================================================================
    Entry* first_input = input_tensors + item.input_start;
    // =======================================================================

    outputs.clear();

    TensorReferenceVector accessed_tensors;

    DeviceContext* device_context = nullptr;

    // Only execute this node if it is not dead or it is a send/recv
    // transfer node. For transfer nodes, we need to propagate the "dead"
    // bit even when the node is dead.
    bool launched_asynchronously = false;

    // -----------------------------------------------------------------------
    // 特殊情况处理分支:
    if (tagged_node.is_dead && !IsTransferNode(node)) {
    // 即使是 Send Node
    // p tagged_node.is_dead
    // $43 = false
    // 所以没有进入这个分支
    // -----------------------------------------------------------------------
      // IsTransferNode 函数说明:
      // graph/graph.h
      // inline bool IsTransferNode(const Node* n) { return IsSend(n) || IsRecv(n); }

      outputs.resize(item.num_outputs);

    } else {
      // oridinary case branch:

      // Prepares inputs.
      bool is_input_dead = false;

      // -----------------------------------------------------------------------
      s = PrepareInputs(
            item, // input
            first_input, // input
            &inputs,  // output
            &input_device_contexts, // output
            &input_alloc_attrs, // output
            &is_input_dead); // output
      // -----------------------------------------------------------------------
      // 对于 _Recv Node 而已，没有 Inputs, 所以这个函数几乎是没干什么。

      // 1.
      // inputs 变量说明
      // inputs 是输入这个节点的 input tensors

      if (!s.ok()) {
        // Clear inputs.
        int num_inputs = item.num_inputs;
        for (int i = 0; i < num_inputs; ++i) {
          (first_input + i)->ClearVal();
        }
        MaybeMarkCompleted(input_frame, input_iter, id);
        // Continue to process the nodes in 'inline_ready'.
        completed = NodeDone(s, item.node, ready, stats, &inline_ready);
        continue;
        // 1.
        // QQQ. 为什么还可以 continue ??? 不是这个 node 已经死了吗? 难道对整个图没有影响?
        // AAA.
        //
      }


      ////////////////////////////////////////////////////////////////////////
      // 重要!
      // 接口: ExecutorState 和 NodeItem 之间的接口，插件交互的点，可以分开/切开的点
      // Set up compute params.
      OpKernel* op_kernel = item.kernel;
      params.op_kernel = op_kernel;
      // 1.
      // params: OpKernelContext::Params
      ////////////////////////////////////////////////////////////////////////


      params.frame_iter = FrameAndIter(input_frame->frame_id, input_iter);
      // FrameAndIter 数据结构
      // - frame_id : int
      // - iter_id : int

      params.is_input_dead = is_input_dead;
      // 对于 y/RSN/_0 send node 而言，is_input_dead == false

      params.output_attr_array = item.output_attrs();
      // 1.
      // params.output_attr_array 变量说明:
      // params.output_attr_array: const AllocatorAttributes*
      // 含义 Array indexed by output number for this node

      // 2.
      // struct AllocatorAttributes
      // tensorflow/core/framework/allocator.h
      //
      //  00000000
      //  ^^^^^^^^
      //  ||||||||
      //  |||||||+----+on host
      //  ||||||+-----+nic compatible
      //  |||||+------+gpu compatible
      //  ||||+-------+
      //  |||+--------+
      //  ||+---------+
      //  |+----------+
      //  +-----------+


      params.forward_from_array = item.forward_from();
      // params.forward_from_array 变量说明:
      // const int* forward_from_array = nullptr;
      // Values in [0,...) represent reservations for the indexed output.


      //////////////////////////////////////////////////////////////////////////
      // 执行计算
      if (item.kernel_is_async) {

        ///////////////////////////////////////////////////////////////////////
        // Asynchronous computes.
        ///////////////////////////////////////////////////////////////////////

        // 参考
        // https://docs.google.com/document/d/1lXvQQ_ZCyzeElZ-VcySGm95r96H6mDnRkUyph-p1B4M/edit#
        // Recv Node 是 异步的

        AsyncOpKernel* async = item.kernel->AsAsync();

        DCHECK(async != nullptr);

        launched_asynchronously = true;

        AsyncState* state =
            new AsyncState(params, tagged_node, &item, first_input, stats);
        // 1.
        // AsyncState 数据结构
        // executor.cc
        // - saved_inputs: TensorValueVec
        // - saved_input_device_contexts: DeviceContextVec
        // - saved_input_alloc_attrs: AllocatorAttributeVec
        // - params: OpKernelContext::Params
        // - tagged_node: TaggedNode
        // - item: const NodeItem*
        // - first_input: Entry*
        // - ctx: OpKernelContext # 重要
        // - stats: NodeExecStatsInterface*

        // 2.
        // struct ExecutorState::AsyncState 构造函数说明
        // executor.cc
        // State kept alive for executing an asynchronous node in another
        // thread.  NOTE: We need to make a copy of p.input,
        // p.input_device_contexts, and p.input_alloc_attrs for asynchronous
        // kernels because OpKernelContext methods like input_type(i) needs
        // the param points to valid input type vector. It's not an issue for
        // sync kernels because these vectors are kept on the stack.

        // auto: std::function<void()>
        auto done = [this, state]() {

          Device* device = impl_->params_.device;

          NodeExecStatsInterface* stats = state->stats;  // Shorthand

          Entry* first_input = state->first_input;       // Shorthand

          nodestats::SetOpEnd(stats);

          EntryVector outputs;

          // Process Outputs
          // 取出 &state->ctx 内的 outputs
          Status s = ProcessOutputs(*state->item, &state->ctx, &outputs, stats);

          nodestats::SetMemory(stats, &state->ctx);

          if (vlog_) {
            VLOG(2) << "Async kernel done: " << state->item->node->id()
                    << " step " << step_id_ << " "
                    << SummarizeNode(*state->item->node)
                    << (state->tagged_node.is_dead ? " is dead" : "")
                    << " device: " << device->name();
          }

          // Clears inputs.
          const int num_inputs = state->item->num_inputs;
          for (int i = 0; i < num_inputs; ++i) {
            (first_input + i)->ClearVal();
          }
          // Clears inputs Done.

          FrameState* input_frame = state->tagged_node.input_frame;
          const int64 input_iter = state->tagged_node.input_iter;

          const int id = state->tagged_node.node->id();
          MaybeMarkCompleted(input_frame, input_iter, id);

          TaggedNodeSeq ready;

          // Propagate Outputs
          if (s.ok()) {
            PropagateOutputs(state->tagged_node, // input
                             state->item, // input
                             &outputs, // input
                             &ready); // output
          }

          outputs.clear();
          if (s.ok() && impl_->device_record_tensor_accesses_) {
            // Get the list of all tensors accessed during the execution
            TensorReferenceVector accessed;
            state->ctx.retrieve_accessed_tensors(&accessed);
            nodestats::SetReferencedTensors(stats, accessed);
            // callee takes ownership of the vector
            device->ConsumeListOfAccessedTensors(state->ctx.op_device_context(),
                                                 accessed);
          }

          // NodeDone
          const bool completed =
              NodeDone(s, state->item->node, ready, stats, nullptr);

          delete state;

          if (completed) ScheduleFinish();

        };


        nodestats::SetOpStart(stats);
        // 1.
        // nodestats::SetOpStart 函数说明:
        // tensorflow/core/common_runtime/step_stats_collector.cc
        // 概述: 仅仅只是记录调用这个函数被调用时刻的时间于 NodeExecStats::op_start_rel_micros 内

        device->ComputeAsync(
                  async,  // input
                  &state->ctx, // input
                  done); // input
        // 1.
        // ComputeAsync 函数说明
        // common_runtime/device.h
        // Device:ComputeAsync
        //
        // Asynchronous kernel's compute.
        // virtual void ComputeAsync(
        //                AsyncOpKernel* op_kernel,
        //                OpKernelContext* context,
        //                AsyncOpKernel::DoneCallback done) {
        //   op_kernel->ComputeAsync(context, std::move(done));
        // }

        // 2.
        // Recv op compute 调用栈
        // core/kernels/sendrecv_ops.cc
        // void RecvOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done)

        // 3
        // async 变量说明:
        // async: AsyncOpKernel*

        // 4.
        // state 变量说明:
        // state: AsyncState*

        // 5.
        // struct ExecutorState::AsyncState 数据结构
        // executor.cc
        // - saved_inputs: TensorValueVec
        // - saved_input_device_contexts: DeviceContextVec
        // - saved_input_alloc_attrs: AllocatorAttributeVec
        // - params: OpKernelContext::Params
        // - tagged_node: TaggedNode
        // - item: const NodeItem*
        // - first_input: Entry*
        // - ctx: OpKernelContext # 重要
        // - stats: NodeExecStatsInterface*

        // 6.
        // class OpKernelContext
        // * typedef std::pair<Allocator*, TrackingAllocator*> WrappedAllocator;
        // - struct Params
        //    * step_id: int64
        //    * op_kernel: OpKernel*
        //    * device: DeviceBase*
        //    * eigen_gpu_device: PerOpGpuDevice*
        //    * track_allocations: bool
        //    * log_memory: bool
        //    * record_tensor_accesses: bool
        //    * output_attr_array: const AllocatorAttributes*
        //    * resource_manager: ResourceMgr*  # 这个和 tensor 是怎么联系起来的？
        //    * step_container: ScopedStepContainer*
        //      Per-step resources accessible by this op kernel invocation should be stored in this container.
        //    * rendezvous: Rendezvous*
        //      Mechanism used by this op kernel invocation to communicate with computations running on other devices.
        //    * collective_executor: CollectiveExecutor*
        //      Mechanism for executing a collective op that needs to coordinate with parallel instances running on other devices.
        //    * session_state: SessionState*
        //      The session state for this op.
        //    * session_handle: string
        //      Unique session identifier. Can be empty.
        //    * tensor_store: TensorStore*  # 留意这个
        //      The tensor store for this op.
        //
        //    ------------------------------------------------------------------
        //    * cancellation_manager: CancellationManager*
        //    ------------------------------------------------------------------
        //
        //    * inputs: const gtl::InlinedVector<TensorValue, 4>*  # 关注一下
        //    * is_input_dead: bool
        //    * input_alloc_attrs: const gtl::InlinedVector<AllocatorAttributes, 4>*
        //    * input_device_contexts: const gtl::InlinedVector<DeviceContext*, 4>*
        //    * op_device_context: DeviceContext*
        //    * frame_iter: FrameAndIter
        //      Control-flow op supports.
        //    * call_frame: CallFrameInterface*
        //    * function_library: FunctionLibraryRuntime*
        //    * runner: std::function<void(std::function<void()>)>*
        //    * stats_collector: StepStatsCollectorInterface*
        //    * graph_collector: GraphCollector*
        //    * slice_reader_cache: checkpoint::TensorSliceReaderCacheWrapper*
        //    * forward_from_array: const int*
        //    * inc_num_deferred_ops_function: std::function<void()>
        //    * dec_num_deferred_ops_function: std::function<void()>
        // - params_: Params*
        // - status_: Status
        // - wrapped_allocators_: gtl::InlinedVector<WrappedAllocator, 4>
        // - outputs_: gtl::InlinedVector<TensorValue, 4>
        // - referenced_tensors_: ManualConstructor<UniqueTensorReferences>
        // - temp_memory_allocated_: int64
        // - persistent_memory_allocated_: int64
        // - temp_tensor_buffer_and_size_: std::unique_ptr<gtl::InlinedVector<std::pair<const void*, int64>, 2>>
        // - persistent_alloc_ids_: std::unique_ptr<gtl::InlinedVector<int64, 2>>

      } else {

        ///////////////////////////////////////////////////////////////////////
        // Synchronous computes.
        ///////////////////////////////////////////////////////////////////////

        // ====================================================================
        // ====================================================================
        OpKernelContext ctx(&params, item.num_outputs);
        // ====================================================================
        // ====================================================================
        // 最最重要的参数

        // 1.
        // item.num_outputs 变量说明
        // NodeItem& instance 的 num_outputs
        // 这个节点的输出个数

        // 2.
        // tensorflow/core/framework/op_kernel.h:567:
        // class OpKernelContext

        // 3.
        // ctx: OpKernelContext 的生命周期:
        // QQQ.ctx 出了这个 scope 后会没有吗？
        // AAA. 是的， ctx 出了这个 {} 后就析构了。
        // if (!launched_asynchronously) 之前的那个 {} 。

        nodestats::SetOpStart(stats);
        // 1.
        // nodestats::SetOpStart 函数说明:
        // tensorflow/core/common_runtime/step_stats_collector.cc
        // 概述: 仅仅只是记录调用这个函数被调用时刻的时间于 NodeExecStats::op_start_rel_micros 内

        if (TF_PREDICT_FALSE(
          // 非此即彼的关系 由 MightTrace 控制
          // gpu_device.h:73:bool TraceUsingAnnotations() const { return true; }
          // GPU 的 kernel 是一定 只要是 tracing 就是 Annotations 这个分支。
          MightTrace(item, event_collector_, trace_using_annotations_))) {

          const string& op_name = op_kernel->name();

          tracing::ScopedRegion region(tracing::EventCategory::kCompute,
                                       op_name);

          // =============================================================
          // 也就是说，其实这里是非此即彼的关系
          // 不是 ScopedAnnotation 就是 ScopedActivity
          // 其中，我发现 ScopedActivity
          // IsEnabledForActivities 的 profiling 更像是 CPU
          // =============================================================
          if (trace_using_annotations_) {
            // The OpKernel may create child activities (such as GPU kernel
            // launches), so use a `ScopedAnnotation` to relate these activities
            // in the trace.
            tracing::ScopedAnnotation activity(
                op_name, strings::StrCat(op_kernel->type_string(),
                                         "#id=", step_id_, "#"));

            // -------------------------------------------------------------------

            device->Compute(op_kernel, &ctx);

            // -------------------------------------------------------------------

          } else {
            // Use the cheaper `ScopedActivity` to trace just the OpKernel
            // execution.

            /// tracing::ScopedActivity, tensorflow/core/platform/tracing.h
            tracing::ScopedActivity activity(
                op_name,
                strings::StrCat(op_kernel->type_string(), "#id=", step_id_,
                                "#"),
                item.kernel->IsExpensive());
            // -------------------------------------------------------------------

            device->Compute(op_kernel, &ctx);

            // -------------------------------------------------------------------

          }
          // =============================================================

        } else {

          // In the common case, avoid creating any tracing objects.

          if (op_kernel->IsExpensive()) {
            // op_kernel: OpKernel*
            // ConstantOp::IsExpensive() 对于 const op 而言
            // core/kernels/constant_op.h
            // 直接 return false

            KernelTimer timer;

            // -------------------------------------------------------------------

            device->Compute(op_kernel, &ctx);

            // -------------------------------------------------------------------
            // 1.
            // 如果是 send op 那么会进入这个分支
            // SendOp::Compute(OpKernelContext* ctx)
            // tensorflow/core/kernels/sendrecv_ops.cc

            op_kernel->UpdateCostEstimate(timer.ElapsedCycles());

          } else {

            // -------------------------------------------------------------------
            device->Compute(
              op_kernel, // input
              &ctx); // output
            // -------------------------------------------------------------------

            // 1.
            // Device::Compute 函数说明:
            // tensorflow/core/common_runtime/device.h
            // virtual void Compute(OpKernel* op_kernel, OpKernelContext* context)

            // 1.1
            // Device::Compute 的 real function
            // void BaseGPUDevice::Compute(OpKernel* op_kernel, OpKernelContext* context)
            // tensorflow/core/common_runtime/gpu/gpu_device.cc

            // 1.2
            // BaseGPUDevice::ComputeHelper(OpKernel* op_kernel, OpKernelContext* context)
            // tensorflow/core/common_runtime/gpu/gpu_device.cc

            // 2.
            // op_kernel: OpKernel*
            // class OpKernel 数据结构
            // tensorflow/core/framework/op_kernel.h
            // 和 device 相关的
            // - input_memory_types_: const MemoryTypeVector
            // - output_memory_types_: const MemoryTypeVector

            // 3.
            // ctx: OpKernelContext
            //
            // class OpKernelContext 数据结构
            // tensorflow/core/framework/op_kernel.h
            // * typedef std::pair<Allocator*, TrackingAllocator*> WrappedAllocator;
            // - struct Params
            //    * step_id: int64
            //    * op_kernel: OpKernel*
            //    * device: DeviceBase*
            //    * eigen_gpu_device: PerOpGpuDevice*
            //    * track_allocations: bool
            //    * log_memory: bool
            //    * record_tensor_accesses: bool
            //    * output_attr_array: const AllocatorAttributes*
            //    * resource_manager: ResourceMgr*  # 这个和 tensor 是怎么联系起来的？
            //    * step_container: ScopedStepContainer*
            //      Per-step resources accessible by this op kernel invocation should be stored in this container.
            //    * rendezvous: Rendezvous*
            //      Mechanism used by this op kernel invocation to communicate with computations running on other devices.
            //    * collective_executor: CollectiveExecutor*
            //      Mechanism for executing a collective op that needs to coordinate with parallel instances running on other devices.
            //    * session_state: SessionState*
            //      The session state for this op.
            //    * session_handle: string
            //      Unique session identifier. Can be empty.
            //    * tensor_store: TensorStore*  # 留意这个
            //      The tensor store for this op.
            //    * cancellation_manager: CancellationManager*
            //    * inputs: const gtl::InlinedVector<TensorValue, 4>*  # 关注一下
            //    * is_input_dead: bool
            //    * input_alloc_attrs: const gtl::InlinedVector<AllocatorAttributes, 4>*
            //    * input_device_contexts: const gtl::InlinedVector<DeviceContext*, 4>*
            //    * op_device_context: DeviceContext*
            //    * frame_iter: FrameAndIter
            //      Control-flow op supports.
            //    * call_frame: CallFrameInterface*
            //    * function_library: FunctionLibraryRuntime*
            //    * runner: std::function<void(std::function<void()>)>*
            //    * stats_collector: StepStatsCollectorInterface*
            //    * graph_collector: GraphCollector*
            //    * slice_reader_cache: checkpoint::TensorSliceReaderCacheWrapper*
            //    * forward_from_array: const int*
            //    * inc_num_deferred_ops_function: std::function<void()>
            //    * dec_num_deferred_ops_function: std::function<void()>
            // - params_: Params*
            // - status_: Status
            // - wrapped_allocators_: gtl::InlinedVector<WrappedAllocator, 4>
            // - outputs_: gtl::InlinedVector<TensorValue, 4>
            // - referenced_tensors_: ManualConstructor<UniqueTensorReferences>
            // - temp_memory_allocated_: int64
            // - persistent_memory_allocated_: int64
            // - temp_tensor_buffer_and_size_: std::unique_ptr<gtl::InlinedVector<std::pair<const void*, int64>, 2>>
            // - persistent_alloc_ids_: std::unique_ptr<gtl::InlinedVector<int64, 2>>

          }
        }

        nodestats::SetOpEnd(stats);

        // -------------------------------------------------------------------
        s = ProcessOutputs(
              item,  // input
              &ctx,  // input
              &outputs, // output
              stats);   // output

        // ProcessOutputs 函数说明:
        // ********************************************************************
        // 这个函数把计算得到的结果从 ctx 取出来，放到 outputs 里面，作为这个函数的输出。
        // ********************************************************************
        // Status ExecutorState::ProcessOutputs(
        //   const NodeItem& item, // input
        //   OpKernelContext* ctx, // input
        //   EntryVector* outputs, // output
        //   NodeExecStatsInterface* stats) // output
        // -------------------------------------------------------------------

        if (s.ok() && impl_->device_record_tensor_accesses_) {
          // Get the list of all tensors accessed during the execution
          ctx.retrieve_accessed_tensors(&accessed_tensors);
          device_context = ctx.op_device_context();
        }

        nodestats::SetMemory(stats, &ctx);

      }
    }


    // -----------------------------------------------------------------------
    // Synchronous kernel done:
    // 上面逻辑主要是为了得到 outputs: EntryVector
    // -----------------------------------------------------------------------
    if (!launched_asynchronously) {
      if (vlog_) {
        VLOG(2) << "Synchronous kernel done: " << id << " step "
                << params.step_id << " " << SummarizeNode(*node)
                << (tagged_node.is_dead ? " is dead: " : "")
                << " device: " << device->name();
      }

      // Clears inputs.
      const int num_inputs = item.num_inputs;
      for (int i = 0; i < num_inputs; ++i) {
        (first_input + i)->ClearVal();
      }

      // -----------------------------------------------------------------------
      MaybeMarkCompleted(input_frame, input_iter, id);
      // -----------------------------------------------------------------------

      // Propagates outputs.
      if (s.ok()) {

        // ---------------------------------------------------------------------
        PropagateOutputs(
          tagged_node, // input
          &item, // input
          &outputs,  // input
          &ready); // output
        // ---------------------------------------------------------------------
        // 1.
        // PropagateOutputs 函数说明:
        // After processing the outputs, propagates the outputs to their dsts.
        // Contents of *outputs are left in an indeterminate state after
        // returning from this method.
        //
        // 函数定义:
        // void PropagateOutputs(
        //   const TaggedNode& tagged_node, // input
        //   const NodeItem* item, // input
        //   EntryVector* outputs, // input
        //   TaggedNodeSeq* ready); // output

        // 2.
        // EntryVector 数据结构
        // tensorflow/core/common_runtime/executor.h
        // typedef gtl::InlinedVector<Entry, 4> EntryVector;

        // 3.
        // struct Entry 数据结构
        // class ExecutorState::struct Entry
        // tensorflow/core/common_runtime/executor.cc
        //
        // 概述:
        // 可以和 Tensor 等价了
        //
        // - val: ManualConstructor<Tensor>
        //   A tensor value, if val_field_is_set.
        // - ref: Tensor*
        //   A tensor reference.
        // - ref_mu: mutex*
        //   mutex for *ref if ref is not nullptr.
        // - has_value: bool
        //   Whether the value exists, either in <val> or <ref>.
        // - val_field_is_set: bool , default value: false
        //   A tensor value if val_field_is_set
        // - alloc_attr: AllocatorAttributes
        //   The attributes of the allocator that creates the tensor.
        // - device_context: DeviceContext* , default value: nullptr
        //   Every entry carries an optional DeviceContext containing
        //   Device-specific information about how the Tensor was produced.

      }

      // -----------------------------------------------------------------------
      outputs.clear();
      // -----------------------------------------------------------------------
      // 1.
      // QQQ.
      // 居然会把他们给删了？？那么 backpropagation 呢？

      if (!accessed_tensors.empty()) {
        nodestats::SetReferencedTensors(stats, accessed_tensors);
        // device_context is set above in synchronous computes
        device->ConsumeListOfAccessedTensors(device_context, accessed_tensors);
      }

      if (stats) {
        scheduled_nsec = nodestats::NowInNsec();
      }

      // Postprocess.
      completed = NodeDone(
                    s, // input
                    item.node, // input
                    ready, // input
                    stats,  // input
                    &inline_ready); // output
      // 1.
      // 即使说 completed 为 true ，但是还是要在这个 while loop 里面把所有的 node 都
      // 执行完为止，不可以这样！

    } // if not asynchronously branch done !
  }  // while !inline_ready.empty

  // This thread of computation is done if completed = true.
  if (completed) ScheduleFinish();
}


///////////////////////////////////////////////////////////////////////////////



// 主要逻辑是把输入从 first_input (数组中的一个) 开始的片段中拷贝到 inputs , 并收集一些其他的信息。
// 在调用的时候，first_input 指向 IterationState 中 inputs 数组中的一个元素
// 在 Process 里调用结束后，inputs 、 input_device_contexts 、 input_alloc_attrs 、 is_input_dead
// 这些数据会传递给 OpKernelContext ，在 Kernel::Compute 里被使用。
// ExecutorState::ProcessOutputs 把计算地 OpKernelContext 里地输出拷贝出来

// class OpKernelContext 数据结构
// tensorflow/core/framework/op_kernel.h:567:

Status ExecutorState::PrepareInputs(
  const NodeItem& item, // input
  Entry* first_input,   // input  重要
  TensorValueVec* inputs, // output
  DeviceContextVec* input_device_contexts, // output
  // input 的 device context
  AllocatorAttributeVec* input_alloc_attrs, // output
  // input 的 AllocatorAttribute
  bool* is_input_dead) { // output
  // 1.
  // inputs 变量说明:
  // inputs: TensorValueVec*
  // inputs 是输入这个节点的 input tensors

  const Node* node = item.node;

  inputs->clear();
  inputs->resize(item.num_inputs);

  input_device_contexts->clear();
  input_device_contexts->resize(item.num_inputs);

  input_alloc_attrs->clear();
  input_alloc_attrs->resize(item.num_inputs);

  *is_input_dead = false;

  bool is_merge = item.is_merge;

  for (int i = 0; i < item.num_inputs; ++i) {
    const bool expect_ref = IsRefType(item.input_type(i));

    Entry* entry = first_input + i;

    (*input_device_contexts)[i] = entry->device_context;
    (*input_alloc_attrs)[i] = entry->alloc_attr;

    // i-th input.
    TensorValue* inp = &(*inputs)[i]; // inp 是代理 &(*inputs)[i] 的 指针

    // 1.
    // TensorValue 数据结构
    // framework/op_kernel.h
    //
    // 简述:
    // Holds a tensor or tensor reference. For tensor references, we need
    // a mutex to prevent concurrent access to the tensor.
    // struct TensorValue {
    //   TensorValue() : mutex_if_ref(nullptr), tensor(nullptr) {}
    //   TensorValue(Tensor* t)  // NOLINT(runtime/explicit)
    //       : mutex_if_ref(nullptr), tensor(t) {}
    //   TensorValue(mutex* mu, Tensor* t) : mutex_if_ref(mu), tensor(t) {}
    //   Tensor* operator->() const { return tensor; }
    //   bool is_ref() const { return mutex_if_ref != nullptr; }
    //   mutex* mutex_if_ref;  // nullptr if not a ref, != nullptr if a ref
    //   Tensor* tensor;
    // };


    // Only merge and transfer nodes can have no-value inputs.
    if (!entry->has_value) {
      if (!is_merge) {
        DCHECK(IsTransferNode(node)) << node->name() << " - input " << i;
        DCHECK(!entry->val_field_is_set) << node->name() << " - input " << i;
        entry->has_value = true;
        entry->val_field_is_set = true;
        entry->val.Init(*kEmptyTensor);
        inp->tensor = entry->val.get();
        *is_input_dead = true;
      }
      continue;
    }

    if (entry->ref == nullptr) {
      if (expect_ref) {
        return AttachDef(
            errors::InvalidArgument(i, "-th input expects a ref type"),
            item.kernel->def());
      }

      // -----------------------------------------------------------------------
      inp->tensor = entry->val.get();
      // -----------------------------------------------------------------------

    } else {

      {
        tf_shared_lock ml(*entry->ref_mu);
        if (!entry->ref->IsInitialized() && !IsInitializationOp(item.node)) {
          return AttachDef(errors::FailedPrecondition(
                               "Attempting to use uninitialized value ",
                               item.kernel->requested_input(i)),
                           item.kernel->def());
        }
      }

      if (expect_ref) {
        inp->mutex_if_ref = entry->ref_mu;
        inp->tensor = entry->ref;
      } else {

        // Automatically deref the tensor ref when the op expects a
        // tensor but is given a ref to a tensor.  Need to deref it
        // under the mutex.
        {
          tf_shared_lock l(*(entry->ref_mu));
          DCHECK(!entry->val_field_is_set);
          entry->val.Init(*entry->ref);
          entry->val_field_is_set = true;
        }

        entry->ref = nullptr;
        entry->ref_mu = nullptr;

        inp->tensor = entry->val.get();
        // The dtype of entry->ref could have been changed by another operation
        // that ran after the operation that "produced" it executed, so
        // re-validate that the type of the dereferenced tensor matches the
        // expected input type.
        if (item.input_type(i) != inp->tensor->dtype()) {
          return AttachDef(
              errors::InvalidArgument(
                  i, "-th input expects type ",
                  DataTypeString(item.input_type(i)),
                  " but automatically dereferenced input tensor has type ",
                  DataTypeString(inp->tensor->dtype())),
              item.kernel->def());
        }

      }
    }
  }
  return Status::OK();

  /*
  输出的打印

  p inputs.size()
  $7 = 1

  p inputs[0].tensor->DebugString()
  $8 = "Tensor<type: float shape: [30,20] values: [-1.39755154 -0.0128526222 0.0272815395...]...>"
  */
}




Status ExecutorState::ProcessOutputs(
  const NodeItem& item, // input
  OpKernelContext* ctx, // input
  EntryVector* outputs, // output
  NodeExecStatsInterface* stats) // output
{
  // class OpKernelContext 数据结构:
  // tensorflow/core/framework/op_kernel.h:567:class OpKernelContext
  // 真真正正的图计算的集大成者，上千行代码描述。

  // EntryVector 数据结构
  // tensorflow/core/common_runtime/executor.h:614:
  // typedef gtl::InlinedVector<Entry, 4> EntryVector;

  // struct Entry 数据结构:
  // tensorflow/core/common_runtime/executor.cc:531
  // class ExecutorState::struct Entry
  // 和 device, op kernel 相关的
  // - alloc_attr: AllocatorAttributes
  // - device_context: DeviceContext*

  const Node* node = item.node;

  DCHECK_EQ(0, outputs->size());

  outputs->resize(item.num_outputs);
  /*
  _Send Op
  item.num_outputs == 0
  */

  Status s = ctx->status();

  // 异常处理，不想看
  if (!s.ok()) {
    s = AttachDef(s, item.kernel->def());
    // TODO(misard) Replace with a finer-grain enabling flag once we
    // add better optional debugging support.
    if (vlog_ && VLOG_IS_ON(1)) {
      LOG(WARNING) << this << " Compute status: " << s;
      DumpState();
    }
    if (s.code() == error::RESOURCE_EXHAUSTED) {
      // -----------------------------------------------------------------------
      if (stats_collector_) {
        // Profiling Memory
        string err = stats_collector_->ReportAllocsOnResourceExhausted(
            s.error_message());
        s = Status(s.code(), strings::StrCat(s.error_message(), err));
      // -----------------------------------------------------------------------
      } else {
        s = Status(
            s.code(),
            strings::StrCat(
                s.error_message(),
                "\nHint: If you want to see a list of allocated tensors when "
                "OOM happens, add report_tensor_allocations_upon_oom "
                "to RunOptions for current allocation info.\n"));
      }
    }
    return s;
  } // 异常处理结束


  // Get the device_context for this node id, if it exists.
  DeviceContext* device_context = nullptr;
  // class DeviceContext 数据结构:
  // tensorflow/core/framework/device_base.h


  // Get the device_context for this node id, if it exists.
  if (node->id() < device_context_map_.size()) {
    device_context = device_context_map_[node->id()];
    // 1.
    // device_context_map_ 变量说明:
    // common_runtime/executor.cc
    // device_context_map_: DeviceContextMap
    // 说明: Contains a value for [node->id()] for the device context assigned by the
    //      device at the beginning of a step.

    // 2.
    // DeviceContextMap 数据结构说明:
    // typedef std::vector<DeviceContext*> DeviceContextMap

    // 3.
    // class DeviceContext 数据结构
    // map[i] is the DeviceContext* for the node with id i, if i < map.size().
    // framework/device_base.h
    // 虚函数为主
    // 说明：A class that devices can subclass to pass around Device-specific context to OpKernels.

    /*
    对于 CPU 的情况
    打印

    p device_context_map_
    $25 = std::vector of length 0, capacity 0
    */
  }

  // -----------------------------------------------------------------------
  // 遍历所有的存在 OpKernelContext 内的输出
  // -----------------------------------------------------------------------
  for (int i = 0; i < item.num_outputs; ++i) {

    const TensorValue val = ctx->release_output(i);
    // 提醒: val 这个变量在本函数最后被 delete 了

    // 1.
    // OpKernelContext::release_output 函数说明
    // tensorflow/core/framework/op_kernel.h
    // inline TensorValue OpKernelContext::release_output(int index)

    // 2.
    // struct TensorValue 数据结构
    // tensorflow/core/framework/op_kernel.h
    // tensor: Tensor*

    if (val.tensor == nullptr) {
      // ***********************************************************************
      // Unless it's a Switch or a Recv, the node must produce a
      // tensor value at i-th output.
      // ***********************************************************************
      if (!IsSwitch(node) && !IsRecv(node)) {
        s.Update(errors::Internal("Missing ", i, "-th output from ",
                                  FormatNodeForError(*node)));
      }

    } else {

      // 首先，取出 i-th output from outputs 并且用 out 指针代理。
      Entry* out = &((*outputs)[i]);
      // 1.
      // outputs: EntryVector*

      // 2.
      // EntryVector 数据结构
      // tensorflow/core/common_runtime/executor.h:614:
      // typedef gtl::InlinedVector<Entry, 4> EntryVector;

      // 3.
      // struct Entry 数据结构
      // Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
      // executor.cc
      // ref: Tensor*
      // device_context: DeviceContext*
      // ... 其他


      // 然后，填充初始化所有的 out 的内容。

      // Set the device context of the output entry.
      out->device_context = device_context;
      // 1.
      // device_context: DeviceContext*

      // 2.
      // DeviceContext 数据结构:
      // 虚函数
      // - stream()
      // - MaintainLifetimeOnStream()
      // - CopyCPUTensorToDevice()
      // - CopyTensorInSameDevice()
      // - CopyDeviceTensorToCPU()
      // - ThenExecute()

      // Set the allocator attributes of the output entry.
      out->alloc_attr = ctx->output_alloc_attr(i);
      // 1.
      // out->alloc_attr: AllocatorAttributes

      // 2.
      // AllocatorAttributes 数据结构
      // - set_on_host
      // - on_host
      // - set_nic_compatible
      // - nic_compatible
      // - set_gpu_compatible
      // - gpu_compatible
      // - Merge
      // - IsEqualOrLessRestrictiveThan
      // - uint32 value = 0;
      // - int32 scope_id = 0;

      // Sanity(明智；头脑清楚；精神健全；通情达理/normal or sound powers of mind) check of output tensor types.
      DataType dtype; // 下面初始化 dtype

      if (val.is_ref()) {
        // val is ref

        // 1.
        // val : class TensorValue
        //       - tensor: Tensor*

        // 2.
        // TensorValue::is_ref() 函数说明:
        // tensorflow/core/framework/op_kernel.h
        // ref 的含义是 tensor reference

        tf_shared_lock ml(*val.mutex_if_ref);

        dtype = MakeRefType(val->dtype());

      } else {
        // val is not ref
        dtype = val->dtype();

      }

      if (dtype == item.output_type(i)) {
        // 正常处理

        if (stats && val.tensor->IsInitialized()) {
          // 1.
          // stats: NodeExecStatsInterface*
          // 我的例子里面这个是 nullptr
          // tracing 打开时会进入
          nodestats::SetOutput(stats, i, val.tensor);
        }

        if (val.is_ref()) {
          // val is tensor reference 的分支

          out->has_value = true;
          out->ref = val.tensor;
          out->ref_mu = val.mutex_if_ref;
          if (log_memory_) {
            Tensor to_log;
            {
              // Dereference the tensor under the lock.
              tf_shared_lock l(*out->ref_mu);
              to_log = *out->ref;
            }
            LogMemory::RecordTensorOutput(ctx->op_kernel().name(),
                                          ctx->step_id(), i, to_log);
            // 打印实例:
            // 2019-10-01 22:33:58.235769: I
            // tensorflow/core/framework/log_memory.cc:35]
            // __LOG_MEMORY__ MemoryLogTensorOutput { step_id: 1 kernel_name: "v4" tensor { dtype: DT_FLOAT shape { dim { size: 25 } dim { size: 15 } } } }
          }
        } else {
          // tensor 不是 reference tensor 的分支

          // NOTE that std::move is used here, so val.tensor goes to
          // uninitialized state (val.tensor->IsInitialized return false).
          DCHECK(!out->val_field_is_set);

          out->has_value = true;  // TRUE 哦
          out->val_field_is_set = true; // TRUE 哦

          // ==================================================================
          // 最核心的一步
          // new 一个 tensor ，然后赋值给 out->val
          out->val.Init(std::move(*val.tensor));
          // ==================================================================

          // 1.
          // val: const TensorValue

          // 2.
          // val.tensor: TensorValue::Tensor*

          // 3.
          // out: Entry*

          // 4.
          // out->val: ExecutorState::Entry::ManualConstructor<Tensor>;
          // A tensor value, if val_field_is_set

          // 5.
          // class ManualConstructor 数据结构
          // tensorflow/core/lib/gtl/manual_constructor.h
          //                 -------

          // 6.
          // ManualConstructor::Init 函数说明
          // inline void Init(const T1& p1) { new (space_) Type(p1); }

          if (log_memory_) {
            // 写到 vlog 3, 这个分支有进入
            LogMemory::RecordTensorOutput(ctx->op_kernel().name(),
                                          ctx->step_id(), i, *out->val);
            // LogMemory::RecordTensorOutput 函数说明
            // 2019-09-26 20:28:58.119028: I tensorflow/core/framework/log_memory.cc:35] __LOG_MEMORY__ MemoryLogTensorOutput { step_id: 1 kernel_name: "a/shape" tensor { dtype: DT_INT32 shape { dim { size: 2 } } allocation_description { requested_bytes: 8 allocated_bytes: 8 allocator_name: "mklcpu" allocation_id: 2 ptr: 94886958997504 } } }
            // tensorflow/core/framework/log_memory.cc
          }
        }

      } else {
        // 异常处理
        s.Update(errors::Internal("Output ", i, " of type ",
                                  DataTypeString(dtype),
                                  " does not match declared output type ",
                                  DataTypeString(item.output_type(i)),
                                  " for node ", FormatNodeForError(*node)));
      }
    } // if branch


    if (!val.is_ref()) {
      // If OpKernelContext returns outputs via pass-by-value, we
      // don't need this trouble.
      delete val.tensor;
    }


  } // End For, 遍历这个节点的所有输出

  return s;
}



void ExecutorState::PropagateOutputs(
  const TaggedNode& tagged_node, // input
  const NodeItem* item, // input
  EntryVector* outputs, // input, 这个节点的 output tensor 作为下一个节点的输入
  TaggedNodeSeq* ready) // output
{

  // 如果 这个节点会追踪，那么会构造一个 trace_collector->CreateActivityHandle
  // activity_handle = CreateActivityHandle() instance 出了本函数范围，然后析构，
  // 那时 profiling 结束。
  auto activity_handle =
      [&]() -> std::unique_ptr<tracing::TraceCollector::Handle> {
    auto* trace_collector = tracing::GetTraceCollector();
    if (TF_PREDICT_FALSE(trace_collector != nullptr &&
                         trace_collector->IsEnabledForActivities(
                             false /* is_expensive */))) {
      const string& op_name = item->kernel->name();
      // Intentionally using ExecutorPropagateOutputs as the first key so that
      // users are aware that it's not the op invocation.
      return trace_collector->CreateActivityHandle(
          "ExecutorPropagateOutputs",
          strings::StrCat(op_name, "#id=", step_id_, "#"),
          false /* is_expensive */);
    } else {
      return nullptr;
    }
  }();


  const Node* node = tagged_node.node;

  // 和 while loop 相关的两个
  // Frame <==> loop
  // FrameState 协助 Frame
  // IterationState 协助 FrameState
  // Frames maintain a variety of data structures to hold the state of each iteration.
  // 参考 https://topic.alibabacloud.com/a/tensorflow-source-code-analysis-of-the-common_runtime-executor-_8_8_30000147.html
  // 强烈推荐: https://www.imooc.com/article/73372
  FrameState* input_frame = tagged_node.input_frame;
  // FrameState 数据结构
  // tensorflow/core/common_runtime/executor.cc:673:  struct FrameState

  // struct Frame 数据结构
  // tensorflow/core/graph/control_flow.cc:29:struct Frame

  const int64 input_iter = tagged_node.input_iter;
  const bool is_dead = tagged_node.is_dead;

  // Propagates outputs along out edges, and puts newly ready nodes
  // into the ready queue.
  ready->clear(); // 居然 clear 了 ready queue?
  // ready 变量说明:
  // ready: TaggedNodeSeq
  // 这个变量是在 ExecutorState::Process 一开始被构造的。
  // 在 synchronous compute 的前提下第一次被使用就是在这个函数 PropagateOutputs 上

  bool is_frame_done = false;

  FrameState* output_frame = input_frame;
  int64 output_iter = input_iter;
  // output_frame 和 output_iter 说明:
  // 全部复制 input_frame, input_iter 的，奇怪

  if (!item->is_enter_exit_or_next_iter) {
    // 关于 output_frame, output_iter 说明:
    // 如果不是 enter, exit, next iteraton node 的话，那么就 全部复制 input_frame , input_iter 的。

    // Fast path for nodes types that don't need special handling
    DCHECK_EQ(input_frame, output_frame);
    // Normal path for most nodes
    mutex_lock l(input_frame->mu);

    // =======================================================================
    // Activate the successors of a node. Contents of *outputs are left in an
    // indeterminate state after returning from this method.
    // ** ActivateNodes 主要功能**
    // 把 NodeItem* item 的 各条 out edge 的 destination node 放入 ready queue 准备执行。
    // 感觉这个才是执行流程的起点
    output_frame->ActivateNodes(
      item,         // input
      is_dead,      // input
      output_iter,  // input
      outputs,      // input
      ready);       // output
    // =======================================================================
    // ExecutorState::FrameState::ActivateNodes 函数说明
    // void ExecutorState::FrameState::ActivateNodes(const NodeItem* item,
    //                                               const bool is_dead,
    //                                               int64 iter,
    //                                               EntryVector* outputs,
    //                                               TaggedNodeSeq* ready) // output
    //

    is_frame_done = input_frame->DecrementOutstandingOpsLocked(
        &impl_->gview_, input_iter, ready);

  } else if (item->is_enter) {

    FindOrCreateChildFrame(input_frame, input_iter, node, &output_frame);
    output_iter = 0;
    {
      const NodeItem* item = impl_->gview_.node(node->id());
      mutex_lock l(output_frame->mu);
      if (item->is_constant_enter) {
        // Propagate to all active iterations if this is a loop invariant.
        output_frame->AddLoopInv(item, (*outputs)[0], ready);
      } else {
        output_frame->ActivateNodes(
                        item,
                        is_dead,
                        output_iter,
                        outputs,
                        ready); // output
      }
      output_frame->num_pending_inputs--;
    }
    is_frame_done =
        input_frame->DecrementOutstandingOps(&impl_->gview_, input_iter, ready);

  } else if (item->is_exit) {

    if (is_dead) {
      mutex_lock l(input_frame->mu);
      // Stop and remember this node if it is a dead exit.
      if (input_iter == input_frame->iteration_count) {
        input_frame->dead_exits.push_back(node);
      }
      is_frame_done = input_frame->DecrementOutstandingOpsLocked(
          &impl_->gview_, input_iter, ready);
    } else {
      output_frame = input_frame->parent_frame;
      output_iter = input_frame->parent_iter;
      {
        mutex_lock l(output_frame->mu);
        output_frame->ActivateNodes(
                        item,
                        is_dead,
                        output_iter,
                        outputs,
                        ready); // output
      }
      is_frame_done = input_frame->DecrementOutstandingOps(&impl_->gview_,
                                                           input_iter, ready);
    }
  } else {
    // IsNextIteration 分支
    DCHECK(IsNextIteration(node));

    mutex_lock l(input_frame->mu);
    if (is_dead) {
      // Stop the deadness propagation.
      output_frame = nullptr;
    } else {
      if (input_iter == input_frame->iteration_count &&
          input_frame->num_outstanding_iterations ==
              input_frame->max_parallel_iterations) {
        // Reached the maximum for parallel iterations.
        input_frame->next_iter_roots.push_back({node, (*outputs)[0]});
        output_frame = nullptr;
      } else {
        // If this is a new iteration, start it.
        if (input_iter == input_frame->iteration_count) {
          input_frame->IncrementIteration(&impl_->gview_, ready);
        }
        output_iter = input_iter + 1;
      }
    }
    if (output_frame != nullptr) {
      // This is the case when node is not Enter, Exit, or NextIteration.
      DCHECK(input_frame == output_frame);
      output_frame->ActivateNodes(
                      item,
                      is_dead,
                      output_iter,
                      outputs,
                      ready); // output
    }
    is_frame_done = input_frame->DecrementOutstandingOpsLocked(
        &impl_->gview_, input_iter, ready);
  }

  // At this point, this node is completely done. We also know if the
  // completion of this node makes its frame completed.
  if (is_frame_done) {
    FrameState* parent_frame = input_frame->parent_frame;
    const int64 parent_iter = input_frame->parent_iter;
    DeleteFrame(input_frame, ready);
    if (parent_frame != nullptr) {
      // The completion of frame may cause completions in its parent frame.
      // So clean things up recursively.
      CleanupFramesIterations(parent_frame, parent_iter, ready);
    }
  }
}


// 1.
// NodeDone 后不予调度的规则设计
// https://docs.google.com/document/d/1r0zSvKuu7TqIJadaRpbiu3vq0YYS8GZ3KXAQoNPj9lc/edit

bool ExecutorState::NodeDone(
  const Status& s, // input
  const Node* node, // input
  const TaggedNodeSeq& ready, // input
  NodeExecStatsInterface* stats, // output
  TaggedNodeReadyQueue* inline_ready) { // output
  // 1.
  // ready 变量说明
  // QQQ. ready 是在哪里补给的?
  // AAA. PropagateOutputs(..., ready) 里面补给了 后继节点 到 ready queue 里面

  nodestats::SetAllEnd(stats);
  // 1.
  // nodestats::SetAllEnd 功能:
  // take down profling 信息

  // take down profling 信息分支
  if (stats) {
    if (stats_collector_) {
      stats->Done(impl_->params_.device->name());
    } else {
      delete stats;
    }
  }

  bool abort_run = false;

  if (!s.ok()) {
    // Some error happened. This thread of computation is done.
    mutex_lock l(mu_);

    if (status_.ok()) {
      abort_run = true;
      status_ = s; // 最后还是被赋值了。
    }

    // 1.
    // QQQ. s.ok() 和 status_.ok() 的区别?
    // AAA.
    // s 来自 s = ProcessOutputs(...)
    // ExecutorState::status_
  }

  // I am looking for ExecutorState::step_id_ to debug this

  if (abort_run) {
    // 1.
    // QQQ. 进入 abort_run 分支以后还会执行这个分支后面的逻辑吗?
    // AAA.

    TRACEPRINTF("StartAbort: %s", s.ToString().c_str());
    if (rendezvous_) {
      rendezvous_->StartAbort(s);
      // 1.
      //
      // 功能说明:
      // Aborts all pending and future Send/Recv with the given "status".
      //
      // StartAbort() does not wait for ongoing calls to finish.
      // REQUIRES: !status.ok()
      // virtual void StartAbort(const Status& status) = 0;

      // 2.
      // see :
      // IntraProcessRendezvous::StartAbort, common_runtime/rendezvous_mgr.cc
      //    -> LocalRendezvousImpl::StartAbort, tensorflow/core/framework/rendezvous.cc
      // https://docs.google.com/document/d/11wKKWYIY-IadlkhpCTHyRrgtH5ChZgmzzhEDsLxWvjE/edit#

      // 3.
      // -----------------------------------------------------------------------
      // 调用栈学习:
      // python code + 模拟多线程切入的代码:
      // https://docs.google.com/document/d/1nXZyZAtSesrJwn7824xjSpkbh-f-M-dfUKiHj-qi8rI/edit#
      // code: https://gist.github.com/shizukanaskytree/3f5d89928736f8b1895aa1d9ba4d3b14
      // -----------------------------------------------------------------------

      // 3.1
      // -----------------------------------------------------------------------
      // node id 对应
      // https://gist.githubusercontent.com/shizukanaskytree/64f764896f1d32aa34eb8e17a6eed4b7/raw/0c4ca393bd97028b5f57e0fa48afbd2ef3150981/step_id_node_id_resnet50.log
      // -----------------------------------------------------------------------

    }

    if (collective_executor_) {
      collective_executor_->StartAbort(s);
      // 1.
      // collective_executor_ 变量说明
      // ExecutorState::collective_executor_: CollectiveExecutor* , default value : nullptr

      // 2.
      // class CollectiveExecutor 数据结构:
      // tensorflow/core/framework/collective.h
      // class CollectiveExecutor : public PeerAccessInterface, public core::RefCounted
      //
      // - StartAbort is not implemented in this function.

      // 3.
      // class BaseCollectiveExecutor
      // tensorflow/core/common_runtime/base_collective_executor.h
      // class BaseCollectiveExecutor : public CollectiveExecutor
      //
      // - remote_access_->StartAbort(s);

      // 3.1
      // remote_access_ 变量说明
      // remote_access_: std::unique_ptr<PerStepCollectiveRemoteAccess>
    }

    // -----------------------------------------------------------------------
    if (cancellation_manager_) {
    // -----------------------------------------------------------------------
    // 1.
    // cancellation_manager_ 概述:
    // 提前 abort 这个 sess.run

    // 2.
    // QQQ. 在哪里被 初始化, updated 的？
    // AAA.
    // 在 ExecutorState::ExecutorState 内被
    // cancellation_manager_(args.cancellation_manager) 初始化

    // 3.
    // QQQ. 如果只是被初始化, 那么第一个 op->Compute 肯定跑不了，怎么还会让它跑一个 op->Compute 呢?
    // AAA. 懂了，如果是起始的 sess.run , 第一个 op node 是 SOURCE_ Node 所以执行到 NodeDone 时也没有执行有价值的计算，所以到了这里就开始推出了。

      cancellation_manager_->StartCancel();
      // 1.
      // ExecutorState::cancellation_manager_ 变量说明
      // 在 ExecutorState::ExecutorState 内被
      // cancellation_manager_(args.cancellation_manager) 初始化

      // 1.
      // class CancellationManager 数据结构
      // tensorflow/core/framework/cancellation.h
      // 成员变量
      // - is_cancelling_: bool
      // - is_cancelled_: std::atomic_bool
      // - mu_: mutex
      // - cancelled_notification_: Notification
      // - next_cancellation_token_: CancellationToken
      // - callbacks_: gtl::FlatMap<CancellationToken, CancelCallback>
      //
      // 部分的接口函数
      // - StartCancel()
      //   Run all callbacks associated with this manager.
      // - IsCancelled()
      //   Returns true iff StartCancel() has been called.
      // - Reset()
      //   Resets the cancellation manager to its original pre-cancelled state.
      // - get_cancellation_token()
      //   Returns a token that must be used in calls to RegisterCallback
      //   and DeregisterCallback.
      // - RegisterCallback
      //   Attempts to register the given callback to be invoked when this
      //   manager is cancelled.

      // 2.
      // class Notification
      // - cv_ : condition_variable
      //   signaled when notified_ becomes non-zero
      // - notified_: std::atomic<bool>
      //   mutations under mu_
      // - mu_: mutex
      //
      // 接口函数
      // - Notify()
      // - HasBeenNotified()
      // - WaitForNotification()

    }

  }

  bool completed = false;

  const size_t ready_size = ready.size();

  if (ready_size == 0 || !s.ok()) {

    completed = (num_outstanding_ops_.fetch_sub(1) == 1);
    // 1.
    // num_outstanding_ops_ 变量说明:
    // num_outstanding_ops_ 表示 当前要执行的 后继 Nodes 总数。
    // NodeDone 以后，算是完成了这个 Node, 所以是减掉 1 个。
    // 下面的那个分支，表示，这个 node 有几个后继节点，所以说还没有结束，所以要累计上去。

    // 2.
    // fetch_sub 函数说明:
    // https://en.cppreference.com/w/cpp/atomic/atomic/fetch_sub
    // The value immediately **preceding the effects** of this function in the
    // modification order of *this.
    // 所以返回的是在减 1 之前的值。所以是 1

    // 2.1
    // num_outstanding_ops_ log
    // num_outstanding_ops_ 在 return completed 为 true, 下一步要执行 ExecutorState::Finish()
    // 时的值为 0！
    // https://gist.github.com/shizukanaskytree/73d2b617d7553a39db1d1108597d320e

  } else if (ready_size > 1) {

    num_outstanding_ops_.fetch_add(ready_size - 1, std::memory_order_relaxed);

  }

  // Schedule the ready nodes in 'ready'.

  if (s.ok()) {

    ScheduleReady(ready, inline_ready);
    // 1.
    // ready 变量说明
    // 这个 ready 只是包含了这个 Node 对应的 output edge dst nodes

    // 2.
    // ScheduleReady 函数说明

  }

  return completed;
}

/** \brief
 *   Schedule all the expensive nodes in 'ready', and put all the inexpensive
 *   nodes in 'ready' into 'inline_ready'.
 *
 *  \param ready: const TaggedNodeSeq&;
 *         In essense, it is a list of TaggedNode struct.
 *
 *  \param inline_ready: TaggedNodeReadyQueue*;
 *         In essense, it is also a list of TaggedNode struct.
 *
 *  \details 4 cases for ready and inline_ready list of Nodes.
 *           Thinking in this way helps me organize the code logic.
 *           ready      inline_ready    case
 *           0          0               0
 *           0          1               1
 *           1          0               2
 *           1          1               3
 *
 *           Only expensive Node will be processed by another thread from the
 *           ThreadPool. For inexpensive nodes, they are not processed by
 *           Process function.
 *
 */
void ExecutorState::ScheduleReady(const TaggedNodeSeq& ready, // input
                                  TaggedNodeReadyQueue* inline_ready) { // output
  // 1.
  // QQQ. inline_ready 为什么会为 nullptr?
  // inline_ready 之所以会为 nullptr 是因为在 Process 里面会让 inline_ready pop
  // pop 光了后就为 nullptr 了。

  // case 0, 1: ready is empty; no matter inline_ready;
  if (ready.empty()) return;

  int64 scheduled_nsec = 0;

  // tracing, profiling
  if (stats_collector_) {
    scheduled_nsec = nodestats::NowInNsec();
  }

  /// case 2: inline_ready is empty.
  if (inline_ready == nullptr) {
    // Schedule to run all the ready ops in thread pool.
    for (auto& tagged_node : ready) {
      /// 这个线程 负责到 eigen lib 里面的 ScheduleWithHint Notify 为止。
      /// 这个 runner_ 主要负责的就是这个。
      runner_([=]() { Process(tagged_node, scheduled_nsec); });
    }
    return;
  }

  // \note
  // case 3:  ready has nodes, inline_ready has nodes.
  //
  //  GraphView is a struct only storing node id. So, it can find NodeItem via
  //  node index.
  const GraphView& gview = impl_->gview_;
  /// curr_expensive_node is used to point to process that Node.
  const TaggedNode* curr_expensive_node = nullptr;

  for (auto& tagged_node : ready) {

    const NodeItem& item = *gview.node(tagged_node.node->id());

    if (tagged_node.is_dead || !item.kernel->IsExpensive()) {

      // Inline this inexpensive node.
      inline_ready->push_back(tagged_node);

    } else {

      if (curr_expensive_node) {
        // Dispatch to another thread since there is plenty of work to
        // do for this thread.
        runner_(std::bind(&ExecutorState::Process,
                          this,
                          *curr_expensive_node,
                          scheduled_nsec));
      }

      curr_expensive_node = &tagged_node;
      // 如果连续取出/碰到两个 expensive node, 那么就要让一个线程去执行这个 node 了
      // 否则暂时存着这个 expensive node ， 尽量把 inexpensive node 放入 inline_ready 中
    }
  }

  // 最后两个是 ... -> expensive --> inexpensive 这样的序列
  //                      |
  //            寄存在 curr_expensive_node 那里
  if (curr_expensive_node) {

    if (inline_ready->empty()) {
      // Tail recursion optimization
      inline_ready->push_back(*curr_expensive_node);

    } else {
      // There are inline nodes to run already. We dispatch this expensive
      // node to other thread.
      runner_(std::bind(&ExecutorState::Process, this, *curr_expensive_node,
                        scheduled_nsec));
      // 1.
      // 代码思路
      // ScheduleWithHint 里面把这个 SchedClosure 构造成一个 Task 然后，调度。
    }

  }
}


inline void ExecutorState::MaybeMarkCompleted(FrameState* frame, int64 iter,
                                              int64 node_id) {
  // TODO(misard) Replace with a finer-grain enabling flag once we
  // add better optional debugging support.
  if (vlog_ && VLOG_IS_ON(1)) {
    const NodeItem* item = impl_->gview_.node(node_id);
    mutex_lock l(frame->mu);
    frame->GetIteration(iter)->mark_completed(item->pending_id);
  }
}

const Tensor* ExecutorState::GetTensorValueForDump(const Entry& input) {
  if (!input.has_value) {
    return kEmptyTensor;
  } else if (input.ref == nullptr) {
    return input.val.get();
  } else {
    return input.ref;
  }
}

void ExecutorState::DumpPendingNodeState(
    const int node_id, const Entry* input_vector,
    const bool show_nodes_with_no_ready_inputs) {
  const NodeItem& node_item = *impl_->gview_.node(node_id);
  const Node& node = *node_item.node;
  const int input_base = node_item.input_start;
  if (!show_nodes_with_no_ready_inputs) {
    bool has_ready_input = false;
    for (int i = 0; i < node.num_inputs(); ++i) {
      const Entry& input = input_vector[input_base + i];
      const Tensor* tensor = GetTensorValueForDump(input);
      if (tensor->IsInitialized()) {
        has_ready_input = true;
        break;
      }
    }
    if (!has_ready_input) {
      return;
    }
  }
  LOG(WARNING) << "    Pending Node: " << node.DebugString();
  for (int i = 0; i < node.num_inputs(); ++i) {
    const Entry& input = input_vector[input_base + i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "      Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(), ">");
    } else {
      LOG(WARNING) << "      Input " << i << ": not present";
    }
  }
}

void ExecutorState::DumpActiveNodeState(const int node_id,
                                        const Entry* input_vector) {
  const NodeItem& node_item = *impl_->gview_.node(node_id);
  const Node& node = *node_item.node;
  LOG(WARNING) << "    Active Node: " << node.DebugString();
  const int input_base = node_item.input_start;
  for (int i = 0; i < node.num_inputs(); ++i) {
    const Entry& input = input_vector[input_base + i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "      Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(), ">");
    } else {
      LOG(WARNING) << "      Input " << i << ": not present";
    }
  }
}

void ExecutorState::DumpIterationState(const FrameState* frame,
                                       IterationState* iteration) {
  const std::vector<const Node*>* nodes = frame->nodes;
  // Dump any waiting nodes that are holding on to tensors.
  for (const Node* node : *nodes) {
    const int node_id = node->id();
    PendingCounts::Handle pending_id = impl_->gview_.node(node_id)->pending_id;
    if (iteration->node_state(pending_id) == PendingCounts::PENDING_NOTREADY ||
        iteration->node_state(pending_id) == PendingCounts::PENDING_READY) {
      DumpPendingNodeState(node_id, iteration->input_tensors, false);
    }
  }
  // Then the active nodes.
  for (const Node* node : *nodes) {
    const int node_id = node->id();
    PendingCounts::Handle pending_id = impl_->gview_.node(node_id)->pending_id;
    if (iteration->node_state(pending_id) == PendingCounts::STARTED) {
      DumpActiveNodeState(node_id, iteration->input_tensors);
    }
  }
  // Show all input tensors in use.
  const int total_input_tensors = frame->total_input_tensors;
  size_t total_bytes = 0;
  for (int i = 0; i < total_input_tensors; ++i) {
    const Entry& input = iteration->input_tensors[i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "    Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(),
                          ", bytes: ", tensor->TotalBytes(), ">");
      total_bytes += tensor->TotalBytes();
    }
  }
  LOG(WARNING) << "    Total bytes " << total_bytes;
}

void ExecutorState::DumpState() {
  mutex_lock l(mu_);
  if (!dumped_on_error_) {
    LOG(WARNING) << "Dumping state";
    for (auto& frame : outstanding_frames_) {
      LOG(WARNING) << frame.first;
      FrameState* frame_state = frame.second;
      mutex_lock frame_lock(frame_state->mu);
      for (IterationState* iteration : frame_state->iterations) {
        LOG(WARNING) << "  Iteration:";
        DumpIterationState(frame_state, iteration);
      }
    }
    dumped_on_error_ = true;
  }
}

void ExecutorState::ScheduleFinish() {
  int num_deferred_ops;

  {
    mutex_lock lock(num_deferred_ops_mu_);
    num_deferred_ops = num_deferred_ops_;
  }

  if (num_deferred_ops > 0) {
    // Finish() may be blocked waiting for deferred async ops to complete. The
    // execution of deferred async ops may be waiting for non-enqueued ops of
    // other executors to complete. So running Finish() on the current thread
    // (inter-op threadpool thread) may lead to a deadlock due to threadpool
    // exhaustion. Instead, we run it on a separate thread to unblock the
    // threadpool thread.
    Env::Default()->SchedClosure([this]() { Finish(); });
  } else {
    Finish();
  }
}

void ExecutorState::Finish() {
  mu_.lock();
  auto status = status_;
  auto done_cb = std::move(done_cb_);
  // 1.
  // done_cb 变量说明
  // 来自 `item.executor->RunAsync(args, barrier->Get());`

  // 2.
  // DoneCallback done 说明
  // tensorflow/core/common_runtime/executor.h
  //
  // typedef std::function<void(const Status&)> DoneCallback;
  //
  // 由 `item.executor->RunAsync(args, barrier->Get());` 的 barrier->Get() 传入

  // 3.1
  // barrier->Get() 函数说明:
  // 概述:
  // Returns a closure that Executors must call when they are done
  // computing, passing the status of their execution as an argument.
  //
  // 定义:
  // executor.h
  // StatusCallback Get() {
  //   return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1);
  // }

  // 3.2
  // StatusCallback 数据结构
  // tensorflow/core/common_runtime/executor.h:150:
  // typedef std::function<void(const Status&)> StatusCallback;

  // 3.3
  // ExecutorBarrier::WhenDone 函数说明:
  // tensorflow/core/common_runtime/executor.h
  // 概述:
  // 更新 Status, 然后如果 !s.ok() 就 StartAbort 否则 就 执行 callback done 函数

  // 4.
  // barrier 变量说明:
  // ExecutorBarrier* barrier

  // 5.
  // class ExecutorBarrier 数据结构
  // tensorflow/core/common_runtime/executor.h
  //
  // 概述:
  // A class to help run multiple executors in parallel and wait until
  // all of them are complete.
  //
  // ExecutorBarrier deletes itself after the function returned by Get()
  // is called.
  //
  // - rendez_: Rendezvous* , default value : nullptr
  // - done_cb_: StatusCallback, default value : nullptr
  // - mu_: mutable mutex
  // - pending_: int
  // - status_group_: StatusGroup
  //
  // 重要接口:
  // - WhenDone(const Status& s)
  // - StatusCallback Get()

  auto runner = std::move(runner_);
  // 1.
  // runner, runner_ 变量说明:
  // Executor::Args 数据结构
  //  struct Args {
  //    int64 step_id = 0;
  //    ...
  //    typedef std::function<void()> Closure;
  //    typedef std::function<void(Closure)> Runner;
  //    Runner runner = nullptr;
  //  }

  // 2.
  // runner, runner_ 的使用是
  // 强行构造一个 signature 为 input void, output void 的函数
  // 比如, [=]() { done_cb(status); }

  mu_.unlock();
  CHECK(done_cb != nullptr);
  Device* device = impl_->params_.device;

  // There are several potential race conditions below. To name a few:
  // 1. Even if the device's status is OK at the precise moment when
  // num_deferred_ops_ reaches 0, it could go bad before device->RefreshStatus()
  // is called below, caused by work enqueued onto the same device by other
  // concurrent ExecutorState objects.
  // 2. Some implementations of Device::RefreshStatus, such as
  // XlaDevice::RefreshStatus, may be inherently racy because it releases the
  // device mutex after a stream pointer is acquired and before the stream is
  // queried for status.
  // 3. It's the same for some implementations of Device::Sync, such as
  // XlaDevice::Sync.
  //
  // However, these race conditions are acceptable because a stream (and
  // therefore an XlaDevice) can only go from OK to not-OK, never the opposite,
  // which means we will at worst report errors when there isn't any, never the
  // opposite.

  // If inc_num_deferred_ops_function has ever been called, ExecutorState must
  // wait for all corresponding dec_num_deferred_ops_function calls to happen
  // regardless of status. This ensures that dec_num_deferred_ops_function can
  // safely use ExecutorState's resources.
  {
    mutex_lock lock(num_deferred_ops_mu_);
    while (num_deferred_ops_ > 0) {
      no_deferred_ops_cv_.wait(lock);
    }
  }

  // An early exit for devices don't allow sync on completion. Ops that run on
  // these devices should have used num_deferred_ops correctly to ensure the
  // device has finished all relevant work at this point.
  if (!device->AllowsSyncOnCompletion()) {
    status.Update(device->RefreshStatus());
    if (!status.ok()) {
      // In device async execution mode, it's possible for device execution to
      // lag behind ExecutorState scheduling so much that this is the first
      // place a device execution error surfaces.
      // If so, all ExecutorState::NodeDone calls have already happened with OK
      // status. This is the last defense where StartCancel must be called to
      // abort all computation still running on any device.
      // TODO(b/124523000): Always call Finish in a separate thread, so even if
      // StartCancel blocks the current thread's execution, we won't encounter
      // deadlocks caused by inter-op thread exhaustion.
      if (cancellation_manager_) {
        cancellation_manager_->StartCancel();
      }
    }
    delete this;
    // 1.
    // QQQ. what is ~ExecutorState?
    // AAA.
    // - delete name_frame.second
    // - device_context_map_->Unref()
    // - delete slice_reader_cache_

    // 2.
    // 在自己里面 把 自己 ExecutorState 给删除了，我以为全部都没了，包括这个成语函数，
    // 但是下面👇还有执行语句。


    runner([=]() { done_cb(status); });
    return;
    // 1.
    // 学习:
    // 原来 signature output 为 void 的也可以强行 return
  }

  if (sync_on_finish_ && status.ok()) {
    // Block until the device has finished all queued operations. For
    // devices like GPUs that continue to execute Ops after their Compute
    // methods have completed, this ensures that control is not returned to
    // the user until the step (and its side-effects) has actually completed.
    device->Sync([=](Status new_status) mutable {
      status.Update(new_status);
      delete this;
      runner([=]() { done_cb(status); });
    });
  } else {
    delete this;
    runner([=]() { done_cb(status); });
  }
}

void ExecutorState::FindOrCreateChildFrame(FrameState* frame, int64 iter,
                                           const Node* node,
                                           FrameState** child) {
  // Get the child frame name.
  string enter_name;
  Status s = GetNodeAttr(node->attrs(), "frame_name", &enter_name);
  DCHECK(s.ok()) << s;
  const string child_name = MakeFrameName(frame, iter, enter_name);

  {
    mutex_lock executor_lock(mu_);
    auto it = outstanding_frames_.find(child_name);
    if (it != outstanding_frames_.end()) {
      *child = it->second;
      return;
    }
  }

  // Need to create a new frame instance.
  // Note that this new frame instance is created without any locks.
  if (vlog_) VLOG(2) << "Create frame: " << child_name;

  int parallel_iters;
  s = GetNodeAttr(node->attrs(), "parallel_iterations", &parallel_iters);
  DCHECK(s.ok()) << s;
  FrameState* temp = new FrameState(impl_, parallel_iters);
  temp->frame_name = child_name;
  temp->frame_id = Hash64(child_name);
  temp->parent_frame = frame;
  temp->parent_iter = iter;
  temp->InitializeFrameInfo(enter_name);

  // 'iterations' is a fixed-length circular buffer.
  temp->iterations.resize(temp->max_parallel_iterations + 1);
  // Initialize iteration 0.
  temp->iterations[0] =
      new IterationState(temp->pending_counts, temp->total_input_tensors);

  {
    mutex_lock executor_lock(mu_);
    auto it = outstanding_frames_.find(child_name);
    if (it != outstanding_frames_.end()) {
      *child = it->second;
    } else {
      mutex_lock frame_lock(frame->mu);
      frame->GetIteration(iter)->outstanding_frame_count++;
      outstanding_frames_[child_name] = temp;
      *child = temp;
      temp = nullptr;
    }
  }
  delete temp;  // Not used so delete it.
}

void ExecutorState::DeleteFrame(FrameState* frame, TaggedNodeSeq* ready) {
  // First, propagate dead_exits (if any) to the parent frame.
  FrameState* parent_frame = frame->parent_frame;
  const int64 parent_iter = frame->parent_iter;
  if (parent_frame != nullptr) {
    mutex_lock parent_frame_lock(parent_frame->mu);
    // Propagate all the dead exits to the parent frame.
    mutex_lock this_frame_lock(frame->mu);
    for (const Node* node : frame->dead_exits) {
      auto parent_iter_state = parent_frame->GetIteration(parent_iter);
      for (const Edge* e : node->out_edges()) {
        const Node* dst_node = e->dst();

        const auto dst_pending_id =
            impl_->gview_.node(dst_node->id())->pending_id;

        // TODO(yuanbyu): We don't need this if we require the subgraph
        // given to an executor not to contain a sink node.
        if (dst_node->IsSink()) continue;

        bool dst_dead = true;
        bool dst_ready = false;
        // We know this is a dead input to dst.
        if (IsMerge(dst_node)) {
          if (e->IsControlEdge()) {
            parent_iter_state->decrement_pending(dst_pending_id, 2);
            int count = parent_iter_state->pending(dst_pending_id);
            int dead_cnt = parent_iter_state->dead_count(dst_pending_id);
            dst_dead = (dead_cnt == dst_node->num_inputs());
            dst_ready = (count == 0) || ((count == 1) && dst_dead);
          } else {
            parent_iter_state->increment_dead_count(dst_pending_id);
            const int dead_cnt = parent_iter_state->dead_count(dst_pending_id);
            dst_dead = (dead_cnt == dst_node->num_inputs());
            dst_ready =
                (parent_iter_state->pending(dst_pending_id) == 1) && dst_dead;
          }
        } else {
          parent_iter_state->increment_dead_count(dst_pending_id);
          dst_ready =
              (parent_iter_state->decrement_pending(dst_pending_id, 1) == 0);
        }
        if (dst_ready) {
          if (IsControlTrigger(dst_node)) dst_dead = false;
          ready->emplace_back(dst_node, parent_frame, parent_iter, dst_dead);
          parent_iter_state->outstanding_ops++;
        }
      }
    }
  }

  // Delete the frame.
  const string& frame_name = frame->frame_name;
  if (vlog_) VLOG(2) << "Delete frame " << frame_name;
  {
    mutex_lock executor_lock(mu_);
    outstanding_frames_.erase(frame_name);
  }
  delete frame;
}

void ExecutorState::CleanupFramesIterations(FrameState* frame, int64 iter,
                                            TaggedNodeSeq* ready) {
  bool is_frame_done = false;
  {
    mutex_lock frame_lock(frame->mu);
    frame->GetIteration(iter)->outstanding_frame_count--;
    is_frame_done = frame->CleanupIterations(&impl_->gview_, iter, ready);
  }
  if (is_frame_done) {
    FrameState* parent_frame = frame->parent_frame;
    const int64 parent_iter = frame->parent_iter;
    DeleteFrame(frame, ready);
    if (parent_frame != nullptr) {
      // The completion of frame may cause completions in its parent frame.
      // So clean things up recursively.
      CleanupFramesIterations(parent_frame, parent_iter, ready);
    }
  }
}

// ** ActivateNodes 主要功能**
// 把 NodeItem* item 的 各条 out edge 的 destination node 放入 ready queue 准备执行。
// 感觉这个才是执行流程的起点

// _Send Op 的 dst node 没有放入 ready ，因为 _SINK node 没必要放进 ready queue.

// 说明帖子
// http://jcf94.com/2018/01/23/2018-01-23-tfunpacking2/
// https://www.twblogs.net/a/5b822d562b717737e032d359
void ExecutorState::FrameState::ActivateNodes(const NodeItem* item, // input
                                              const bool is_dead, // input
                                              int64 iter, // input
                                              EntryVector* outputs, // input
                                              TaggedNodeSeq* ready) { // output
  // 1.
  // ready 变量目的说明:
  // ready queue 给我的感觉其主要目的是为了存放 这个 已经处理完的 NodeItem* item 的后继节点

  const GraphView& gview = executor->gview_;
  // 1.
  // executor 变量说明
  // executor: ExecutorState::FrameState::const ExecutorImpl*
  // 含义 The executor the frame is in.

  // 拿到 当前这个 ExecutorState::FrameState 的 第 iter 次 對應的 IterationState
  IterationState* iter_state = GetIteration(iter);
  // 1.
  // struct IterationState 数据结构
  // 含义: The state of an iteration.
  // - input_tensors: Entry*
  // - outstanding_ops: size_t
  // - outstanding_frame_count: int
  // - counts_: PendingCounts
  // 用途:
  // To get input_tensors from IterationState

  // 2.
  // GetIteration 函数说明:
  //

  // 從NodeItem 拿到輸出邊的信息
  const size_t num_output_edges = item->num_output_edges;
  /*
  p item->node->DebugString()
  $12 = "{name:'_SOURCE' id:0 source}"
  p num_output_edges
  $13 = 6
  */

  const EdgeInfo* edges = item->output_edge_list();
  // 只是返回第一个 EdgeInfo 的 base 作为起点
  // output_edge_base()

  // 從 iter_state 拿到 Entry 數組 的起始指針, 下面有一种情况讨论到，
  // if (dst_need_input) 怎么怎么样. 这时会用到 input_tensors 。
  Entry* input_tensors = iter_state->input_tensors;

  // 遍历所有的 output edge
  for (size_t out_index = 0; out_index < num_output_edges; out_index++) {

    const EdgeInfo& e = edges[out_index];
    // edges: EdgeInfo*
    // e: const EdgeInfo&

    const int dst_id = e.dst_id;
    // 得到 这条边的 dst node id

    ////////////////////////////////////////////////////////////////////////
    // 重要！
    const NodeItem* dst_item = gview.node(dst_id);
    // 根据 ID 得到这条边的 dst node item
    //
    // 这个数据结构用来表示这个 out edge 的 destination node item
    // 这个 destination node item 会被放入 ready queue 内进行周而复始的执行
    ////////////////////////////////////////////////////////////////////////

    // 一個 handle 用來從 PendingCounts 結構裏存取 counts
    // 一個 Iteration 維持一個 PeningCounts 對象，用來保存 NodeItem 在本次 iteration 的
    // pendingcount 數據
    //
    const PendingCounts::Handle dst_pending_id = dst_item->pending_id;
    // 1.
    // PendingCounts 数据结构
    // common_runtime/pending_counts.h

    // 2.
    // PendingCounts::Handle 数据结构
    // common_runtime/pending_counts.h
    // - byte_offset_: int  : 31; // Byte offset of the rep in PendingCounts object
    // - is_large_: bool  : 1;  // If true, rep is LargeCounts; otherwise PackedCounts

    // 3.
    /*
    // 以 _Send node 的 dst node _SINK Node 为例:
    p dst_pending_id
    $66 = {
      byte_offset_ = 1,
      is_large_ = false
    }
    */

    const int src_slot = e.output_slot;
    // 打印
    // src_slot == -1
    // 因为是 SINK_ Node

    // TODO(yuanbyu): We don't need this if we require the subgraph
    // given to an executor not to contain a sink node.
    if (dst_item->is_sink) continue; // 对于 send node , 就止步于此了。

    bool dst_dead = false;
    bool dst_ready = false;

    // True iff this input for dst is needed. We only set this input for
    // dst if this flag is true. This is needed to make the thread safety
    // analysis happy.
    const bool is_control_edge = (src_slot == Graph::kControlSlot);

    // 是否需要這個輸入數據
    bool dst_need_input = !is_control_edge;

    if (dst_item->is_merge) {
      // 如果是 merge node
      // A merge node is ready if all control inputs have arrived and either
      // a) a live data input becomes available or b) all data inputs are
      // dead. For Merge, pending's LSB is set iff a live data input has
      // arrived.
      if (is_control_edge) {

        iter_state->decrement_pending(dst_pending_id, 2);
        // iter_state: IterationState*

        int count = iter_state->pending(dst_pending_id);

        int dead_cnt = iter_state->dead_count(dst_pending_id);

        dst_dead = (dead_cnt == dst_item->num_inputs);

        dst_ready = (count == 0) || ((count == 1) && dst_dead);

      } else {

        if ((*outputs)[src_slot].has_value) {
          // outputs 变量说明:
          // outputs: EntryVector*
          // src_slot: slot id of a node

          // This is a live data input.
          int count = iter_state->pending(dst_pending_id);

          iter_state->mark_live(dst_pending_id);

          // Only the first live edge sets the input and (potentially)
          // triggers execution. The low bit of count is set if and
          // only if no live input has been used yet (mark_live clears
          // it). The node should be started if and only if this is
          // the first live input and there are no pending control
          // edges, i.e. count == 1.
          dst_ready = (count == 1);

          dst_need_input = ((count & 0x1) == 1);

        } else {

          // This is a dead data input. Note that dst_node is dead if node is
          // a dead enter. We need this to handle properly a while loop on
          // the untaken branch of a conditional.
          // TODO(yuanbyu): This is a bit hacky, but a good solution for
          // now.
          iter_state->increment_dead_count(dst_pending_id);

          const int dead_cnt = iter_state->dead_count(dst_pending_id);

          dst_dead = (dead_cnt == dst_item->num_inputs) || item->is_enter;

          dst_ready = (iter_state->pending(dst_pending_id) == 1) && dst_dead;

          dst_need_input = false;

        }
      }

    } else {
      // 如果不是 merge node
      const bool increment_dead =
          (is_dead || (!is_control_edge && !(*outputs)[src_slot].has_value));

      int pending, dead;

      iter_state->adjust_for_activation(dst_pending_id, increment_dead,
                                        &pending, &dead);
      dst_dead = (dead > 0);
      dst_ready = (pending == 0);

    }

    if (dst_need_input) {

      const int dst_slot = e.input_slot;
      const int dst_loc = dst_item->input_start + dst_slot;
      if (e.is_last) {
        input_tensors[dst_loc] = std::move((*outputs)[src_slot]);
      } else {
        input_tensors[dst_loc] = (*outputs)[src_slot];
      }

    }

    // Add dst to the ready queue if it's ready
    if (dst_ready) {
      if (dst_item->is_control_trigger) dst_dead = false;

      // 重要！！！
      // 概述:
      // 把 dst_item->node 放入 ready queue 进行下一轮的迭代
      // 在遍历所有边的过程中，把当前这条边的
      // 这里是传递 edge 的下一个节点进入 ready queue
      //////////////////////////////////////////////////////////////////////
      ready->emplace_back(dst_item->node, this, iter, dst_dead);
      //                  --------------    |     |      |
      //                  Node* t_node      |     |      bool dead
      //                                    |     int64 in_iter
      //                                    FrameState* in_frame
      //                 --------------------------------------
      //                       全部：TaggedNode instance
      //////////////////////////////////////////////////////////////////////
      // 1.
      // 构造了 TaggedNode
      // 构造函数:
      // TaggedNode(const Node* t_node, FrameState* in_frame, int64 in_iter,
      //            bool dead) {
      //   node = t_node;
      //   input_frame = in_frame;
      //   input_iter = in_iter;
      //   is_dead = dead;
      // }
      // 打印:
      // this 是 ExecutorState::FrameState
      // iter == 0
      // dst_dead == false

      // FrameState* output_frame = input_frame;
      // ==> output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
      //  ==> output_frame 就是 input_frame 的，当然就是那个 this 的值了。

      // 2.
      // ready 变量说明:
      // ready : TaggedNodeSeq*

      // 3.
      // TaggedNodeSeq 数据结构
      // typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq

      // 4.
      // TaggedNode 数据结构
      // node: const Node*, default : nullptr;
      // input_frame: FrameState*, default : nullptr;
      // input_iter: int64, default : -1;
      // is_dead: bool, default : false;

      // 5.
      // dst_item 变量说明:
      // dst_item: const NodeItem*

      iter_state->outstanding_ops++;
    }

  }
}



void ExecutorState::FrameState::ActivateNexts(const GraphView* gview,
                                              int64 iter,
                                              TaggedNodeSeq* ready) {
  // Propagate the deferred NextIteration nodes to the new iteration.
  for (auto& node_entry : next_iter_roots) {
    const Node* node = node_entry.first;
    const Entry& entry = node_entry.second;
    const bool is_dead = !entry.has_value;
    const NodeItem* item = gview->node(node->id());
    EntryVector outputs{entry};
    ActivateNodes(item, is_dead, iter, &outputs, ready);
  }
  next_iter_roots.clear();
}

void ExecutorState::FrameState::ActivateLoopInvs(const GraphView* gview,
                                                 int64 iter,
                                                 TaggedNodeSeq* ready) {
  // Propagate loop invariants to the new iteration.
  for (auto& node_entry : inv_values) {
    const Node* node = node_entry.first;
    const Entry& entry = node_entry.second;
    const bool is_dead = !entry.has_value;
    const NodeItem* item = gview->node(node->id());
    EntryVector outputs{entry};
    ActivateNodes(item, is_dead, iter, &outputs, ready);
  }
}

void ExecutorState::FrameState::AddLoopInv(const NodeItem* item,
                                           const Entry& entry,
                                           TaggedNodeSeq* ready) {
  // Store this value.
  inv_values.push_back({item->node, entry});

  // Make this value available to all iterations.
  const bool is_dead = !entry.has_value;
  for (int i = 0; i <= iteration_count; ++i) {
    EntryVector outputs{entry};
    ActivateNodes(item, is_dead, i, &outputs, ready);
  }
}

bool ExecutorState::FrameState::IsIterationDone(int64 iter) {

  IterationState* iter_state = GetIteration(iter);

  if (iter_state->outstanding_ops == 0 &&
      iter_state->outstanding_frame_count == 0) {
    if (iter == 0) {
      // The enclosing frame has no pending input.
      return num_pending_inputs == 0;
    } else {
      // The preceding iteration is deleted (and therefore done).
      return (GetIteration(iter - 1) == nullptr);
    }
  }
  return false;
}

void ExecutorState::FrameState::IncrementIteration(const GraphView* gview,
                                                   TaggedNodeSeq* ready) {
  iteration_count++;
  const int64 next_iter = iteration_count;

  // Initialize the next iteration.
  IterationState* iter_state =
      new IterationState(pending_counts, total_input_tensors);
  SetIteration(next_iter, iter_state);
  num_outstanding_iterations++;
  dead_exits.clear();

  // Activate the successors of the deferred roots in the new iteration.
  ActivateNexts(gview, next_iter, ready);

  // Activate the loop invariants in the new iteration.
  ActivateLoopInvs(gview, next_iter, ready);
}

// QQQ. 如何理解一个 iteration 的结束?
//
bool ExecutorState::FrameState::CleanupIterations(const GraphView* gview,
                                                  int64 iter,
                                                  TaggedNodeSeq* ready) {
  int64 curr_iter = iter;
  while (curr_iter <= iteration_count && IsIterationDone(curr_iter)) {
    // Delete the iteration curr_iter.
    delete GetIteration(curr_iter);
    SetIteration(curr_iter, nullptr);
    --num_outstanding_iterations;
    ++curr_iter;

    // When one iteration is completed, we check for deferred iteration,
    // and start it if there is one.
    if (!next_iter_roots.empty()) {
      IncrementIteration(gview, ready);
    }
  }
  return IsFrameDone();
}


void ExecutorImpl::RunAsync(
      const Args& args,
      DoneCallback done) {

  (new ExecutorState(args, this))->RunAsync(std::move(done));

}
// 1.
// args 变量说明:
// defined above : Executor::Args args;
// 1.
// Executor::Args 数据结构
//  struct Args {
//    int64 step_id = 0;
//    Rendezvous* rendezvous = nullptr;
//    StepStatsCollectorInterface* stats_collector = nullptr;
//    CallFrameInterface* call_frame = nullptr;
//    CancellationManager* cancellation_manager = nullptr;
//    SessionState* session_state = nullptr;
//
//    // Unique session identifier. Can be empty.
//    string session_handle;
//    TensorStore* tensor_store = nullptr;
//    ScopedStepContainer* step_container = nullptr;
//    CollectiveExecutor* collective_executor = nullptr;
//
//    // If true, calls Sync() on the device.
//    bool sync_on_finish = false;
//
//    typedef std::function<void()> Closure;
//    typedef std::function<void(Closure)> Runner;
//    Runner runner = nullptr;
//  };

// 2.
// ExecutorState::RunAsync(...) 函数说明:

// 3.
// DoneCallback done 说明
// tensorflow/core/common_runtime/executor.h
//
// typedef std::function<void(const Status&)> DoneCallback;
//
// 由 `item.executor->RunAsync(args, barrier->Get());` 的 barrier->Get() 传入

// 3.1
// barrier->Get() 函数说明:
// 概述:
// Returns a closure that Executors must call when they are done
// computing, passing the status of their execution as an argument.
//
// 定义:
// executor.h
// StatusCallback Get() {
//   return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1);
// }

// 3.2
// StatusCallback 数据结构
// tensorflow/core/common_runtime/executor.h:150:
// typedef std::function<void(const Status&)> StatusCallback;

// 3.3
// ExecutorBarrier::WhenDone 函数说明:
// tensorflow/core/common_runtime/executor.h
// 概述:
// 更新 Status, 然后如果 !s.ok() 就 StartAbort 否则 就 执行 callback done 函数

// 4.
// barrier 变量说明:
// ExecutorBarrier* barrier

// 5.
// class ExecutorBarrier 数据结构
// tensorflow/core/common_runtime/executor.h
//
// 概述:
// A class to help run multiple executors in parallel and wait until
// all of them are complete.
//
// ExecutorBarrier deletes itself after the function returned by Get()
// is called.
//
// - rendez_: Rendezvous* , default value : nullptr
// - done_cb_: StatusCallback, default value : nullptr
// - mu_: mutable mutex
// - pending_: int
// - status_group_: StatusGroup
//
// 重要接口:
// - WhenDone(const Status& s)
// - StatusCallback Get()

// 6.
// class CancellationManager 数据结构
// tensorflow/core/framework/cancellation.h
// 成员变量
// - is_cancelling_: bool
// - is_cancelled_: std::atomic_bool
// - mu_: mutex
// - cancelled_notification_: Notification
// - next_cancellation_token_: CancellationToken
// - callbacks_: gtl::FlatMap<CancellationToken, CancelCallback>
//
// 部分的接口函数
// - StartCancel()
//   Run all callbacks associated with this manager.
// - IsCancelled()
//   Returns true iff StartCancel() has been called.
// - Reset()
//   Resets the cancellation manager to its original pre-cancelled state.
// - get_cancellation_token()
//   Returns a token that must be used in calls to RegisterCallback
//   and DeregisterCallback.
// - RegisterCallback
//   Attempts to register the given callback to be invoked when this
//   manager is cancelled.
// - ...

// 6.1
// CancellationManager 构造函数说明:
// CancellationManager::CancellationManager()
// : is_cancelling_(false),
//   is_cancelled_(false),
//   next_cancellation_token_(0) {}
// - is_cancelling_ : false
// - is_cancelled_: false
// - next_cancellation_token_: 0


}  // namespace

Status NewLocalExecutor(const LocalExecutorParams& params,
                        std::unique_ptr<const Graph> graph,
                        Executor** executor) {

  // struct LocalExecutorParams 数据结构
  // 概述:
  // "params" provides a set of context for the executor. We expect that
  // different context would provide different implementations.
  //
  // 打印 log: https://gist.github.com/shizukanaskytree/c6bb691de2b710e8a47a394fb6139eef
  //
  // tensorflow/core/common_runtime/executor.h:84:struct LocalExecutorParams
  // - device: Device*
  // - function_library: FunctionLibraryRuntime*
  // - create_kernel: std::function<Status(const NodeDef&, OpKernel**)>
  // - delete_kernel: std::function<void(OpKernel*)>

  ExecutorImpl* impl = new ExecutorImpl(params, std::move(graph));

  const Status s = impl->Initialize();

  if (s.ok()) {
    *executor = impl;
  } else {
    delete impl;
  }
  return s;
}

/** \brief 根据输入的参数，创建 OpKernel
 *
 *  \param [out] kernel: OpKernel**
 */
Status CreateNonCachedKernel(
  Device* device, // input
  FunctionLibraryRuntime* flib, // input
  const NodeDef& ndef, // input
  int graph_def_version, // input
  OpKernel** kernel) { // output

  const auto device_type = DeviceType(device->attributes().device_type());
  /*
  打印
  p device_type
  $22 = {type_ = "CPU"}

  p ndef.DebugString()
  $23 = "name: \"_SOURCE\"\nop: \"NoOp\"\n"

  p *device
  https://gist.github.com/shizukanaskytree/12f66845cfacc8175d6a7831d39cf413
  */

  auto allocator = device->GetAllocator(AllocatorAttributes());
  // AllocatorAttributes() 构造函数说明:
  // framework/allocator.h
  // struct AllocatorAttributes 的构造函数
  //
  // AllocatorAttributes 数据结构 说明
  // A tensorflow Op may need access to different kinds of memory that
  // are not simply a function of the device to which the Op has been
  // assigned.  For example, an Op executing on a GPU may still need
  // to allocate CPU RAM for some purpose.  Internal to the tensorflow
  // runtime we may choose to allocate CPU ram from special regions
  // that have been prepared for higher performance in some use
  // contexts, e.g. doing DMA with particular devices.  For these
  // reasons, the Device interface does not expose just one memory
  // Allocator, but instead provides an accessor that takes a
  // specification of the desired memory attributes in order to select
  // an Allocator.
  //
  // Example use:
  //  // Allocator for ordinary device memory:
  //  Allocator* a = allocator(AllocatorAttributes());
  // ...
  //  // Allocator for CPU RAM, regardless of where Op is executing:
  //  AllocatorAttributes attr;
  //  attr.set_on_host(true);
  //  Allocator* a = allocator(attr);

  // -----------------------------------------
  // tensorflow/core/framework/op_kernel.cc:1265
  // Status CreateOpKernel(DeviceType device_type, DeviceBase* device,

  // device->GetAllocator 函数说明
  // device 的真实类型是 GPUCompatibleCPUDevice* , 所以
  // GetAllocator() 在 common_runtime/gpu/gpu_device_factory.cc
  // Allocator* GPUCompatibleCPUDevice::GetAllocator(AllocatorAttributes attr) override {...}
  // 再调用 ThreadPoolDevice::GetAllocator at common_runtime/threadpool_device.cc

  // class Allocator 数据结构
  // Allocator is an abstract interface for allocating and deallocating
  // device memory.
  // framework/allocator.h

  return CreateOpKernel(
          device_type, // input
          device, // input
          allocator, // input
          flib, // input
          ndef, // input
          graph_def_version, // input
          kernel); // output
  // -----------------------------------------
}

void DeleteNonCachedKernel(OpKernel* kernel) { delete kernel; }

namespace {

//
class DefaultExecutorRegistrar {
 public:
  // 在哪里被调用?
  //
  DefaultExecutorRegistrar() {
    Factory* factory = new Factory;
    ExecutorFactory::Register("", factory);
    ExecutorFactory::Register("DEFAULT", factory);
  }

 private:
  class Factory : public ExecutorFactory {
    Status NewExecutor(const LocalExecutorParams& params,
                       std::unique_ptr<const Graph> graph,
                       std::unique_ptr<Executor>* out_executor) override {
      Executor* ret = nullptr;
      TF_RETURN_IF_ERROR(NewLocalExecutor(params, std::move(graph), &ret));
      out_executor->reset(ret);
      return Status::OK();
    }
  };
};
static DefaultExecutorRegistrar registrar;

}  // namespace

}  // namespace tensorflow
