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
#include <thread>
#include <iostream>

#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/common_runtime/direct_session.h" // wxf
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
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#include "tensorflow/core/util/env_var.h"

//#include "tensorflow/c/c_api.h"

namespace tensorflow {
// wxf
DirectSessionsManager* direct_sessions_manager_ = new DirectSessionsManager();

// wxf
bool ExecutorState::catched_ = false;
// wxf
//bool ExecutorState::entered_ = false;

// wxf
std::vector<ExecutorState::Entry> reuse_entry_inputs;
std::atomic<bool> matmul01_is_ok(false);
std::atomic<bool> matmul02_is_ok(false);

//namespace {

// 1-D, 0 element tensor.
static const Tensor* const kEmptyTensor = new Tensor;

bool IsInitializationOp(const Node* node) {
  return node->op_def().allows_uninitialized_input();
}

// Helper routines for collecting step stats.
namespace nodestats {
inline int64 NowInNsec() { return Env::Default()->NowNanos(); }

void SetScheduled(NodeExecStatsInterface* stats, int64 micros) {
  if (!stats) return;
  stats->SetScheduled(micros * EnvTime::kMicrosToNanos);
}

void SetAllStart(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordExecutorStarted();
}

void SetOpStart(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordComputeStarted();
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

//class ExecutorImpl;
//class ExecutorsPoolManager;
//class GraphView;

// ------------------------------------------------------------------------------ 

// Maybe we can set a global instance of ExecutorManager below and
// add all of the Executors into it.
ExecutorsPoolManager* ExecutorImpl::executors_pool_manager_ = new ExecutorsPoolManager();

// -----------------------------------------------------------------------------

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

size_t GraphView::NodeItemBytes(const Node* n) {
  const size_t num_output_edges = n->out_edges().size();
  const int num_inputs = n->num_inputs();
  const int num_outputs = n->num_outputs();

  // Compute number of bytes needed for NodeItem and variable length data.
  // We do not subtract sizeof(var) since num_inputs/num_outputs might
  // both be zero.
  const size_t raw_bytes =
      sizeof(NodeItem)                             // Fixed
      + num_output_edges * sizeof(EdgeInfo)        // output_edges[...]
      + num_outputs * sizeof(AllocatorAttributes)  // output_attr[...]
      + num_outputs * sizeof(int)                  // forward_from[num_outputs]
      + num_inputs * sizeof(uint8)                 // input_type[num_inputs]
      + num_outputs * sizeof(uint8);               // output_type[num_outputs]
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
  const size_t bytes =
      ((raw_bytes + kItemAlignment - 1) / kItemAlignment) * kItemAlignment;
  return bytes;
}

char* GraphView::InitializeNode(char* ptr, const Node* n) {
  const int id = n->id();
  CHECK(node_offsets_[id] == kuint32max);  // Initial value in constructor

  const size_t bytes = NodeItemBytes(n);
  constexpr size_t kItemAlignment = sizeof(NodeItem*);
  CHECK_EQ(reinterpret_cast<uintptr_t>(ptr) % kItemAlignment, 0);
  NodeItem* item = reinterpret_cast<NodeItem*>(ptr);

  // We store a 32-bit offset relative to the beginning of space_, so that we
  // only need an array of 32-bit values to map from node id to the NodeItem*,
  // (versus 64 bits on most machines if we just stored an array of NodeItem*
  // pointers). Casting to int64 is needed on 32bit CPU to avoid comparing
  // values as "int" vs "size_t" in CHECK_LE.
  CHECK_LE(static_cast<int64>(ptr - space_), kuint32max);
  const uint32 offset = static_cast<uint32>(ptr - space_);
  node_offsets_[id] = offset;
  ptr += bytes;

  const size_t num_output_edges = n->out_edges().size();
  const int num_inputs = n->num_inputs();
  const int num_outputs = n->num_outputs();

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
  EdgeInfo* dst_edge = item->output_edge_base();
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
  for (int i = 0; i < num_outputs; i++) {
    new (&output_attrs[i]) AllocatorAttributes();
  }

  DCHECK_LT(DataType_MAX, 255);  // Must fit in uint8
  uint8* input_types = item->input_type_base();
  for (int i = 0; i < num_inputs; i++) {
    input_types[i] = static_cast<uint8>(n->input_type(i));
    DCHECK_EQ(item->input_type(i), n->input_type(i));
  }

  // Check ScopedAllocatorAttrs and forward_from.  Also assign output_types.
  {
    std::vector<int> forward_input;
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
  num_nodes_ = num_nodes;
  size_t total_bytes = 0;
  for (const Node* n : g->nodes()) {
    total_bytes += NodeItemBytes(n);
  }

  node_offsets_ = new uint32[num_nodes];
  for (int i = 0; i < num_nodes; i++) {
    node_offsets_[i] = kuint32max;
  }

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
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph_.get(), &cf_info));

  // Cache this value so we make this virtual function call once, rather
  // that O(# steps * # nodes per step) times.
  device_record_tensor_accesses_ =
      params_.device->RequiresRecordingAccessedTensors();

  for (auto& it : cf_info.unique_frame_names) {
    EnsureFrameInfo(it)->nodes = new std::vector<const Node*>;
  }

  // Preprocess every node in the graph to create an instance of op
  // kernel for each node.
  for (const Node* n : graph_->nodes()) {
    const int id = n->id();
    const string& frame_name = cf_info.frame_names[id];
    FrameInfo* frame_info = EnsureFrameInfo(frame_name);

    // See if this node is a root node, and if so, add to root_nodes_.
    if (n->in_edges().empty()) {
      root_nodes_.push_back(n);
    }

    NodeItem* item = gview_.node(id);
    item->node = n;

    item->input_start = frame_info->total_inputs;
    frame_info->total_inputs += n->num_inputs();

    Status s = params_.create_kernel(n->def(), &item->kernel);
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

// ---------------------------------------------------------------------

ExecutorState::ExecutorState(const Executor::Args& args, ExecutorImpl* impl)
    : 
      // wxf
      // Priority of this ExecutorState instance inherited from DirectSession which
      // creates this ExecutorState instance. 
      // It should be initialized by const Executor::Args& args.
      executor_state_priority_(args.executor_state_priority),

      // wxf
      // Tell the ExecutorState which device type it is running on now.
      // It is used to construct the ExecutorState instances.
      device_type_executing_on_(args.device_type_executing_on),

      vlog_(VLOG_IS_ON(1)),
      log_memory_(LogMemory::IsEnabled()),
      step_id_(args.step_id),
      rendezvous_(args.rendezvous),
      collective_executor_(args.collective_executor),
      session_state_(args.session_state),
      session_handle_(args.session_handle),
      tensor_store_(args.tensor_store),
      step_container_(args.step_container),
      stats_collector_(args.stats_collector),
      event_collector_(
          tracing::GetEventCollector(tracing::EventCategory::kCompute)),
      context_(ContextKind::kThread),
      slice_reader_cache_(new checkpoint::TensorSliceReaderCacheWrapper),
      call_frame_(args.call_frame),
      impl_(impl),
      cancellation_manager_(args.cancellation_manager),
      runner_(args.runner),
      sync_on_finish_(args.sync_on_finish),
      trace_using_annotations_(impl->params_.device->TraceUsingAnnotations()),
      num_outstanding_ops_(0) {
  // We start the entire execution in iteration 0 of the root frame
  // so let us create the root frame and the state for iteration 0.
  // We assume root_frame_->frame_name.empty().
  root_frame_ = new FrameState(impl_, 1);
  root_frame_->frame_id = 0;  // must be 0
  root_frame_->InitializeFrameInfo(root_frame_->frame_name);

  // Initialize iteration 0.
  root_frame_->iterations.resize(root_frame_->max_parallel_iterations);
  root_frame_->iterations[0] = new IterationState(
      root_frame_->pending_counts, root_frame_->total_input_tensors);

  outstanding_frames_.insert({root_frame_->frame_name, root_frame_});

  // wxf
  // reset the global catched_ var after this iteration.
  if (executor_state_priority_ == ExecutorStatePriority::LOW) {
    catched_ = false; 
  }
  //~wxf
}

ExecutorState::~ExecutorState() {
  for (auto name_frame : outstanding_frames_) {
    delete name_frame.second;
  }
  for (auto it : device_context_map_) {
    it->Unref();
  }
  delete slice_reader_cache_;
}

Status ExecutorImpl::BuildControlFlowInfo(const Graph* g,
                                          ControlFlowInfo* cf_info) {
  const int num_nodes = g->num_node_ids();
  cf_info->frame_names.resize(num_nodes);
  std::vector<Node*> parent_nodes;
  parent_nodes.resize(num_nodes);
  std::vector<bool> visited;
  visited.resize(num_nodes);

  string frame_name;
  std::deque<Node*> ready;

  // Initialize with the root nodes.
  for (Node* n : g->nodes()) {
    if (n->in_edges().empty()) {
      visited[n->id()] = true;
      cf_info->unique_frame_names.insert(frame_name);
      ready.push_back(n);
    }
  }

  while (!ready.empty()) {
    Node* curr_node = ready.front();
    int curr_id = curr_node->id();
    ready.pop_front();

    Node* parent = nullptr;
    if (IsEnter(curr_node)) {
      // Enter a child frame.
      TF_RETURN_IF_ERROR(
          GetNodeAttr(curr_node->attrs(), "frame_name", &frame_name));
      parent = curr_node;
    } else if (IsExit(curr_node)) {
      // Exit to the parent frame.
      parent = parent_nodes[curr_id];
      frame_name = cf_info->frame_names[parent->id()];
      parent = parent_nodes[parent->id()];
    } else {
      parent = parent_nodes[curr_id];
      frame_name = cf_info->frame_names[curr_id];
    }

    for (const Edge* out_edge : curr_node->out_edges()) {
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

void ExecutorState::RunAsync(Executor::DoneCallback done) {
  const Graph* graph = impl_->graph_.get();
  TaggedNodeSeq ready;

  // Ask the device to fill in the device context map.
  Device* device = impl_->params_.device;
  const Status fill_status =
      device->FillContextMap(graph, &device_context_map_);
  if (!fill_status.ok()) {
    delete this;
    done(fill_status);
    return;
  }

  // Initialize the ready queue.
  for (const Node* n : impl_->root_nodes_) {
    DCHECK_EQ(n->in_edges().size(), 0);
    ready.push_back(TaggedNode{n, root_frame_, 0, false});
  }
  if (ready.empty()) {
    delete this;
    done(Status::OK());
  } else {
    // wxf
//    VLOG(0) << std::this_thread::get_id() << ", 1. ExecutorState::RunAsync: num_outstanding_ops_: " << num_outstanding_ops_ ;

    num_outstanding_ops_ = ready.size();

    // wxf
//    VLOG(0) << std::this_thread::get_id() << ", 2. ExecutorState::RunAsync: num_outstanding_ops_: " << num_outstanding_ops_ ;

    root_frame_->iterations[0]->outstanding_ops = ready.size();
    done_cb_ = std::move(done);
    // Schedule to run all the ready ops in thread pool.
    ScheduleReady(ready, nullptr);
  }
}


// Returns true if `item` might be traced by the given trace and event
// collectors. Returns false only if `item` definitely will not be traced.
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
  auto* trace_collector = tracing::GetTraceCollector();
  if (trace_collector) {
    if (using_annotations) {
      return trace_collector->IsEnabledForAnnotations();
    } else {
      return trace_collector->IsEnabledForActivities(
          item.kernel->IsExpensive());
    }
  }
  return false;
}

void ExecutorState::Process(TaggedNode tagged_node, int64 scheduled_nsec) {

  // wxf
  // useful to test terminal of graph
  // ===
  //VLOG(0) << "Process node: " << tagged_node.node->id() << " step " << step_id_ << " ";
  // ~~~
//         << SummarizeNode(*(tagged_node.node));

// // python step id ==2 //  0, 1, "2"
// if (step_id_ == 11 && tagged_node.node->id() == 25) {
//   // simulate high task enters
//   direct_sessions_manager_->high_priority_direct_session_count_.fetch_add(1, std::memory_order_relaxed);
// } 
//
// if (step_id_ == 11 && tagged_node.node->id() == 1530) {
//   // simulate high task exit 
//   direct_sessions_manager_->high_priority_direct_session_count_.fetch_sub(1, std::memory_order_relaxed);
// } 
 

  //VLOG(0) << std::this_thread::get_id() << ", === ExecutorState::Process: num_outstanding_ops_: " << num_outstanding_ops_ ;



  WithContext wc(context_);
  const GraphView& gview = impl_->gview_;
  TaggedNodeSeq ready;
  TaggedNodeReadyQueue inline_ready;

  // Parameters passed to OpKernel::Compute.
  TensorValueVec inputs;
  DeviceContextVec input_device_contexts;
  AllocatorAttributeVec input_alloc_attrs;

  OpKernelContext::Params params;
  params.step_id = step_id_;

  Device* device = impl_->params_.device;

  params.device = device;
  params.log_memory = log_memory_;
  params.record_tensor_accesses = impl_->device_record_tensor_accesses_;
  params.rendezvous = rendezvous_;
  params.collective_executor = collective_executor_;
  params.session_state = session_state_;
  params.session_handle = session_handle_;
  params.tensor_store = tensor_store_;
  params.cancellation_manager = cancellation_manager_;
  params.call_frame = call_frame_;
  params.function_library = impl_->params_.function_library;
  params.resource_manager = device->resource_manager();
  params.step_container = step_container_;
  params.slice_reader_cache = slice_reader_cache_;
  params.inputs = &inputs;
  params.input_device_contexts = &input_device_contexts;
  params.input_alloc_attrs = &input_alloc_attrs;
  params.runner = &runner_;
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

  EntryVector outputs;
  bool completed = false;
  inline_ready.push_back(tagged_node);
  while (!inline_ready.empty()) {
    tagged_node = inline_ready.front();
    inline_ready.pop_front();

//    // wxf
//    if ( (executor_state_priority_ == ExecutorStatePriority::LOW &&
//          device_type_executing_on_ == "HPU" &&
//          direct_sessions_manager_->high_priority_direct_session_count_.\
//            load(std::memory_order_relaxed)) || 
//          (catched_ && (executor_state_priority_ == ExecutorStatePriority::LOW)) 
//       ){
//      catched_ = true;
//    
//      VLOG(0) << std::this_thread::get_id() << ", *** Low priority abort branch Debugging message.";
//      bool finished = false;
//      {
//        mutex_lock l(mu_);
//        num_outstanding_ops_.fetch_sub(inline_ready.size());
//        VLOG(0) << std::this_thread::get_id() << ", ExecutorState::Process: num_outstanding_ops_: " << num_outstanding_ops_ ;
//        if (num_outstanding_ops_.load(std::memory_order_relaxed) == 0) {
//          finished = true;
//        }
//        VLOG(0) << std::this_thread::get_id() << ", ExecutorState::Process: finished: " << finished;
//        if (finished) {
//          done_cb_(Status());
//          //done_cb_(Status());
//        } 
//      }
//      
//      return;
//    }
//    //~wxf

    const Node* node = tagged_node.node;
    FrameState* input_frame = tagged_node.input_frame;
    const int64 input_iter = tagged_node.input_iter;
    const int id = node->id();
    const NodeItem& item = *gview.node(id);

    // TODO(misard) Replace with a finer-grain enabling flag once we
    // add better optional debugging support.
    if (vlog_ && VLOG_IS_ON(1)) {
      mutex_lock l(input_frame->mu);
      input_frame->GetIteration(input_iter)->mark_started(item.pending_id);
    }

    // Set the device_context for this node id, if it exists.
    if (id < device_context_map_.size()) {
      params.op_device_context = device_context_map_[id];
    }

    params.track_allocations = false;
    stats = nullptr;
    if (stats_collector_ && !tagged_node.is_dead) {
      stats = stats_collector_->CreateNodeExecStats(node);
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

    Entry* input_tensors = GetInputTensors(input_frame, input_iter);
    Entry* first_input = input_tensors + item.input_start;
    outputs.clear();

    TensorReferenceVector accessed_tensors;
    DeviceContext* device_context = nullptr;

    // Only execute this node if it is not dead or it is a send/recv
    // transfer node. For transfer nodes, we need to propagate the "dead"
    // bit even when the node is dead.
    bool launched_asynchronously = false;
    if (tagged_node.is_dead && !IsTransferNode(node)) {
      outputs.resize(item.num_outputs);
    } else {
      // Prepares inputs.
      bool is_input_dead = false;

      // wxf
      if (node->name() == "matmul02") {
        //VLOG(0) << "Before while(!matmul01_is_ok)";
        // wait until matmul 01 is ok.
        while(!matmul01_is_ok.load()){
          //VLOG(0) << matmul01_is_ok.load() << matmul02_is_ok.load();
          // loop waiting until matmul 01 is ok
        }
        //VLOG(0) << "After while(!matmul01_is_ok)";

        first_input = &(reuse_entry_inputs[0]);
        Entry* temp = first_input + 1;
        temp = &(reuse_entry_inputs[1]);
        //matmul02_is_ok.store(true);
        //VLOG(0) << "After matmul02_is_ok = true";
        matmul01_is_ok.store(false);
      }
      //~wxf

      s = PrepareInputs(item, first_input, &inputs, &input_device_contexts,
                        &input_alloc_attrs, &is_input_dead);

      // wxf
      if (node->name() == "matmul01") {
        for (int i = 0; i < item.num_inputs; ++i) {
          Entry temp;
          temp = *(first_input + i);
          reuse_entry_inputs.push_back(temp);
        }

        matmul01_is_ok.store(true);
        //VLOG(0) << "Before while(!matmul02_is_ok)";
        //while(!matmul02_is_ok.load()){
        //  //VLOG(0) << matmul01_is_ok.load() << matmul02_is_ok.load();
        //}
        //VLOG(0) << "After while(!matmul02_is_ok)";

        //matmul02_is_ok.store(false);
      }
      //~wxf

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
      }

      // Set up compute params.
      OpKernel* op_kernel = item.kernel;
      params.op_kernel = op_kernel;
      params.frame_iter = FrameAndIter(input_frame->frame_id, input_iter);
      params.is_input_dead = is_input_dead;
      params.output_attr_array = item.output_attrs();
      params.forward_from_array = item.forward_from();

      if (item.kernel_is_async) {
        // Asynchronous computes.
        AsyncOpKernel* async = item.kernel->AsAsync();
        DCHECK(async != nullptr);
        launched_asynchronously = true;
        AsyncState* state =
            new AsyncState(params, tagged_node, &item, first_input, stats);

        auto done = [this, state]() {
          Device* device = impl_->params_.device;
          NodeExecStatsInterface* stats = state->stats;  // Shorthand
          Entry* first_input = state->first_input;       // Shorthand

          nodestats::SetOpEnd(stats);
          EntryVector outputs;
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
          FrameState* input_frame = state->tagged_node.input_frame;
          const int64 input_iter = state->tagged_node.input_iter;
          const int id = state->tagged_node.node->id();
          MaybeMarkCompleted(input_frame, input_iter, id);
          TaggedNodeSeq ready;
          if (s.ok()) {
            PropagateOutputs(state->tagged_node, state->item, &outputs, &ready);
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
          const bool completed =
              NodeDone(s, state->item->node, ready, stats, nullptr);
          delete state;
          if (completed) ScheduleFinish();
        };
        nodestats::SetOpStart(stats);
        device->ComputeAsync(async, &state->ctx, done);
      } else {
        // Synchronous computes.
        OpKernelContext ctx(&params, item.num_outputs);
        nodestats::SetOpStart(stats);

        if (TF_PREDICT_FALSE(
                MightTrace(item, event_collector_, trace_using_annotations_))) {
          const string& op_name = op_kernel->name();
          tracing::ScopedRegion region(tracing::EventCategory::kCompute,
                                       op_name);
          if (trace_using_annotations_) {
            // The OpKernel may create child activities (such as GPU kernel
            // launches), so use a `ScopedAnnotation` to relate these activities
            // in the trace.
            tracing::ScopedAnnotation activity(
                op_name, strings::StrCat(op_kernel->type_string(),
                                         "#id=", step_id_, "#"));
            device->Compute(op_kernel, &ctx);
          } else {
            // Use the cheaper `ScopedActivity` to trace just the OpKernel
            // execution.
            tracing::ScopedActivity activity(
                op_name,
                strings::StrCat(op_kernel->type_string(), "#id=", step_id_,
                                "#"),
                item.kernel->IsExpensive());
            device->Compute(op_kernel, &ctx);
          }
        } else {
          // In the common case, avoid creating any tracing objects.
          if (op_kernel->IsExpensive()) {
            KernelTimer timer;
            device->Compute(op_kernel, &ctx);
            op_kernel->UpdateCostEstimate(timer.ElapsedCycles());
          } else {
            device->Compute(op_kernel, &ctx);
          }
        }

        nodestats::SetOpEnd(stats);
        s = ProcessOutputs(item, &ctx, &outputs, stats);
        if (s.ok() && impl_->device_record_tensor_accesses_) {
          // Get the list of all tensors accessed during the execution
          ctx.retrieve_accessed_tensors(&accessed_tensors);
          device_context = ctx.op_device_context();
        }
        nodestats::SetMemory(stats, &ctx);
      }
    }

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

//      // wxf
//      if (node->name() == "matmul01") {
//        // Don't delete matmul's inputs
//        //reuse_entry_inputs.clear();
//      } else {
//        // Clears inputs.
//        const int num_inputs = item.num_inputs;
//        for (int i = 0; i < num_inputs; ++i) {
//          (first_input + i)->ClearVal();
//        }
//      }

      if (node->name() == "matmul02") {
        reuse_entry_inputs.clear();
      }
      //~wxf 

      MaybeMarkCompleted(input_frame, input_iter, id);
      // Propagates outputs.
      if (s.ok()) {
        PropagateOutputs(tagged_node, &item, &outputs, &ready);
      }
      outputs.clear();
      if (!accessed_tensors.empty()) {
        nodestats::SetReferencedTensors(stats, accessed_tensors);
        // device_context is set above in synchronous computes
        device->ConsumeListOfAccessedTensors(device_context, accessed_tensors);
      }
      if (stats) {
        scheduled_nsec = nodestats::NowInNsec();
      }
      // Postprocess.
      completed = NodeDone(s, item.node, ready, stats, &inline_ready);
    }
  }  // while !inline_ready.empty()

  // wxf
  //VLOG(0) << std::this_thread::get_id() << ", Process(), Do ScheduleFinish? : completed: " << completed;

  // This thread of computation is done if completed = true.
  if (completed) ScheduleFinish();

  // wxf
//  VLOG(0) << std::this_thread::get_id() << ", Process(), End!";
}

Status ExecutorState::PrepareInputs(const NodeItem& item, Entry* first_input,
                                    TensorValueVec* inputs,
                                    DeviceContextVec* input_device_contexts,
                                    AllocatorAttributeVec* input_alloc_attrs,
                                    bool* is_input_dead) {
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
    TensorValue* inp = &(*inputs)[i];

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
      inp->tensor = entry->val.get();
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
}

Status ExecutorState::ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                                     EntryVector* outputs,
                                     NodeExecStatsInterface* stats) {
  const Node* node = item.node;
  DCHECK_EQ(0, outputs->size());
  outputs->resize(item.num_outputs);

  Status s = ctx->status();
  if (!s.ok()) {
    s = AttachDef(s, item.kernel->def());
    // TODO(misard) Replace with a finer-grain enabling flag once we
    // add better optional debugging support.
    if (vlog_ && VLOG_IS_ON(1)) {
      LOG(WARNING) << this << " Compute status: " << s;
      DumpState();
    }
    if (s.code() == error::RESOURCE_EXHAUSTED) {
      if (stats_collector_) {
        string err = stats_collector_->ReportAllocsOnResourceExhausted(
            s.error_message());
        s = Status(s.code(), strings::StrCat(s.error_message(), err));
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
  }

  // Get the device_context for this node id, if it exists.
  DeviceContext* device_context = nullptr;
  if (node->id() < device_context_map_.size()) {
    device_context = device_context_map_[node->id()];
  }

  for (int i = 0; i < item.num_outputs; ++i) {
    const TensorValue val = ctx->release_output(i);
    if (val.tensor == nullptr) {
      // Unless it's a Switch or a Recv, the node must produce a
      // tensor value at i-th output.
      if (!IsSwitch(node) && !IsRecv(node)) {
        s.Update(errors::Internal("Missing ", i, "-th output from ",
                                  FormatNodeForError(*node)));
      }
    } else {
      Entry* out = &((*outputs)[i]);

      // Set the device context of the output entry.
      out->device_context = device_context;

      // Set the allocator attributes of the output entry.
      out->alloc_attr = ctx->output_alloc_attr(i);

      // Sanity check of output tensor types.
      DataType dtype;
      if (val.is_ref()) {
        tf_shared_lock ml(*val.mutex_if_ref);
        dtype = MakeRefType(val->dtype());
      } else {
        dtype = val->dtype();
      }
      if (dtype == item.output_type(i)) {
        if (stats && val.tensor->IsInitialized()) {
          nodestats::SetOutput(stats, i, val.tensor);
        }
        if (val.is_ref()) {
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
          }
        } else {
          // NOTE that std::move is used here, so val.tensor goes to
          // uninitialized state (val.tensor->IsInitialized return false).
          DCHECK(!out->val_field_is_set);
          out->has_value = true;
          out->val_field_is_set = true;
          out->val.Init(std::move(*val.tensor));
          if (log_memory_) {
            LogMemory::RecordTensorOutput(ctx->op_kernel().name(),
                                          ctx->step_id(), i, *out->val);
          }
        }
      } else {
        s.Update(errors::Internal("Output ", i, " of type ",
                                  DataTypeString(dtype),
                                  " does not match declared output type ",
                                  DataTypeString(item.output_type(i)),
                                  " for node ", FormatNodeForError(*node)));
      }
    }
    if (!val.is_ref()) {
      // If OpKernelContext returns outputs via pass-by-value, we
      // don't need this trouble.
      delete val.tensor;
    }
  }
  return s;
}

void ExecutorState::PropagateOutputs(const TaggedNode& tagged_node,
                                     const NodeItem* item, EntryVector* outputs,
                                     TaggedNodeSeq* ready) {
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
  FrameState* input_frame = tagged_node.input_frame;
  const int64 input_iter = tagged_node.input_iter;
  const bool is_dead = tagged_node.is_dead;

  // Propagates outputs along out edges, and puts newly ready nodes
  // into the ready queue.
  ready->clear();
  bool is_frame_done = false;
  FrameState* output_frame = input_frame;
  int64 output_iter = input_iter;

  if (!item->is_enter_exit_or_next_iter) {
    // Fast path for nodes types that don't need special handling
    DCHECK_EQ(input_frame, output_frame);
    // Normal path for most nodes
    mutex_lock l(input_frame->mu);
    output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
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
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
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
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
      }
      is_frame_done = input_frame->DecrementOutstandingOps(&impl_->gview_,
                                                           input_iter, ready);
    }
  } else {
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
      output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
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

bool ExecutorState::NodeDone(const Status& s, const Node* node,
                             const TaggedNodeSeq& ready,
                             NodeExecStatsInterface* stats,
                             TaggedNodeReadyQueue* inline_ready) {
  nodestats::SetAllEnd(stats);
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
      status_ = s;
    }
  }

  if (abort_run) {
    TRACEPRINTF("StartAbort: %s", s.ToString().c_str());
    if (rendezvous_) {
      rendezvous_->StartAbort(s);
    }
    if (collective_executor_) {
      collective_executor_->StartAbort(s);
    }
    if (cancellation_manager_) {
      cancellation_manager_->StartCancel();
    }
  }

  bool completed = false;

  // wxf
  // study how to use cancellation_manager_ StartCancel to abort 
  // current node execution
  // 1. If low priority ExecutorState, then we need to consider abort this node execution.
  //    (short-circuit condition) give the Condition 2 an assumption: it is low priority executor
  // 2. If the low priority is executing on HPU and high priority tasks exist, must abort.
  //    (short-circuit condition) Otherwise, it will not skip executing the current node 
  //    and abort the left graph nodes.
  // 3. Only high and low co-exist, low will abort. Only high or low will not go into 
  //    this branch. 
  // If the data is transferred, only abort the 1st time. Then don't 
  // go into this branch and abort again and again!
  if ( (executor_state_priority_ == ExecutorStatePriority::LOW &&
        device_type_executing_on_ == "HPU" &&
        direct_sessions_manager_->high_priority_direct_session_count_.\
          load(std::memory_order_relaxed)) || 
        (catched_ && (executor_state_priority_ == ExecutorStatePriority::LOW)) 
     ){
    catched_ = true;
    VLOG(0) << std::this_thread::get_id() << ", *** Low priority abort branch Debugging message.";
    VLOG(0) << std::this_thread::get_id() << ", 1. ExecutorState::NodeDone: :num_outstanding_ops_ " << num_outstanding_ops_;

    if (rendezvous_) {
      rendezvous_->StartAbort(Status(tensorflow::error::SESS_RUN_CANCELLED, "wxf cancelled"));
    }
    if (collective_executor_) {
      collective_executor_->StartAbort(Status(tensorflow::error::SESS_RUN_CANCELLED, "wxf cancelled"));
    }
    if (cancellation_manager_) {
      cancellation_manager_->StartCancel();
    }

    if (inline_ready && !inline_ready->empty()) {
//      VLOG(0) << std::this_thread::get_id() << ", inline_ready->size()" << inline_ready->size();
      num_outstanding_ops_.fetch_sub(inline_ready->size() + 1); 
      // pop all inline_ready nodes
      inline_ready->clear();
    } else {
      num_outstanding_ops_.fetch_sub(1);
    }

//    VLOG(0) << std::this_thread::get_id() << ", 2. ExecutorState::NodeDone: :num_outstanding_ops_ " << num_outstanding_ops_;

    if (num_outstanding_ops_.load(std::memory_order_relaxed) == 0) {
      completed = true;
    }

    return completed;
  }
  //~wxf

  const size_t ready_size = ready.size();
  if (ready_size == 0 || !s.ok()) {
    // wxf
//    VLOG(0) << std::this_thread::get_id() << ", 1. ExecutorState::NodeDone: num_outstanding_ops_: " << num_outstanding_ops_ ;

    completed = (num_outstanding_ops_.fetch_sub(1) == 1);

    // wxf
//    VLOG(0) << std::this_thread::get_id() << ", 2. ExecutorState::NodeDone: num_outstanding_ops_: " << num_outstanding_ops_ ;

  } else if (ready_size > 1) {
    // wxf
//    VLOG(0) << std::this_thread::get_id() << ", 3. ExecutorState::NodeDone: num_outstanding_ops_: " << num_outstanding_ops_ ;
    
    num_outstanding_ops_.fetch_add(ready_size - 1, std::memory_order_relaxed);

    // wxf
//    VLOG(0) << std::this_thread::get_id() << ", 4. ExecutorState::NodeDone: num_outstanding_ops_: " << num_outstanding_ops_ ;
  }

  // Schedule the ready nodes in 'ready'.
  if (s.ok()) {
    ScheduleReady(ready, inline_ready);
  }

//  // wxf
//  // python step id ==3 //  0, 1, 2, "3"
//  if (step_id_ == 11 && node->id() == 0) {
//    // simulate high task enters
//    direct_sessions_manager_->high_priority_direct_session_count_.fetch_add(1, std::memory_order_relaxed);
//  } 
// 
//  if (step_id_ == 11 && node->id() == 1530) {
//    // simulate high task exit 
//    direct_sessions_manager_->high_priority_direct_session_count_.fetch_sub(1, std::memory_order_relaxed);
//  } 
//  // test passed
//  //~wxf

  return completed;
}


void ExecutorState::ScheduleReady(const TaggedNodeSeq& ready,
                                  TaggedNodeReadyQueue* inline_ready) {

  /////////////////////////////////////////////////////
  // wxf
  //bool stop = false;
  //ReadBoolFromEnvVar("TF_TO_STOP_SCHED", false, &stop);
  //if (stop) {
  //	  while(1);
  //}
  /////////////////////////////////////////////////////

  // Inquire the current ExecutorImpl's priority, if it is low(1),
  // further ask ExecutorsPoolManager and wait until there is no
  // high priority executors (2).
  // For default priority(0) generated from ThreadPool, we don't stop them.
  // Those Op Nodes are Dataset Op, RemoteCallOp, IfOp etc.
//  if (impl_->executors_pool_manager_->InquirePriorityByExecutorImpl(impl_) == 1) {
	// -- debug
	//std::cout << "ScheduleReady Low Pirority" << std::endl;
	//std::cout << impl_->executors_pool_manager_->high_priority_executors_ref_count_.load(std::memory_order_relaxed) << std::endl;
	// ~~ debug

  if (ready.empty()) return;

  int64 scheduled_nsec = 0;
  if (stats_collector_) {
    scheduled_nsec = nodestats::NowInNsec();
  }

  if (inline_ready == nullptr) {
    // Schedule to run all the ready ops in thread pool.
    for (auto& tagged_node : ready) {
      runner_([=]() { Process(tagged_node, scheduled_nsec); });
    }
    return;
  }

  const GraphView& gview = impl_->gview_;
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
        runner_(std::bind(&ExecutorState::Process, this, *curr_expensive_node,
                          scheduled_nsec));
      }
      curr_expensive_node = &tagged_node;
    }
  }
  if (curr_expensive_node) {
    if (inline_ready->empty()) {
      // Tail recursion optimization
      inline_ready->push_back(*curr_expensive_node);
    } else {
      // There are inline nodes to run already. We dispatch this expensive
      // node to other thread.
      runner_(std::bind(&ExecutorState::Process, this, *curr_expensive_node,
                        scheduled_nsec));
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
  // wxf 
  //VLOG(0) << std::this_thread::get_id() << ", ExecutorState::Finish()";
  //~wxf
  mu_.lock();
  auto status = status_;
  auto done_cb = std::move(done_cb_);
  auto runner = std::move(runner_);
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
    runner([=]() { done_cb(status); });
    return;
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

void ExecutorState::FrameState::ActivateNodes(const NodeItem* item,
                                              const bool is_dead, int64 iter,
                                              EntryVector* outputs,
                                              TaggedNodeSeq* ready) {
  const GraphView& gview = executor->gview_;
  IterationState* iter_state = GetIteration(iter);
  const size_t num_output_edges = item->num_output_edges;
  const EdgeInfo* edges = item->output_edge_list();
  Entry* input_tensors = iter_state->input_tensors;
  for (size_t out_index = 0; out_index < num_output_edges; out_index++) {
    const EdgeInfo& e = edges[out_index];
    const int dst_id = e.dst_id;
    const NodeItem* dst_item = gview.node(dst_id);
    const PendingCounts::Handle dst_pending_id = dst_item->pending_id;
    const int src_slot = e.output_slot;

    // TODO(yuanbyu): We don't need this if we require the subgraph
    // given to an executor not to contain a sink node.
    if (dst_item->is_sink) continue;

    bool dst_dead = false;
    bool dst_ready = false;
    // True iff this input for dst is needed. We only set this input for
    // dst if this flag is true. This is needed to make the thread safety
    // analysis happy.
    const bool is_control_edge = (src_slot == Graph::kControlSlot);
    bool dst_need_input = !is_control_edge;
    if (dst_item->is_merge) {
      // A merge node is ready if all control inputs have arrived and either
      // a) a live data input becomes available or b) all data inputs are
      // dead. For Merge, pending's LSB is set iff a live data input has
      // arrived.
      if (is_control_edge) {
        iter_state->decrement_pending(dst_pending_id, 2);
        int count = iter_state->pending(dst_pending_id);
        int dead_cnt = iter_state->dead_count(dst_pending_id);
        dst_dead = (dead_cnt == dst_item->num_inputs);
        dst_ready = (count == 0) || ((count == 1) && dst_dead);
      } else {
        if ((*outputs)[src_slot].has_value) {
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
      ready->emplace_back(dst_item->node, this, iter, dst_dead);
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

void ExecutorImpl::RunAsync(const Args& args, DoneCallback done) {
  // wxf: TODO
  // Refactor the following code
  // 1. create ExecutorState instance
  // 2. add the ExecutorState instance into the ExecutorStatesManager's pool
  // 3. call RunAsync
  (new ExecutorState(args, this))->RunAsync(std::move(done));
}


//}  // namespace


// ----------------------------------------------------------------------------

// tid_execution_priority_map_ stores tid and its executors' priority
// Note: syntax detour from c_api.cc
std::unordered_map<std::thread::id, int> tid_execution_priority_map_;


void ExecutorsPoolManager::AddExecutorAndPriority(ExecutorImpl** executor){
	// lock before modifing the shared variable.
	add_mu_.lock();
	std::thread::id tid = std::this_thread::get_id();
	auto iter = tid_execution_priority_map_.find(tid);
	if (iter == tid_execution_priority_map_.end()){
	  // If we cannot find this tid, it means that the priority is not set,
	  // i.e. 0 by default for ThreadPool threads.
	  executor_priority_map_.insert({*executor, 0});

	  // -- debug
	  //std::cout << "Add::ThreadPool Pirority (0)" << std::endl;
	  // ~~ debug
	}else {
	  // add this Executor* executor with its priority from the current
	  // client primary thread id.
	  int execution_priority = iter->second;
	  executor_priority_map_.insert({*executor, execution_priority});

    /// Low Priority ExecutorImpl instance
	  if (execution_priority == 1){
		  low_priority_executors_ref_count_.fetch_add(1, std::memory_order_relaxed);

	    // -- debug
      //std::cout << "Add::Low Pirority (1), #Low: " <<
		  //  low_priority_executors_ref_count_.load(std::memory_order_relaxed) << std::endl;
	    // ~~ debug
	  }

	  // Since the high priority is defined as 2 and low is 1, we counts how
	  // many high priority ExecutorImpl instances.
	  if (execution_priority == 2) {
		  high_priority_executors_ref_count_.fetch_add(1, std::memory_order_relaxed);
      // -- debug
      //std::cout << "Add::High Pirority (2), #High: " << 
      //  high_priority_executors_ref_count_.load(std::memory_order_relaxed) << std::endl;
      // ~~ debug
	  }
	}
	add_mu_.unlock();
}

void ExecutorsPoolManager::DeleteExecutor(ExecutorImpl* executor){
  delete_mu_.lock();
  int execution_priority = executor_priority_map_[executor];
  // Since the high priority is defined as 2 and low is 1, we decrease the
  // number of high priority ExecutorImpl instances.
  if (execution_priority == 1) {
	  low_priority_executors_ref_count_.fetch_sub(1, std::memory_order_relaxed);
	  // -- debug
	  //std::cout << "Delete::Low Pirority (1)" << std::endl;
	  //std::cout << high_priority_executors_ref_count_.load(std::memory_order_relaxed) << std::endl;
	  // ~~ debug
  }
  if (execution_priority == 2) {
	  high_priority_executors_ref_count_.fetch_sub(1, std::memory_order_relaxed);
	  // -- debug
    //std::cout << "Delete::High Pirority (2)" << std::endl;
	  //std::cout << high_priority_executors_ref_count_.load(std::memory_order_relaxed) << std::endl;
	  // ~~ debug
  }

  // -- debug
//  if (execution_priority == 0){
//  	std::cout << "Delete::ThreadPool Pirority (0)" << std::endl;
//	  std::cout << high_priority_executors_ref_count_.load(std::memory_order_relaxed) << std::endl;
//  }
  // ~~ debug

  // Delete the ExecutorImpl instance pointer from the map.
  executor_priority_map_.erase(executor);
  delete_mu_.unlock();
}
  
int ExecutorsPoolManager::InquirePriorityByExecutorImpl(const ExecutorImpl* executor) {
  //read_mu_.lock();
  //int priority = executor_priority_map_[const_cast<ExecutorImpl*>(executor)];
  //read_mu_.unlock();
  //return priority;

  return executor_priority_map_[const_cast<ExecutorImpl*>(executor)];

  //ExecutorImpl* e = const_cast<ExecutorImpl*>(executor);
  //auto search = executor_priority_map_.find(e);
  //if (search != executor_priority_map_.end()) {
  //  return search->second;
  //} else {
  //	//errors::NotFound("No ExecutorImpl* found in the ExecutorsPoolManager::executor_priority_map_ map");
  //	return -1;
  //}
}


// ----------------------------------------------------------------------------

// Before NewLocalExecutor, I need to create a ExecutorsManager
Status NewLocalExecutor(const LocalExecutorParams& params,
                        std::unique_ptr<const Graph> graph,
                        Executor** executor) {

  ExecutorImpl* impl = new ExecutorImpl(params, std::move(graph));

  // Add ExecutorImpl pointer into the executors_pool_manager_
  ExecutorImpl::executors_pool_manager_->AddExecutorAndPriority(&impl);

  const Status s = impl->Initialize();
  if (s.ok()) {
    *executor = impl;
  } else {
    delete impl;
  }
  return s;
}

Status CreateNonCachedKernel(Device* device, FunctionLibraryRuntime* flib,
                             const NodeDef& ndef, int graph_def_version,
                             OpKernel** kernel) {
  const auto device_type = DeviceType(device->attributes().device_type());
  auto allocator = device->GetAllocator(AllocatorAttributes());
  return CreateOpKernel(device_type, device, allocator, flib, ndef,
                        graph_def_version, kernel);
}

void DeleteNonCachedKernel(OpKernel* kernel) { delete kernel; }

namespace {

class DefaultExecutorRegistrar {
 public:
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
