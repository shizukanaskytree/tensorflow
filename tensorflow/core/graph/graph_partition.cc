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

#include "tensorflow/core/graph/graph_partition.h"

#include <deque>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

namespace {

inline bool IsMerge(const NodeDef& node_def) {
  return node_def.op() == "Merge" || node_def.op() == "RefMerge";
}

inline bool IsNextIteration(const NodeDef& node_def) {
  return node_def.op() == "NextIteration" ||
         node_def.op() == "RefNextIteration";
}

struct DupRecvKey {
  int src_node_id;           // Edge's src node id
  int src_output_slot;       // Edge's src node output slot
  GraphDef* dst_graph;       // Edge's dst node is in this subgraph
  bool recv_output_on_host;  // The output of recv is on host

  template <typename H>
  friend H AbslHashValue(H h, const DupRecvKey& c) {
    return H::combine(std::move(h), c.src_node_id, c.src_output_slot,
                      reinterpret_cast<std::uintptr_t>(c.dst_graph),
                      c.recv_output_on_host);
  }

  friend bool operator==(const DupRecvKey& x, const DupRecvKey& y) {
    return (x.src_node_id == y.src_node_id) &&
           (x.src_output_slot == y.src_output_slot) &&
           (x.dst_graph == y.dst_graph) &&
           (x.recv_output_on_host == y.recv_output_on_host);
  }
};

// struct used to store the recvs, so that start times can be properly updated
struct RecvInfo {
  NodeDef* recv;
  NodeDef* real_recv;
  int64 start_time;
};

typedef absl::flat_hash_map<DupRecvKey, RecvInfo> DupRecvTable;

// A map used to store memory types for the inputs/outputs of every node.
// The key is a pair of ints consisting of a node id and input/output index.
// TODO(power): migrate back to std::pair when absl::Hash is fixed for MSVC.
struct NodePort {
  int node_id;
  int index;

  friend bool operator==(const NodePort& x, const NodePort& y) {
    return x.node_id == y.node_id && x.index == y.index;
  }

  template <typename H>
  friend H AbslHashValue(H h, const NodePort& c) {
    return H::combine(std::move(h), c.node_id, c.index);
  }
};

typedef absl::flat_hash_map<NodePort, MemoryType> MemoryTypeMap;

// We collect the following information about the graph before performing
// graph partitioning.
// QQQ. GraphInfo 的用途是什么？
// AAA.
struct GraphInfo {
  // info->device_types.resize(g.num_node_ids(), DEVICE_CPU);
  std::vector<DeviceType> device_types;
  MemoryTypeMap input_types;
  MemoryTypeMap output_types;
  std::vector<ControlFlowInfo> cf_info;
};
// 1.
// GraphInfo 数据结构
// tensorflow/core/graph/graph_partition.cc:108:
// struct GraphInfo
// - device_types: std::vector<DeviceType>
// - input_types: MemoryTypeMap
// - output_types: MemoryTypeMap
// - cf_info: std::vector<ControlFlowInfo>

// 2.
// class DeviceType: tensorflow/core/framework/types.h
//  - string type_;
//    const char* const DEVICE_CPU, DEVICE_GPU, DEVICE_SYCL

// 3.
// struct ControlFlowInfo 数据结构
// tensorflow/core/graph/control_flow.h
// - frame: const Node*
// - parent_frame: const Node*
// - frame_name: string

// 4.
// MemoryTypeMap 数据结构
// typedef absl::flat_hash_map<NodePort, MemoryType> MemoryTypeMap;

// 5.
// struct NodePort 数据结构
// - node_id: int
// - index: int

// 6.
// MemoryType 数据结构
// tensorflow/core/framework/types.h:47:enum MemoryType
//
// MemoryType is used to describe whether input or output Tensors of
// an OpKernel should reside in "Host memory" (e.g., CPU memory) or
// "Device" Memory (CPU memory for CPU devices, GPU memory for GPU
// devices).
// enum MemoryType {
//   DEVICE_MEMORY = 0,
//   HOST_MEMORY = 1,
//};


DataType EdgeType(const Edge* e) {
  if (e->IsControlEdge()) {
    return DT_FLOAT;
  } else {
    return e->dst()->input_type(e->dst_input());
  }
}

// Return true iff we need to add the same device send/recv for 'edge'.
bool NeedSameDeviceSendRecv(
  const Edge* edge,  // input
  const GraphInfo& info) { // input

  if (edge->IsControlEdge()) {
    return false;
  }

  const Node* src = edge->src();
  const Node* dst = edge->dst();

  // need same device send recv node 的前提是 他们属于同一种 device 类型
  if (src->assigned_device_name() == dst->assigned_device_name()) {

    int src_port = edge->src_output();
    int dst_port = edge->dst_input();

    if (info.device_types[src->id()] != DEVICE_CPU) {

      auto src_it = info.output_types.find({src->id(), src_port});
      DCHECK(src_it != info.output_types.end());

      auto dst_it = info.input_types.find({dst->id(), dst_port});
      DCHECK(dst_it != info.input_types.end());

      return src_it->second != dst_it->second;
    }

  }
  return false;
}

// Return true iff (dst, dst_input) is specified on host memory.
bool IsDstInputOnHost(const Edge* edge, const GraphInfo& info) {
  const Node* dst = edge->dst();
  int dst_port = edge->dst_input();
  if (info.device_types[dst->id()] != DEVICE_CPU) {
    if (edge->IsControlEdge()) return false;
    auto dst_it = info.input_types.find({dst->id(), dst_port});
    DCHECK(dst_it != info.input_types.end());
    return dst_it->second == HOST_MEMORY;
  }
  return true;
}




/*

   +----------+
   |0         |
   |          |
   |1       0 +--------+
   |          |        |
   |2         |        |
   +----------+        |
                       |
                       |                +----------+
                       v--------------->|0         |
                                        |          |
                       +--------------->|1  dst    |
                                        |          |
                       +--------------->|2         |
                                        +----------+

                                   inputs edges
*/

// Add an input to dst that comes from the "src_slot" output of the
// node named by "src_name".
void AddInput(
  NodeDef* dst,
  StringPiece src_name,
  int src_slot) {

  if (src_slot == Graph::kControlSlot) {

    dst->add_input(strings::StrCat("^", src_name));

  } else if (src_slot == 0) {

    dst->add_input(src_name.data(), src_name.size());

  } else {

    dst->add_input(strings::StrCat(src_name, ":", src_slot));

  }

}

// Add a control edge from each input to each recv.
void AddReadControl(const std::vector<NodeDef*>& recvs,
                    const std::vector<string>& inputs) {
  for (NodeDef* recv : recvs) {
    for (const string& input : inputs) {
      recv->add_input(strings::StrCat("^", input));
    }
  }
}



void SetSendRecvAttrs(
  const PartitionOptions& opts,
  const Edge* edge,
  NodeDefBuilder* builder)
{
  builder->Attr("tensor_name",
                strings::StrCat("edge_", edge->id(), "_", edge->src()->name()));
  builder->Attr("send_device", edge->src()->assigned_device_name());
  builder->Attr("send_device_incarnation",
                static_cast<int64>(
                    opts.get_incarnation(edge->src()->assigned_device_name())));
  builder->Attr("recv_device", edge->dst()->assigned_device_name());
  builder->Attr("client_terminated", false);
}


////////////////////////////////////////////////////////////////////////

// AddSend 概述:
// 往 GraphDef* gdef 里面增加一个 send 节点。
// 这个 send 节点的 起始点 叫做 send_from 节点。
NodeDef* AddSend(
  const PartitionOptions& opts, // input
  const GraphInfo& g_info,  // input
  GraphDef* gdef, // input
  const Edge* edge, // input
  NodeDefBuilder::NodeOut send_from, // input
  int64 start_time, // input
  Status* status) // output
{

  /*

  p edge->DebugString()
  $39 = "[id=8 y/RandomStandardNormal:0 -> z:1]"


  ptype opts
  type = const struct tensorflow::PartitionOptions {
    NodeToLocFunc node_to_loc;
    NewNameFunc new_name;
    static const tensorflow::uint64 kIllegalIncarnation;
    GetIncarnationFunc get_incarnation;
    const tensorflow::FunctionLibraryDefinition *flib_def;
    bool control_flow_added;
    ShouldCastFunc should_cast;
    bool scheduling_for_recvs;
    bool need_to_record_start_times;
    std::vector<tensorflow::gtl::IntType<tensorflow::Microseconds_tag_, long long>> start_times;

    typedef std::function<std::basic_string<char, std::char_traits<char>, std::allocator<char> >(const tensorflow::Node*)> NodeToLocFunc;
    typedef std::function<std::basic_string<char, std::char_traits<char>, std::allocator<char> >(const std::basic_string<char, std::char_traits<char>, std::allocator<char> >&)> NewNameFunc;
    typedef std::function<long long unsigned int(const std::basic_string<char, std::char_traits<char>, std::allocator<char> >&)> GetIncarnationFunc;
    typedef std::function<tensorflow::DataType(const tensorflow::Edge*)> ShouldCastFunc;
  } &


  */

  const DataType dtype = send_from.data_type;
  const DataType cast_dtype = opts.should_cast ? opts.should_cast(edge) : dtype;

  const Node* src = edge->src();
  const int src_port = edge->src_output();

  // host_memory = true iff we need to use HostSend/HostCast.
  bool host_memory = false;

  if (!edge->IsControlEdge()) {

    auto src_it = g_info.output_types.find({src->id(), src_port});
    DCHECK(src_it != g_info.output_types.end());

    host_memory = (src_it->second == HOST_MEMORY);
  }

  // Add a cast node that casts dtype to cast_dtype.
  // NOTE(yuanbyu): Only cast for cross-device send/recv.
  if (dtype != cast_dtype && !NeedSameDeviceSendRecv(edge, g_info)) {
    const string cast_op = (host_memory) ? "_HostCast" : "Cast";

    NodeDefBuilder cast_builder(opts.new_name(src->name()), cast_op,
                                NodeDebugInfo(*src));

    cast_builder.Device(src->assigned_device_name()).Input(send_from);

    if (opts.scheduling_for_recvs) {
      cast_builder.Attr("_start_time", start_time);
    }

    cast_builder.Attr("DstT", cast_dtype);

    if (cast_dtype == DT_BFLOAT16) {
      // the below attribute specifies that the cast to bfloat16 should use
      // truncation. This is needed to retain legacy behavior when we change
      // the default bfloat16 casts to use rounding instead of truncation
      cast_builder.Attr("Truncate", true);
    }

    NodeDef* cast = gdef->add_node();

    *status = cast_builder.Finalize(cast);

    if (!status->ok()) return nullptr;

    // Connect the Send op to the cast.
    send_from.Reset(cast->name(), 0, cast_dtype);
  }


  // Add the send node.
  const string send_op = (host_memory) ? "_HostSend" : "_Send";
  // p send_op, $42 = "_Send"

  NodeDefBuilder send_builder(
    opts.new_name(src->name()),
    send_op,
    NodeDebugInfo(*src));

  // 详细描述了 send node device, receive node device.
  SetSendRecvAttrs(opts, edge, &send_builder);

  send_builder.Device(src->assigned_device_name()).Input(send_from);

  if (opts.scheduling_for_recvs) {
    send_builder.Attr("_start_time", start_time);
  }

  // 往子图里构造一个新的节点，最后通过 Finalize 把 send node 构造好。
  NodeDef* send = gdef->add_node();

  *status = send_builder.Finalize(send);

  return send;
}



NodeDef* AddRecv(const PartitionOptions& opts, const GraphInfo& g_info,
                 GraphDef* gdef, const Edge* edge, NodeDef** real_recv,
                 Status* status) {
  const DataType dtype = EdgeType(edge);
  const Node* src = edge->src();
  const Node* dst = edge->dst();
  const int dst_port = edge->dst_input();
  DataType cast_dtype = dtype;

  // NOTE(yuanbyu): Only cast for cross-device send/recv.
  if (opts.should_cast && !NeedSameDeviceSendRecv(edge, g_info)) {
    cast_dtype = opts.should_cast(edge);
  }

  // host_memory = true iff we need to use HostRecv/HostCast.
  bool host_memory = false;
  if (!edge->IsControlEdge()) {
    auto dst_it = g_info.input_types.find({dst->id(), dst_port});
    DCHECK(dst_it != g_info.input_types.end());
    host_memory = (dst_it->second == HOST_MEMORY);
  }

  // Add the recv node.
  const string recv_op = (host_memory) ? "_HostRecv" : "_Recv";
  NodeDefBuilder recv_builder(opts.new_name(src->name()), recv_op,
                              NodeDebugInfo(*src));
  SetSendRecvAttrs(opts, edge, &recv_builder);
  recv_builder.Device(dst->assigned_device_name())
      .Attr("tensor_type", cast_dtype);
  NodeDef* recv = gdef->add_node();
  *status = recv_builder.Finalize(recv);
  if (!status->ok()) return nullptr;
  *real_recv = recv;

  // Add the cast node (from cast_dtype to dtype) or an Identity node.
  if (dtype != cast_dtype) {
    const string cast_op = (host_memory) ? "_HostCast" : "Cast";
    NodeDefBuilder cast_builder(opts.new_name(src->name()), cast_op,
                                NodeDebugInfo(*src));
    cast_builder.Attr("DstT", dtype);
    cast_builder.Device(dst->assigned_device_name())
        .Input(recv->name(), 0, cast_dtype);
    NodeDef* cast = gdef->add_node();
    *status = cast_builder.Finalize(cast);
    if (!status->ok()) return nullptr;
    return cast;
  } else if (edge->IsControlEdge()) {
    // An Identity is only needed for control edges.
    NodeDefBuilder id_builder(opts.new_name(src->name()), "Identity",
                              NodeDebugInfo(*src));
    id_builder.Device(dst->assigned_device_name())
        .Input(recv->name(), 0, cast_dtype);
    NodeDef* id = gdef->add_node();
    *status = id_builder.Finalize(id);
    if (!status->ok()) return nullptr;
    return id;
  } else {
    return recv;
  }
}

NodeDef* AddDummyConst(const PartitionOptions& opts, GraphDef* gdef,
                       const Edge* edge, Status* status) {
  const Node* src = edge->src();
  Tensor tensor(DT_FLOAT, TensorShape({0}));
  NodeDef* result = gdef->add_node();
  *status = NodeDefBuilder(opts.new_name(src->name()), "Const")
                .Device(src->assigned_device_name())
                .Attr("dtype", DT_FLOAT)
                .Attr("value", tensor)
                .Finalize(result);
  return result;
}

// A dummy node for scheduling.
NodeDef* AddControlTrigger(const PartitionOptions& opts, GraphDef* gdef,
                           const string& assigned_device_name, int64 epoch,
                           int64 starttime, Status* status) {
  NodeDef* result = gdef->add_node();
  *status = NodeDefBuilder(opts.new_name(strings::StrCat("synch_", epoch)),
                           "ControlTrigger")
                .Device(assigned_device_name)
                .Attr("_start_time", starttime)
                .Finalize(result);
  return result;
}

// Optimize colocation for control flow nodes. 1. For cond, we want the
// switch nodes to colocate with its data input. This is particularly
// needed for conditional reading of a remote variable. It may also
// reduce the number of devices involved in a loop.
// TODO(yuanbyu): In this case, we don't respect the requested device in
// the GraphDef for these nodes. Ideally, the placer would enforce the
// colocation to render this unnecessary.
void OptimizeControlFlowColocation(Graph* graph) {

  auto visit = [](Node* node) {

    if (IsSwitch(node)) {
      // Switch Node 分支
      for (const Edge* in_edge : node->in_edges()) {
        if (in_edge->dst_input() == 0) {
          // Colocate with the data input.
          node->set_assigned_device_name(
              in_edge->src()->assigned_device_name());
          return;
        }
      }
    } else if (IsExit(node)) {
      // 不是 Switch Node 分支，Exit Node 分支
      for (const Edge* in_edge : node->in_edges()) {
        if (!in_edge->IsControlEdge()) {
          // Colocate with upstream node.
          node->set_assigned_device_name(
              in_edge->src()->assigned_device_name());
          return;
        }
      }
    } else {
      // 不是 Switch Node 分支，不是 Exit Node 分支
      if ((IsEnter(node) && !IsRefType(node->input_type(0))) ||
          IsNextIteration(node)) {
        // 不是 Switch Node 分支，不是 Exit Node 分支，
        // 是 Enter Node 或者 NextIteration Node 的分支
        const Edge* data_edge = nullptr;
        for (const Edge* out_edge : node->out_edges()) {
          if (!out_edge->IsControlEdge()) {
            data_edge = out_edge;
            break;
          }
        }
        // Colocate with the first downstream data node.
        if (data_edge) {
          node->set_assigned_device_name(
              data_edge->dst()->assigned_device_name());
        }
      }

    }
  };

  DFS(*graph, visit, {});
}

string ControlLoopName(const string& name) {
  return strings::StrCat("_cloop", name);
}

bool IsControlLoop(const Node* node) {
  const string& name = node->name();
  return str_util::StartsWith(name, "_cloop");
}

// An enter node for control flow.
Node* AddControlEnter(Graph* g, const string& node_name,
                      const string& device_name, const string& frame_name,
                      const int parallel_iterations, Status* status) {
  NodeBuilder node_builder(node_name, "Enter", g->op_registry());
  node_builder.Input({"dummy", 0, DT_FLOAT});
  node_builder.Attr("frame_name", frame_name);
  node_builder.Attr("parallel_iterations", parallel_iterations);
  Node* res_node;
  *status = node_builder.Finalize(g, &res_node);
  if (!status->ok()) return nullptr;
  res_node->set_assigned_device_name(device_name);
  return res_node;
}

// A merge node for control flow.
Node* AddControlMerge(const string& in_name1, const string& in_name2, Graph* g,
                      const string& node_name, const string& device_name,
                      Status* status) {
  NodeBuilder node_builder(node_name, "Merge", g->op_registry());
  node_builder.Input({{in_name1, 0, DT_FLOAT}, {in_name2, 0, DT_FLOAT}});
  Node* res_node;
  *status = node_builder.Finalize(g, &res_node);
  if (!status->ok()) return nullptr;
  res_node->set_assigned_device_name(device_name);
  return res_node;
}

// A switch node for control flow.
Node* AddControlSwitch(NodeBuilder::NodeOut input1, NodeBuilder::NodeOut input2,
                       const string& device_name,
                       const GraphDefBuilder::Options& bopts) {
  Node* res_node =
      ops::BinaryOp("Switch", std::move(input1), std::move(input2), bopts);
  if (bopts.HaveError()) return nullptr;
  res_node->set_assigned_device_name(device_name);
  return res_node;
}

// A next_iteration node for control flow.
Node* AddControlNext(NodeBuilder::NodeOut input, const string& device_name,
                     const GraphDefBuilder::Options& bopts) {
  Node* res_node = ops::UnaryOp("NextIteration", std::move(input), bopts);
  if (bopts.HaveError()) return nullptr;
  res_node->set_assigned_device_name(device_name);
  return res_node;
}

Node* EmptyConst(const GraphDefBuilder::Options& options) {
  if (options.HaveError()) return nullptr;
  NodeBuilder node_builder(options.GetNameForOp("Const"), "Const",
                           options.op_registry());
  const DataType dt = DataTypeToEnum<float>::v();
  TensorProto proto;
  proto.set_dtype(dt);
  TensorShape empty_shape({0});
  empty_shape.AsProto(proto.mutable_tensor_shape());
  node_builder.Attr("dtype", dt).Attr("value", proto);
  return options.FinalizeBuilder(&node_builder);
}

// A dummy const node for control flow.
Node* AddControlConst(const string& device_name,
                      const GraphDefBuilder::Options& bopts) {
  Node* res_node = EmptyConst(bopts);
  if (bopts.HaveError()) return nullptr;
  res_node->set_assigned_device_name(device_name);
  return res_node;
}

// A synthetic loop, made up of dummy nodes. It performs control-flow actions
// on behalf of a leader on a different device.
struct ControlLoop {
  Node* enter = nullptr;
  Node* merge = nullptr;
  Node* switch_node = nullptr;
};

// Add the control flow info of a new node added during partitioning.
// The new node has the same control flow info as src.
void AddControlFlowInfo(const Node* node, const Node* src,
                        std::vector<ControlFlowInfo>* cf_info) {
  int id = node->id();
  if (static_cast<size_t>(id) >= cf_info->size()) {
    cf_info->resize(id + 1);
  }
  const ControlFlowInfo& src_info = (*cf_info)[src->id()];
  ControlFlowInfo* info = &(*cf_info)[id];
  info->frame = src_info.frame;
  info->parent_frame = src_info.parent_frame;
  info->frame_name = src_info.frame_name;
}

// Constructs a control loop. Returns a struct containing the newly created
// enter, merge, and switch nodes. The enter and merge nodes are used in the
// recursive construction of control loops for nested frames (loops). The
// switch node will be connected to the LoopCond node. The merge node will
// be connected to all the recvs of the same frame by control edges when
// the actual partitioning happens.
Status AddControlLoop(const PartitionOptions& opts, Graph* g, const Node* src,
                      const Edge* edge, Node* loop_cond,
                      std::vector<ControlFlowInfo>* cf_info,
                      ControlLoop* loop) {
  Status status;
  GraphDefBuilder::Options bopts(g, &status);
  const ControlFlowInfo& src_info = (*cf_info)[src->id()];
  const string& device_name = edge->dst()->assigned_device_name();
  const string& frame_name = src_info.frame_name;
  int parallel_iterations;
  status = GetNodeAttr(src_info.frame->attrs(), "parallel_iterations",
                       &parallel_iterations);
  if (!status.ok()) return status;

  // The names of the nodes to be added.
  const string& enter_name =
      ControlLoopName(opts.new_name(edge->dst()->name()));
  const string& merge_name =
      ControlLoopName(opts.new_name(edge->dst()->name()));
  const string& switch_name =
      ControlLoopName(opts.new_name(edge->dst()->name()));
  const string& next_name = ControlLoopName(opts.new_name(edge->dst()->name()));

  // Add the nodes to the graph g.
  Node* enter = AddControlEnter(g, enter_name, device_name, frame_name,
                                parallel_iterations, &status);
  if (!status.ok()) return status;
  Node* merge = AddControlMerge(enter_name, next_name, g, merge_name,
                                device_name, &status);
  if (!status.ok()) return status;
  Node* switch_node = AddControlSwitch(merge, loop_cond, device_name,
                                       bopts.WithName(switch_name));
  if (!status.ok()) return status;
  Node* next =
      AddControlNext({switch_node, 1}, device_name, bopts.WithName(next_name));
  if (!status.ok()) return status;

  // Add control flow info for these new nodes:
  AddControlFlowInfo(enter, src, cf_info);
  AddControlFlowInfo(merge, src, cf_info);
  AddControlFlowInfo(switch_node, src, cf_info);
  AddControlFlowInfo(next, src, cf_info);

  // Add input edges for the newly created merge node:
  g->AddEdge(enter, 0, merge, 0);
  g->AddEdge(next, 0, merge, 1);

  loop->enter = enter;
  loop->merge = merge;
  loop->switch_node = switch_node;
  return Status::OK();
}




// 这个函数目的:
// Build memory and device type info
// for every node in the graph.
// 这些都存在了 GraphInfo info 里面

// QQQ. 这些被初始化的 memory and device type of each node 在哪里被怎么使用了？
//      我想不要改动这一部分，所以我打算加一份 low priority memory and device type
//      of each node

// Build memory and device type info for every node in the graph.
// comment: 我是不是应该写两个 GraphInfo* info_high, info_low?

// TODO(yuanbyu): It might be simpler if we convert MemoryType to
// DeviceType for the inputs/outputs of each node.
Status BuildMemoryDeviceInfo(
  const Graph& g,  // input
  GraphInfo* info) // output
{
  // GraphInfo 数据结构:
  // tensorflow/core/graph/graph_partition.cc:108:
  // struct GraphInfo
  // - device_types : std::vector<DeviceType>
  // - input_types: MemoryTypeMap
  // - output_types: MemoryTypeMap
  // - cf_info: std::vector<ControlFlowInfo>
  //    * struct ControlFlowInfo 数据结构
  //      tensorflow/core/common_runtime/executor.cc
  //      + unique_frame_names: gtl::FlatSet<string>
  //      + frame_names: std::vector<string>

  // MemoryTypeVector 数据结构
  // tensorflow/core/framework/types.h:100:
  // typedef gtl::InlinedVector<MemoryType, 4> MemoryTypeVector;
  //   enum MemoryType, tensorflow/core/framework/types.h
  //     DEVICE_MEMORY = 0
  //     HOST_MEMORY = 1

  MemoryTypeVector input_memory_types;
  MemoryTypeVector output_memory_types;

  // info->device_types: std::vector<DeviceType>
  // tensorflow/core/framework/types.h:54:
  // class DeviceType
  //   A DeviceType is just a string
  //   string type_;
  // 取值:
  //   // Convenient constants that can be passed to a DeviceType constructor
  //   TF_EXPORT extern const char* const DEVICE_CPU;   // "CPU"
  //   TF_EXPORT extern const char* const DEVICE_GPU;   // "GPU"
  //   TF_EXPORT extern const char* const DEVICE_SYCL;  // "SYCL"
  info->device_types.resize(g.num_node_ids(), DEVICE_CPU);


  // g.op_nodes(): gtl::iterator_range<NodeIter>
  // 含义是 Access to the list of all nodes, excluding the Source and Sink nodes.
  for (const Node* node : g.op_nodes()) {

    DeviceNameUtils::ParsedName parsed;

    if (!DeviceNameUtils::ParseFullName(node->assigned_device_name(),
                                        &parsed)) {
      return errors::Internal("Malformed assigned device '",
                              node->assigned_device_name(), "'");
    }


    // MemoryTypesForNode 函数说明:
    // tensorflow/core/framework/memory_types.h:31
    // tensorflow/core/framework/memory_types.cc

    // // Returns into *{input,output}_memory_types the memory type of each
    // // {input,output} tensor.
    // //
    // // REQUIRES: * '*_memory_types' is not nullptr.
    // //           * def has all attrs specified (e.g. using AddDefaultsToNodeDef()).
    // Status MemoryTypesForNode(const OpRegistryInterface* op_registry,
    //                           const DeviceType& device_type,
    //                           const NodeDef& ndef,
    //                           MemoryTypeVector* input_memory_types,
    //                           MemoryTypeVector* output_memory_types);
    // 提醒:
    // 在调用这个函数时，device type 已经确定了。所以是根据是 device type 确定 memory type.
    // 所以，如果我要增加双份的 CPU 版本，我就要提前告知 CPU device, 或者在函数下层调用栈时跑两遍
    // 这些函数达到我的目的。

    // 这个函数，会明确每个 node 的 input / output 的 memory type
    TF_RETURN_IF_ERROR(
      MemoryTypesForNode(
        // g.op_registry(): const OpRegistryInterface*
        g.op_registry(),         // input

        // class DeviceType 见本文件上面一点点
        DeviceType(parsed.type), // input

        node->def(),             // input
        &input_memory_types,     // output
        &output_memory_types));  // output
        // ------------------------------------------------------------------
        // 这个函数的 output 用于赋值 info->output_types
        // ------------------------------------------------------------------

    int node_id = node->id();

    // --------------------------------------------------------------------
    // Summary:
    // node 的 device 赋值是 info->device_types[node_id]
    // node 的 每个 input arg 的 memory type 赋值是 info->input_types
    // node 的 每个 output arg 的 memory type 赋值是 info->input_types
    // --------------------------------------------------------------------

    info->device_types[node_id] = DeviceType(parsed.type);

    for (int i = 0; i < input_memory_types.size(); ++i) {
      // input_memory_types 来自于 上面的 MemoryTypesForNode 函数
      // 这里的 i 是 多个 input_arg OR output_arg : OpDef::ArgDef
      info->input_types[{node_id, i}] = input_memory_types[i];
    }


    for (int i = 0; i < output_memory_types.size(); ++i) {
      // output_memory_types 来自于 上面的 MemoryTypesForNode 函数
      info->output_types[{node_id, i}] = output_memory_types[i];
    }


  }
  return Status::OK();
}

const Node* InputFrame(const Node* node,
                       const std::vector<ControlFlowInfo>& cf_info) {
  // An input is in the same frame as the node except for Enter nodes.
  // The input of Enter is in the parent frame of the Enter node.
  if (!node->IsEnter()) {
    return node;
  }
  return cf_info[node->id()].parent_frame;
}

const Node* OutputFrame(const Node* node,
                        const std::vector<ControlFlowInfo>& cf_info) {
  // An output is in the same frame as the node except for Exit nodes.
  // The output of Exit is in the parent frame of the Exit node.
  if (!node->IsExit()) {
    return node;
  }
  return cf_info[node->id()].parent_frame;
}



// Each participating device needs to decide
// a) if there is a next iteration,
// and
// b) if the loop terminates.
// We take the approach to encode this control flow logic in the dataflow graph.
// There are at least two possible encodings.
// 1. In a completely decentralized encoding, the participants communicate peer
// to peer. 2. The other encoding uses a frame leader (the participant who owns
// the pivot termination predicate) to broadcast the termination condition to
// all the participants.
// For now we take the latter (2.) because it is simpler.
//
// TODO(yuanbyu): The correctness of this construction is rather subtle. I got
// it wrong many times so it would be nice to write a proof to be sure.

// 1.
// QQQ. GraphInfo* g_info 谁的？
// AAA. g_info 是临时变量一样
// 在 Partition 函数内被构造。tensorflow/core/graph/graph_partition.cc

Status AddControlFlow(
  const PartitionOptions& opts,  // input
  Graph* g, // input
  GraphInfo* g_info) { // output
  // 1.
  // GraphInfo 数据结构
  // tensorflow/core/graph/graph_partition.cc:108:
  // struct GraphInfo
  // - device_types: std::vector<DeviceType>
  // - input_types: MemoryTypeMap
  // - output_types: MemoryTypeMap
  // - cf_info: std::vector<ControlFlowInfo> # AddControlFlow 负责初始化这项

  // 2.
  // class DeviceType: tensorflow/core/framework/types.h
  //  - string type_;
  //    const char* const DEVICE_CPU, DEVICE_GPU, DEVICE_SYCL

  // 3.
  // struct ControlFlowInfo 数据结构
  // tensorflow/core/graph/control_flow.h
  // - frame: const Node*
  // - parent_frame: const Node*
  // - frame_name: string

  // 4.
  // MemoryTypeMap 数据结构
  // typedef absl::flat_hash_map<NodePort, MemoryType> MemoryTypeMap;

  // 5.
  // struct NodePort 数据结构
  // - node_id: int
  // - index: int

  // 6.
  // MemoryType 数据结构
  // tensorflow/core/framework/types.h:47:enum MemoryType
  //
  // MemoryType is used to describe whether input or output Tensors of
  // an OpKernel should reside in "Host memory" (e.g., CPU memory) or
  // "Device" Memory (CPU memory for CPU devices, GPU memory for GPU
  // devices).
  // enum MemoryType {
  //   DEVICE_MEMORY = 0,
  //   HOST_MEMORY = 1,
  //};

  // 7.
  // struct PartitionOptions 数据结构
  // tensorflow/core/graph/graph_partition.h:31:struct PartitionOptions
  //
  // - NodeToLocFunc: std::function<string(const Node*)>
  // - node_to_loc: NodeToLocFunc
  // - NewNameFunc: std::function<string(const string&)>
  // - new_name: NewNameFunc
  // - GetIncarnationFunc: std::function<uint64(const string&)>
  // - get_incarnation: GetIncarnationFunc
  // - flib_def: const FunctionLibraryDefinition*
  // - control_flow_added: bool, default: false
  // - ShouldCastFunc: std::function<DataType(const Edge*)>
  // - should_cast: ShouldCastFunc
  // - scheduling_for_recvs: bool, default: false
  // - need_to_record_start_times: bool, default: false
  // - start_times: std::vector<Microseconds>
  //
  // 打印: https://gist.github.com/shizukanaskytree/7bbf6ef24dae11fb582abc78a1adf748

  // 8.
  // FunctionLibraryDefinition 数据结构
  // tensorflow/core/framework/function.h:313:
  // class FunctionLibraryDefinition : public OpRegistryInterface
  // - default_registry_ : const OpRegistryInterface* const
  // - function_defs_ : gtl::FlatMap<string, std::unique_ptr<FunctionDefAndOpRegistration>>
  // - func_grad_ : gtl::FlatMap<string, string>

  // 9.
  // FunctionDefAndOpRegistration 数据结构
  // - fdef: FunctionDef
  // - op_registration_data: OpRegistrationData

  Status status;

  GraphDefBuilder::Options bopts(g, &status);
  // 1.
  // GraphDefBuilder::Options 数据结构
  // tensorflow/core/graph/graph_def_builder.h
  // tensorflow/core/graph/graph_def_builder.cc
  // GraphDefBuilder::Options::Options 构造函数
  // - graph_: Graph* const;
  // - status_: Status* const;
  // - name_: string;
  // - device_: string;
  // - control_inputs_: std::vector<Node*>;
  // - attrs_: std::vector<std::pair<string, AttrValue>>;

  // 2.
  // GraphDefBuilder 函数说明
  // tensorflow/core/graph/graph_def_builder.h
  // - Graph graph_;
  // - Status status_;
  // - Options opts_;

  std::vector<ControlFlowInfo>& cf_info = g_info->cf_info;
  // 1.
  // struct ControlFlowInfo 数据结构:
  // Control flow info for a graph node.
  // tensorflow/core/graph/control_flow.h
  // - frame: const Node*
  //    * frame of a node
  // - parent_frame: const Node*
  //    * parent frame of a node
  // - frame_name: string
  //    * frame name of a node
  //
  // 上面的 cf_info 打印是 $13 = std::vector of length 0, capacity 0

  // -----------------------------------------------------------------------
  // Build the control flow info for every node.
  // -----------------------------------------------------------------------

  status = BuildControlFlowInfo(
    g, // input
    &cf_info); // output
  // 1.
  // BuildControlFlowInfo 函数说明
  // tensorflow/core/graph/control_flow.h
  // tensorflow/core/graph/control_flow.cc
  //
  // Clear and populate `cf_info` with each node's frame and the level it belongs to.
  // We check the well-formedness of the graph:
  // 1) All inputs to a node must come from the same frame and have the same
  //    "static" iteration level.
  // 2) Each frame has at most one LoopCond node.
  // 3) Each frame has a single parent frame.
  // If `unreachable_nodes` is set, return names of nodes unreachable from the
  // source node. We cannot build ControlFlowInfo for such nodes. They might be
  // pruned later.
  //
  // NOTE(yuanbyu): For now, we require all sends/recvs have iteration level 0.
  // This essentially means there can't be multiple serial Nexts in an iteration,
  // which all sane front-ends should satisfy.

  // 2.
  // 打印 cf_info 的结果, frame 和 parent_frame 都是 SOURCE_ Node 的 指针, frame_name = ""
  // 和预料的一样。
  // 打印: https://gist.github.com/shizukanaskytree/b42d709acc435ce5906678e2792ae872

  if (!status.ok()) return status;

  // OptimizeControlFlowColocation 只对 Switch, Enter, Exit, NextIteration 节点进行节点 device placement 优化
  OptimizeControlFlowColocation(g);

  // 我的例子没有 loop 节点
  // The map from frames to their LoopCond nodes.
  std::unordered_map<string, Node*> frame_cond_map;

  int num_node_ids = g->num_node_ids();

  for (int i = 0; i < num_node_ids; ++i) {
    Node* node = g->FindNodeId(i);
    if (node == nullptr) continue;

    if (IsLoopCond(node)) {
      const string& frame_name = cf_info[node->id()].frame_name;
      DCHECK(!frame_name.empty());
      frame_cond_map[frame_name] = node;
    }
  }

  // -----------------------------------------------------------------------
  // 讨论 cross-device edge，不过因为 frame_name 都是空的原因，没有起到真正的作用
  // -----------------------------------------------------------------------
  // Add all control loops for cross-device frames.
  // A control loop is added only when there is a **cross-device edge** in a
  // non-root frame. Nothing is added if there is no loops. We also don't
  // add anything for a frame that is completely local to a device. For
  // nested loops, we stack the control loops together by connecting
  // the merge of the outer loop to the enter of the inner loop.
  //
  // A map from <frame_name, device_name> to ControlLoop.
  std::unordered_map<string, ControlLoop> control_loops;

  int num_edge_ids = g->num_edge_ids();

  for (int i = 0; i < num_edge_ids; ++i) {
    const Edge* edge = g->FindEdgeId(i);

    /*
    p edge->DebugString()
    $50 = "[id=0 _SOURCE:-1 -> _SINK:-1]"

    p edge->DebugString()
    $51 = "[id=1 _SOURCE:-1 -> _SINK:-1]"

    p edge->DebugString()
    $52 = "[id=2 _SOURCE:-1 -> _SINK:-1]"

    p edge->DebugString()
    $53 = "[id=3 y/shape:0 -> y/RandomStandardNormal:0]" 同一个 device

    p edge->DebugString()
    $54 = "[id=4 x/shape:0 -> x/RandomStandardNormal:0]" 同一个 device

    p edge->DebugString()
    $55 = "[id=5 b/shape:0 -> b/RandomStandardNormal:0]" 同一个 device

    p edge->DebugString()
    $56 = "[id=6 a/shape:0 -> a/RandomStandardNormal:0]" 同一个 device

    p edge->DebugString()
    $57 = "[id=7 x/RandomStandardNormal:0 -> z:0]" 同一个 device

    p edge->DebugString()
    $58 = "[id=8 y/RandomStandardNormal:0 -> z:1]"
    */

    if (edge == nullptr) continue;

    const Node* src = edge->src();
    const Node* dst = edge->dst();

    // Skip Sink/Source nodes.
    if (!src->IsOp() || !dst->IsOp()) continue;

    const string& src_device = src->assigned_device_name();
    const string& dst_device = dst->assigned_device_name();

    // Skip local edges.
    if (src_device == dst_device) continue;

    // OutputFrame 函数说明:
    const Node* src_frame = OutputFrame(src, cf_info);
    const Node* dst_frame = InputFrame(dst, cf_info);
    const string& src_frame_name = cf_info[src_frame->id()].frame_name;
    const string& dst_frame_name = cf_info[dst_frame->id()].frame_name;

    // Skip if src and dst are not in the same frame.
    if (src_frame_name.empty() || src_frame_name != dst_frame_name) {
      // src_frame_name.empty() 是空，所以进入这个分支，然后 continue
      // 因为这个原因，没有能够执行到下面的
      continue;
    }

    // Add the control loop. Start by adding the control loop for the
    // current frame if needed, and recursively adding the control loop
    // for its outer frame when nested.
    ControlLoop child_loop;

    while (true) {
      const string& curr_frame_name = cf_info[src_frame->id()].frame_name;
      if (curr_frame_name.empty()) {
        // We have reached the root frame.
        if (child_loop.merge != nullptr) {
          const string& node_name = opts.new_name(edge->dst()->name());
          const string& device_name = edge->dst()->assigned_device_name();

          Node* const_node =
              AddControlConst(device_name, bopts.WithName(node_name));
          if (!status.ok()) return status;
          AddControlFlowInfo(const_node, src_frame, &cf_info);
          g->AddEdge(const_node, 0, child_loop.enter, 0);
        }
        break;
      }

      const string& cl_key = strings::StrCat(curr_frame_name, "$$", dst_device);
      auto it = control_loops.find(cl_key);
      if (it != control_loops.end()) {
        if (child_loop.enter != nullptr) {
          g->AddEdge(it->second.merge, 0, child_loop.enter, 0);
        }
        break;
      }

      // Get the frame's LoopCond.
      auto cond_it = frame_cond_map.find(curr_frame_name);
      if (cond_it == frame_cond_map.end()) {
        return errors::InvalidArgument(
            "A cross-device loop must have a pivot predicate: ",
            curr_frame_name);
      }
      Node* loop_cond = cond_it->second;

      // Add the control loop.
      ControlLoop curr_loop;
      status = AddControlLoop(opts, g, src_frame, edge, loop_cond, &cf_info,
                              &curr_loop);
      if (!status.ok()) return status;
      control_loops[cl_key] = curr_loop;

      if (child_loop.enter != nullptr) {
        // Connect the merge of the outer loop to the enter of the inner.
        g->AddEdge(curr_loop.merge, 0, child_loop.enter, 0);
      }
      src_frame = cf_info[src_frame->id()].parent_frame;
      child_loop = curr_loop;
    }
  }

  // For a cross-device edge, on the dst device, add a control edge
  // from the merge node of the control loop to dst. If a send/recv is
  // introduced for this edge in future partitioning, we delete this
  // control edge and add a new control edge from the merge to the recv.
  num_edge_ids = g->num_edge_ids();
  for (int i = 0; i < num_edge_ids; ++i) {
    const Edge* edge = g->FindEdgeId(i);
    if (edge == nullptr) continue;

    const Node* src = edge->src();
    Node* dst = edge->dst();
    // Skip Sink/Source nodes.
    if (!src->IsOp() || !dst->IsOp()) continue;

    const string& src_device = src->assigned_device_name();
    const string& dst_device = dst->assigned_device_name();
    if (src_device != dst_device) {
      const Node* src_frame = OutputFrame(src, cf_info);
      const Node* dst_frame = InputFrame(dst, cf_info);
      const string& src_frame_name = cf_info[src_frame->id()].frame_name;
      const string& dst_frame_name = cf_info[dst_frame->id()].frame_name;

      // 所有的都没能进入这个分支，因为 !src_frame_name.empty() 这个条件不满足
      if (!src_frame_name.empty() && src_frame_name == dst_frame_name) {
        const string& cl_key =
            strings::StrCat(dst_frame_name, "$$", dst_device);
        ControlLoop loop = control_loops[cl_key];
        DCHECK(loop.enter != nullptr);
        // Note that we'll create multiple duplicate edges if dst has multiple
        // cross-device inputs. This is expected by the logic in Partition(), so
        // it can add control edges to the recv nodes once they're created.
        g->AddControlEdge(loop.merge, dst, /*allow_duplicates=*/true);
      } // end if
    } // end if
  } // end for

  return Status::OK();
}

struct PriorityTopoSortNode {
  PriorityTopoSortNode(const NodeDef* n, int64 st) : node(n), start_time(st) {}

  const NodeDef* node;
  int64 start_time;
};

struct PriorityTopoSortNodeGreater {
  bool operator()(const PriorityTopoSortNode& left,
                  const PriorityTopoSortNode& right) {
    return left.start_time > right.start_time;
  }
};

}  // namespace

// Returns in <nodes> the nodes that should participate in epoch-based recv
// scheduling, along with their times; <nodes> is ordered by increasing
// start_time. Returns in <node_to_start_time_out> the timing for all nodes,
// even those not in <nodes>.
//
// Comparing to sorting on the node's start time only, this also processes the
// nodes in dependency order, and updates start times to ensure a node's
// start_time > the start time for all dependencies.
//
// Note that graph_partition_test.cc accesses this function for testing, even
// though it's not declared in the header.
Status TopologicalSortNodesWithTimePriority(
    const GraphDef* gdef, std::vector<std::pair<const NodeDef*, int64>>* nodes,
    std::unordered_map<const NodeDef*, int64>* node_to_start_time_out) {
  // Queue of nodes to process; lowest start time is returned first.
  std::priority_queue<PriorityTopoSortNode, std::vector<PriorityTopoSortNode>,
                      PriorityTopoSortNodeGreater>
      q;
  std::unordered_map<const NodeDef*, int64> node_to_start_time;
  auto enqueue = [&q, &node_to_start_time](const NodeDef* node) {
    const int64 start_time = node_to_start_time[node];
    q.emplace(node, start_time);
  };

  // Build initial structures, initial contents of queue.
  std::unordered_map<string, std::vector<const NodeDef*>> node_to_output_nodes;
  std::unordered_map<const NodeDef*, int> inputs_needed;
  for (int n = 0; n < gdef->node_size(); ++n) {
    const NodeDef* ndef = &gdef->node(n);
    for (int i = 0; i < ndef->input_size(); ++i) {
      node_to_output_nodes[string(ParseTensorName(ndef->input(i)).first)]
          .push_back(ndef);
    }
    int64 start_time;
    TF_RETURN_IF_ERROR(GetNodeAttr(*ndef, "_start_time", &start_time));
    node_to_start_time[ndef] = start_time;
    inputs_needed[ndef] = ndef->input_size();
    if (ndef->input_size() == 0) {
      enqueue(ndef);
    }
  }

  // Determine which merge nodes are parts of loops; these
  // need to happen in the traversal after all non-NextIteration inputs
  // are run.
  for (int n = 0; n < gdef->node_size(); ++n) {
    const NodeDef* ndef = &gdef->node(n);
    if (IsNextIteration(*ndef)) {
      for (const NodeDef* n : node_to_output_nodes[ndef->name()]) {
        if (IsMerge(*n)) {
          // n is a merge that is part of a loop structure.
          // It doesn't need to wait for this NextIteration loop
          // when doing the traversal.
          --inputs_needed[n];
        }
      }
    }
  }

  // Traverse.
  std::vector<std::pair<const NodeDef*, int64>> start_times;
  start_times.reserve(gdef->node_size());
  while (!q.empty()) {
    PriorityTopoSortNode cur = q.top();
    q.pop();

    start_times.emplace_back(cur.node, cur.start_time);

    for (const NodeDef* n : node_to_output_nodes[cur.node->name()]) {
      auto& output_start_time = node_to_start_time[n];
      if (output_start_time <= cur.start_time) {
        output_start_time = cur.start_time + 1;
      }
      if (--inputs_needed[n] == 0) {
        enqueue(n);
      }
    }
  }

  // Done.
  nodes->swap(start_times);
  node_to_start_time_out->swap(node_to_start_time);
  return Status::OK();
}

Status AddControlEdges(const PartitionOptions& opts,
                       std::unordered_map<string, GraphDef>* partitions) {
  Status status;
  // TODO(yuanbyu): Very naive for now. To be improved.
  const int num_epochs = 100;
  const int prefetch = 6;

  for (auto& part : *partitions) {
    GraphDef* gdef = &part.second;
    std::vector<std::pair<const NodeDef*, int64>> start_times;
    std::unordered_map<const NodeDef*, int64> node_to_start_time;
    status = TopologicalSortNodesWithTimePriority(gdef, &start_times,
                                                  &node_to_start_time);
    if (!status.ok()) {
      return status;
    }

    // Add a dummy node for every epoch, and add a control edge from the
    // "last" node in the preceding epoch to the dummy node.
    string device_name = gdef->node(0).device();
    int64 makespan = start_times.back().second;
    int64 resolution = (makespan / num_epochs) + 1;

    int i = 0;
    int j = 0;
    std::vector<NodeDef*> dummys;
    while (i < num_epochs && static_cast<size_t>(j) < start_times.size()) {
      if (i * resolution > start_times[j].second) {
        j++;
      } else {
        NodeDef* dummy = AddControlTrigger(opts, gdef, device_name, i,
                                           i * resolution, &status);
        if (!status.ok()) {
          return status;
        }
        dummys.push_back(dummy);
        if (j > 0) {
          string src_name = start_times[j - 1].first->name();
          AddInput(dummy, src_name, Graph::kControlSlot);
        }
        i++;
      }
    }

    // Finally, add the control edges to recvs.
    for (int n = 0; n < gdef->node_size(); ++n) {
      NodeDef* ndef = gdef->mutable_node(n);
      if (ndef->op() == "_Recv") {
        const int64 start_time = node_to_start_time[ndef];
        const int recv_epoch = start_time / resolution;
        if (recv_epoch >= prefetch) {
          NodeDef* dummy = dummys[recv_epoch - prefetch];
          AddInput(ndef, dummy->name(), Graph::kControlSlot);
        }
      }
    }
  }
  return Status::OK();
}

// If 'ndef' is a Send or Recv, fills its attr send_device_incarnation
// if possible.
void SetIncarnation(const PartitionOptions& opts, NodeDef* ndef) {
  StringPiece op(ndef->op());
  if (op != "_Send" && op != "_Recv") {
    // Not related to send/recv.
    return;
  }
  string send_device;
  if (!GetNodeAttr(*ndef, "send_device", &send_device).ok()) {
    // No known send_device. The runtime will detect it later.
    return;
  }
  int64 incarnation = PartitionOptions::kIllegalIncarnation;
  if (!GetNodeAttr(*ndef, "send_device_incarnation", &incarnation).ok() ||
      (incarnation == PartitionOptions::kIllegalIncarnation)) {
    incarnation = opts.get_incarnation(send_device);
    SetAttrValue(incarnation,
                 &((*ndef->mutable_attr())["send_device_incarnation"]));
  }
}

// Sets attribute send_device_incarnation of all Send/Recv nodes in
// 'gdef', if possible.
void SetIncarnation(const PartitionOptions& opts, GraphDef* gdef) {
  for (NodeDef& ndef : *gdef->mutable_node()) {
    SetIncarnation(opts, &ndef);
  }
  for (FunctionDef& fdef : *gdef->mutable_library()->mutable_function()) {
    for (NodeDef& ndef : *fdef.mutable_node_def()) {
      SetIncarnation(opts, &ndef);
    }
  }
}




Status Partition(
  const PartitionOptions& opts, // input
  // client_graph->graph, 是以 targets 为输出的最小依赖的图
  Graph* g, // input
  std::unordered_map<string, GraphDef>* partitions) // output
{
  // 1.
  // struct PartitionOptions 数据结构
  // tensorflow/core/graph/graph_partition.h:31:struct PartitionOptions
  //
  // - NodeToLocFunc: std::function<string(const Node*)>
  // - node_to_loc: NodeToLocFunc
  // - NewNameFunc: std::function<string(const string&)>
  // - new_name: NewNameFunc
  // - GetIncarnationFunc: std::function<uint64(const string&)>
  // - get_incarnation: GetIncarnationFunc
  // - flib_def: const FunctionLibraryDefinition*
  // - control_flow_added: bool, default: false
  // - ShouldCastFunc: std::function<DataType(const Edge*)>
  // - should_cast: ShouldCastFunc
  // - scheduling_for_recvs: bool, default: false
  // - need_to_record_start_times: bool, default: false
  // - start_times: std::vector<Microseconds>
  //
  // 打印: https://gist.github.com/shizukanaskytree/7bbf6ef24dae11fb582abc78a1adf748

  // 2.
  // FunctionLibraryDefinition 数据结构
  // tensorflow/core/framework/function.h:313:
  // class FunctionLibraryDefinition : public OpRegistryInterface
  // - default_registry_ : const OpRegistryInterface* const
  // - function_defs_ : gtl::FlatMap<string, std::unique_ptr<FunctionDefAndOpRegistration>>
  // - func_grad_ : gtl::FlatMap<string, string>

  // 3.
  // FunctionDefAndOpRegistration 数据结构
  // - fdef: FunctionDef
  // - op_registration_data: OpRegistrationData

  Status status;
  partitions->clear();

  GraphInfo g_info;
  // 1.
  // GraphInfo 数据结构
  // tensorflow/core/graph/graph_partition.cc:108:
  // struct GraphInfo
  // - device_types: std::vector<DeviceType>
  //    * class DeviceType: tensorflow/core/framework/types.h
  //       + string type_;
  //          - const char* const DEVICE_CPU, DEVICE_GPU, DEVICE_SYCL
  // - input_types: MemoryTypeMap
  // - output_types: MemoryTypeMap
  // - cf_info: std::vector<ControlFlowInfo>

  // 2.
  // g_info 变量说明:
  // g_info 的 MemoryTypeMap input_types 和 MemoryTypeMap output_types
  // 在 BuildMemoryDeviceInfo() 里面赋值

  // 3.
  // struct ControlFlowInfo 数据结构
  // tensorflow/core/graph/control_flow.h
  // - frame: const Node*
  // - parent_frame: const Node*
  // - frame_name: string

  // 进入了这个分支
  if (!opts.control_flow_added) {
    // Add the "code" for distributed execution of control flow. Code is
    // added only for the frames that are placed on multiple devices. The
    // new graph is an equivalent transformation of the original graph and
    // has the property that it can be subsequently partitioned arbitrarily
    // (down to the level of individual device) for distributed execution.

    status = AddControlFlow(
      opts,  // input
      g, // input
      &g_info); // output
    // AddControlFlow 函数说明:
    // tensorflow/core/graph/graph_partition.cc:608:
    // Status AddControlFlow(
    //  const PartitionOptions& opts,
    //  Graph* g,
    //  GraphInfo* g_info)

    if (!status.ok()) return status;
  }

  // -----------------------------------------------------------------------
  // Memory Device Info 存在 g_info 这里面
  //////////////////////////////////////////////////////////////////////////
  // At this point, all the graph mutations have been done. Build memory
  // and device type info for every node and edge in the graph.
  //////////////////////////////////////////////////////////////////////////
  status = BuildMemoryDeviceInfo(
    *g,  // input
    &g_info); // output

  if (!status.ok()) return status;
  // -----------------------------------------------------------------------

  string dstp;
  std::vector<const Edge*> inputs;

  // DupRecvTable 数据结构:
  // typedef absl::flat_hash_map<DupRecvKey, RecvInfo> DupRecvTable;
  // graph/graph_partition.cc

  // DupRecvKey 数据结构:
  // graph/graph_partition.cc

  // RecvInfo 数据结构:
  // graph/graph_partition.cc

  DupRecvTable dup_recv(3);

  // QQQ. 依然是很模糊的感觉
  // For a node dst, 'ref_recvs' remembers the recvs introduced by a ref
  // edge to dst. 'ref_control_inputs' remembers the inputs by a non-ref
  // edge to dst. We will add a control edge for every pair in
  // (ref_recvs x ref_control_inputs).
  std::vector<NodeDef*> ref_recvs;
  std::vector<string> ref_control_inputs;

  int32 num_data = 0;
  int32 num_control = 0;

  // 造子图开始
  // step 1: 遍历每个 node
  // step 2 : 遍历 dst node 的 每条 input edge
  for (const Node* dst : g->op_nodes()) {
    // node_to_loc
    //   tensorflow/core/common_runtime/direct_session.cc-1535
    // lambda 函数:
    // popts.node_to_loc = [](const Node* node) {
    //   return node->assigned_device_name();
    // };
    dstp = opts.node_to_loc(dst);

    GraphDef* dst_graph = &(*partitions)[dstp];

    // QQQ. 这一步是在 子图与子图直接添加 什么 node？
    // AAA. 这是在重头构造子图，不是子图与子图直接的什么节点。
    NodeDef* dst_def = dst_graph->add_node(); // 构造

    *dst_def = dst->def(); // 初始化各个 fields

    MergeDebugInfo(NodeDebugInfo(dst->def()), dst_def);

    // -----------------------------------------------------------------------
    // dst_def: NodeDef*
    // dst: Node*
    // 多少感觉是重复了这个 device 的赋值
    // NodeDef::device : string
    // dst->assigned_device_name() : device_names_[node.assigned_device_name_index()]
    dst_def->set_device(dst->assigned_device_name());
    // -----------------------------------------------------------------------

    dst_def->clear_input();  // Inputs are filled below

    if (opts.need_to_record_start_times) {
      int64 start_time;
      status = GetNodeAttr(*dst_def, "_start_time", &start_time);
      if (errors::IsNotFound(status)) {
        start_time = opts.start_times[dst->id()].value();
        AddNodeAttr("_start_time", start_time, dst_def);
      } else if (!status.ok()) {
        return status;
      }
    }

    // Arrange the incoming edges to dst so that input[i] holds the
    // input flowing into slot numbered i. Trailing 后面的；拖尾的 entries in input[]
    // hold control edges.

    // inputs 的含义是 the incoming edges to dst node
    inputs.clear(); // clear() 是因为 inputs edges 是重复使用的一个变量，对于每个 node 而言
    inputs.resize(dst->num_inputs(), nullptr);

    ref_recvs.clear();
    ref_control_inputs.clear();

    const Edge* control_flow_edge = nullptr;

    int32 num_control_flow_edges = 0;
    int32 num_input_edges = 0;

    // 遍历 dst node 的 每条 input edge ，加入 inputs，用作下一轮处理。
    for (const Edge* edge : dst->in_edges()) {

      // IsControlEdge 函数定义:
      // graph/graph.h,
      // if either src_output_ or dst_input_ is kControlSlot,
      // return src_output_ == Graph::kControlSlot;
      if (edge->IsControlEdge()) {

        if (IsMerge(edge->src()) && IsControlLoop(edge->src())) {
          // This is one of the control edges added for control flow. There
          // can be multiple such edges as the dest node may have multiple
          // remote inputs. We keep track of the number of such edges.
          control_flow_edge = edge;
          ++num_control_flow_edges;

        } else {
          // p edge->DebugString()
          // $29 = "[id=14 _SOURCE:-1 -> x/shape:-1]"
          // 这个就落入此
          // 因为是 -1 作为 node slot id , 所以无法像下面一样用数组访问的方式去存入 inputs
          inputs.push_back(edge);
        }
        // edge 是 Control Edge 分支结束
      } else {
        // edge 不是 Control Edge 分支
        DCHECK(inputs[edge->dst_input()] == nullptr);

        /*
                             +----------+
            +--------------->|0         |
                             |          |
            +--------------->|1  dst    |
                             |          |
            +--------------->|2         |
                             +----------+

                        inputs edges
        */

        // Return the index of the destination input that consumes the data
        // carried by this edge.  The special value kControlSlot is used
        // for control dependencies.

        /*
        进入这个分支的 edge 打印信息:

        p edge->DebugString()
        $35 = "[id=3 y/shape:0 -> y/RandomStandardNormal:0]"
        */

        inputs[edge->dst_input()] = edge;
        ++num_input_edges;
      }
    } // 遍历结束所有的 dst 的 input edges


    if (num_input_edges != dst->num_inputs()) {
      return errors::InvalidArgument("Incomplete graph, missing ",
                                     (dst->num_inputs() - num_input_edges),
                                     " inputs for ", dst->name());
    }

    /*
                 +----------+
                 |0         |
                 |          |
                 |1         +--------+
                 |          |        |
                 |2         |        |
                 +----------+        |
                                     |
                                     |                +----------+
                                     v--------------->|0         |
                                                      |          |
                                     +--------------->|1  dst    |
                                                      |          |
                                     +--------------->|2         |
                                                      +----------+

                                                 inputs edges
    */

    // Process in order so that all data edges are added as inputs to
    // dst in Edge::dst_input() order.
    // 如果这条边 cross device 了的话，就要构造 send 和 recv 节点
    for (const Edge* edge : inputs) {

      const Node* src = edge->src();

      if (!src->IsOp()) continue;  // Skip Sink/Source nodes.

      // 因为 device type : GraphDef instance ，所以 取出 src_graph
      GraphDef* src_graph = &(*partitions)[opts.node_to_loc(src)];

      // NeedSameDeviceSendRecv 函数说明:
      // tensorflow/core/graph/graph_partition.cc
      // 大致的含义是: 即使在 same device 上也想要增加 send recv node 的判读
      if (src_graph == dst_graph && !NeedSameDeviceSendRecv(
                                      edge,     // input
                                      g_info))  // input
      {
        // AddInput 函数说明:
        // Add an input to dst that comes from the "src_slot" output of the
        // node named by "src_name".
        // AddInput 函数接口:
        // void AddInput(NodeDef* dst, StringPiece src_name, int src_slot)

        // Same partition and compatible memory types:
        // 给 dst_def / dst 节点加一条边
        AddInput(dst_def, src->name(), edge->src_output());

        /*
                     +----------+
                     |0         |
                     |          |
                     |1       0 +--------+
                     |          |        |
                     |2         |        |
                     +----------+        |
                                         |
                                         |                +----------+
                                         v--------------->|0         |
                                                          |          |
                                         +--------------->|1  dst    |
                                                          |          |
                                         +--------------->|2         |
                                                          +----------+

                                                     inputs edges
        */

        if (edge->IsControlEdge() ||
            !IsRefType(src->output_type(edge->src_output()))) {

          ref_control_inputs.push_back(src->name());

        }

        continue;
      }

      // QQQ. 为什么要计时？
      //
      int64 send_start_time = 0;
      int64 recv_start_time = 0;

      if (opts.scheduling_for_recvs) {

        status = GetNodeAttr(src->attrs(), "_start_time", &send_start_time);

        if (errors::IsNotFound(status) && opts.need_to_record_start_times) {

          send_start_time = opts.start_times[src->id()].value();

        } else if (!status.ok()) {

          return status;

        }

        status = GetNodeAttr(dst->attrs(), "_start_time", &recv_start_time);

        if (errors::IsNotFound(status) && opts.need_to_record_start_times) {

          recv_start_time = opts.start_times[dst->id()].value();

        } else if (!status.ok()) {

          return status;

        }
      }

      // Check whether there is already a send/recv pair transferring
      // the same tensor/control from the src to dst partition.
      const bool on_host = IsDstInputOnHost(edge, g_info);

      // 两个 key 才能确定一个 edge id: node id + edge slot id of that node
      // DupRecvKey 数据结构
      DupRecvKey key{src->id(), edge->src_output(), dst_graph, on_host};

      // 之所以叫 duplicated 是因为如果已经存在了的话不就是重复的吗。
      auto iter = dup_recv.find(key);

      if (iter != dup_recv.end()) {

        // We found one. Reuse the data/control transferred already.
        const string& recv_node_name = iter->second.recv->name();

        if (edge->IsControlEdge()) {

          AddInput(dst_def, recv_node_name, Graph::kControlSlot);

        } else {

          AddInput(dst_def, recv_node_name, 0);

        }

        ref_control_inputs.push_back(recv_node_name);

        // We want the start_time for the recv to be the smallest of the start
        // times of it's consumers. So we update this whenever we use a recv,
        // and write it out to the attribute at the end of the subroutine
        if (iter->second.start_time > recv_start_time) {
          iter->second.start_time = recv_start_time;
        }
        continue;
      }

      // 如果没有进入上面的分支，那么就要构造 send 节点

      // NodeOut 数据结构
      NodeDefBuilder::NodeOut send_from;
      if (edge->IsControlEdge()) {
        // Insert a dummy const node that will generate a tiny
        // data element to be sent from send to recv.
        VLOG(1) << "Send/Recv control: " << src->assigned_device_name() << "["
                << src->name() << "] -> " << dst->assigned_device_name() << "["
                << dst->name() << "]";
        NodeDef* dummy = AddDummyConst(opts, src_graph, edge, &status);
        if (!status.ok()) return status;
        // Set the start time for this dummy node.
        if (opts.scheduling_for_recvs) {
          AddNodeAttr("_start_time", send_start_time, dummy);
        }
        AddInput(dummy, src->name(), Graph::kControlSlot);
        send_from.Reset(dummy->name(), 0, DT_FLOAT);
      } else {
        // send 节点的 send_from 节点
        send_from.Reset(src->name(), edge->src_output(), EdgeType(edge));

      }

      // ----------------------------------------------------------------------
      // ----------------------------------------------------------------------
      // 增加 send node
      // QQQ. send node 和 recv node 里面都用到了 GraphInfo g_info device and memory type 信息吧?
      // AAA. send node 和 recv node 里面都用到了 GraphInfo g_info device and memory type 信息吧?

      // Need to split edge by placing matching send/recv nodes on
      // the src/dst sides of the edge.

      // AddSend 函数说明:
      // graph/graph_partition.cc
      // NodeDef* AddSend(const PartitionOptions& opts,
      //                  const GraphInfo& g_info,
      //                  GraphDef* gdef,
      //                  const Edge* edge,
      //                  NodeDefBuilder::NodeOut send_from,
      //                  int64 start_time,
      //                  Status* status)
      //
      NodeDef* send = AddSend(
        opts, // input
        g_info, // input
        src_graph, // input
        edge, // input
        send_from, // input
        send_start_time, // input
        &status); // output
      // -----------------------------------------------------------------------

      if (!status.ok()) return status;

      // -----------------------------------------------------------------------
      // 增加 recv node
      //
      NodeDef* real_recv = nullptr;
      NodeDef* recv =
          AddRecv(opts, g_info, dst_graph, edge, &real_recv, &status);
      // -----------------------------------------------------------------------

      if (!status.ok()) return status;

      // Fix up the control flow edge.
      // NOTE(yuanbyu): 'real_recv' must be the real recv node.
      if (src_graph == dst_graph) {
        // For same device send/recv, add a control edge from send to recv.
        // This prevents the asynchronous recv kernel from being scheduled
        // before the data is available.
        AddInput(real_recv, send->name(), Graph::kControlSlot);
      } else if (control_flow_edge != nullptr) {
        // Redirect control edge to the real recv since this is not the same
        // device send/recv.
        --num_control_flow_edges;
        AddInput(real_recv, control_flow_edge->src()->name(),
                 Graph::kControlSlot);
      }

      if (!edge->IsControlEdge() &&
          IsRefType(src->output_type(edge->src_output()))) {
        AddNodeAttr("_start_time", recv_start_time, recv);
        if (real_recv != recv) {
          AddNodeAttr("_start_time", recv_start_time, real_recv);
        }
        // If src is of ref type and the edge is not a control edge, dst has
        // read semantics and therefore we must control the recv.
        ref_recvs.push_back(real_recv);
      } else {
        // Memorize the send/recv pair, only if this is not a "ref" edge.
        // NOTE(yuanbyu): Collapsing ref edges requires extreme care so
        // for now we don't do it.
        dup_recv[key] = {recv, real_recv, recv_start_time};
        ref_control_inputs.push_back(recv->name());
      }

      if (edge->IsControlEdge()) {
        ++num_control;
        AddInput(dst_def, recv->name(), Graph::kControlSlot);
      } else {
        ++num_data;
        AddInput(dst_def, recv->name(), 0);
      }
    } // end for loop of each a node's inputs' edge

    // Add control edges from 'ref_control_inputs' to 'ref_recvs'.
    // NOTE(yuanbyu): Adding these control edges should not introduce
    // deadlocks. 'dst' has implicit "read" nodes that, when we split
    // across devices, are made explicit; Retargeting the dependencies
    // to 'dst' to those nodes would not introduce cycles if there isn't
    // one before the transformation.
    // NOTE(yuanbyu): This may impact performance because it defers the
    // execution of recvs until all the other inputs become available.
    AddReadControl(ref_recvs, ref_control_inputs);

    // Add back the control edges for control flow that are not used.
    if (control_flow_edge != nullptr) {
      for (int i = 0; i < num_control_flow_edges; ++i) {
        AddInput(dst_def, control_flow_edge->src()->name(),
                 Graph::kControlSlot);
      }
    }
  } // for loop of each op node End!

  const FunctionLibraryDefinition* flib_def = opts.flib_def;
  if (flib_def == nullptr) {
    flib_def = &g->flib_def();
  }

  // Set versions, function library and send/recv incarnation.
  for (auto& it : *partitions) {

    GraphDef* gdef = &it.second;
    // gpu_graph_def.log
    // https://gist.github.com/shizukanaskytree/f33ee989f956e6a063949e1048725313
    // search : op: "_Recv", op: "_Send"

    // cpu_graph_def.log
    // https://gist.github.com/shizukanaskytree/c5af4564c1c76b6bc3421b268ac18d43
    // search:
    // "b/RandomStandardNormal/_3"
    // op: "MatMul"                         乘法的输入都因此而变了
    // input: "a/RandomStandardNormal"
    // input: "b/RandomStandardNormal/_3"

    *gdef->mutable_versions() = g->versions();
    // Prune unreachable functions from `flib_def` before adding them to `gdef`.
    *gdef->mutable_library() = flib_def->ReachableDefinitions(*gdef).ToProto();

    // Traverse the graph to fill every send/recv op's incarnation
    // information.
    SetIncarnation(opts, gdef);
  }

  // Set the start times for recvs at the very end.
  if (opts.scheduling_for_recvs) {
    for (auto& it : dup_recv) {
      AddNodeAttr("_start_time", it.second.start_time, it.second.recv);
      if (it.second.real_recv != it.second.recv) {
        AddNodeAttr("_start_time", it.second.start_time, it.second.real_recv);
      }
    }
  }

  VLOG(1) << "Added send/recv: controls=" << num_control
          << ", data=" << num_data;
  return Status::OK();
}

}  // namespace tensorflow
