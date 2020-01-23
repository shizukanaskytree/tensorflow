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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_EXECUTION_STATE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_EXECUTION_STATE_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/build_graph_options.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
struct SessionOptions;

namespace subgraph {
struct RewriteGraphMetadata;
}

struct GraphExecutionStateOptions {
  const DeviceSet* device_set = nullptr;
  const SessionOptions* session_options = nullptr;
  // Unique session identifier. Can be empty.
  string session_handle;
  // A map from node name to device name, representing the unchangeable
  // placement of stateful nodes.
  std::unordered_map<string, string> stateful_placements;
};
// 1.
// struct GraphExecutionStateOptions 数据结构
// tensorflow/core/common_runtime/graph_execution_state.h
// - device_set: const DeviceSet*
// - session_options: const SessionOptions*
// - session_handle: string
//   Unique session identifier. Can be empty.
// - stateful_placements
//   A map from node name to device name, representing the unchangeable
//   placement of stateful nodes.

// 2.
// class DeviceSet 数据结构
// 概述:
// DeviceSet is a container class for managing the various types of
// devices used by a model.
//
// tensorflow/core/common_runtime/device_set.h
//
// - devices_: td::vector<Device*> , not owned
// - device_by_name_: std::unordered_map<string, Device*>
//   Fullname -> device* for device in devices_.
// - client_device_: Device*
//   client_device 指的是 host CPU , 下面打印部分印证。
//   The device designated as the "client".
//   client_device_ points to an element of devices_ that we consider
//   to be the client device (in this local process).

// 2.
// 打印
// 2.1
// 只有 CPU , 没有 GPU 的情况:
/*
$2 = {
  devices_ = std::vector of length 2,
  capacity 2 = {
    0x5573b1bf6300,
    0x5573b37f6570
  },
  device_by_name_ = std::unordered_map with 4 elements = {
    ["/job:localhost/replica:0/task:0/xla_cpu:0"] = 0x5573b37f6570,
    ["/job:localhost/replica:0/task:0/device:XLA_CPU:0"] = 0x5573b37f6570,
    ["/job:localhost/replica:0/task:0/device:CPU:0"] = 0x5573b1bf6300,
    ["/job:localhost/replica:0/task:0/cpu:0"] = 0x5573b1bf6300
  },
  client_device_ = 0x5573b1bf6300
}
*/

// 2.2
// 4 个 GPU 都可见
/*
$2 = {
  devices_ = std::vector of length 10,
  capacity 16 = {
    0x5600134775d0,
    0x560015081440,
    0x5600152d7ca0,
    0x5600152e07c0,
    0x5600152e8a90,
    0x5600152f1a80,
    0x5600153086d0,
    0x560015319fc0,
    0x5600153277f0,
    0x5600153350f0
  },
  device_by_name_ = std::unordered_map with 20 elements = {
    ["/job:localhost/replica:0/task:0/gpu:3"] = 0x5600153350f0,
    ["/job:localhost/replica:0/task:0/device:GPU:3"] = 0x5600153350f0,
    ["/job:localhost/replica:0/task:0/device:GPU:2"] = 0x5600153277f0,
    ["/job:localhost/replica:0/task:0/device:XLA_GPU:0"] = 0x5600152d7ca0,
    ["/job:localhost/replica:0/task:0/xla_cpu:0"] = 0x560015081440,
    ["/job:localhost/replica:0/task:0/device:XLA_GPU:3"] = 0x5600152f1a80,
    ["/job:localhost/replica:0/task:0/device:XLA_CPU:0"] = 0x560015081440,
    ["/job:localhost/replica:0/task:0/device:CPU:0"] = 0x5600134775d0,
    ["/job:localhost/replica:0/task:0/cpu:0"] = 0x5600134775d0,
    ["/job:localhost/replica:0/task:0/gpu:2"] = 0x5600153277f0,
    ["/job:localhost/replica:0/task:0/xla_gpu:3"] = 0x5600152f1a80,
    ["/job:localhost/replica:0/task:0/xla_gpu:0"] = 0x5600152d7ca0,
    ["/job:localhost/replica:0/task:0/xla_gpu:1"] = 0x5600152e07c0,
    ["/job:localhost/replica:0/task:0/device:XLA_GPU:1"] = 0x5600152e07c0,
    ["/job:localhost/replica:0/task:0/gpu:1"] = 0x560015319fc0,
    ["/job:localhost/replica:0/task:0/gpu:0"] = 0x5600153086d0,
    ["/job:localhost/replica:0/task:0/device:XLA_GPU:2"] = 0x5600152e8a90,
    ["/job:localhost/replica:0/task:0/xla_gpu:2"] = 0x5600152e8a90,
    ["/job:localhost/replica:0/task:0/device:GPU:1"] = 0x560015319fc0,
    ["/job:localhost/replica:0/task:0/device:GPU:0"] = 0x5600153086d0
  },
  client_device_ = 0x5600134775d0
}
*/

// TODO 提示:
// Device* client_device() const { return client_device_; }
// 我改写的只需要 client_device 就行了。
/*
p device_set_.client_device()
$3 = (tensorflow::GPUCompatibleCPUDevice *) 0x5600134775d0

p device_set_.client_device()->name()
$4 = "/job:localhost/replica:0/task:0/device:CPU:0"
*/

// 3.
// Device 数据结构
// class Device : public DeviceBase
// tensorflow/core/common_runtime/device.h
// 重要接口:
// - Compute
// - ComputeAsync
// - FillContextMap
// - resource_manager
// 成员变量:
// - device_mgr_: DeviceMgr*
// - device_attributes_: const DeviceAttributes
// - parsed_name_: DeviceNameUtils::ParsedName
// - op_seg_: OpSegment
// - rmgr_: ResourceMgr*
// friend class:
// friend class DeviceMgr;

// 3.1
// message DeviceAttributes 数据结构说明:
// tensorflow/core/framework/device_attributes.proto
// - name: string
// - device_type: string
// - memory_limit: int64
// - locality: DeviceLocality
// - incarnation: fixed64
// - physical_device_desc: string



// A ClientGraph is simply a sub-graph of the full graph as induced by
// BuildGraphOptions.
struct ClientGraph {
  explicit ClientGraph(std::unique_ptr<FunctionLibraryDefinition> flib,
                       DataTypeVector feed_types, DataTypeVector fetch_types,
                       int64 collective_graph_key)
      : flib_def(std::move(flib)),
        graph(flib_def.get()),
        feed_types(std::move(feed_types)),
        fetch_types(std::move(fetch_types)),
        collective_graph_key(collective_graph_key) {}

  // Each client-graph gets its own function library since optimization passes
  // post rewrite for execution might want to introduce new functions.
  std::unique_ptr<FunctionLibraryDefinition> flib_def;

  // owned by itself.
  // Also, the device placement information is store in the Graph class.
  Graph graph;

  DataTypeVector feed_types;
  DataTypeVector fetch_types;
  // DataTypeVector 数据结构
  // tensorflow/core/framework/types.h:103:
  // typedef gtl::InlinedVector<DataType, 4> DataTypeVector;
  int64 collective_graph_key;
};

// 1.
// QQQ. 何时被构造?
// AAA. 在 DirectSession::CreateGraphs 内使用
// 输入:
// class DirectSession
//  ::class GraphExecutionState execution_state_
//  ::Graph* graph_
// 使用函数:
// GraphExecutionState::BuildGraph()
// 得到:
// ClientGraph instance .





// 1.
// QQQ.GraphExecutionState 用途?
// AAA.
// 1. 浏览一下成员函数，略知概况。
// 2. DirectSession::CreateGraphs 内使用 execution_state = execution_state_.get();
//    用于 execution_state->BuildGraph(...)

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
//
// QQQ. 怎么做到的？GraphDef 没有 device 相关的信息。
// AAA. NodeDef 里面存在 请求的 device placement 信息。NodeDef is within the GraphDef.
//
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
//
// GraphExecutionState is thread-safe.
class GraphExecutionState {
 public:
  virtual ~GraphExecutionState();

  // Creates a new `GraphExecutionState` for the given
  // `graph_def`, which represents the entire graph for a session.
  //
  // N.B. This method uses `GraphDef::Swap()` and leaves `graph_def`
  // in an undefined state. If it is necessary to use `*graph_def`
  // after this call, make an explicit copy of the graph before
  // calling this method.
  static Status MakeForBaseGraph(
      GraphDef* graph_def,
      const GraphExecutionStateOptions& options,
      std::unique_ptr<GraphExecutionState>* out_state);

  // Creates a new `GraphExecutionState` and `SimpleClientGraph`
  // for the subgraph of `original_graph_def` defined by
  // `subgraph_options`.
  static Status MakeForPrunedGraph(
      const FunctionDefLibrary& func_def_lib,
      const GraphExecutionStateOptions& options,
      const GraphDef& original_graph_def,
      const BuildGraphOptions& subgraph_options,
      std::unique_ptr<GraphExecutionState>* out_state,
      std::unique_ptr<ClientGraph>* out_client_graph);

  // Creates a new GraphExecutionState representing the
  // concatenation of this graph, and the graph defined by
  // "extension_def". The same name may not be used to define a node
  // in both this graph and "extension_def".
  //
  // If successful, returns OK and the caller takes ownership of "*out".
  // Otherwise returns an error and does not modify "*out".
  //
  // After calling `old_state->Extend()`, `old_state` may no longer be
  // used.
  //
  // NOTE(mrry): This method respects the placement of stateful nodes in
  // in *this, but currently does not transfer any other placement
  // or cost model information to the new graph.
  Status Extend(const GraphDef& extension_def,
                std::unique_ptr<GraphExecutionState>* out) const;

  // Builds a ClientGraph (a sub-graph of the full graph as induced by
  // the Node set specified in "options").  If successful, returns OK
  // and the caller takes the ownership of "*out". Otherwise, returns
  // an error.
  Status BuildGraph(const BuildGraphOptions& options,
                    std::unique_ptr<ClientGraph>* out);
  // - OptimizeGraph
  // - PruneGraph

  // The graph returned by BuildGraph may contain only the pruned
  // graph, whereas some clients may want access to the full graph.
  const Graph* full_graph() { return graph_; }

  // Returns the node with the given name, or null if it does not exist.
  const Node* get_node_by_name(const string& name) const {
    NodeNameToCostIdMap::const_iterator iter =
        node_name_to_cost_id_map_.find(name);
    if (iter != node_name_to_cost_id_map_.end()) {
      return graph_->FindNodeId(iter->second);
    } else {
      return nullptr;
    }
  }

  // Returns a reference to the current graph_def.  Use must
  // not extend beyond lifetime of GrahExecutionState object.
  const GraphDef& original_graph_def() { return original_graph_def_; }

  // Returns the map of stateful placements as a map of
  // node name to placement string.
  std::unordered_map<string, string> GetStatefulPlacements() const {
    return stateful_placements_;
  }

 //-----------------------------------------------------------------------
 private:
 //-----------------------------------------------------------------------
  GraphExecutionState(GraphDef* graph_def,
                      const GraphExecutionStateOptions& options);

  Status InitBaseGraph(const BuildGraphOptions& options);

  // Map of placed stateful nodes, i.e. nodes for which is_stateful()
  // is true, such as "params" and "queue" nodes.  Once placed these
  // nodes can not be moved to a different device.  Maps node names to
  // device names.
  std::unordered_map<string, string> stateful_placements_;  // Immutable after
                                                            // ctor.
  void SaveStatefulNodes(Graph* graph);
  void RestoreStatefulNodes(Graph* graph);

  // Extract the subset of the graph that needs to be run, adding feed/fetch
  // ops as needed.
  Status PruneGraph(const BuildGraphOptions& options,
                    Graph* graph,
                    subgraph::RewriteGraphMetadata* out_rewrite_metadata);

  Status OptimizeGraph(
      const BuildGraphOptions& options,
      std::unique_ptr<Graph>* optimized_graph,
      std::unique_ptr<FunctionLibraryDefinition>* optimized_flib);

  GraphDef original_graph_def_;            // Immutable after ctor.
  const DeviceSet* device_set_;            // Not owned
  const SessionOptions* session_options_;  // Not owned
  // Unique session identifier. Can be empty.
  string session_handle_;

  // Map from name to Node for the full graph in placed_.
  NodeNameToCostIdMap node_name_to_cost_id_map_;

  // 'flib_def_' is initialized from the initial graph def's library,
  // and may be updated by a graph optimization pass.
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;

  // `rewrite_metadata_` is only set for GraphExecutionState
  // objects created by `MakeForPrunedGraph()`.
  std::unique_ptr<subgraph::RewriteGraphMetadata> rewrite_metadata_;

  // The dataflow graph owned by this object.
  Graph* graph_;

  TF_DISALLOW_COPY_AND_ASSIGN(GraphExecutionState);
};
// class GraphExecutionState END

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
// 注意：
// class GraphExecutionState 数据结构 内不含有 ClientGraph 对象或者它的指针。


}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_EXECUTION_STATE_H_
