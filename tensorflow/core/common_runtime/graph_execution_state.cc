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

#include "tensorflow/core/common_runtime/graph_execution_state.h"

#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb_text.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/collective_order.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/util.h"

#ifndef IS_MOBILE_PLATFORM
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#endif  // IS_MOBILE_PLATFORM

namespace tensorflow {

GraphExecutionState::GraphExecutionState(
    GraphDef* graph_def,
    const GraphExecutionStateOptions& options)

    : stateful_placements_(options.stateful_placements),
      device_set_(options.device_set),
      session_options_(options.session_options),
      session_handle_(options.session_handle),
      flib_def_(new FunctionLibraryDefinition(OpRegistry::Global(),
                                              graph_def->library())),
      graph_(nullptr) {

  // NOTE(mrry): GraphDef does not have a move constructor, so we pass
  // a non-const pointer and use `Swap()` to transfer the contents
  // without copying.
  original_graph_def_.Swap(graph_def);
  // 1.
  // 提醒: graph_def 是来自于 TF_Session 内的 Graph 转换成 GraphDef 的原图

  // 2.
  // GraphDef original_graph_def_;
  // 居然 GraphExecutionState 内是一个独立完整的 GraphDef 啊

  // 3.
  // QQQ. class GraphExecutionState 有哪些成员变量？
  // AAA.
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

  // 3.
  // Swap 函数说明
  // 使用 Swap 的意图:
  // Swap 的目的是因为传入的参数 graph_def 复制后就没有利用价值了，所以就还不如 Swap 正好变成了一个全部为空的变量。
  // 理由: 在 DirectSession::MaybeInitializeExecutionState 内
  // 传给 GraphExecutionState::MakeForBaseGraph 的一个参数是 temp, GraphDef temp(graph)
  // 所以，那时就准备让这个 temp Swap 变成 一个空白变量了。

  // TODO(mrry): Publish placement visualizations or handle the log
  // placement option.
}

GraphExecutionState::~GraphExecutionState() {
  node_name_to_cost_id_map_.clear();
  delete graph_;
}


// 1.
// 一句人话总结:
// 构造 GraphExecutionState instance, 并把 GraphExecutionState instance 内的图 device placement 做好.
// 图的效果是: https://gist.github.com/shizukanaskytree/bfd21d79a7c13bf0b945a3710c1c13ef
/* static */ Status GraphExecutionState::MakeForBaseGraph(
    GraphDef* graph_def, // input, temp GraphDef var
    const GraphExecutionStateOptions& options, // input
    std::unique_ptr<GraphExecutionState>* out_state) { // output, 实际过程中 out_state 是 DirectSession::execution_state_

#ifndef __ANDROID__
  VLOG(4) << "Graph proto is \n" << graph_def->DebugString();
#endif  // __ANDROID__

  std::unique_ptr<GraphExecutionState> ret(
      new GraphExecutionState(graph_def, options));
  // 1.
  // graph_def 变量说明:
  // 这是一个 temp GraphDef var

  // 2.
  // GraphDef 数据结构
  // - node: repeated NodeDef
  // - versions: VersionDef
  // - library: FunctionDefLibrary # EXPERIMENTAL. DO NOT USE OR DEPEND ON THIS YET.
  //   注意: 这一项通常都为空

  // 3.
  // struct GraphExecutionStateOptions 数据结构说明:
  // tensorflow/core/common_runtime/graph_execution_state.h
  //
  // - device_set: const DeviceSet*
  // - session_options: const SessionOptions*
  // - session_handle: string
  //   Unique session identifier. Can be empty.
  // - stateful_placements
  //   A map from node name to device name, representing the unchangeable
  //   placement of stateful nodes.

  // 3.1
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

  // 3.2
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

  // 3.3
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

  // 3.4
  // TODO 提示:
  // Device* client_device() const { return client_device_; }
  // 我改写的只需要 client_device 就行了。
  /*
  p device_set_.client_device()
  $3 = (tensorflow::GPUCompatibleCPUDevice *) 0x5600134775d0

  p device_set_.client_device()->name()
  $4 = "/job:localhost/replica:0/task:0/device:CPU:0"
  */

  // 4.
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

  // 4.1
  // message DeviceAttributes 数据结构说明:
  // tensorflow/core/framework/device_attributes.proto
  // - name: string
  // - device_type: string
  // - memory_limit: int64
  // - locality: DeviceLocality
  // - incarnation: fixed64
  // - physical_device_desc: string


  TF_RETURN_IF_ERROR(
      AddDefaultAttrsToGraphDef(
        &ret->original_graph_def_,  // input and output
        *ret->flib_def_, // output
        0)); // input
      // 1.
      // AddDefaultAttrsToGraphDef 函数说明
      // tensorflow/core/framework/graph_def_util.cc
      // Status AddDefaultAttrsToGraphDef(
      //   GraphDef* graph_def,
      //   const OpRegistryInterface& op_registry,
      //   int node_offset)


  // TODO(mrry): Refactor InitBaseGraph() so that we don't have to
  // pass an empty BuildGraphOptions (that isn't going to be used when
  // place_pruned_graph is false).
  if (!ret->session_options_->config.graph_options().place_pruned_graph()) {
    TF_RETURN_IF_ERROR(

      // -----------------------------------------------------------------------
      ret->InitBaseGraph(BuildGraphOptions()));
      // -----------------------------------------------------------------------
      // 1.
      // InitBaseGraph 函数说明
      // GraphExecutionState::InitBaseGraph(const BuildGraphOptions& options)
      // tensorflow/core/common_runtime/graph_execution_state.cc

      // 2.
      // struct BuildGraphOptions 数据结构
      // tensorflow/core/common_runtime/build_graph_options.h
      // - callable_options: CallableOptions
      // - use_function_convention: bool, default: false
      // - kNoCollectiveGraphKey: static const int64, default: 0
      // - collective_graph_key: int64, default: kNoCollectiveGraphKey
      // - collective_order: GraphCollectiveOrder

      // 3.
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

  }

  *out_state = std::move(ret);
  return Status::OK();
}
// 1.
// where are we ?
//
// Thread #1 [python] 15583 [core: 55] (Suspended : Step)
// 	tensorflow::GraphExecutionState::MakeForBaseGraph() 👀 at graph_execution_state.cc:99 0x7efc1b23ac2a
// 	tensorflow::DirectSession::MaybeInitializeExecutionState() 👀 at direct_session.cc:1,694 0x7efc1708f659
// 	tensorflow::DirectSession::ExtendLocked() at direct_session.cc:1,740 0x7efc1708fa41
// 	tensorflow::DirectSession::Extend() at direct_session.cc:1,733 0x7efc1708f9fc
// 	tensorflow::SessionRef::Extend() at session_ref.cc:441 0x7efc12b73833
// 	tensorflow::ExtendSessionGraphHelper() at c_api.cc:815 0x7efc171137c3
// 	tensorflow::ExtendSession() at python_api.cc:118 0x7efc12bac2e6
// 	_wrap_ExtendSession() at pywrap_tensorflow_internal.cc:19,726 0x7efc12ae1b45
// 	_PyCFunction_FastCallDict() at methodobject.c:234 0x55be63d17681
// 	call_function() at ceval.c:4,851 0x55be63d9e610
// 	<...more frames...>



/* static */ Status GraphExecutionState::MakeForPrunedGraph(
    const FunctionDefLibrary& func_def_lib, // input
    const GraphExecutionStateOptions& options, // input
    const GraphDef& graph_def, // input
    const BuildGraphOptions& subgraph_options, // input
    std::unique_ptr<GraphExecutionState>* out_state, // output
    std::unique_ptr<ClientGraph>* out_client_graph) { // output

  DCHECK(options.session_options->config.graph_options().place_pruned_graph());
  // NOTE(mrry): This makes a copy of `graph_def`, which is
  // regrettable. We could make `GraphDef` objects sharable between
  // execution states to optimize pruned graph execution, but since
  // this case is primarily used for interactive sessions, we make the
  // bet that graph construction is not performance-critical. (Note
  // also that the previous version used `Extend()`, which is strictly
  // more expensive than copying a `GraphDef`.)
  GraphDef temp(graph_def);

  std::unique_ptr<GraphExecutionState> ret(
      new GraphExecutionState(&temp, options));

  TF_RETURN_IF_ERROR(
      AddDefaultAttrsToGraphDef(&ret->original_graph_def_, *ret->flib_def_, 0));

  TF_RETURN_IF_ERROR(ret->InitBaseGraph(subgraph_options));

  TF_RETURN_IF_ERROR(ret->BuildGraph(subgraph_options, out_client_graph));

  *out_state = std::move(ret);

  return Status::OK();
}


Status GraphExecutionState::Extend(
    const GraphDef& extension_def,  // input
    std::unique_ptr<GraphExecutionState>* out) const {  // output

  GraphDef gdef;

  // 1. Copy the function library.
  TF_RETURN_IF_ERROR(flib_def_->AddLibrary(extension_def.library()));
  *gdef.mutable_library() = flib_def_->ToProto();

  // 2. Build an index of the new node names.
  std::unordered_set<string> new_names;
  for (const NodeDef& node : extension_def.node()) {
    new_names.insert(node.name());
  }

  // 3. Add the non-duplicates from the old graph to the new graph.
  //    Return an error if the same node name appears in both the
  //    old graph and the extension.
  for (const NodeDef& node : original_graph_def_.node()) {
    if (new_names.count(node.name()) == 0) {
      *gdef.add_node() = node;
    } else {
      return errors::InvalidArgument(tensorflow::strings::Printf(
          "GraphDef argument to Extend includes node '%s', which was created "
          "by a previous call to Create or Extend in this session.",
          node.name().c_str()));
    }
  }

  // 4. Merge the versions field.
  int old_node_size = gdef.node_size();
  gdef.mutable_node()->MergeFrom(extension_def.node());
  TF_RETURN_IF_ERROR(
      AddDefaultAttrsToGraphDef(&gdef, *flib_def_, old_node_size));
  // Merge versions
  if (gdef.has_versions()) {
    if (gdef.versions().producer() != extension_def.versions().producer()) {
      return errors::InvalidArgument(
          "Can't extend GraphDef at version ", gdef.versions().producer(),
          " with graph at version ", extension_def.versions().producer());
    }
    VersionDef* versions = gdef.mutable_versions();
    versions->set_min_consumer(std::max(
        versions->min_consumer(), extension_def.versions().min_consumer()));
    if (extension_def.versions().bad_consumers_size()) {
      // Add new bad_consumers that aren't already marked bad.
      //
      // Note: This implementation is quadratic time if there are many calls to
      // ExtendLocked with many bad consumers.  Since this is unlikely, and
      // fixing it would require data structures outside of this routine,
      // quadratic time it is.
      auto* bad_consumers = versions->mutable_bad_consumers();
      const std::unordered_set<int> existing(bad_consumers->begin(),
                                             bad_consumers->end());
      for (const int v : extension_def.versions().bad_consumers()) {
        if (existing.find(v) == existing.end()) {
          bad_consumers->Add(v);
        }
      }
    }

  } else {
    gdef.mutable_versions()->CopyFrom(extension_def.versions());
  }

  // 5. Validate that the final graphdef is valid.
  if (gdef.versions().producer() >= 5) {
    // Validate the graph: we assume that merging two valid graphs
    // should maintain graph validity.
    TF_RETURN_IF_ERROR(graph::ValidateGraphDef(gdef, *flib_def_));
  }

  // 6. Add the extension.
  GraphExecutionStateOptions combined_options;
  combined_options.device_set = device_set_;
  combined_options.session_options = session_options_;
  combined_options.session_handle = session_handle_;
  combined_options.stateful_placements = stateful_placements_;

  // NOTE(mrry): `gdef` is no longer valid after the constructor
  // executes.
  std::unique_ptr<GraphExecutionState> new_execution_state(
      new GraphExecutionState(&gdef, combined_options));

  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(
      &new_execution_state->original_graph_def_, *flib_def_, 0));
  if (!session_options_->config.graph_options().place_pruned_graph()) {
    // TODO(mrry): Refactor InitBaseGraph() so that we don't have to
    // pass an empty BuildGraphOptions (that isn't going to be used
    // when place_pruned_graph is false).
    TF_RETURN_IF_ERROR(new_execution_state->InitBaseGraph(BuildGraphOptions()));
  }
  *out = std::move(new_execution_state);

  // TODO(mrry): This is likely to be used for non-throughput-sensitive
  // interactive workloads, but in future we may want to transfer other
  // parts of the placement and/or cost model.
  return Status::OK();
}


void GraphExecutionState::SaveStatefulNodes(Graph* graph) {
  for (Node* n : graph->nodes()) {
    if (n->op_def().is_stateful()) {
      // 1.
      // is_stateful() 说明:
      // As others have mentioned, stateful objects are those holding a state. Now, a state, in TensorFlow terms, is some value or data that is saved between different calls to tf.Session.run .
      // Indeed, even if your graph consists only on operations with constant values, tensor operations will be evaluated every time you call run, even though the result will always be the same. When you assign a value to a variable, however, it will "stick" (and, by the way, take the corresponding memory and, if you choose so, be serialized on checkpoints).
      // https://stackoverflow.com/questions/52636943/what-is-a-stateful-object-in-tensorflow

      // 2.
      // https://www.tensorflow.org/api_docs/python/tf/Graph

      // 3.
      // 目的: 因为我想深度复制 state , 所以我想知道 state 在哪里
      VLOG(2) << "Saving " << n->DebugString();
      stateful_placements_[n->name()] = n->assigned_device_name();

      // stateful_placements_ 变量说明
      // tensorflow/core/common_runtime/direct_session.h:395:
      // std::unordered_map<string, string> stateful_placements_
      //
      // QQQ. 在哪里被使用的?
      // AAA.
      // tensorflow/core/common_runtime/graph_execution_state.cc
      // void GraphExecutionState::RestoreStatefulNodes(Graph* graph) {
      //   ...
      //

      // 4.
      // https://stackoverflow.com/questions/42783909/internals-of-variable-in-tensorflow

      // 5.
      // https://stackoverflow.com/questions/52636943/what-is-a-stateful-object-in-tensorflow

    }
  }
}

void GraphExecutionState::RestoreStatefulNodes(Graph* graph) {
  for (Node* n : graph->nodes()) {
    if (n->op_def().is_stateful()) {
      // 1.
      // n->op_def().is_stateful() 是什么?
      // 去 tensorflow/core/framework/op_def.proto 看
      // bool is_stateful = 17;  // for things like variables, queue
      // **for things like variables, queue**
      // Ops are marked as stateful if their behavior depends on some state beyond
      // their input tensors (e.g. variable reading op
      // 不是特别懂, 但是感觉理解了.

      auto iter = stateful_placements_.find(n->name());
      if (iter != stateful_placements_.end()) {
        n->set_assigned_device_name(iter->second);
        // 1.
        // 设置 Node 的 device_name
        VLOG(2) << "Restored " << n->DebugString();
      }
    }
  }
}

namespace {

class TensorConnectionPruneRewrite : public subgraph::PruneRewrite {
 public:
  TensorConnectionPruneRewrite(const string* endpoint_name,
                               NodeBuilder::NodeOut from_tensor)
      : subgraph::PruneRewrite(endpoint_name, nullptr /* device_info */),
        from_tensor_(std::move(from_tensor)) {}

  Status AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                 Node** out_node) override {
    Status s;
    auto check_no_cycle_fn = [this, feed_tensor, &s](Node* n) {
      if (n == feed_tensor.node) {
        s.Update(errors::InvalidArgument(
            "Requested Tensor connection between nodes \"",
            feed_tensor.node->name(), "\" and \"", from_tensor_.node->name(),
            "\" would create a cycle."));
      }
    };
    ReverseDFSFrom(*g, {from_tensor_.node}, std::move(check_no_cycle_fn),
                   nullptr);
    TF_RETURN_IF_ERROR(s);

    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat("_identity_", feed_tensor.node->name(), "_",
                                    feed_tensor.index),
                    "Identity")
            .Input(from_tensor_)
            .Attr("T",
                  BaseType(from_tensor_.node->output_type(from_tensor_.index)))
            .Finalize(g, out_node));

    (*out_node)->set_assigned_device_name(
        feed_tensor.node->assigned_device_name());
    return Status::OK();
  }

 private:
  NodeBuilder::NodeOut from_tensor_;
};

template <class Map>
Status LookupDevice(
  const DeviceSet& device_set, // input
  const string& tensor_name, // input
  const Map& tensor2device, // input
  const tensorflow::DeviceAttributes** out_device_attrs) { //output

  *out_device_attrs = nullptr;

  if (tensor2device.empty()) {
    // 1.
    // 进入
    // 对于 feed nodes 而言.

    // 2.
    // 对于 feed nodes 而言, 由于 tensor2device.empty() 为 True, 致使如下 *out_device_attrs 使用了 CPU!

    // 3.
    // device_set: class DeviceSet
    // tensorflow/core/common_runtime/device_set.h:32:class DeviceSet {
    // - client_device_: Device*
    //    * client_device_ points to an element of devices_ that we consider
    //    * to be the client device (in this local process).
    // - device_by_name_: std::unordered_map<string, Device*>
    //    * Fullname -> device* for device in devices_.
    // - devices_: std::vector<Device*>

    // 4.
    // &device_set.client_device(): Device*

    // 5.
    // &device_set.client_device()->attributes() : tensorflow::DeviceAttributes**

    // 5.1
    // DeviceAttributes 数据结构
    // tensorflow/core/framework/device_attributes.proto:32:
    // message DeviceAttributes
    // - name : string
    // - device_type: string
    // - memory_limit: int64
    // - locality : DeviceLocality
    // - incarnation : fixed64
    // - physical_device_desc: string

    *out_device_attrs = &device_set.client_device()->attributes();

    // 1.
    // 打印 *out_device_attrs
    // p (*out_device_attrs )->DebugString()
    //
    // "name: \"/job:localhost/replica:0/task:0/device:CPU:0\"\ndevice_type: \"CPU\"\nmemory_limit: 268435456\nlocality {\n}\nincarnation: 6649073119670775225\n"
    //
    // 即:
    //
    // name: "/job:localhost/replica:0/task:0/device:CPU:0"
    // device_type: "CPU"
    // memory_limit: 268435456
    // locality {
    // }
    // incarnation: 6649073119670775225

    return Status::OK();
  } // 解释返回了.


  const auto it = tensor2device.find(tensor_name);
  if (it == tensor2device.end()) {
    *out_device_attrs = &device_set.client_device()->attributes();
    return Status::OK();
  }

  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(it->second, &parsed_name)) {
    return errors::InvalidArgument("Invalid device name ('", it->second,
                                   "') provided for the tensor '", tensor_name,
                                   "' in CallableOptions");
  }

  Device* device = device_set.FindDeviceByName(
      DeviceNameUtils::ParsedNameToString(parsed_name));
  if (device == nullptr) {
    return errors::InvalidArgument("Device '", it->second,
                                   "' specified for tensor '", tensor_name,
                                   "' in CallableOptions does not exist");
  }

  *out_device_attrs = &device->attributes();

  return Status::OK();
}

struct TensorAndDevice {
  // WARNING: backing memory for the 'tensor' field is NOT owend.
  const TensorId tensor;
  // 1.
  // struct TensorId  数据结构
  // tensorflow/core/graph/tensor_id.h
  //   : public std::pair<StringPiece, int>
  // 含义是 first == operation_name, second == output_index

  // WARNING: device pointer is not owned, so must outlive TensorAndDevice.
  const DeviceAttributes* device;
};
// 1.
// struct TensorAndDevice 数据结构
// tensorflow/core/common_runtime/graph_execution_state.cc
// - tensor: const TensorId
//    * Base: typedef std::pair<StringPiece, int> Base
//    * first == operation_name, second == output_index
//      打印: strings::StrCat(first, ":", second);
// - device: const DeviceAttributes*

// 2.
// message DeviceAttributes 数据结构
// tensorflow/core/framework/device_attributes.proto
// - name: string
// - device_type: string
// - memory_limit: int64
// - locality: DeviceLocality
// - incarnation: fixed64
// - physical_device_desc: string


// Tensors of some DataTypes cannot placed in device memory as feeds or
// fetches. Validate against a whitelist of those known to work.
bool IsFeedAndFetchSupported(DataType dtype, const string& device_type) {
  // The mechanism for supporting feeds of device-backed Tensors requires
  // the _Arg kernel to be registered for the corresponding type (and that
  // the input to the kernel be in device and not host memory).
  //
  // The mechanism for supporting fetches of device-backed Tensors requires
  // the _Retval kernel to be registered for the corresponding type (and
  // that the output is produced in device and not host memory).
  //
  // For now, we return true iff there are _Arg AND _Retval kernels for dtype on
  // the device. False negatives are okay, false positives would be bad.
  //
  // TODO(ashankar): Instead of a whitelist here, perhaps we could query
  // the kernel registry for _Arg and _Retval kernels instead.
  if (device_type == DEVICE_CPU) return true;
  if (device_type != DEVICE_GPU) return false;
  switch (dtype) {
    case DT_BFLOAT16:
    case DT_BOOL:
    case DT_COMPLEX128:
    case DT_COMPLEX64:
    case DT_DOUBLE:
    case DT_FLOAT:
    case DT_HALF:
    case DT_INT16:
    case DT_INT64:
    case DT_INT8:
    case DT_UINT16:
    case DT_UINT8:
      return true;
    default:
      return false;
  }
}

Status ValidateFeedAndFetchDevices(
    const Graph& graph,
    const std::vector<TensorAndDevice>& tensors_and_devices) {
  // 1.
  // struct TensorAndDevice 数据结构
  // tensorflow/core/common_runtime/graph_execution_state.cc
  // - tensor: const TensorId
  //    * Base: typedef std::pair<StringPiece, int> Base
  //    * first == operation_name, second == output_index
  //      打印: strings::StrCat(first, ":", second);
  // - device: const DeviceAttributes*

  // 2.
  // message DeviceAttributes 数据结构
  // tensorflow/core/framework/device_attributes.proto
  // - name: string
  // - device_type: string
  // - memory_limit: int64
  // - locality: DeviceLocality
  // - incarnation: fixed64
  // - physical_device_desc: string

  if (tensors_and_devices.empty()) return Status::OK();

  std::vector<bool> found(tensors_and_devices.size(), false);

  for (const Node* node : graph.nodes()) {
    // Linearly looping through all nodes and then all feed+fetch tensors isn't
    // quite efficient. At the time of this writing, the expectation was that
    // tensors_and_devices.size() is really small in practice, so this won't be
    // problematic.
    // Revist and make a more efficient lookup possible if needed (e.g., perhaps
    // Graph can maintain a map from node name to Node*).
    for (int i = 0; i < tensors_and_devices.size(); ++i) {
      const TensorAndDevice& td = tensors_and_devices[i];
      if (td.tensor.first != node->name()) continue;
      // 1.
      // td.tensor.first 变量说明:
      // 是 operation_name

      found[i] = true;

      TF_RETURN_IF_ERROR(graph.IsValidOutputTensor(node, td.tensor.second));
      const DataType dtype = node->output_type(td.tensor.second);
      if (!IsFeedAndFetchSupported(dtype, td.device->device_type())) {
        return errors::Unimplemented(
            "Cannot feed or fetch tensor '", td.tensor.ToString(),
            "' from device ", td.device->name(), " as feeding/fetching from ",
            td.device->device_type(), " devices is not yet supported for ",
            DataTypeString(dtype), " tensors");
      }
    }
  }

  for (int i = 0; i < found.size(); ++i) {
    if (!found[i]) {
      return errors::InvalidArgument(
          "Tensor ", tensors_and_devices[i].tensor.ToString(),
          ", specified in either feed_devices or fetch_devices was not found "
          "in the Graph");
    }
  }
  return Status::OK();
}

Status GetFeedShapeAndTypeFromAttribute(const NodeDef& node,
                                        PartialTensorShape* shape,
                                        DataType* type) {
  static const gtl::FlatSet<string>* const kHasExplicitShapeAttribute =
      CHECK_NOTNULL((new gtl::FlatSet<string>{
          "Placeholder", "PlaceholderV2", "PlaceholderWithDefault",
          "ParallelConcat", "ImmutableConst", "_ParallelConcatStart",
          "InfeedDequeue", "OutfeedDequeue", "CollectiveBcastSend",
          "CollectiveBcastRecv", "AccumulateNV2", "VariableV2", "Variable",
          "TemporaryVariable", "NcclBroadcast", "_ScopedAllocator",
          "_ScopedAllocatorConcat"}));

  // All the node types handled here have their output datatype set in
  // either attribute 'dtype' or 'T'.
  if (!GetNodeAttr(node, "dtype", type).ok() &&
      !GetNodeAttr(node, "T", type).ok()) {
    return errors::InvalidArgument(
        "Could not determine output type for feed node: ", node.name(),
        " of type ", node.op());
  }

  // First handle the case of feeding a const node.
  if (node.op() == "Const" && HasNodeAttr(node, "value")) {
    *shape =
        PartialTensorShape(node.attr().at("value").tensor().tensor_shape());
  } else if (kHasExplicitShapeAttribute->find(node.op()) !=
             kHasExplicitShapeAttribute->end()) {
    TF_RETURN_IF_ERROR(GetNodeAttr(node, "shape", shape));
  } else {
    return errors::InvalidArgument("Could not determine shape for feed node: ",
                                   node.name(), " of type ", node.op());
  }
  return Status::OK();
}

}  // namespace

// Where am I ?
//
// Thread #1 [python] 17387 [core: 3] (Suspended : Step)
// 	tensorflow::GraphExecutionState::PruneGraph() 👀 at graph_execution_state.cc:564 0x7f855b23e460
// 	tensorflow::GraphExecutionState::BuildGraph() 👀 at graph_execution_state.cc:806 0x7f855b240643
// 	tensorflow::DirectSession::CreateGraphs() 👀 at direct_session.cc:3,191 0x7f855709a515
// 	tensorflow::DirectSession::CreateExecutors() 👀 at direct_session.cc:2,627 0x7f8557095f7e
// 	tensorflow::DirectSession::GetOrCreateExecutors() at direct_session.cc:3,032 0x7f8557098fdc
// 	tensorflow::DirectSession::Run() at direct_session.cc:2,147 0x7f8557092802
// 	tensorflow::SessionRef::Run() at session_ref.cc:414 0x7f8552b72f8a
// 	TF_Run_Helper() at c_api.cc:878 0x7f8557113b96
// 	TF_SessionRun() at c_api.cc:2,752 0x7f855711d45e
// 	tensorflow::TF_SessionRun_wrapper_helper() at tf_session_helper.cc:407 0x7f8552b673e1
// 	<...more frames...>
Status GraphExecutionState::PruneGraph(
    const BuildGraphOptions& options, // input
    Graph* graph, // input
    subgraph::RewriteGraphMetadata* out_rewrite_metadata) // output
{
  // 1.
  // 进来时的图 graph:
  //
  // node {
  //   name: "x"
  //   op: "Placeholder"
  //   device: "/job:localhost/replica:0/task:0/device:GPU:0"
  //   attr {
  //     key: "dtype"
  //     value {
  //       type: DT_FLOAT
  //     }
  //   }
  //   attr {
  //     key: "shape"
  //     value {
  //       shape {
  //         dim {
  //           size: 2
  //         }
  //         dim {
  //           size: 5
  //         }
  //       }
  //     }
  //   }
  // }
  // node {
  //   name: "y"
  //   op: "Placeholder"
  //   device: "/job:localhost/replica:0/task:0/device:GPU:0"
  //   attr {
  //     key: "dtype"
  //     value {
  //       type: DT_FLOAT
  //     }
  //   }
  //   attr {
  //     key: "shape"
  //     value {
  //       shape {
  //         dim {
  //           size: 5
  //         }
  //         dim {
  //           size: 3
  //         }
  //       }
  //     }
  //   }
  // }
  // node {
  //   name: "MatMul"
  //   op: "MatMul"
  //   input: "x"
  //   input: "y"
  //   device: "/job:localhost/replica:0/task:0/device:GPU:0"
  //   attr {
  //     key: "T"
  //     value {
  //       type: DT_FLOAT
  //     }
  //   }
  //   attr {
  //     key: "transpose_a"
  //     value {
  //       b: false
  //     }
  //   }
  //   attr {
  //     key: "transpose_b"
  //     value {
  //       b: false
  //     }
  //   }
  // }
  // library {
  // }
  // versions {
  //   producer: 27
  // }

  std::vector<std::unique_ptr<subgraph::PruneRewrite>> feed_rewrites;
  feed_rewrites.reserve(options.callable_options.feed_size());
  std::vector<std::unique_ptr<subgraph::PruneRewrite>> fetch_rewrites;
  fetch_rewrites.reserve(options.callable_options.fetch_size());
  // 1.
  // 一句话概括: feed_rewrites, fetch_rewrites of the client graph by adding _Arg
  // and _Retval and by removing placeholders.

  // 2.
  // class PruneRewrite 数据结构
  // tensorflow/core/graph/subgraph.h
  // - endpoint_name_: const string* const
  // - device_info_: const DeviceAttributes* const

  // 3.
  // PruneRewrite 虚函数，继承的函数是
  // - class ArgFeedRewrite
  // - class RecvFeedRewrite
  // - class RetvalFetchRewrite
  // - class SendFetchRewrite
  // - class TensorConnectionPruneRewrite

  // 4.
  // options 变量说明:
  // options: const BuildGraphOptions&
  // 打印:
  // BuildGraphOptions options 打印:
  //
  // Feed endpoints:
  // Fetch endpoints: out:0,
  // Target nodes:
  // collective_order: none

  if (options.use_function_convention) {
    // 进入

    // 1.
    // options.use_function_convention 含义:
    // If `true`, uses Arg/Retval to implement feeds/fetches; otherwise
    // uses Recv/Send to implement feeds/fetches.

    std::vector<TensorAndDevice> tensors_and_devices;
    // 1.
    // tensorflow/core/common_runtime/graph_execution_state.cc:322:
    // struct TensorAndDevice
    //  - tensor: const TensorId
    //  - device: const DeviceAttributes*

    // === feed ===
    for (int i = 0; i < options.callable_options.feed_size(); ++i) {
      // 1.
      // (gdb) p options.callable_options.feed_size()
      // $3 = 2

      // WARNING: feed MUST be a reference, since ArgFeedRewrite and
      // tensors_and_devices holds on to its address.
      const string& feed = options.callable_options.feed(i);
      const DeviceAttributes* device_info;
      TF_RETURN_IF_ERROR(
        LookupDevice(*device_set_, // input
                     feed, // input
                     options.callable_options.feed_devices(), // input 空的 const Map& tensor2devic
                     &device_info));  // output, &device_info == const tensorflow::DeviceAttributes** out_device_attrs
      // 1.
      // LookupDevice 函数说明:
      // tensorflow/core/common_runtime/graph_execution_state.cc ✅
      // Status LookupDevice(const DeviceSet& device_set, const string& tensor_name,
      //               const Map& tensor2device,
      //               const tensorflow::DeviceAttributes** out_device_attrs)

      // 2.
      // options.callable_options.feed_devices()
      //
      // inline const ::google::protobuf::Map< ::std::string, ::std::string >&
      // CallableOptions::feed_devices() const {
      //   // @@protoc_insertion_point(field_map:tensorflow.CallableOptions.feed_devices)
      //   return feed_devices_.GetMap();
      // }

      // 2.1
      // message CallableOptions {
      //   ...
      //   map<string, string> feed_devices = 6;
      //   ...
      // }
      // tensorflow/core/protobuf/config.proto

      // 2.2
      // p options.callable_options.DebugString()
      // $17 = "feed: \"x:0\"\nfeed: \"y:0\"\nfetch: \"MatMul:0\"\nrun_options {\n  debug_options {\n  }\n  experimental {\n  }\n}\n"
      //
      // 即:
      //
      // feed: "x:0"
      // feed: "y:0"
      // fetch: "MatMul:0"
      // run_options {
      //   debug_options {
      //   }
      //   experimental {
      //   }
      // }

      // 3.
      // (gdb) p device_info->DebugString()
      // $23 = "name: \"/job:localhost/replica:0/task:0/device:CPU:0\"\ndevice_type: \"CPU\"\nmemory_limit: 268435456\nlocality {\n}\nincarnation: 6649073119670775225\n"

      feed_rewrites.emplace_back(
          new subgraph::ArgFeedRewrite(&feed, device_info, i));
      // 1.
      // ArgFeedRewrite 构造函数
      // 这样, 后面才能用上它里面的 ArgFeedRewrite::AddNode()

      tensors_and_devices.push_back({ParseTensorName(feed), device_info});
    }

    if (!options.callable_options.fetch_devices().empty() &&
        !options.callable_options.fetch_skip_sync()) {
      return errors::Unimplemented(
          "CallableOptions.fetch_skip_sync = false is not yet implemented. You "
          "can set it to true instead, but MUST ensure that Device::Sync() is "
          "invoked on the Device corresponding to the fetched tensor before "
          "dereferencing the Tensor's memory.");
    }

    // === fetch ===
    for (int i = 0; i < options.callable_options.fetch_size(); ++i) {
      // WARNING: fetch MUST be a reference, since RetvalFetchRewrite and
      // tensors_and_devices holds on to its address.
      const string& fetch = options.callable_options.fetch(i);
      const DeviceAttributes* device_info;
      // DeviceAttributes 数据结构
      // tensorflow/core/framework/device_attributes.proto:32:
      // message DeviceAttributes
      // - name : string
      // - device_type: string
      // - memory_limit: int64
      // - locality : DeviceLocality
      // - incarnation : fixed64
      // - physical_device_desc: string

      TF_RETURN_IF_ERROR(
        // LookupDevice 接口说明
        // tensorflow/core/common_runtime/graph_execution_state.cc:292:
        // Status LookupDevice(
        //   const DeviceSet& device_set,
        //   const string& tensor_name,
        //   const Map& tensor2device,
        //   const tensorflow::DeviceAttributes** out_device_attrs)
        LookupDevice(
          *device_set_, // input
          fetch, // input, string, e.g. p fetch, "out:0"
          options.callable_options.fetch_devices(), // input
          &device_info)); // output

        /**
        // 1.
        p *device_set_
        $4= {
          devices_=std::vector of length 10,
          capacity 16= {
            0x560996a06610,
            0x560998615a60,
            0x56099886c990,
            0x5609988754b0,
            0x56099887d780,
            0x560998886730,
            0x56099889d370,
            0x5609988aec80,
            0x5609988bc4b0,
            0x5609988c9e30
          },
          device_by_name_=std::unordered_map with 20 elements= {
            ["/job:localhost/replica:0/task:0/gpu:3"]=0x5609988c9e30,
            ["/job:localhost/replica:0/task:0/device:GPU:3"]=0x5609988c9e30,
            ["/job:localhost/replica:0/task:0/device:GPU:2"]=0x5609988bc4b0,
            ["/job:localhost/replica:0/task:0/device:XLA_GPU:0"]=0x56099886c990,
            ["/job:localhost/replica:0/task:0/xla_cpu:0"]=0x560998615a60,
            ["/job:localhost/replica:0/task:0/device:XLA_GPU:3"]=0x560998886730,
            ["/job:localhost/replica:0/task:0/device:XLA_CPU:0"]=0x560998615a60,
            ["/job:localhost/replica:0/task:0/device:CPU:0"]=0x560996a06610,
            ["/job:localhost/replica:0/task:0/cpu:0"]=0x560996a06610,
            ["/job:localhost/replica:0/task:0/gpu:2"]=0x5609988bc4b0,
            ["/job:localhost/replica:0/task:0/xla_gpu:3"]=0x560998886730,
            ["/job:localhost/replica:0/task:0/xla_gpu:0"]=0x56099886c990,
            ["/job:localhost/replica:0/task:0/xla_gpu:1"]=0x5609988754b0,
            ["/job:localhost/replica:0/task:0/device:XLA_GPU:1"]=0x5609988754b0,
            ["/job:localhost/replica:0/task:0/gpu:1"]=0x5609988aec80,
            ["/job:localhost/replica:0/task:0/gpu:0"]=0x56099889d370,
            ["/job:localhost/replica:0/task:0/device:XLA_GPU:2"]=0x56099887d780,
            ["/job:localhost/replica:0/task:0/xla_gpu:2"]=0x56099887d780,
            ["/job:localhost/replica:0/task:0/device:GPU:1"]=0x5609988aec80,
            ["/job:localhost/replica:0/task:0/device:GPU:0"]=0x56099889d370
          },
          client_device_=0x560996a06610
        }

        // 2.

        p fetch
        $5 = "out:0"

        // 3.

        message CallableOptions, tensorflow/core/protobuf/config.proto

        map<string, string> feed_devices = 6;
        ---
        map<string, string> fetch_devices = 7;
        ---

        p options.callable_options.DebugString()
        fetch: "out:0"
        run_options {
          debug_options {
          }
          experimental {
          }
        }

        这里面没有提前写 device

        // 4.

        p device_info->DebugString()

        name: "/job:localhost/replica:0/task:0/device:CPU:0"
        device_type: "CPU"
        memory_limit: 268435456
        locality {
        }
        incarnation: 6778765248398393778

        */
        // 5.
        // class RetvalFetchRewrite 数据结构:
        // tensorflow/core/graph/subgraph.h

      fetch_rewrites.emplace_back(
        new subgraph::RetvalFetchRewrite(
          &fetch, // input, "out:0"
          device_info, // input, 见上面
          i) // input
      );
      // 1.
      // fetch_rewrites 变量说明
      // fetch_rewrites: std::vector<std::unique_ptr<subgraph::PruneRewrite>>

      tensors_and_devices.push_back(
        {
          ParseTensorName(fetch),
          device_info
        }
      );
      // 1.
      // tensors_and_devices 数据结构
      // tensors_and_devices: struct TensorAndDevice

    } // fetch part done!

    TF_RETURN_IF_ERROR(
        ValidateFeedAndFetchDevices(*graph, tensors_and_devices));

  // another branch of if statement
  } else {
    // 未进入

    if (!options.callable_options.feed_devices().empty() ||
        !options.callable_options.fetch_devices().empty()) {
      return errors::Unimplemented(
          "CallableOptions::feed_devices and CallableOptions::fetch_devices "
          "to configure feeding/fetching tensors to/from device memory is not "
          "yet supported when using a remote session.");
    }
    const DeviceAttributes* device_info =
        &device_set_->client_device()->attributes();

    for (const string& feed : options.callable_options.feed()) {
      feed_rewrites.emplace_back(
          new subgraph::RecvFeedRewrite(&feed, device_info));
    }

    for (const string& fetch : options.callable_options.fetch()) {
      fetch_rewrites.emplace_back(
          new subgraph::SendFetchRewrite(&fetch, device_info));
    }
  }
  // if end

  for (const TensorConnection& tensor_connection :
       options.callable_options.tensor_connection()) {
    // 未进入

    Node* from_node = nullptr;
    TensorId from_id(ParseTensorName(tensor_connection.from_tensor()));

    for (Node* n : graph->nodes()) {
      if (n->name() == from_id.first) {
        from_node = n;
        break;
      }
    }

    if (from_node == nullptr) {
      return errors::InvalidArgument(
          "Requested tensor connection from unknown node: \"",
          tensor_connection.to_tensor(), "\".");
    }

    if (from_id.second >= from_node->num_outputs()) {
      return errors::InvalidArgument(
          "Requested tensor connection from unknown edge: \"",
          tensor_connection.to_tensor(),
          "\" (actual number of outputs = ", from_node->num_outputs(), ").");
    }

    feed_rewrites.emplace_back(new TensorConnectionPruneRewrite(
        &tensor_connection.to_tensor(), {from_node, from_id.second}));
  }

  std::vector<string> target_node_names(
      options.callable_options.target().begin(),
      options.callable_options.target().end());
  // 1.
  // target_node_names
  // (gdb) p target_node_names
  // $24 = std::vector of length 0, capacity 0

  // 进入 subgraph::RewriteGraphForExecution() 前的图:
  // p graph->ToGraphDefDebug().DebugString()
  // node {
  //   name: "x"
  //   op: "Placeholder"
  //   device: "/job:localhost/replica:0/task:0/device:GPU:0"
  //   attr {
  //     key: "dtype"
  //     value {
  //       type: DT_FLOAT
  //     }
  //   }
  //   attr {
  //     key: "shape"
  //     value {
  //       shape {
  //         dim {
  //           size: 2
  //         }
  //         dim {
  //           size: 5
  //         }
  //       }
  //     }
  //   }
  // }
  // node {
  //   name: "y"
  //   op: "Placeholder"
  //   device: "/job:localhost/replica:0/task:0/device:GPU:0"
  //   attr {
  //     key: "dtype"
  //     value {
  //       type: DT_FLOAT
  //     }
  //   }
  //   attr {
  //     key: "shape"
  //     value {
  //       shape {
  //         dim {
  //           size: 5
  //         }
  //         dim {
  //           size: 3
  //         }
  //       }
  //     }
  //   }
  // }
  // node {
  //   name: "MatMul"
  //   op: "MatMul"
  //   input: "x"
  //   input: "y"
  //   device: "/job:localhost/replica:0/task:0/device:GPU:0"
  //   attr {
  //     key: "T"
  //     value {
  //       type: DT_FLOAT
  //     }
  //   }
  //   attr {
  //     key: "transpose_a"
  //     value {
  //       b: false
  //     }
  //   }
  //   attr {
  //     key: "transpose_b"
  //     value {
  //       b: false
  //     }
  //   }
  // }
  // library {
  // }
  // versions {
  //   producer: 27
  // }
  //

  TF_RETURN_IF_ERROR(
    subgraph::RewriteGraphForExecution(
      graph, // input and output
      feed_rewrites, // input
      fetch_rewrites, // input
      target_node_names, // input
      out_rewrite_metadata)); // output
  // 1.
  // 函数原型:
  // Status RewriteGraphForExecution(
  //     Graph* g,
  //     const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites,
  //     const std::vector<std::unique_ptr<PruneRewrite>>& fetch_rewrites,
  //     const gtl::ArraySlice<string>& target_node_names,
  //     RewriteGraphMetadata* out_metadata)
  //
  // 这个函数的效果
  // 出来后的图
  // https://gist.github.com/shizukanaskytree/50c61811865c9b79f8965e0769d12b9b

  CHECK_EQ(
    out_rewrite_metadata->feed_types.size(),
    options.callable_options.feed_size() +
    options.callable_options.tensor_connection_size());

  for (int i = 0; i < options.callable_options.tensor_connection_size(); ++i) {
    // 未进入

    out_rewrite_metadata->feed_types.pop_back();
  }

  return Status::OK();
}
// Where are we?
// Thread #1 [python] 17387 [core: 1] (Suspended : Step)
// 	tensorflow::GraphExecutionState::PruneGraph() at graph_execution_state.cc:574 0x7f855b23e60d
// 	tensorflow::GraphExecutionState::BuildGraph() at graph_execution_state.cc:806 0x7f855b240643
// 	tensorflow::DirectSession::CreateGraphs() at direct_session.cc:3,191 0x7f855709a515
// 	tensorflow::DirectSession::CreateExecutors() at direct_session.cc:2,627 0x7f8557095f7e
// 	tensorflow::DirectSession::GetOrCreateExecutors() at direct_session.cc:3,032 0x7f8557098fdc
// 	tensorflow::DirectSession::Run() at direct_session.cc:2,147 0x7f8557092802
// 	tensorflow::SessionRef::Run() at session_ref.cc:414 0x7f8552b72f8a
// 	TF_Run_Helper() at c_api.cc:878 0x7f8557113b96
// 	TF_SessionRun() at c_api.cc:2,752 0x7f855711d45e
// 	tensorflow::TF_SessionRun_wrapper_helper() at tf_session_helper.cc:407 0x7f8552b673e1
// 	<...more frames...>



// 1.
// QQQ. 如果我要构造两个 Executor, 我需要构造两个 GraphExecutionState::GraphExecutionState 吗？
//      我需要构造两个所谓的  Base Graph 吗?
// AAA. 需要

// 2.
// 一句人话总结:
// GraphExecutionState::InitBaseGraph() 进行了 node 的 device assignment placement
// 图的输出效果是 https://gist.github.com/shizukanaskytree/bfd21d79a7c13bf0b945a3710c1c13ef
// 上图初始化了 GraphExecutionState::graph_ : Graph*

// 3.
// 函数末尾有 call stack
Status GraphExecutionState::InitBaseGraph(const BuildGraphOptions& options) {
  // 1.
  // const BuildGraphOptions& options 语法
  // const BuildGraphOptions& options = BuildGraphOptions()
  // 表示 reference "代理"

  // 2.
  // Thread #1 [python] 12410 [core: 7] (Suspended : Breakpoint)
  // 	tensorflow::(anonymous namespace)::AssignAndLog at placer.cc:70 0x7f9b5acf0bb3
  // 	tensorflow::Placer::Run() at placer.cc:184 0x7f9b5acf1673
  //
  // 	tensorflow::GraphExecutionState::InitBaseGraph() at graph_execution_state.cc:616 0x7f9b6b23eaea
  //
  // 	tensorflow::GraphExecutionState::MakeForBaseGraph() at graph_execution_state.cc:97 0x7f9b6b23abc2
  // 	tensorflow::DirectSession::MaybeInitializeExecutionState() at direct_session.cc:1,694 0x7f9b6708f659
  // 	tensorflow::DirectSession::ExtendLocked() at direct_session.cc:1,740 0x7f9b6708fa41
  // 	tensorflow::DirectSession::Extend() at direct_session.cc:1,733 0x7f9b6708f9fc
  // 	tensorflow::SessionRef::Extend() at session_ref.cc:441 0x7f9b62b73833
  // 	tensorflow::ExtendSessionGraphHelper() at c_api.cc:815 0x7f9b671137c3
  // 	tensorflow::ExtendSession() at python_api.cc:118 0x7f9b62bac2e6
  // 	_wrap_ExtendSession() at pywrap_tensorflow_internal.cc:19,726 0x7f9b62ae1b45
  // 	_PyCFunction_FastCallDict() at methodobject.c:234 0x55909b42e681
  // 	call_function() at ceval.c:4,851 0x55909b4b5610

  const GraphDef* graph_def = &original_graph_def_;
  // 1.
  // graph_def 变量说明
  // graph_def "代理" GraphExecutionState::original_graph_def_, 使用指针，为了避免复制。

  // 2.
  // original_graph_def_ 变量说明:
  // GraphExecutionState::original_graph_def_: GraphDef original_graph_def_;

  // 3.
  // QQQ. GraphExecutionState::original_graph_def_ 什么时候初始化的?
  // AAA.
  // 在 GraphExecutionState::GraphExecutionState() 构造函数内被初始化
  // tensorflow/core/common_runtime/graph_execution_state.cc
  //
  // tensorflow/core/common_runtime/graph_execution_state.cc:92:
  // Status GraphExecutionState::MakeForBaseGraph 内
  // AddDefaultAttrsToGraphDef(&ret->original_graph_def_, *ret->flib_def_, 0));

  std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));

  GraphConstructorOptions opts;
  // 1.
  // struct GraphConstructorOptions 数据结构
  // tensorflow/core/graph/graph_constructor.h
  // - allow_internal_ops: bool , default : false
  //     If true, allows internal ops in the GraphDef.
  // - expect_device_spec: bool, default : false
  //     If true, the graph def is expected to have fully specified
  //     devices for all nodes. A node in the resulting graph "g" has the
  //     device name set accordingly.
  //
  // 概述:
  // Construct a Graph *g out of a GraphDef gdef. Returns non-OK on
  // error, in which case *g is left in an incomplete state.
  //
  // *g is expected to be an empty graph (with no more than a source and sink
  // nodes) when provided to ConvertGraphDefToGraph. To enhance an existing Graph,
  // see ImportGraphDef.

  TF_RETURN_IF_ERROR(
    ConvertGraphDefToGraph(opts, // input
                           *graph_def, // input
                           new_graph.get())); // output
  // 1.
  // 此刻的图并未安排 device assignment
  // log: https://gist.github.com/shizukanaskytree/cbf37674fb1ca66d4fb3fe9ac59fa000

  if (session_options_ &&
      session_options_->config.graph_options().place_pruned_graph()) {
    // 未进入
    // 因为: session_options_->config.graph_options().place_pruned_graph() == false

    // 1.
    // session_options_ 是什么?
    // const SessionOptions*
    // tensorflow/core/public/session_options.h

    // Rewrite the graph before placement.
    rewrite_metadata_.reset(new subgraph::RewriteGraphMetadata);
    TF_RETURN_IF_ERROR(
        PruneGraph(options, new_graph.get(), rewrite_metadata_.get()));
  }

  // Save stateful placements before placing.
  RestoreStatefulNodes(new_graph.get());
  // 1.
  // RestoreStatefulNodes 意图:
  // 下面马上要执行 Placer 了, 所以对于 stateful op node 提前设定好 device 保证在下面的
  // placer 过程中不会去改变它.

  GraphOptimizationPassOptions optimization_options;
  optimization_options.session_handle = session_handle_;
  optimization_options.session_options = session_options_;
  optimization_options.graph = &new_graph;
  optimization_options.flib_def = flib_def_.get();
  optimization_options.device_set = device_set_;

  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::PRE_PLACEMENT, optimization_options));

  Placer placer(
    new_graph.get(),
    device_set_,
    /* default_device= */ nullptr,
    /*allow_soft_placement=*/session_options_ == nullptr || session_options_->config.allow_soft_placement(),
    /*log_device_placement=*/session_options_ != nullptr && session_options_->config.log_device_placement());
  // 1.
  // Placer 构造函数:
  // Placer::Placer(Graph* graph,
  //                const DeviceSet* devices,
  //                const Device* default_device,
  //                bool allow_soft_placement,
  //                bool log_device_placement)


  // TODO(mrry): Consider making the Placer cancelable.
  TF_RETURN_IF_ERROR(placer.Run());
  // 1.
  // 执行完这个后的 log 和 效果:
  // MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
  // 2020-03-04 22:57:37.270443: I tensorflow/core/common_runtime/placer.cc:61] MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
  // 2020-03-04 23:30:04.449519: I tensorflow/core/common_runtime/placer.cc:61] x: (Placeholder)/job:localhost/replica:0/task:0/device:GPU:0
  // x: (Placeholder): /job:localhost/replica:0/task:0/device:GPU:0
  // y: (Placeholder): /job:localhost/replica:0/task:0/device:GPU:0
  // 2020-03-04 23:39:56.480328: I tensorflow/core/common_runtime/placer.cc:61] y: (Placeholder)/job:localhost/replica:0/task:0/device:GPU:0

  // 2.
  // 原来都在 GPU 上.

  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_PLACEMENT, optimization_options));
  // 1.
  // POST_PLACEMENT 只有唯一一个 图优化 pass, 是 NcclReplacePass

  for (const Node* n : new_graph->nodes()) {
    VLOG(2) << "Mapping " << n->name() << " to " << n->cost_id();
    node_name_to_cost_id_map_[n->name()] = n->cost_id();
  }
  // 1.
  // vlog
  //
  // Mapping _SOURCE to 0
  // Mapping _SINK to 1
  // Mapping x to 2
  // Mapping y to 3
  // Mapping MatMul to 4

  // 1.1
  // 什么是 cost_id
  // if there is no corresponding cost accounting node
  //
  // Node* Graph::AllocateNode(std::shared_ptr<NodeProperties> props,
  //                           const Node* cost_node)
  //
  // int cost_id = cost_node ? cost_node->cost_id() : id;
  // cost_id 和 id 等效

  // 2.
  // 此刻的图
  // p new_graph.get()->ToGraphDefDebug().DebugString()
  // log:
  // https://gist.github.com/shizukanaskytree/bfd21d79a7c13bf0b945a3710c1c13ef

  SaveStatefulNodes(new_graph.get());
  // 1.
  // 对于 z = x * y 这个例子, 没有节点是 stateful 的, 这个函数没有作用.
  // https://gist.github.com/shizukanaskytree/bfd21d79a7c13bf0b945a3710c1c13ef

  graph_ = new_graph.release();
  // 1.
  // graph_ 变量说明:
  // GraphExecutionState::graph_ : Graph*

  // 2.
  // 效果是:
  // 初始化了 GraphExecutionState::graph_ : Graph*

  return Status::OK();
}
// 1.
// where are we?
//
// Thread #1 [python] 15583 [core: 55] (Suspended : Step)
// 	tensorflow::GraphExecutionState::InitBaseGraph() 👀 at graph_execution_state.cc:632 0x7efc1b23edd7
// 	tensorflow::GraphExecutionState::MakeForBaseGraph() at graph_execution_state.cc:97 0x7efc1b23abc2
// 	tensorflow::DirectSession::MaybeInitializeExecutionState() at direct_session.cc:1,694 0x7efc1708f659
// 	tensorflow::DirectSession::ExtendLocked() at direct_session.cc:1,740 0x7efc1708fa41
// 	tensorflow::DirectSession::Extend() at direct_session.cc:1,733 0x7efc1708f9fc
// 	tensorflow::SessionRef::Extend() at session_ref.cc:441 0x7efc12b73833
// 	tensorflow::ExtendSessionGraphHelper() at c_api.cc:815 0x7efc171137c3
// 	tensorflow::ExtendSession() at python_api.cc:118 0x7efc12bac2e6
// 	_wrap_ExtendSession() at pywrap_tensorflow_internal.cc:19,726 0x7efc12ae1b45
// 	_PyCFunction_FastCallDict() at methodobject.c:234 0x55be63d17681
// 	<...more frames...>

// 2.
// 一句人话总结:
// GraphExecutionState::InitBaseGraph() 进行了 node 的 device assignment placement
// 图的输出效果是 https://gist.github.com/shizukanaskytree/bfd21d79a7c13bf0b945a3710c1c13ef
// 上图初始化了 GraphExecutionState::graph_ : Graph*

Status GraphExecutionState::OptimizeGraph(
    const BuildGraphOptions& options, // input
    std::unique_ptr<Graph>* optimized_graph, // output
    std::unique_ptr<FunctionLibraryDefinition>* optimized_flib) // output
{

#ifndef IS_MOBILE_PLATFORM
  if (session_options_->config.graph_options().place_pruned_graph()) {
    return errors::InvalidArgument("Can't optimize a pruned graph");
  }

  if (grappler::MetaOptimizerEnabled(session_options_->config)) {
    grappler::GrapplerItem item;
    item.id = "tf_graph";
    graph_->ToGraphDef(&item.graph);

    // It's ok to skip invalid device annotations in Grappler.
    Status inferred_devices = item.InferDevicesFromGraph();
    if (!inferred_devices.ok()) {
      VLOG(3) << inferred_devices.error_message();
    }

    // TODO(b/114748242): Add a unit test to test this bug fix.
    if (flib_def_) {
      *item.graph.mutable_library() = flib_def_->ToProto();
    }

    item.fetch.insert(item.fetch.end(),
                      options.callable_options.fetch().begin(),
                      options.callable_options.fetch().end());
    item.fetch.insert(item.fetch.end(),
                      options.callable_options.target().begin(),
                      options.callable_options.target().end());

    for (const TensorConnection& tensor_connection :
         options.callable_options.tensor_connection()) {
      item.fetch.push_back(tensor_connection.from_tensor());
    }

    if (!(options.callable_options.feed().empty() &&
          options.callable_options.tensor_connection().empty())) {
      std::unordered_set<string> feeds;
      for (const string& feed : options.callable_options.feed()) {
        TensorId id = ParseTensorName(feed);
        if (id.second != 0) {
          return errors::InvalidArgument("Unsupported feed: ", feed);
        }
        feeds.emplace(id.first);
      }
      for (const TensorConnection& tensor_connection :
           options.callable_options.tensor_connection()) {
        TensorId id = ParseTensorName(tensor_connection.to_tensor());
        if (id.second != 0) {
          return errors::InvalidArgument("Unsupported feed: ",
                                         tensor_connection.to_tensor());
        }
        feeds.emplace(id.first);
      }
      for (const NodeDef& node : original_graph_def_.node()) {
        if (feeds.find(node.name()) == feeds.end()) {
          continue;
        }
        // Get the type and shape of the feed node.
        PartialTensorShape partial_shape;
        DataType type;
        TF_RETURN_IF_ERROR(
            GetFeedShapeAndTypeFromAttribute(node, &partial_shape, &type));
        // If the shape of the placeholder is only partially known, we are free
        // to set unknown dimensions of its shape to any value we desire. We
        // choose 0 to minimize the memory impact. Note that this only matters
        // if an optimizer chooses to run the graph.
        TensorShape shape;
        if (partial_shape.unknown_rank()) {
          shape = TensorShape({0});
        } else {
          for (int i = 0; i < partial_shape.dims(); ++i) {
            if (partial_shape.dim_size(i) < 0) {
              partial_shape.set_dim(i, 0);
            }
          }
          if (!partial_shape.AsTensorShape(&shape)) {
            return errors::InvalidArgument(
                "Could not derive shape for feed node: ", node.DebugString());
          }
        }

        Tensor fake_input(type, shape);
        item.feed.emplace_back(node.name(), fake_input);
      }
    }

    Device* cpu_device = nullptr;
    for (const auto& device : device_set_->devices()) {
      if (device->parsed_name().id == 0 &&
          StringPiece(device->parsed_name().type) == "CPU" &&
          device->GetAllocator(AllocatorAttributes()) != nullptr) {
        cpu_device = device;
      }
    }

    grappler::VirtualCluster cluster(device_set_);

    GraphDef new_graph;

    TF_RETURN_IF_ERROR(grappler::RunMetaOptimizer(
        item,
        session_options_->config,
        cpu_device,
        &cluster,
        &new_graph));

    //////////////////////////////////////////////////////////////////
    // Merge optimized graph function library with an original library.
    // Optimized graph might have new functions specialized for it's
    // instantiation context (see Grappler function optimizer), and modified
    // function body for the existing functions.
    optimized_flib->reset(new FunctionLibraryDefinition(*flib_def_));

    // 如果还有更好的 function kernel，替换到更好的。
    for (const FunctionDef& fdef : new_graph.library().function()) {
      const string& func_name = fdef.signature().name();

      if ((*optimized_flib)->Contains(func_name)) {
        VLOG(3) << "Replace function: name=" << func_name;
        // ReplaceFunction 函数说明
        // tensorflow/core/framework/function.cc:1220:
        // Status FunctionLibraryDefinition::ReplaceFunction(const string& func, const FunctionDef& fdef)
        //
        TF_RETURN_IF_ERROR((*optimized_flib)->ReplaceFunction(func_name, fdef));
      } else {
        VLOG(3) << "Add new function: name=" << func_name;
        TF_RETURN_IF_ERROR((*optimized_flib)->AddFunctionDef(fdef));
      }
    }

    // 处理优化的图
    optimized_graph->reset(new Graph(OpRegistry::Global()));

    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;

    TF_RETURN_IF_ERROR(
        ConvertGraphDefToGraph(opts, new_graph, optimized_graph->get()));

    // The graph conversion sets the requested device names but not the
    // assigned device names. However, since at this point the graph is placed
    // TF expects an assigned device name for every node. Therefore we copy
    // the requested device into the assigned device field.
    for (Node* node : optimized_graph->get()->nodes()) {
      node->set_assigned_device_name(node->requested_device());
    }

    return Status::OK();

  } else {

    return errors::InvalidArgument("Meta Optimizer disabled");

  }

#else
  return errors::InvalidArgument("Mobile platforms not supported");
#endif  // IS_MOBILE_PLATFORM
}

/** \brief Build a sub-graph of the full graph as induced by BuildGraphOptions.
 *
 *  \param options: const BuildGraphOptions&
 *
 *  \param out: [out] std::unique_ptr<ClientGraph>*
 *         ClientGraph is simply a sub-graph of the full graph as induced by
 *         BuildGraphOptions.
 */
Status GraphExecutionState::BuildGraph(
  const BuildGraphOptions& options, // input
  std::unique_ptr<ClientGraph>* out) { // output
  // 1.
  // options 变量说明:
  // BuildGraphOptions& options 打印:
  //
  // Feed endpoints:
  // Fetch endpoints: out:0,
  // Target nodes:
  // collective_order: none

  // 2.
  // ClientGraph 数据结构
  // A ClientGraph is simply a sub-graph of the full graph as induced by
  // BuildGraphOptions.
  // tensorflow/core/common_runtime/graph_execution_state.h
  //
  // struct ClientGraph
  // - flib_def : std::unique_ptr<FunctionLibraryDefinition>
  //     Each client-graph gets its own function library since optimization passes
  //     post rewrite for execution might want to introduce new functions.
  // - graph: Graph
  // - feed_types: DataTypeVector
  // - fetch_types: DataTypeVector
  // - collective_graph_key: int64

  VLOG(1) << "BuildGraph";

  const uint64 start_time_usecs = Env::Default()->NowMicros();

  if (!graph_) {
    // 未进入

    // It is only valid to call this method directly when the original graph
    // was created with the option `place_pruned_graph == false`.
    return errors::Internal(
        "Attempted to prune a graph that has not been fully initialized.");
  }

  // 1.
  // 此刻的图:
  // p graph_->ToGraphDefDebug().DebugString()
  // https://gist.github.com/shizukanaskytree/b23525eecfa6543441b5ad0986b0e7e7

  // Grappler optimization might change the structure of a graph itself, and
  // also it can add/prune functions to/from the library.
  std::unique_ptr<Graph> optimized_graph;

  // FunctionLibraryDefinition 数据结构
  // ./tensorflow/core/framework/function.h:313:
  // class FunctionLibraryDefinition : public OpRegistryInterface
  // - default_registry_ : const OpRegistryInterface* const
  // - function_defs_ : gtl::FlatMap<string, std::unique_ptr<FunctionDefAndOpRegistration>>
  //  * FunctionDefAndOpRegistration
  //    + fdef: FunctionDef
  //    + op_registration_data: OpRegistrationData
  // - func_grad_ : gtl::FlatMap<string, string>
  std::unique_ptr<FunctionLibraryDefinition> optimized_flib;

  /// OptimizeGraph has a lot to do.
  Status s = OptimizeGraph(options, // input  打印见上面
                           &optimized_graph, // output
                           &optimized_flib); // output
  // 1.
  // 优化图
  // 参考日志中的时间戳:
  // ✅ BuildGraph 和 ✅ 图优化
  // https://gist.github.com/shizukanaskytree/3c875d642406b61356865064835695f2

  // 2.
  // 执行上面 OptimizeGraph() 所对应的 log:
  // https://gist.github.com/shizukanaskytree/033da3ad0919cd2b4cc93a16f34f2e07

  if (!s.ok()) {
    // 未进入

    VLOG(2) << "Grappler optimization failed. Error: " << s.error_message();
    // Simply copy the original graph and the function library if we couldn't
    // optimize it.
    optimized_graph.reset(new Graph(flib_def_.get()));
    CopyGraph(*graph_, optimized_graph.get());
    optimized_flib.reset(new FunctionLibraryDefinition(*flib_def_));
  }

  subgraph::RewriteGraphMetadata rewrite_metadata;
  // 1.
  // RewriteGraphMetadata 数据结构
  // tensorflow/core/graph/subgraph.h:32: struct RewriteGraphMetadata
  // - feed_types: DataTypeVector
  //               The element type of each tensor fed to this subgraph.
  // - fetch_types: DataTypeVector

  if (session_options_ == nullptr ||
      !session_options_->config.graph_options().place_pruned_graph()) {
    // 进入

    // 1.
    // 此刻 optimized_graph 的图, 未执行 PruneGraph()
    // https://gist.github.com/shizukanaskytree/0bd24751a6f79730afaf47efae5912b3

    TF_RETURN_IF_ERROR(
        PruneGraph(
          options, // input
          optimized_graph.get(), // input
          &rewrite_metadata)); // output
    // 1.
    // ✅一句话概括:
    // 把用户输入的图转变为一个带 _Arg, _Retval nodes 的图,
    // 消掉以前的 placeholder 节点.
    // 暂不涉及 send recv op

    // 2.
    // optimized_graph after PruneGraph:
    // https://gist.github.com/shizukanaskytree/50c61811865c9b79f8965e0769d12b9b

  } else {
    // 未进入

    // This GraphExecutionState represents a graph that was
    // pruned when this was constructed, so we copy the metadata from
    // a member variable.
    CHECK(rewrite_metadata_);
    rewrite_metadata = *rewrite_metadata_;
  }

  CHECK_EQ(options.callable_options.feed_size(),
           rewrite_metadata.feed_types.size());
  CHECK_EQ(options.callable_options.fetch_size(),
           rewrite_metadata.fetch_types.size());

  // TODO(andydavis): Clarify optimization pass requirements around CostModel.
  GraphOptimizationPassOptions optimization_options;
  optimization_options.session_options = session_options_;
  optimization_options.graph = &optimized_graph;
  optimization_options.flib_def = optimized_flib.get();
  optimization_options.device_set = device_set_;

  TF_RETURN_IF_ERROR(
    OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_REWRITE_FOR_EXEC,
      optimization_options)
  );
  // 1.
  // 此刻的图
  // node {
  //   name: "MatMul"
  //   op: "MatMul"
  //   input: "_arg_x_0_0"
  //   input: "_arg_y_0_1"
  //   device: "/job:localhost/replica:0/task:0/device:GPU:0"  ✅ 发现没, 这里已经把 node device assignment 给分配好了.
  //   attr {
  //     key: "T"
  //     value {
  //       type: DT_FLOAT
  //     }
  //   }
  //   attr {
  //     key: "transpose_a"
  //     value {
  //       b: false
  //     }
  //   }
  //   attr {
  //     key: "transpose_b"
  //     value {
  //       b: false
  //     }
  //   }
  // }
  // node {
  //   name: "_arg_x_0_0"
  //   op: "_Arg"
  //   device: "/job:localhost/replica:0/task:0/device:CPU:0"
  //   attr {
  //     key: "T"
  //     value {
  //       type: DT_FLOAT
  //     }
  //   }
  //   attr {
  //     key: "index"
  //     value {
  //       i: 0
  //     }
  //   }
  // }
  // node {
  //   name: "_arg_y_0_1"
  //   op: "_Arg"
  //   device: "/job:localhost/replica:0/task:0/device:CPU:0"
  //   attr {
  //     key: "T"
  //     value {
  //       type: DT_FLOAT
  //     }
  //   }
  //   attr {
  //     key: "index"
  //     value {
  //       i: 1
  //     }
  //   }
  // }
  // node {
  //   name: "_retval_MatMul_0_0"
  //   op: "_Retval"
  //   input: "MatMul"
  //   device: "/job:localhost/replica:0/task:0/device:CPU:0"
  //   attr {
  //     key: "T"
  //     value {
  //       type: DT_FLOAT
  //     }
  //   }
  //   attr {
  //     key: "index"
  //     value {
  //       i: 0
  //     }
  //   }
  // }
  // library {
  // }
  // versions {
  //   producer: 27
  // }

  // 2.
  // 进展到了这里: L339 (中间态)
  // https://gist.github.com/shizukanaskytree/3c875d642406b61356865064835695f2#file-analysis_level_1_log-cc-L339

  int64 collective_graph_key = options.collective_graph_key;
  if (collective_graph_key == BuildGraphOptions::kNoCollectiveGraphKey) {
    // 进入

    // BuildGraphOptions does not specify a collective_graph_key.  Check all
    // nodes in the Graph and FunctionLibraryDefinition for collective ops and
    // if found, initialize a collective_graph_key as a hash of the ordered set
    // of instance keys.
    std::set<int32> instance_key_set;

    for (Node* node : optimized_graph->nodes()) {
      if (node->IsCollective()) {
        // 未进入

        int32 instance_key;
        TF_RETURN_IF_ERROR(
            GetNodeAttr(node->attrs(), "instance_key", &instance_key));
        instance_key_set.emplace(instance_key);
      } else {
        // 进入

        // 但是，没有那个 node 的 fdef 是存在的，**目前都是 空指针**
        const FunctionDef* fdef = optimized_flib->Find(node->def().op());

        if (fdef != nullptr) {
          // 未进入
          // 因为 fdef == nullptr 

          for (const NodeDef& ndef : fdef->node_def()) {
            if (ndef.op() == "CollectiveReduce" ||
                ndef.op() == "CollectiveBcastSend" ||
                ndef.op() == "CollectiveBcastRecv") {
              int32 instance_key;
              TF_RETURN_IF_ERROR(
                  GetNodeAttr(ndef, "instance_key", &instance_key));
              instance_key_set.emplace(instance_key);
            }
          }
        }

      }
    }

    if (!instance_key_set.empty()) {
      uint64 hash = 0x8774aa605c729c72ULL;
      for (int32 instance_key : instance_key_set) {
        hash = Hash64Combine(instance_key, hash);
      }
      collective_graph_key = hash;
    }
  }

  // Make collective execution order deterministic if needed.
  if (options.collective_order != GraphCollectiveOrder::kNone) {
    // 未进入
    TF_RETURN_IF_ERROR(
        OrderCollectives(optimized_graph.get(), options.collective_order));
  }

  // Copy the extracted graph in order to make its node ids dense,
  // since the local CostModel used to record its stats is sized by
  // the largest node id.
  std::unique_ptr<ClientGraph> dense_copy(
      new ClientGraph(
        std::move(optimized_flib),
        rewrite_metadata.feed_types,
        rewrite_metadata.fetch_types,
        collective_graph_key));
  // 1.
  // ClientGraph

  // 2.
  // optimized_flib: std::unique_ptr<FunctionLibraryDefinition>

  // FunctionLibraryDefinition 数据结构
  // ./tensorflow/core/framework/function.h:313:
  // class FunctionLibraryDefinition : public OpRegistryInterface
  // - default_registry_ : const OpRegistryInterface* const
  // - function_defs_ : gtl::FlatMap<string, std::unique_ptr<FunctionDefAndOpRegistration>>
  //  * FunctionDefAndOpRegistration
  //    + fdef: FunctionDef
  //    + op_registration_data: OpRegistrationData
  // - func_grad_ : gtl::FlatMap<string, string>

  // int64 collective_graph_key = options.collective_graph_key;
  // p collective_graph_key
  // $33 = 0

  CopyGraph(*optimized_graph, &dense_copy->graph);
  // 1.
  // 此刻的图 optimized_graph 和 dense_copy->graph
  /*
  node {
    name: "MatMul"
    op: "MatMul"
    input: "_arg_x_0_0"
    input: "_arg_y_0_1"
    device: "/job:localhost/replica:0/task:0/device:GPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "transpose_a"
      value {
        b: false
      }
    }
    attr {
      key: "transpose_b"
      value {
        b: false
      }
    }
  }
  node {
    name: "_arg_x_0_0"
    op: "_Arg"
    device: "/job:localhost/replica:0/task:0/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "index"
      value {
        i: 0
      }
    }
  }
  node {
    name: "_arg_y_0_1"
    op: "_Arg"
    device: "/job:localhost/replica:0/task:0/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "index"
      value {
        i: 1
      }
    }
  }
  node {
    name: "_retval_MatMul_0_0"
    op: "_Retval"
    input: "MatMul"
    device: "/job:localhost/replica:0/task:0/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "index"
      value {
        i: 0
      }
    }
  }
  library {
  }
  versions {
    producer: 27
  }
  */

  // TODO(vrv): We should check invariants of the graph here.
  metrics::UpdateGraphBuildTime(
    Env::Default()->NowMicros() - start_time_usecs);

  *out = std::move(dense_copy);

  return Status::OK();
}
// where are we?



}  // namespace tensorflow
