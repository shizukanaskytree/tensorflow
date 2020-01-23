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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_BUILD_GRAPH_OPTIONS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_BUILD_GRAPH_OPTIONS_H_

#include <vector>

#include "tensorflow/core/graph/collective_order.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

/** \struct BuildGraphOptions
 *
 *  \brief Build a sub-graph of the full graph as induced by BuildGraphOptions.
 *
 *  \note
 *  - callable_options: CallableOptions, protobuf message.
 *    callable is a subset of graph object GraphDef with its feed and fetch. It
 *    also specifie the target and device to run.
 *
 *  - collective_order: GraphCollectiveOrder
 *    Introduces a deterministic execution order between potentially concurrent
 *    CollectiveOps.  This may be used to execute collectives in the same order
 *    across all workers in a distributed execution,if all workers are executing
 *    the same graph.
 */
struct BuildGraphOptions {
  CallableOptions callable_options;

  // If `true`, uses Arg/Retval to implement feeds/fetches; otherwise
  // uses Recv/Send to implement feeds/fetches.
  // TODO(mrry): Remove this when the distributed runtime supports Arg/Retval.
  bool use_function_convention = false;

  static const int64 kNoCollectiveGraphKey = 0;

  int64 collective_graph_key = kNoCollectiveGraphKey;

  // If not `kNone`, order all CollectiveReduce operations statically and
  // deterministically.  If `kEdges`, encode dependencies as explicit control
  // edges, if `kAttrs` encode as attribute on collective op.
  GraphCollectiveOrder collective_order = GraphCollectiveOrder::kNone;

  string DebugString() const;
};
// 1.
// struct BuildGraphOptions 数据结构
// tensorflow/core/common_runtime/build_graph_options.h
// - callable_options: CallableOptions
// - use_function_convention: bool, default: false
// - kNoCollectiveGraphKey: static const int64, default: 0
// - collective_graph_key: int64, default: kNoCollectiveGraphKey
// - collective_order: GraphCollectiveOrder

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

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_BUILD_GRAPH_OPTIONS_H_
