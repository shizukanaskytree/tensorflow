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

#include "tensorflow/core/graph/subgraph.h"

#include <algorithm>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace subgraph {

// ----------------------------------------------------------------------------
// Subgraph construction-related routines
// ----------------------------------------------------------------------------
// TODO(vrv): Profile the unordered_set and unordered_map use in this file to
// see if we should use an alternative implementation.

namespace {

typedef std::unordered_map<StringPiece, Node*, StringPieceHasher> NameIndex;

// Rewrite graph by replacing the output tensors specified in
// "fed_outputs" with special feed nodes for each specified output
// tensor, and removing any nodes that are now disconnected from the
// part of the graph that reaches the sink node.  The set of special
// feed nodes added to the graph are returned in "*feed_nodes".
//
// Return true on success.  On error, return false and sets *error to
// an appropriate error message (and *g is left in an indeterminate
// state).
Status FeedInputs(
    Graph* g, // input
    const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites, // input
    NameIndex* name_index, // input
    DataTypeVector* out_feed_types) // output
{
  out_feed_types->clear();
  out_feed_types->reserve(feed_rewrites.size());
  for (size_t i = 0; i < feed_rewrites.size(); ++i) {
    const string& t = feed_rewrites[i]->endpoint_name();
    TensorId id(ParseTensorName(t));

    auto iter = name_index->find(id.first);
    if (iter == name_index->end()) {
      return errors::NotFound("FeedInputs: unable to find feed output ", t);
    }
    Node* n = iter->second;
    DCHECK_EQ(n->name(), id.first);
    if (id.second >= n->num_outputs()) {
      return errors::InvalidArgument(
          "FeedInputs: ", t, " should have output index < ", n->num_outputs());
    }

    Node* feed_node;
    TF_RETURN_IF_ERROR(
        feed_rewrites[i]->AddNode(g, {n, id.second}, &feed_node));

    // Update name_index
    (*name_index)[feed_node->name()] = feed_node;
    // Duplicate control edges aren't allowed, but feed_node was *just* created
    // so there's no need to check for a duplicate.
    g->AddControlEdge(g->source_node(), feed_node, true);

    // Look through edges coming out of "n" for edges whose src_output() index
    // matches "output_index".  If found, replace the edges with a connection
    // from the special feed node.
    std::vector<const Edge*> to_remove;
    for (const Edge* e : n->out_edges()) {
      if (e->src_output() == id.second) {
        to_remove.emplace_back(e);
      } else if (e->src_output() == Graph::kControlSlot &&
                 (n->type_string() == "Placeholder" ||
                  n->type_string() == "PlaceholderV2")) {
        // When feeding a Placeholder node, any outgoing control edges
        // will be replaced with a control edge from the replacement
        // feed_node.
        // TODO(josh11b,mrry): Come up with a more elegant way of addressing
        // the general version of this problem.
        to_remove.emplace_back(e);
      }
    }

    for (const Edge* e : to_remove) {
      if (e->src_output() == id.second) {
        g->AddEdge(feed_node, 0, e->dst(), e->dst_input());
      } else {
        CHECK_EQ(Graph::kControlSlot, e->src_output());
        // Duplicate control edges aren't allowed, but feed_node was *just*
        // created so there's no need to check for a duplicate.
        g->AddControlEdge(feed_node, e->dst(), true);
      }
      g->RemoveEdge(e);
    }
    out_feed_types->push_back(BaseType(n->output_type(id.second)));
  }
  return Status::OK();
}

Status FetchOutputs(
    Graph* g, // input, https://docs.google.com/document/d/19JJMlSxdq2PvwbfxJ2jtKAkAOQq_Q_wYNeekZFlFioc/edit#
    const std::vector<std::unique_ptr<PruneRewrite>>& fetch_rewrites, // input
    NameIndex* name_index, // output (add item into this map)
    std::vector<Node*>* out_fetch_nodes,  // output
    DataTypeVector* out_fetch_types) { // output

  out_fetch_nodes->clear();
  out_fetch_nodes->reserve(fetch_rewrites.size());

  for (size_t i = 0; i < fetch_rewrites.size(); ++i) {
    const string& t = fetch_rewrites[i]->endpoint_name();
/*
比如说:

p t
$5 = "out:0"
*/

    // Parse t into node_name and output_index.
    TensorId id(ParseTensorName(t));

    // Find node in graph with that name.
    auto iter = name_index->find(id.first);
    if (iter == name_index->end()) {
      return errors::NotFound("FetchOutputs node ", t, ": not found");
    }
    Node* n = iter->second;
    DCHECK_EQ(n->name(), id.first);
    VLOG(2) << "Found fetch node for " << t;

    // Validate output_index
    if (n->num_outputs() == 0) {
      return errors::InvalidArgument(
          "Tried to fetch data for '", t,
          "', which produces no output.  To run to a node but not fetch any "
          "data, pass '",
          t,
          "' as an argument to the 'target_node_names' argument of the "
          "Session::Run API.");
    } else if (id.second >= n->num_outputs()) {
      return errors::InvalidArgument("FetchOutputs ", t,
                                     ": output index too large, must be < ",
                                     n->num_outputs());
    }

    // Create the fetch Node and connect it up
    Node* fetch_node;
    TF_RETURN_IF_ERROR(
        fetch_rewrites[i]->AddNode(g, {n, id.second}, &fetch_node));

    // Update the index.
    (*name_index)[fetch_node->name()] = fetch_node;

    /**
    name_index 比如

    p *name_index

    {
      [{
          static npos = 18446744073709551615,
          static kMaxSize = 9223372036854775807,
          ptr_ = 0x55b643d083f8 "_retval_out_0_0", // 这个应该是新增的
          length_ = 15
        }

      ] = 0x55b643d3b198,
      [{
          static npos = 18446744073709551615,
          static kMaxSize = 9223372036854775807,
          ptr_ = 0x55b641e21578 "out",
          length_ = 3
        }

      ] = 0x55b643d3b040,
      [{
          static npos = 18446744073709551615,
          static kMaxSize = 9223372036854775807,
          ptr_ = 0x55b641d7a0b8 "c",
          length_ = 1
        }

      ] = 0x55b643d3af88,
      [{
          static npos = 18446744073709551615,
          static kMaxSize = 9223372036854775807,
          ptr_ = 0x55b6411f4e08 "a/RandomStandardNormal",
          length_ = 22
        }

      ] = 0x55b643d3ae38,
      [{
          static npos = 18446744073709551615,
          static kMaxSize = 9223372036854775807,
          ptr_ = 0x55b641dd96d8 "b/RandomStandardNormal",
          length_ = 22
        }

      ] = 0x55b643d3ada0,
      [{
          static npos = 18446744073709551615,
          static kMaxSize = 9223372036854775807,
          ptr_ = 0x55b641d66068 "z",
          length_ = 1
        }

      ] = 0x55b643d3aed0,
      [{
          static npos = 18446744073709551615,
          static kMaxSize = 9223372036854775807,
          ptr_ = 0x55b641d590d8 "y/RandomStandardNormal",
          length_ = 22
        }

      ] = 0x55b643d3ac70,
      [{
          static npos = 18446744073709551615,
          static kMaxSize = 9223372036854775807,
          ptr_ = 0x55b641829a58 "a/shape",
          length_ = 7
        }

      ] = 0x55b643d3abf8,
      [{
          static npos = 18446744073709551615,
          static kMaxSize = 9223372036854775807,
          ptr_ = 0x55b6418fe378 "x/RandomStandardNormal",
          length_ = 22
        }

      ] = 0x55b643d3ad08,
      [{
          static npos = 18446744073709551615,
          static kMaxSize = 9223372036854775807,
          ptr_ = 0x55b641d53ce8 "b/shape",
          length_ = 7
        }

      ] = 0x55b643d3ab80,
      [{
          static npos = 18446744073709551615,
          static kMaxSize = 9223372036854775807,
          ptr_ = 0x55b6411f1c88 "y/shape",
          length_ = 7
        }

      ] = 0x55b643d3aa90,
      [{
          static npos = 18446744073709551615,
          static kMaxSize = 9223372036854775807,
          ptr_ = 0x55b6411f5378 "x/shape",
          length_ = 7
        }

      ] = 0x55b643d3ab08,
      [{
          static npos = 18446744073709551615,
          static kMaxSize = 9223372036854775807,
          ptr_ = 0x55b643d28768 "_SINK",
          length_ = 5
        }

      ] = 0x55b643d3a9f8,
      [{
          static npos = 18446744073709551615,
          static kMaxSize = 9223372036854775807,
          ptr_ = 0x55b643d51e78 "_SOURCE",
          length_ = 7
        }

      ] = 0x55b643d3a980
    }

    */

// -----------------------------------------------------------------------
/*
p fetch_node ->DebugString()
{
  name:'_retval_out_0_0'

  id:13

  op device: {
    /job: localhost/replica:0/task:0/device:CPU:0
  }

  def: {
      {
        {
        node _retval_out_0_0
      }
    }

    =_Retval[T=DT_FLOAT,
    index=0](out) // 看这里，这个节点的输入是 out 节点
  }
}
*/


    // Duplicate control edges aren't allowed, but fetch_node was *just* created
    // so there's no need to check for a duplicate.
    g->AddControlEdge(fetch_node, g->sink_node(), true);
    out_fetch_nodes->push_back(fetch_node);
    out_fetch_types->push_back(BaseType(n->output_type(id.second)));
  }

  return Status::OK();
}


// callstack see "AddNodeToTargets callstack" in Doc: Cross Device experiment
// https://docs.google.com/document/d/1tUPyuaG6wlW76FGTPrEFXRMczFVXMZGL_DoH9xt6uQU/edit#heading=h.4irhh7mrp0pf

bool AddNodeToTargets(
  const string& node_or_tensor_name, // input
  // 打印：p node_or_tensor_name, $14 = "_retval_out_0_0"

  const NameIndex& name_index, // input, 打印见下面
  std::unordered_set<const Node*>* targets) // output
{

/*
p name_index

std::unordered_map with 14 elements
{
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b643d083f8 "_retval_out_0_0",
      length_ = 15
    }

  ] = 0x55b643d3b198,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b641e21578 "out",
      length_ = 3
    }

  ] = 0x55b643d3b040,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b641d7a0b8 "c",
      length_ = 1
    }

  ] = 0x55b643d3af88,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b6411f4e08 "a/RandomStandardNormal",
      length_ = 22
    }

  ] = 0x55b643d3ae38,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b641dd96d8 "b/RandomStandardNormal",
      length_ = 22
    }

  ] = 0x55b643d3ada0,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b641d66068 "z",
      length_ = 1
    }

  ] = 0x55b643d3aed0,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b641d590d8 "y/RandomStandardNormal",
      length_ = 22
    }

  ] = 0x55b643d3ac70,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b641829a58 "a/shape",
      length_ = 7
    }

  ] = 0x55b643d3abf8,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b6418fe378 "x/RandomStandardNormal",
      length_ = 22
    }

  ] = 0x55b643d3ad08,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b641d53ce8 "b/shape",
      length_ = 7
    }

  ] = 0x55b643d3ab80,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b6411f1c88 "y/shape",
      length_ = 7
    }

  ] = 0x55b643d3aa90,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b6411f5378 "x/shape",
      length_ = 7
    }

  ] = 0x55b643d3ab08,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b643d28768 "_SINK",
      length_ = 5
    }

  ] = 0x55b643d3a9f8,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b643d51e78 "_SOURCE",
      length_ = 7
    }

  ] = 0x55b643d3a980
}
*/



  // TensorId 数据结构
  // first == operation_name, second == output_index, "out:0"
  // tensorflow/core/graph/tensor_id.h:33:struct TensorId : public std::pair<StringPiece, int>
  TensorId id = ParseTensorName(node_or_tensor_name);
  auto iter = name_index.find(id.first);
  if (iter == name_index.end()) {
    return false;
  }
  const Node* n = iter->second;
  CHECK_EQ(n->name(), id.first);

  // 就这里，把 name_index 里面那个 targets node 放进 targets[output] 里面
  targets->insert(n);

  return true;
}




Status PruneForTargets(
  Graph* g,  // input and output
  const NameIndex& name_index, // input
  const std::vector<Node*>& fetch_nodes, // input
  const gtl::ArraySlice<string>& target_nodes) { // input

  string not_found;

  std::unordered_set<const Node*> targets;

  for (Node* n : fetch_nodes) {
    // -----------------------------------------------------------------------
    // AddNodeToTargets 说明
    if (!AddNodeToTargets(
           n->name(), // input, 打印：p node_or_tensor_name, $14 = "_retval_out_0_0"
           name_index, // input
           &targets)) // output
    // -----------------------------------------------------------------------
    {
      strings::StrAppend(&not_found, n->name(), " ");
    }
  }

  for (const string& s : target_nodes) {
    // -----------------------------------------------------------------------
    if (!AddNodeToTargets(s, name_index, &targets)) {
    // -----------------------------------------------------------------------
      strings::StrAppend(&not_found, s, " ");
    }
  }

  if (!not_found.empty()) {
    return errors::NotFound("PruneForTargets: Some target nodes not found: ",
                            not_found);
  }

  // -----------------------------------------------------------------------
  // PruneForReverseReachability 说明:
  // tensorflow/core/graph/algorithm.h
  // tensorflow/core/graph/algorithm.cc
  // PruneForReverseReachability(Graph* g, std::unordered_set<const Node*> nodes)
  PruneForReverseReachability(
    g,  // input
    targets); // input and output 
  // -----------------------------------------------------------------------


  // -----------------------------------------------------------------------
  // Reconnect nodes with no outgoing edges to the sink node

  // FixupSourceAndSinkEdges 说明：
  // tensorflow/core/graph/algorithm.cc:248:bool FixupSourceAndSinkEdges(Graph* g)
  // Connect all nodes with no incoming edges to source.
  // Connect all nodes with no outgoing edges to sink.
  FixupSourceAndSinkEdges(g);
  // -----------------------------------------------------------------------

  return Status::OK();
}

}  // namespace

Status ArgFeedRewrite::AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                               Node** out_node) {
  // NOTE(mrry): We must include the index as part of the node
  // name, because _Arg is a "stateful" kernel and therefore
  // its name must uniquely identify a kernel instance across all
  // graphs in the same session.
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat("_arg_", feed_tensor.node->name(), "_",
                                  feed_tensor.index, "_", arg_index_),
                  "_Arg")
          .Attr("T", BaseType(feed_tensor.node->output_type(feed_tensor.index)))
          .Attr("index", arg_index_)
          .Finalize(g, out_node));
  (*out_node)->set_assigned_device_name(device_info().name());
  return Status::OK();
}

Status RecvFeedRewrite::AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                                Node** out_node) {
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat("_recv_", feed_tensor.node->name(), "_",
                                  feed_tensor.index),
                  "_Recv")
          .Attr("tensor_type",
                BaseType(feed_tensor.node->output_type(feed_tensor.index)))
          .Attr("tensor_name", endpoint_name())
          .Attr("send_device", device_info().name())
          .Attr("recv_device", device_info().name())
          .Attr("send_device_incarnation",
                static_cast<int64>(device_info().incarnation()))
          .Attr("client_terminated", true)
          .Finalize(g, out_node));

  (*out_node)->set_assigned_device_name(device_info().name());
  return Status::OK();
}

Status RetvalFetchRewrite::AddNode(Graph* g, NodeBuilder::NodeOut fetch_tensor,
                                   Node** out_node) {
  // NOTE(mrry): We must include the index as part of the node
  // name, because _Retval is a "stateful" kernel and therefore
  // its name must uniquely identify a kernel instance across all
  // graphs in the same session.
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat("_retval_", fetch_tensor.node->name(), "_",
                                  fetch_tensor.index, "_", retval_index_),
                  "_Retval")
          .Input(fetch_tensor.node, fetch_tensor.index)
          .Attr("T",
                BaseType(fetch_tensor.node->output_type(fetch_tensor.index)))
          .Attr("index", retval_index_)
          .Finalize(g, out_node));
  (*out_node)->set_assigned_device_name(device_info().name());
  return Status::OK();
}

Status SendFetchRewrite::AddNode(Graph* g, NodeBuilder::NodeOut fetch_tensor,
                                 Node** out_node) {
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat("_send_", fetch_tensor.node->name(), "_",
                                  fetch_tensor.index),
                  "_Send")
          .Input(fetch_tensor.node, fetch_tensor.index)
          .Attr("tensor_name", endpoint_name())
          .Attr("send_device", device_info().name())
          .Attr("recv_device", device_info().name())
          .Attr("send_device_incarnation",
                static_cast<int64>(device_info().incarnation()))
          .Attr("client_terminated", true)
          .Finalize(g, out_node));
  (*out_node)->set_assigned_device_name(device_info().name());
  return Status::OK();
}


Status RewriteGraphForExecution(
    Graph* g,
    const gtl::ArraySlice<string>& fed_outputs,
    const gtl::ArraySlice<string>& fetch_outputs,
    const gtl::ArraySlice<string>& target_node_names,
    const DeviceAttributes& device_info,
    bool use_function_convention,
    RewriteGraphMetadata* out_metadata) {

  std::vector<std::unique_ptr<PruneRewrite>> feed_rewrites;

  feed_rewrites.reserve(fed_outputs.size());

  if (use_function_convention) {
    for (size_t i = 0; i < fed_outputs.size(); ++i) {
      feed_rewrites.emplace_back(new ArgFeedRewrite(
          &fed_outputs[i], &device_info, static_cast<int32>(i)));
    }
  } else {
    for (const string& fed_output : fed_outputs) {
      feed_rewrites.emplace_back(
          new RecvFeedRewrite(&fed_output, &device_info));
    }
  }

  std::vector<std::unique_ptr<PruneRewrite>> fetch_rewrites;
  fetch_rewrites.reserve(fetch_outputs.size());
  if (use_function_convention) {
    for (size_t i = 0; i < fetch_outputs.size(); ++i) {
      fetch_rewrites.emplace_back(new RetvalFetchRewrite(
          &fetch_outputs[i], &device_info, static_cast<int32>(i)));
    }
  } else {
    for (const string& fetch_output : fetch_outputs) {
      fetch_rewrites.emplace_back(
          new SendFetchRewrite(&fetch_output, &device_info));
    }
  }

  return RewriteGraphForExecution(
    g,
    feed_rewrites,
    fetch_rewrites,
    target_node_names,
    out_metadata);
}

namespace {
template <typename StringContainer>
std::vector<string> ConvertToVector(StringContainer field) {
  return std::vector<string>(field.begin(), field.end());
}
}  // namespace





Status RewriteGraphForExecution(
    Graph* g, // input and output
    const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites, // input
    const std::vector<std::unique_ptr<PruneRewrite>>& fetch_rewrites, // inputs
    const gtl::ArraySlice<string>& target_node_names, // inputs
    RewriteGraphMetadata* out_metadata) // output

// RewriteGraphMetadata 数据结构
// tensorflow/core/graph/subgraph.h:32:struct RewriteGraphMetadata
// - feed_types : DataTypeVector
//                The element type of each tensor fed to this subgraph.
// - fetch_types : DataTypeVector
//                 The element type of each tensor fetched from this subgraph.

{
  if (fetch_rewrites.empty() && target_node_names.empty()) {
    return errors::InvalidArgument(
        "Must specify at least one target to fetch or execute.");
  }

  std::unordered_set<string> endpoints;

  // 我的例子没有 feed
  for (const auto& feed_rewrite : feed_rewrites) {
    auto result = endpoints.insert(feed_rewrite->endpoint_name());
    if (!result.second) {
      return errors::InvalidArgument("Endpoint \"",
                                     feed_rewrite->endpoint_name(),
                                     "\" fed more than once.");
    }
  }

  for (const auto& fetch_rewrite : fetch_rewrites) {
    if (endpoints.count(fetch_rewrite->endpoint_name()) > 0) {
      return errors::InvalidArgument(fetch_rewrite->endpoint_name(),
                                     " is both fed and fetched.");
    }
  }

  // A separate index mapping name to Node*, for use by FeedInputs,
  // FetchOutputs, and PruneForTargets

  // NameIndex 数据结构
  // tensorflow/core/graph/subgraph.cc:47:
  // typedef std::unordered_map<StringPiece, Node*, StringPieceHasher> NameIndex;
  //                                 |         |
  //                             node name   node
  // 后面会往里面塞入 {string node_name: Node* node} 来增加 fetch, feed node
  NameIndex name_index;
  // 初始化 name_index
  name_index.reserve(g->num_nodes());
  for (Node* n : g->nodes()) {
    name_index[n->name()] = n;
  }


/**
比如

p *name_index (我是从 FetchOutputs 这个函数里面打印的)

{
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b643d083f8 "_retval_out_0_0",
      length_ = 15
    }

  ] = 0x55b643d3b198,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b641e21578 "out",
      length_ = 3
    }

  ] = 0x55b643d3b040,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b641d7a0b8 "c",
      length_ = 1
    }

  ] = 0x55b643d3af88,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b6411f4e08 "a/RandomStandardNormal",
      length_ = 22
    }

  ] = 0x55b643d3ae38,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b641dd96d8 "b/RandomStandardNormal",
      length_ = 22
    }

  ] = 0x55b643d3ada0,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b641d66068 "z",
      length_ = 1
    }

  ] = 0x55b643d3aed0,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b641d590d8 "y/RandomStandardNormal",
      length_ = 22
    }

  ] = 0x55b643d3ac70,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b641829a58 "a/shape",
      length_ = 7
    }

  ] = 0x55b643d3abf8,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b6418fe378 "x/RandomStandardNormal",
      length_ = 22
    }

  ] = 0x55b643d3ad08,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b641d53ce8 "b/shape",
      length_ = 7
    }

  ] = 0x55b643d3ab80,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b6411f1c88 "y/shape",
      length_ = 7
    }

  ] = 0x55b643d3aa90,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b6411f5378 "x/shape",
      length_ = 7
    }

  ] = 0x55b643d3ab08,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b643d28768 "_SINK",
      length_ = 5
    }

  ] = 0x55b643d3a9f8,
  [{
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x55b643d51e78 "_SOURCE",
      length_ = 7
    }

  ] = 0x55b643d3a980
}

*/


  // 我的例子没有 feed nodes
  // Add the feeds.  This may replace nodes in the graph, including the nodes
  // currently listed in "fetch_rewrites".  We pass "name_index" so the index is
  // kept up to date.
  if (!feed_rewrites.empty()) {
    TF_RETURN_IF_ERROR(
        // FeedInputs 函数:
        // tensorflow/core/graph/subgraph.cc:58:Status FeedInputs
        //
        FeedInputs(
          g, // input
          feed_rewrites, // input
          &name_index, // output
          &out_metadata->feed_types)); // output
  }


  // Add the fetch nodes, also updating "name_index".
  std::vector<Node*> fetch_nodes;
  if (!fetch_rewrites.empty()) {
    TF_RETURN_IF_ERROR(
      FetchOutputs(
        g, // input
        fetch_rewrites, // input
        &name_index, // output
        &fetch_nodes,  // output
        &out_metadata->fetch_types)); // output
  }

  // -----------------------------------------------------------------------
  // Prune the graph to only compute what is needed for the fetch nodes and the
  // target nodes.
  // -----------------------------------------------------------------------
  if (!fetch_nodes.empty() || !target_node_names.empty()) {
    TF_RETURN_IF_ERROR(
        // PruneForTargets 说明:
        // tensorflow/core/graph/subgraph.cc
        PruneForTargets(
          g, // input and output
          name_index, // input
          fetch_nodes, // input
          target_node_names)); // input
  }

  return Status::OK();
}

}  // namespace subgraph

}  // namespace tensorflow
