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

// 1.
// 概述:
// 添加 _Arg node 并连边, 消除 placeholder node 的连接边
//
// _arg_x_0_0    placeholder_x:0 ------ MatMul:0
//      |__________________________________|
//
// _arg_x_0_0    placeholder_x:0 ---X--- MatMul:0
//      |__________________________________|

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
    Graph* g, // input and output
    const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites, // input
    NameIndex* name_index, // input
    DataTypeVector* out_feed_types) // output
{
  out_feed_types->clear();
  out_feed_types->reserve(feed_rewrites.size());
  // 1.
  // feed_rewrites.size()
  // 尺寸为 2

  for (size_t i = 0; i < feed_rewrites.size(); ++i) {
    const string& t = feed_rewrites[i]->endpoint_name();
    // 1.
    // 打印 t
    // $24 = "x:0"
    // $25 = "y:0"

    // 1.1
    // 说明:
    // :0 表示 output slot of the node 'x' or 'y'

    TensorId id(ParseTensorName(t));
    // 1.
    // TensorId 数据结构
    // tensorflow/core/graph/tensor_id.cc
    //
    // (gdb) ptype id
    // type = struct tensorflow::TensorId : public std::pair<absl::string_view, int> {
    //   public:
    //     TensorId(void);
    //     TensorId(const tensorflow::SafeTensorId &);
    //     const tensorflow::StringPiece node(void) const;
    //     int index(void) const;
    //     std::string ToString(void) const;
    // }

    // 1.1
    // ParseTensorName 函数在
    // tensorflow/core/graph/tensor_id.cc

    // 2.
    // TensorId::first 是什么?
    //
    // (gdb) ptype id
    // type = struct tensorflow::TensorId : public std::pair<absl::string_view, int>
    // 所以, TensorId::first 是 absl::string_view
    //
    // 打印
    // (gdb) p id.first
    // $30 = {static npos = 18446744073709551615, static kMaxSize = 9223372036854775807, ptr_ = 0x5641e3c653d8 "x:0", length_ = 1}

    // 3.
    // TensorId::second 是什么?
    // 所以, TensorId::second 是 int
    // 打印
    // (gdb) p id.second
    // $31 = 0
    //
    // short answer: 'name:digits' 中的 digits
    // e.g., 'x:0' 中的 '0'
    // 规则:
    // 来自 tensorflow/core/graph/tensor_id.cc
    // Parse either a name, ^name, or name:digits.  To do so, we go backwards from
    // the end of the string, skipping over a run of digits.  If we hit a ':'
    // character, then we know we are in the 'name:digits' regime.  Otherwise, we
    // see if the name starts with '^', indicating a control edge. If we find
    // neither ':' nor '^' characters, the output index is implicitly 0, and the
    // whole name string forms the first part of the tensor name.

    // 如下是异常检测, 不重要
    auto iter = name_index->find(id.first);
    // 1.
    // id.first 是什么
    // (gdb) p id.first
    // $35 = {static npos = 18446744073709551615, static kMaxSize = 9223372036854775807, ptr_ = 0x5641e3c653d8 "x:0", length_ = 1}

    // 2.
    // name_index 是什么?
    // The pair of node name, node*
    // 这个变量是在 RewriteGraphForExecution 里面被构造和初始化的.
    //
    // 打印:
    // std::unordered_map with 5 elements = {
    //     [{
    //         static npos = 18446744073709551615,
    //         static kMaxSize = 9223372036854775807,
    //         ptr_ = 0x5641e1e64268 "y",
    //         length_ = 1
    //     }] = 0x5641e3c7e968,
    //     [{
    //         static npos = 18446744073709551615,
    //         static kMaxSize = 9223372036854775807,
    //         ptr_ = 0x5641e18b6058 "x",
    //         length_ = 1
    //     }] = 0x5641e3c7e8f0,
    //     [{
    //         static npos = 18446744073709551615,
    //         static kMaxSize = 9223372036854775807,
    //         ptr_ = 0x5641e129ada8 "MatMul",
    //         length_ = 6
    //     }] = 0x5641e3c7e9e0,
    //     [{
    //         static npos = 18446744073709551615,
    //         static kMaxSize = 9223372036854775807,
    //         ptr_ = 0x5641e3c7d2a8 "_SINK",
    //         length_ = 5
    //     }] = 0x5641e3c7e858,
    //     [{
    //         static npos = 18446744073709551615,
    //         static kMaxSize = 9223372036854775807,
    //         ptr_ = 0x5641e3c6ae88 "_SOURCE",
    //         length_ = 7
    //     }] = 0x5641e3c7e7e0
    // }

    if (iter == name_index->end()) {
      return errors::NotFound("FeedInputs: unable to find feed output ", t);
    }
    // 1.
    // QQQ. "feed output" ???

    // 如下是异常检测, 不重要
    Node* n = iter->second;
    DCHECK_EQ(n->name(), id.first);
    if (id.second >= n->num_outputs()) {
      return errors::InvalidArgument(
          "FeedInputs: ", t, " should have output index < ", n->num_outputs());
    }

    Node* feed_node;
    TF_RETURN_IF_ERROR(
        feed_rewrites[i]->AddNode(
          g, // input and output
          {n, id.second}, // input
          &feed_node)); // output
    // 1.
    // AddNode 函数是构造节点的函数.
    // ArgFeedRewrite::AddNode
    //

    // 2.
    // id.second is within [0, n->num_outputs())
    // id.second is number of fanout

    // Update name_index
    (*name_index)[feed_node->name()] = feed_node;
    // 1.
    // feed_node 是什么?
    // feed_node 是从 feed_rewrites[i]->AddNode() 里面构造出来的

    // 1.1
    // 打印 feed_node
    // p feed_node->DebugString()
    // $62 = "{name:'_arg_x_0_0' id:5 op device:{/job:localhost/replica:0/task:0/device:CPU:0} def:{{{node _arg_x_0_0}} = _Arg[T=DT_FLOAT, index=0]()}}"

    // Duplicate control edges aren't allowed, but feed_node was *just* created
    // so there's no need to check for a duplicate.
    g->AddControlEdge(g->source_node(), feed_node, true);
    // 1.
    // 参数对照:
    // const Edge* Graph::AddControlEdge(Node* source, Node* dest, bool allow_duplicates)

    // 2
    // g->source_node() 是什么?
    // ptype g->source_node()
    // Node* source
    // 值:
    // $64 = "{name:'_SOURCE' id:0 source}"

    // 3.
    // 所以这里是打算把 _SOURCE <--> _Arg 连起来
    // 居然真的执行了.
    // https://docs.google.com/document/d/1FdkOkqexWnymuYKqF3HCNlRUAbDXjTHTjsK8wIdxIRQ/edit#heading=h.or282np0alz6

    // Look through edges coming out of "n" for edges whose src_output() index
    // matches "output_index".  If found, replace the edges with a connection
    // from the special feed node.
    std::vector<const Edge*> to_remove;
    // 1.
    // Intent of to_remove
    // 这里是要 remove graph g 里面的 edge !!!

    // 2.
    // 注释里的 "n" 是什么?
    // p n->DebugString()
    // $69 = "{name:'x' id:2 op device:{/job:localhost/replica:0/task:0/device:GPU:0} def:{{{node x}} = Placeholde

    for (const Edge* e : n->out_edges()) {
      // 1.
      // (gdb) p n->out_edges().size()
      // $70 = 1

      // 2
      // 这条边是什么?
      // (gdb) p e->DebugString()
      // $71 = "[id=1 x:0 -> MatMul:0]"

      if (e->src_output() == id.second) {
        // 1.
        // e->src_output() 解释:
        // Return the index of the source output that produces the data
        // carried by this edge.  The special value kControlSlot is used
        // for control dependencies.

        // 2.
        // 进入了

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
      // 1.
      // e ?
      // $76 = "[id=1 x:0 -> MatMul:0]"
      //  placeholder_x:0 ----- MatMul:0

      if (e->src_output() == id.second) {
        // 1.
        // 进入了
        // 意图:
        // _arg_x_0_0    placeholder_x:0 ------ MatMul:0
        //      |__________________________________|
        //
        // 打印 feed_node
        // p feed_node->DebugString()
        // $62 = "{name:'_arg_x_0_0' id:5 op device:{/job:localhost/replica:0/task:0/device:CPU:0} def:{{{node _arg_x_0_0}} = _Arg[T=DT_FLOAT, index=0]()}}"

        g->AddEdge(feed_node, 0, e->dst(), e->dst_input());
        // 1.
        // const Edge* Graph::AddEdge(Node* source, int x, Node* dest, int y)
        // _arg_x_0_0    placeholder_x:0 ------ MatMul:0
        //      |__________________________________|

      } else {
        CHECK_EQ(Graph::kControlSlot, e->src_output());
        // Duplicate control edges aren't allowed, but feed_node was *just*
        // created so there's no need to check for a duplicate.
        g->AddControlEdge(feed_node, e->dst(), true);
      }

      g->RemoveEdge(e);
      // 1.
      // 居然要 remove edge!!!
      // 删除 "[id=1 x:0 -> MatMul:0]" 这条边
        // _arg_x_0_0    placeholder_x:0 ---X--- MatMul:0
        //      |__________________________________|
    }
    out_feed_types->push_back(BaseType(n->output_type(id.second)));
    // 1.
    // out_feed_types 是什么?
    // 这个函数的输入参数 DataTypeVector* out_feed_types

    // 2.
    // DataType Node::output_type(int32 o) const { return props_->output_types[o]; }

    // 3.
    // enum DataType {DT_FLOAT 等}, in types.pb.h

    // 4.
    // inline DataType BaseType(DataType dtype) {
    //   return IsRefType(dtype) ? RemoveRefType(dtype) : dtype;
    // }

    // 5.
    // retval
    // Value returned is $81 = tensorflow::DT_FLOAT

    // 6.
    // out_feed_types->push_back(tensorflow::DT_FLOAT)
  } // for loop: for another feed node that needs to be processed.

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
    // 1.
    // const Edge* Graph::AddControlEdge(Node* source, Node* dest, bool allow_duplicates)

    // 2.
    // context:
    // https://docs.google.com/document/d/1FdkOkqexWnymuYKqF3HCNlRUAbDXjTHTjsK8wIdxIRQ/edit#
    //
    // (gdb) p source->DebugString()
    // $10 = "{name:'_retval_MatMul_0_0' id:7 op device:{/job:localhost/replica:0/task:0/device:CPU:0} def:{{{node _retval_MatMul_0_0}} = _Retval[T=DT_FLOAT, index=0](MatMul)}}"
    // (gdb) p dest->DebugString()
    // $11 = "{name:'_SINK' id:1 sink}"

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

  // 1.
  // PruneForReverseReachability 后
  // 此刻的图:
  //
  // node {
  //   name: "MatMul"
  //   op: "MatMul"
  //   input: "_arg_x_0_0"
  //   input: "_arg_y_0_1"
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

  // Reconnect nodes with no outgoing edges to the sink node
  FixupSourceAndSinkEdges(g);
  // 1.
  // FixupSourceAndSinkEdges 说明：
  // tensorflow/core/graph/algorithm.cc
  // Connect all nodes with no incoming edges to source.
  // Connect all nodes with no outgoing edges to sink.

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
  // 1.
  // NodeBuilder 构造函数
  // strings::StrCat("_arg_", feed_tensor.node->name(), "_", feed_tensor.index, "_", arg_index_)
  //

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
  // 1.
  // struct RewriteGraphMetadata
  // subgraph.h

  // 1.1
  // DataTypeVector
  // typedef gtl::InlinedVector<DataType, 4> DataTypeVector;

  std::vector<std::unique_ptr<PruneRewrite>> feed_rewrites;
  // 1.
  // PruneRewrite
  //

  feed_rewrites.reserve(fed_outputs.size());

  if (use_function_convention) {
    for (size_t i = 0; i < fed_outputs.size(); ++i) {

      feed_rewrites.emplace_back(
        new ArgFeedRewrite(
          &fed_outputs[i], // endpoint_name
          &device_info, // device_info
          static_cast<int32>(i))); // arg_index
      // 1.
      // ArgFeedRewrite 构造函数
      // tensorflow/core/graph/subgraph.h
      //
      // ArgFeedRewrite(
      //   const string* endpoint_name,
      //   const DeviceAttributes* device_info,
      //   int32 arg_index)

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

// 1.
// 一句话概括:
// 把用户输入的图转变为一个带 _Arg, _Retval nodes 的图, 消掉以前的 placeholder 节点.
Status RewriteGraphForExecution(
    Graph* g, // input and output
    const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites, // input
    const std::vector<std::unique_ptr<PruneRewrite>>& fetch_rewrites, // inputs
    const gtl::ArraySlice<string>& target_node_names, // inputs
    RewriteGraphMetadata* out_metadata) // output
{
  // 1.
  // RewriteGraphMetadata 数据结构
  // tensorflow/core/graph/subgraph.h:32:struct RewriteGraphMetadata
  // - feed_types : DataTypeVector
  //                The element type of each tensor fed to this subgraph.
  // - fetch_types : DataTypeVector
  //                 The element type of each tensor fetched from this subgraph.

  // 2.
  // PruneRewrite

  // 3.
  // (gdb) p feed_rewrites.size()
  // $5 = 2
  // (gdb) p fetch_rewrites.size()
  // $6 = 1
  //
  // 背景: https://docs.google.com/document/d/1FdkOkqexWnymuYKqF3HCNlRUAbDXjTHTjsK8wIdxIRQ/edit#heading=h.jv22t2piwshw
  //
  // (gdb) p *(feed_rewrites[0].get()->endpoint_name_)
  // $8 = "x:0"
  //
  // (gdb) p *(feed_rewrites[1].get()->endpoint_name_)
  // $9 = "y:0"
  //
  // (gdb) p *(fetch_rewrites[0].get()->endpoint_name_)
  // $11 = "MatMul:0"

  // QQQ.
  // :0 是什么?
  // a particular tensor endpoint (described by a "<node_name>:<output_index>" pair)

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
  // 1.
  // "if (endpoints.count(fetch_rewrite->endpoint_name()) > 0)" 是多少?
  // (gdb) p endpoints.count(fetch_rewrite->endpoint_name())
  // $17 = 0

  // 1.1
  // 意图: 不能让 fetch node 也成为 feed node

  // A separate index mapping name to Node*, for use by FeedInputs,
  // FetchOutputs, and PruneForTargets
  // 1.
  // Feed, 亦即 Inputs
  // Fetch, 亦即 Outputs

  NameIndex name_index;
  // 1.
  // NameIndex 数据结构
  // tensorflow/core/graph/subgraph.cc:47:
  // typedef std::unordered_map<StringPiece, Node*, StringPieceHasher> NameIndex;
  //                                 |         |
  //                             node name   node
  // 后面会往里面塞入 {string node_name, Node* node} 来增加 fetch, feed node

  name_index.reserve(g->num_nodes());
  // 初始化 name_index 大小

  for (Node* n : g->nodes()) {
    name_index[n->name()] = n;
    // 1.
    // 打印
    // (gdb) p n->name()
    // $18 = "_SOURCE"
    // "_SOURCE", node*
    // $19 = "_SINK"
    // $20 = "x"
    // $21 = "y"
    // $22 = "MatMul"
    // 只有上述这几个
  }
  // 1.
  // 例子
  // context of the graph:
  // https://docs.google.com/document/d/1FdkOkqexWnymuYKqF3HCNlRUAbDXjTHTjsK8wIdxIRQ/edit#heading=h.jv22t2piwshw

  // 2.
  // 打印例子 1
  // 注意
  // ptr_ = 0x55b643d083f8 "_retval_out_0_0"
  // ptr_ = 0x55b641e21578 "out",
  //
  // 比如
  // p *name_index (我是从 FetchOutputs 这个函数里面打印的)
  // {
  //   [{
  //       static npos = 18446744073709551615,
  //       static kMaxSize = 9223372036854775807,
  //       ptr_ = 0x55b643d083f8 "_retval_out_0_0",
  //       length_ = 15
  //     }
  //   ] = 0x55b643d3b198,
  //   [{
  //       static npos = 18446744073709551615,
  //       static kMaxSize = 9223372036854775807,
  //       ptr_ = 0x55b641e21578 "out",
  //       length_ = 3
  //     }
  //   ] = 0x55b643d3b040,
  //   [{
  //       static npos = 18446744073709551615,
  //       static kMaxSize = 9223372036854775807,
  //       ptr_ = 0x55b641d7a0b8 "c",
  //       length_ = 1
  //     }
  //   ] = 0x55b643d3af88,
  //   [{
  //       static npos = 18446744073709551615,
  //       static kMaxSize = 9223372036854775807,
  //       ptr_ = 0x55b6411f4e08 "a/RandomStandardNormal",
  //       length_ = 22
  //     }
  //   ] = 0x55b643d3ae38,
  //   [{
  //       static npos = 18446744073709551615,
  //       static kMaxSize = 9223372036854775807,
  //       ptr_ = 0x55b641dd96d8 "b/RandomStandardNormal",
  //       length_ = 22
  //     }
  //   ] = 0x55b643d3ada0,
  //   [{
  //       static npos = 18446744073709551615,
  //       static kMaxSize = 9223372036854775807,
  //       ptr_ = 0x55b641d66068 "z",
  //       length_ = 1
  //     }
  //   ] = 0x55b643d3aed0,
  //   [{
  //       static npos = 18446744073709551615,
  //       static kMaxSize = 9223372036854775807,
  //       ptr_ = 0x55b641d590d8 "y/RandomStandardNormal",
  //       length_ = 22
  //     }
  //   ] = 0x55b643d3ac70,
  //   [{
  //       static npos = 18446744073709551615,
  //       static kMaxSize = 9223372036854775807,
  //       ptr_ = 0x55b641829a58 "a/shape",
  //       length_ = 7
  //     }
  //   ] = 0x55b643d3abf8,
  //   [{
  //       static npos = 18446744073709551615,
  //       static kMaxSize = 9223372036854775807,
  //       ptr_ = 0x55b6418fe378 "x/RandomStandardNormal",
  //       length_ = 22
  //     }
  //   ] = 0x55b643d3ad08,
  //   [{
  //       static npos = 18446744073709551615,
  //       static kMaxSize = 9223372036854775807,
  //       ptr_ = 0x55b641d53ce8 "b/shape",
  //       length_ = 7
  //     }
  //   ] = 0x55b643d3ab80,
  //   [{
  //       static npos = 18446744073709551615,
  //       static kMaxSize = 9223372036854775807,
  //       ptr_ = 0x55b6411f1c88 "y/shape",
  //       length_ = 7
  //     }
  //   ] = 0x55b643d3aa90,
  //   [{
  //       static npos = 18446744073709551615,
  //       static kMaxSize = 9223372036854775807,
  //       ptr_ = 0x55b6411f5378 "x/shape",
  //       length_ = 7
  //     }
  //   ] = 0x55b643d3ab08,
  //   [{
  //       static npos = 18446744073709551615,
  //       static kMaxSize = 9223372036854775807,
  //       ptr_ = 0x55b643d28768 "_SINK",
  //       length_ = 5
  //     }
  //   ] = 0x55b643d3a9f8,
  //   [{
  //       static npos = 18446744073709551615,
  //       static kMaxSize = 9223372036854775807,
  //       ptr_ = 0x55b643d51e78 "_SOURCE",
  //       length_ = 7
  //     }
  //   ] = 0x55b643d3a980
  // }

  // Add the feeds.  This may replace nodes in the graph, including the nodes
  // currently listed in "fetch_rewrites".  We pass "name_index" so the index is
  // kept up to date.
  if (!feed_rewrites.empty()) {
    TF_RETURN_IF_ERROR(
        FeedInputs(
          g, // input and output
          feed_rewrites, // input
          &name_index, // input and output
          &out_metadata->feed_types)); // output
        // 1.
        // FeedInputs 函数:
        // tensorflow/core/graph/subgraph.cc
  }
  // 1.
  // 在增加了 FeedInputs 后,
  // 此刻的图 g:
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
  //   input: "_arg_x_0_0"
  //   input: "_arg_y_0_1"
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
  // library {
  // }
  // versions {
  //   producer: 27
  // }
  //

  // Add the fetch nodes, also updating "name_index".
  std::vector<Node*> fetch_nodes;
  if (!fetch_rewrites.empty()) {
    TF_RETURN_IF_ERROR(
      FetchOutputs(
        g, // input
        fetch_rewrites, // input
        &name_index, // input and output
        &fetch_nodes,  // output
        &out_metadata->fetch_types)); // output
  }
  // 1.
  // 此刻的 graph
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
  //   input: "_arg_x_0_0"
  //   input: "_arg_y_0_1"
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
  //

  // -----------------------------------------------------------------------
  // 1. 一句话概括:
  // Prune the graph to only compute what is needed for the fetch nodes and the
  // target nodes.
  // 根据 PruneForReverseReachability() 方法.
  // 算法在 tensorflow/core/graph/algorithm.cc
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
  // 1.
  // 此刻的 graph
  // p g->ToGraphDefDebug().DebugString()
  // node {
  //   name: "MatMul"
  //   op: "MatMul"
  //   input: "_arg_x_0_0"
  //   input: "_arg_y_0_1"
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

  return Status::OK();
}

}  // namespace subgraph

}  // namespace tensorflow
