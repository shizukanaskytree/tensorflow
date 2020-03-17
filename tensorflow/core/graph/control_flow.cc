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

#include "tensorflow/core/graph/control_flow.h"

#include <deque>
#include <vector>

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {
// Information about a loop frame structure.
struct Frame {
  string name;

  // Pointer to the parent frame. The root frame has a pointer to itself.
  Frame* parent = nullptr;

  // The loop condition of the loop. There should be exactly one loop condition
  // in every loop.
  const Node* loop_cond = nullptr;
};

// Verify that the ControlFlowInfo of the graph has valid loop structure.
Status ValidateControlFlowInfo(const Graph* graph,
                               const std::vector<ControlFlowInfo>& cf_info) {
  std::unordered_map<string, Frame> frames;
  for (const Node* node : graph->op_nodes()) {
    const ControlFlowInfo& cf = cf_info[node->id()];
    if (!cf.frame || !cf.parent_frame) {
      // Skip nodes unreachable from the source node. They might be pruned
      // later.
      continue;
    }

    Frame& frame = frames[cf.frame_name];
    Frame* parent = &frames[cf_info[cf.parent_frame->id()].frame_name];
    if (frame.parent == nullptr) {
      frame.parent = parent;
      frame.name = cf.frame_name;
    } else if (frame.parent != parent) {
      return errors::Internal(
          "Invalid loop structure: Mismatched parent frames for \"",
          cf.frame_name, "\": \"", parent->name, "\" vs \"", frame.parent->name,
          "\". The node giving this error: ", FormatNodeForError(*node),
          ". This is an internal bug, please file a bug report with "
          "instructions on how to reproduce the error.");
    }
    if (IsLoopCond(node)) {
      // ForwardLoopCounter runs in the same frame as the forward loop and
      // BackPropLoopCounter runs in the same frame as the backprop loop. They
      // are the only cases that multiple loops share the same frame.
      if (frame.loop_cond &&
          !str_util::StrContains(frame.loop_cond->name(), "LoopCounter") &&
          !str_util::StrContains(node->name(), "LoopCounter")) {
        return errors::InvalidArgument(
            "Invalid loop structure: Loop \"", cf.frame_name,
            "\" has more than one LoopCond node: ", FormatNodeForError(*node),
            " and ", FormatNodeForError(*frame.loop_cond),
            ". This is an internal bug, please file a bug report with "
            "instructions on how to reproduce the error.");
      }
      frame.loop_cond = node;
    }
  }
  return Status::OK();
}
}  // namespace


// 1.
// 一句话概括:
// Build the control flow info for every node.

// 2.
// 一句人话总结:
// 由 current node 散射出去达到的 out node, 这个 out node 所属的 control flow 是 out_info
// out node 的 control flow info 的 frame, parent_frame, frame_name 跟它的 predecessor node 姓
// 指向的都是 predecessor node 所属的  frame, parent_frame, frame_name.
// 这里的 predecessor node 是 current node.

// 3.
// 大胆猜测:
// 因为前面发现说 "你发现没，src node 的 frame 和 parent_frame 都是 自己 src_node"
// 所以, 这个 graph 里面 所有的 node 的 frame, parent_frame, frame_name 都是 src_node.

// Clear and populate `info` with each node's frame and the level it belongs to.
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
Status BuildControlFlowInfo(
  const Graph* g, // input
  std::vector<ControlFlowInfo>* info, // output
  std::vector<string>* unreachable_nodes /*=nullptr*/) {
  // 1.
  // 演示图:
  // https://keep.google.com/u/1/#NOTE/11Q9d55Ui19RiiiTSUanHjSLoZWlmdTtcNB1WZBefbZqGpmu5ErXF3SGHR2avSw

  info->clear();
  info->resize(g->num_node_ids());
  // 1.
  // Graph::num_node_ids() 返回的是 图里面所有 nodes 的个数.
  // Returns one more than the maximum id assigned to any node.

  // 2.
  // Graph::nodes_
  // Map from node ids to allocated nodes.  nodes_[id] may be nullptr if
  // the node with that id was removed from the graph.
  // std::vector<Node*> nodes_;

  // 3.
  // 此刻的图: https://gist.github.com/shizukanaskytree/8d83d0f73b6970d33cc2e4b2c4bdd9c6
  // context: https://docs.google.com/document/d/1FdkOkqexWnymuYKqF3HCNlRUAbDXjTHTjsK8wIdxIRQ/edit#

  // 4.
  // p g->num_node_ids()
  // 6

  std::vector<const Node*> parent_nodes;
  // 1.
  // parent_nodes 变量的作用:
  // 用于记录 某个节点的 父节点是哪个节点

  parent_nodes.resize(g->num_node_ids());
  // 1.
  // 这个的意思是指每个 node 都要有一个 parent node 了

  // 初始化 src_node 对应的 ControlFlowInfo instance
  const Node* src_node = g->source_node();
  ControlFlowInfo& src_info = (*info)[src_node->id()];
  src_info.frame = src_node;
  src_info.parent_frame = src_node;
  // 1.
  // comment:
  // 你发现没，src node 的 frame 和 parent_frame 都是 自己 src_node

  string frame_name;
  // 1.
  // 打印 p frame_name, $15 = ""

  std::deque<const Node*> ready;
  ready.push_back(src_node);
  while (!ready.empty()) {
    const Node* curr_node = ready.front();
    ready.pop_front();
    const ControlFlowInfo& curr_info = (*info)[curr_node->id()];
    const Node* frame = curr_info.frame;
    const Node* parent = curr_info.parent_frame;
    frame_name = curr_info.frame_name;
    // 1.
    // comment:
    // 上面产生了 curr_info, frame, parent 其实都是 "代理" 的功用.
    // 下面是对其赋值的. 所以用"代理"方便一些, 毕竟代码都短了很多

    // 2.
    // 目的
    // 对 current node 所"散射"出去的 nodes 所对应的 control flow info 内的 frame, parent_frame 进行赋值.

    // 1.
    // 如果这个 node 是 While Loop 结构中的 Exit 类节点
    if (IsExit(curr_node)) {
      // 1.
      // IsExit 函数说明:
      // bool IsExit() const { return class_ == NC_EXIT; }

      // 1.1
      // SOURCE_ node 未进入

      // 2.
      // tensorflow/core/graph/graph.h, IsExit(const Node* node)
      // 继而调用 graph/graph.h, Node::IsExit()
      // 打印 SOURCE_ node 的 class_: p class_, $16 = tensorflow::Node::NC_OTHER

      // 3.
      // tensorflow::Node::NC_OTHER 意思是 Not a special kind of node

      // Exit to the parent frame.
      // 1.
      // 这句注释特别有启发
      const ControlFlowInfo& parent_info = (*info)[parent->id()];
      frame = parent_info.frame;
      parent = parent_info.parent_frame;
      frame_name = parent_info.frame_name;
    }

    for (const Edge* out_edge : curr_node->out_edges()) {
      // 1.
      // ptype curr_node->out_edges()
      // type = const class tensorflow::EdgeSet

      // 2.
      // p out_edge->DebugString()
      // $26 = "[id=13 _SOURCE:-1 -> y/shape:-1]"

      const Node* out = out_edge->dst();
      // 1.
      // 打印几个例子:
      // p out->DebugString()
      // $25 = "{name:'y/shape' id:2 op device:{/job:localhost/replica:0/task:0/device:CPU:0} def:{{{node y/shape}} = Const[dtype=DT_INT32, value=Tensor<type: int32 shape: [2] values: 30 20>, _device=\"/job:localhost/replica:0/task:0/device:CPU:0\"]()}}"
      // p out->DebugString()
      // $40 = "{name:'y/RandomStandardNormal' id:6 op device:{/job:localhost/replica:0/task:0/device:CPU:0} def:{{{node y/RandomStandardNormal}} = RandomStandardNormal[T=DT_INT32, dtype=DT_FLOAT, seed=0, seed2=0, _device=\"/job:localhost/replica:0/task:0/device:CPU:0\"](y/shape)}}"

      int out_id = out->id();
      // p out_id
      // $42 = 6

      ControlFlowInfo* out_info = &(*info)[out_id];
      // 1.
      // out_info 变量说明
      // 此刻, out_info 是 右边取出的 &(*info)[out_id], 即 out node 的 info 的 "代理"

      const Node* out_parent = out_info->parent_frame;

      bool is_visited = (parent_nodes[out_id] != nullptr);

      // Skip Sink/Source nodes.
      if (!out->IsOp()) continue;

      // Add to ready queue if not seen.
      if (!is_visited) {
        // 记录父节点，具体来说，y/shape 节点， x/shape 节点 的 parent_nodes
        // 对应的 node id 的 Node* 都指向 SOURCE_ 节点。
        parent_nodes[out->id()] = curr_node;
        // 1.
        // 打印

        // 1.1
        // context: https://docs.google.com/document/d/1FdkOkqexWnymuYKqF3HCNlRUAbDXjTHTjsK8wIdxIRQ/edit#
        //
        // 一句人话概括: 4 号 node, _arg_y_0_1, 的 parent node 是 _SOURCE node.
        //
        // (gdb) p curr_node->DebugString()
        // $17 = "{name:'_SOURCE' id:0 source}"
        //
        // p out_edge->DebugString()
        // $16 = "[id=6 _SOURCE:-1 -> _arg_y_0_1:-1]"
        //
        // (gdb) p out->DebugString()
        // $18 = "{name:'_arg_y_0_1' id:4 op device:{/job:localhost/replica:0/task:0/device:CPU:0} def:{{{node _arg_y_0_1}} = _Arg[T=DT_FLOAT, index=1]()}}"
        //
        // (gdb) p out_id
        // $19 = 4

        // 1.2
        // p curr_node->DebugString()
        // $28 = "{name:'_SOURCE' id:0 source}"
        //
        // parent_nodes[out->id()] = curr_node;
        //              ---------    ----------
        //              2, y/shape    _SOURCE

        ready.push_back(out);
        // 1.
        // 意图
        // while 循环, bfs 持续进行, 从而遍历整个图.
      }

      // IsEnter 函数说明:
      // core/graph/graph.h, IsEnter() --> node->IsEnter()
      // 我也不知道定义是什么， Enter 类型的是用来构造 for loop , while loop 用的 node
      // tensorflow/core/common_runtime/lower_while_op.cc
      //
      // Helper to convert a functional While op to its lowered form.
      //
      // Example:
      //
      // Input graph:
      //
      // loop_var -> WhileOp<cond_func, body_func> -> consumer
      //
      // Output graph(top to down flow):
      //
      //                          loop_var
      //                             |
      //                           Enter
      //                             |
      // inlined_cond_func ---<--- Merge -----<----- NextIteration
      //      |                      |                    |
      //      V                      V                    ^
      //      |                      |                    |
      //  LoopCond ------>-------- Switch ---->---- inlined_body_func
      //                             |
      //                           Exit
      //                             |
      //                          consumer

      // Process the node 'out'.
      if (IsEnter(out)) {
        // 1.
        // e.g.
        // curr_node = SOURCE_ node 未进入

        if (is_visited) {
          const string& parent_frame = (*info)[out_parent->id()].frame_name;
          if (parent_frame != frame_name) {
            return errors::InvalidArgument(
                FormatNodeForError(*out),
                " has inputs from different frames. The input ",
                FormatNodeForError(*curr_node), " is in frame '", frame_name,
                "'. The input ", FormatNodeForError(*parent_nodes[out->id()]),
                " is in frame '", parent_frame, "'.");
          }
        } else {
          out_info->frame = out;
          out_info->parent_frame = frame;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(out->attrs(), "frame_name", &out_info->frame_name));
          if (out_info->frame_name.empty()) {
            return errors::InvalidArgument("The Enter ",
                                           FormatNodeForError(*out),
                                           " must have a frame name.");
          }
        }
      } else {
        // 1.
        // 非 Enter Op 分支:
        // e.g.
        // curr_node = SOURCE_ node 进入

        if (is_visited) {
          // 未进入

          // 非 Enter Op 分支: op 已经访问过了
          if (out_info->frame_name != frame_name) {
            return errors::InvalidArgument(
                FormatNodeForError(*out),
                " has inputs from different frames. The input ",
                FormatNodeForError(*curr_node), " is in frame '", frame_name,
                "'. The input ", FormatNodeForError(*parent_nodes[out->id()]),
                " is in frame '", out_info->frame_name, "'.");
          }
        } else {
          // 进入
          // 非 Enter Op 分支: op 还没有访问过

          out_info->frame = frame;
          out_info->parent_frame = parent;
          out_info->frame_name = frame_name;
          // 1.
          // 一句人话总结:
          // 由 current node 散射出去达到的 out node, 这个 out node 所属的 control flow 是 out_info
          // out node 的 control flow info 的 frame, parent_frame, frame_name 跟它的 predecessor node 姓
          // 指向的都是 predecessor node 所属的  frame, parent_frame, frame_name.
          // 这里的 predecessor node 是 current node.

          // 2.
          // 大胆猜测:
          // 因为前面发现说 "你发现没，src node 的 frame 和 parent_frame 都是 自己 src_node"
          // 所以, 这个 graph 里面 所有的 node 的 frame, parent_frame, frame_name 都是 src_node.

        }
      }
    }
  }


  if (unreachable_nodes) {
    for (const Node* node : g->op_nodes()) {
      if (!parent_nodes[node->id()]) {
        unreachable_nodes->push_back(node->name());
      }
    }
  }
  TF_RETURN_IF_ERROR(ValidateControlFlowInfo(g, *info));
  return Status::OK();
}

}  // namespace tensorflow
