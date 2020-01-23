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

  info->clear();
  info->resize(g->num_node_ids());

  // parent_nodes 变量的作用:
  // 用于记录 某个节点的 父节点是哪个节点
  std::vector<const Node*> parent_nodes;
  parent_nodes.resize(g->num_node_ids());
  // p g->num_node_ids(), $14 = 14

  // 初始化 src_node 对应的 ControlFlowInfo instance
  const Node* src_node = g->source_node();
  ControlFlowInfo& src_info = (*info)[src_node->id()];
  src_info.frame = src_node;
  src_info.parent_frame = src_node;
  // 你发现没，src node 的 frame 和 parent_frame 都是 自己 src_node

  string frame_name;

  std::deque<const Node*> ready;

  ready.push_back(src_node);

  while (!ready.empty()) {

    const Node* curr_node = ready.front();
    ready.pop_front();

    const ControlFlowInfo& curr_info = (*info)[curr_node->id()];
    const Node* frame = curr_info.frame;
    const Node* parent = curr_info.parent_frame;
    frame_name = curr_info.frame_name;
    // 打印 p frame_name, $15 = ""

    // IsExit 函数说明
    // tensorflow/core/graph/graph.h, IsExit(const Node* node)
    // 继而调用 graph/graph.h, Node::IsExit()
    // 打印 SOURCE_ node 的 class_: p class_, $16 = tensorflow::Node::NC_OTHER

    // tensorflow::Node::NC_OTHER 意思是 Not a special kind of node

    if (IsExit(curr_node)) {

      // Exit to the parent frame.
      const ControlFlowInfo& parent_info = (*info)[parent->id()];
      frame = parent_info.frame;
      parent = parent_info.parent_frame;
      frame_name = parent_info.frame_name;
    }

    for (const Edge* out_edge : curr_node->out_edges()) {
      /*
      打印信息

      ptype curr_node->out_edges()
      type = const class tensorflow::EdgeSet {
        private:
          static const int kInline;
          const void *ptrs_[4];

        public:
          EdgeSet(void);
        private:
          EdgeSet(const tensorflow::EdgeSet &);
        public:
          ~EdgeSet();
          bool empty(void) const;
          size_type size(void) const;
          void clear(void);
          std::pair<tensorflow::EdgeSet::const_iterator, bool> insert(key_type);
          size_type erase(key_type);
          tensorflow::EdgeSet::const_iterator begin(void) const;
          tensorflow::EdgeSet::const_iterator end(void) const;
        private:
          std::set<const tensorflow::Edge*> * get_set(void) const;
          void RegisterMutation(void);
          void operator=(const tensorflow::EdgeSet &);

        public:
          typedef const tensorflow::Edge *key_type;
          typedef const tensorflow::Edge *value_type;
          typedef size_t size_type;
      } &
      p curr_node->out_edges()
      $19 = (const tensorflow::EdgeSet &) @0x5555d0160700: {static kInline = 4, ptrs_ = {0x5555d0160700, 0x5555ce1d6450, 0x5555d0160da0, 0x5555d0160f00}}
      p curr_node->out_edges().size()
      $20 = 7
      p curr_node->DebugString()
      $21 = "{name:'_SOURCE' id:0 source}"

      ===

      p out->DebugString()
      $25 = "{name:'y/shape' id:2 op device:{/job:localhost/replica:0/task:0/device:CPU:0} def:{{{node y/shape}} = Const[dtype=DT_INT32, value=Tensor<type: int32 shape: [2] values: 30 20>, _device=\"/job:localhost/replica:0/task:0/device:CPU:0\"]()}}"

      p out_edge->DebugString()
      $26 = "[id=13 _SOURCE:-1 -> y/shape:-1]"

      p out_id
      $27 = 2
      */

      /*
      p curr_node->DebugString()
      $41 = "{name:'y/shape' id:2 op device:{/job:localhost/replica:0/task:0/device:CPU:0} def:{{{node y/shape}} = Const[dtype=DT_INT32, value=Tensor<type: int32 shape: [2] values: 30 20>, _device=\"/job:localhost/replica:0/task:0/device:CPU:0\"]()}}"

      p out->DebugString()
      $40 = "{name:'y/RandomStandardNormal' id:6 op device:{/job:localhost/replica:0/task:0/device:CPU:0} def:{{{node y/RandomStandardNormal}} = RandomStandardNormal[T=DT_INT32, dtype=DT_FLOAT, seed=0, seed2=0, _device=\"/job:localhost/replica:0/task:0/device:CPU:0\"](y/shape)}}"

      p out_id
      $42 = 6

      */

      const Node* out = out_edge->dst();
      int out_id = out->id();
      ControlFlowInfo* out_info = &(*info)[out_id];
      const Node* out_parent = out_info->parent_frame;

      bool is_visited = (parent_nodes[out_id] != nullptr);

      // Skip Sink/Source nodes.
      if (!out->IsOp()) continue;

      // Add to ready queue if not seen.
      if (!is_visited) {
        // 记录父节点，具体来说，y/shape 节点， x/shape 节点 的 parent_nodes
        // 对应的 node id 的 Node* 都指向 SOURCE_ 节点。
        parent_nodes[out->id()] = curr_node;

        /*
        打印
        p curr_node->DebugString()
        $28 = "{name:'_SOURCE' id:0 source}"

        parent_nodes[out->id()] = curr_node;
                     ---------    ----------
                     2, y/shape    _SOURCE
        */
        ready.push_back(out);
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
        // 非 Enter Op 分支:
        if (is_visited) {
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
          // 非 Enter Op 分支: op 还没有访问过

          // 打印的情况是吧 y/shape Node 的 control_flow frame, parent_frame 都初始化为 SOURCE_ Node
          out_info->frame = frame;
          /*
          p frame->DebugString()
          $33 = "{name:'_SOURCE' id:0 source}"
          */
          out_info->parent_frame = parent;
          /*
          p parent->DebugString()
          $34 = "{name:'_SOURCE' id:0 source}"
          */
          out_info->frame_name = frame_name;
          /*
          p frame_name
          $35 = ""
          */
          // comment:
          // 即使说，current node 换成了 'y/shape'，由于 y/shape 节点 的 frame, parent_frame 也是 SOURCE_ 节点
          // 所以，这里似乎所有的节点的 frame, parent_frame 都是 SOURCE_ 节点的指针
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
