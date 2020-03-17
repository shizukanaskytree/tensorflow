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

#include "tensorflow/core/graph/algorithm.h"

#include <algorithm>
#include <deque>
#include <vector>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {
template <typename T>
void DFSFromHelper(const Graph& g, gtl::ArraySlice<T> start,
                   const std::function<void(T)>& enter,
                   const std::function<void(T)>& leave,
                   const NodeComparator& stable_comparator,
                   const EdgeFilter& edge_filter) {
  // Stack of work to do.
  struct Work {
    T node;
    bool leave;  // Are we entering or leaving n?
  };
  std::vector<Work> stack(start.size());
  for (int i = 0; i < start.size(); ++i) {
    stack[i] = Work{start[i], false};
  }

  std::vector<bool> visited(g.num_node_ids(), false);
  while (!stack.empty()) {
    Work w = stack.back();
    stack.pop_back();

    T n = w.node;
    if (w.leave) {
      leave(n);
      continue;
    }

    if (visited[n->id()]) continue;
    visited[n->id()] = true;
    if (enter) enter(n);

    // Arrange to call leave(n) when all done with descendants.
    if (leave) stack.push_back(Work{n, true});

    auto add_work = [&visited, &stack](Node* out) {
      if (!visited[out->id()]) {
        // Note; we must not mark as visited until we actually process it.
        stack.push_back(Work{out, false});
      }
    };

    if (stable_comparator) {
      std::vector<Node*> nodes_sorted;
      for (const Edge* out_edge : n->out_edges()) {
        if (!edge_filter || edge_filter(*out_edge)) {
          nodes_sorted.emplace_back(out_edge->dst());
        }
      }
      std::sort(nodes_sorted.begin(), nodes_sorted.end(), stable_comparator);
      for (Node* out : nodes_sorted) {
        add_work(out);
      }
    } else {
      for (const Edge* out_edge : n->out_edges()) {
        if (!edge_filter || edge_filter(*out_edge)) {
          add_work(out_edge->dst());
        }
      }
    }
  }
}
}  // namespace

void DFS(const Graph& g, const std::function<void(Node*)>& enter,
         const std::function<void(Node*)>& leave,
         const NodeComparator& stable_comparator,
         const EdgeFilter& edge_filter) {
  DFSFromHelper(g, {g.source_node()}, enter, leave, stable_comparator,
                edge_filter);
}

void DFSFrom(const Graph& g, gtl::ArraySlice<Node*> start,
             const std::function<void(Node*)>& enter,
             const std::function<void(Node*)>& leave,
             const NodeComparator& stable_comparator,
             const EdgeFilter& edge_filter) {
  DFSFromHelper(g, start, enter, leave, stable_comparator, edge_filter);
}

void DFSFrom(const Graph& g, gtl::ArraySlice<const Node*> start,
             const std::function<void(const Node*)>& enter,
             const std::function<void(const Node*)>& leave,
             const NodeComparator& stable_comparator,
             const EdgeFilter& edge_filter) {
  DFSFromHelper(g, start, enter, leave, stable_comparator, edge_filter);
}

void ReverseDFS(const Graph& g, const std::function<void(Node*)>& enter,
                const std::function<void(Node*)>& leave,
                const NodeComparator& stable_comparator) {
  ReverseDFSFrom(g, {g.sink_node()}, enter, leave, stable_comparator);
}

namespace {

template <typename T>
void ReverseDFSFromHelper(const Graph& g, gtl::ArraySlice<T> start,
                          const std::function<void(T)>& enter,
                          const std::function<void(T)>& leave,
                          const NodeComparator& stable_comparator) {
  // Stack of work to do.
  struct Work {
    T node;
    bool leave;  // Are we entering or leaving n?
  };
  std::vector<Work> stack(start.size());
  for (int i = 0; i < start.size(); ++i) {
    stack[i] = Work{start[i], false};
  }

  std::vector<bool> visited(g.num_node_ids(), false);
  while (!stack.empty()) {
    Work w = stack.back();
    stack.pop_back();

    T n = w.node;
    if (w.leave) {
      leave(n);
      continue;
    }

    if (visited[n->id()]) continue;
    visited[n->id()] = true;
    if (enter) enter(n);

    // Arrange to call leave(n) when all done with descendants.
    if (leave) stack.push_back(Work{n, true});

    auto add_work = [&visited, &stack](T out) {
      if (!visited[out->id()]) {
        // Note; we must not mark as visited until we actually process it.
        stack.push_back(Work{out, false});
      }
    };

    if (stable_comparator) {
      std::vector<T> nodes_sorted;
      for (const Edge* in_edge : n->in_edges()) {
        nodes_sorted.emplace_back(in_edge->src());
      }
      std::sort(nodes_sorted.begin(), nodes_sorted.end(), stable_comparator);
      for (T in : nodes_sorted) {
        add_work(in);
      }
    } else {
      for (const Edge* in_edge : n->in_edges()) {
        add_work(in_edge->src());
      }
    }
  }
}

}  // namespace

void ReverseDFSFrom(const Graph& g, gtl::ArraySlice<const Node*> start,
                    const std::function<void(const Node*)>& enter,
                    const std::function<void(const Node*)>& leave,
                    const NodeComparator& stable_comparator) {
  ReverseDFSFromHelper(g, start, enter, leave, stable_comparator);
}

void ReverseDFSFrom(const Graph& g, gtl::ArraySlice<Node*> start,
                    const std::function<void(Node*)>& enter,
                    const std::function<void(Node*)>& leave,
                    const NodeComparator& stable_comparator) {
  ReverseDFSFromHelper(g, start, enter, leave, stable_comparator);
}

void GetPostOrder(const Graph& g, std::vector<Node*>* order,
                  const NodeComparator& stable_comparator,
                  const EdgeFilter& edge_filter) {
  order->clear();
  DFS(g, nullptr, [order](Node* n) { order->push_back(n); }, stable_comparator,
      edge_filter);
}

/** \brief Get the order sequence from source node to sink node of a graph.
 *   Post order: after visiting all outgoing nodes, visit itself.
 *   Reverse post order: itself first, then others. So, it is right order.
 *
 *  \param g: const Graph&;
 *
 *  \param order: std::vector<Node*>*;
 *
 *  \param stable_comparator: const NodeComparator&;
 *
 *  \param edge_filter: EdgeFilter&;
 */
void GetReversePostOrder(const Graph& g, std::vector<Node*>* order,
                         const NodeComparator& stable_comparator,
                         const EdgeFilter& edge_filter) {
  GetPostOrder(g, order, stable_comparator, edge_filter);
  std::reverse(order->begin(), order->end());
}

bool PruneForReverseReachability(
  Graph* g, // input
  std::unordered_set<const Node*> visited) // input and output
{
  // Compute set of nodes that we need to traverse in order to reach
  // the nodes in "nodes" by performing a breadth-first search from those
  // nodes, and accumulating the visited nodes.
  std::deque<const Node*> queue;
  for (const Node* n : visited) {
    VLOG(2) << "Reverse reach init: " << n->name();
    // Reverse reach init: _retval_out_0_0
    queue.push_back(n);
  }

  while (!queue.empty()) {
    const Node* n = queue.front();
    queue.pop_front();
    for (const Node* in : n->in_nodes()) {
      if (visited.insert(in).second) {
        queue.push_back(in);
        VLOG(2) << "Reverse reach : " << n->name() << " from " << in->name();
        /*
        结构图:
        https://docs.google.com/document/d/1tUPyuaG6wlW76FGTPrEFXRMczFVXMZGL_DoH9xt6uQU/edit#heading=h.gx7euao7hus8

        Reverse reach : _retval_out_0_0 from out
        Reverse reach : out from c
        Reverse reach : out from z
        Reverse reach : c from a/RandomStandardNormal
        Reverse reach : c from b/RandomStandardNormal
        Reverse reach : z from x/RandomStandardNormal
        Reverse reach : z from y/RandomStandardNormal
        Reverse reach : a/RandomStandardNormal from a/shape
        Reverse reach : b/RandomStandardNormal from b/shape
        Reverse reach : x/RandomStandardNormal from x/shape
        Reverse reach : y/RandomStandardNormal from y/shape
        Reverse reach : a/shape from _SOURCE
        */
      }
    }
  }

  // Make a pass over the graph to remove nodes not in "visited"
  // - pass: 指遍历一遍图的术语
  // - 去掉不在 visited 集合里面的节点
  std::vector<Node*> all_nodes;
  all_nodes.reserve(g->num_nodes());
  for (Node* n : g->nodes()) {
    all_nodes.push_back(n);
  }

  bool any_removed = false;
  for (Node* n : all_nodes) {
    if (visited.count(n) == 0 && !n->IsSource() && !n->IsSink()) {
      g->RemoveNode(n);
      any_removed = true;
    }
  }

  return any_removed;
  // 我的例子 p any_removed, $17 = false
}


bool FixupSourceAndSinkEdges(Graph* g) {
  // Connect all nodes with no incoming edges to source.
  // Connect all nodes with no outgoing edges to sink.
  bool changed = false;
  for (Node* n : g->nodes()) {
    if (!n->IsSource() && n->in_edges().empty()) {
      g->AddControlEdge(g->source_node(), n,
                        true /* skip test for duplicates */);
      changed = true;
    }
    if (!n->IsSink() && n->out_edges().empty()) {
      g->AddControlEdge(n, g->sink_node(), true /* skip test for duplicates */);
      changed = true;
    }
  }
  return changed;
  // 1.
  // 我的例子里面:
  // p changed
  // $19 = false
}

}  // namespace tensorflow
