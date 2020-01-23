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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_OPTIMIZER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_OPTIMIZER_H_

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class GraphOptimizer {
 public:
  using NodePredicate = std::function<bool(const Node*)>;

  struct Options {
    // If not null it maps from nodes in graph to partially-known
    // shapes of their outputs, and may be used, e.g., in the constant folding
    // pass. The use of shape_map implies that the mapping from node name to the
    // vector of partial shapes of its outputs is stable, i.e., no optimization
    // pass may replace a node with a different node of the same name that has a
    // different number of outputs, or outputs with different known shapes.
    // TODO(b/65453533) introduce a unique way to name nodes in a graph.
    std::unordered_map<string, std::vector<PartialTensorShape>>* shape_map =
        nullptr;

    // If not null then only nodes for which cse_consider_fn returns true will
    // be considered for CSE.
    NodePredicate cse_consider_fn = nullptr;

    // If not null then only nodes for which cf_consider_fn returns true will be
    // considered for CF.
    NodePredicate cf_consider_fn = nullptr;
  };

  GraphOptimizer(const OptimizerOptions& opts);
  // OptimizerOptions 数据结构
  /*
  // Options passed to the graph optimizer
  message OptimizerOptions {
    // If true, optimize the graph using common subexpression elimination.
    bool do_common_subexpression_elimination = 1;

    // If true, perform constant folding optimization on the graph.
    bool do_constant_folding = 2;

    // Constant folding optimization replaces tensors whose values can be
    // predetermined, with constant nodes. To avoid inserting too large constants,
    // the size of each constant created can be limited. If this value is zero, a
    // default limit of 10 MiB will be applied. If constant folding optimization
    // is disabled, this value is ignored.
    int64 max_folded_constant_in_bytes = 6;

    // If true, perform function inlining on the graph.
    bool do_function_inlining = 4;

    // Optimization level
    enum Level {
      // L1 is the default level.
      // Optimization performed at L1 :
      // 1. Common subexpression elimination
      // 2. Constant folding
      L1 = 0;

      // No optimizations
      L0 = -1;
    }

    // Overall optimization level. The actual optimizations applied will be the
    // logical OR of the flags that this level implies and any flags already set.
    Level opt_level = 3;

    // Control the use of the compiler/jit.  Experimental.
    enum GlobalJitLevel {
      DEFAULT = 0;  // Default setting ("off" now, but later expected to be "on")
      OFF = -1;
      // The following settings turn on compilation, with higher values being
      // more aggressive.  Higher values may reduce opportunities for parallelism
      // and may use more memory.  (At present, there is no distinction, but this
      // is expected to change.)
      ON_1 = 1;
      ON_2 = 2;
    }
    GlobalJitLevel global_jit_level = 5;
  }
  */

  ~GraphOptimizer();

  // Applies optimization passes specified in 'opts' to 'graph'.
  // Maybe replace *graph with a new graph object.  'device' is device
  // on which the 'graph' will execute. It's passed to the optimizers
  // so that they can respect constraints if any, that should be
  // respected.
  void Optimize(FunctionLibraryRuntime* runtime, Env* env, Device* device,
                std::unique_ptr<Graph>* graph,
                const Options& graph_optimizer_options);
  // DEPRECATED: Consider passing a GraphOptimizer::Options object instead.
  void Optimize(
      FunctionLibraryRuntime* runtime, Env* env, Device* device,
      std::unique_ptr<Graph>* graph,
      const std::unordered_map<string, std::vector<PartialTensorShape>>*
          shape_map,
      const NodePredicate& cse_consider_fn = nullptr,
      const NodePredicate& cf_consider_fn = nullptr);

  const OptimizerOptions& options() { return opts_; }

 private:
  OptimizerOptions opts_;

  TF_DISALLOW_COPY_AND_ASSIGN(GraphOptimizer);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_OPTIMIZER_H_
