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

#include "tensorflow/core/common_runtime/placer.h"

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/colocation_graph.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

namespace {

// Returns true if the node has no inputs and produces outputs
// that are consumed by a single node.
//
// TODO(vrv): Currently this handles only nodes with one output, but
// this could be extended to handle the case where a node has many
// outputs that are connected to nodes in the same colocation group.
bool IsGeneratorNode(const Node* node) {
  return node->num_inputs() == 0 && node->num_outputs() == 1 &&
         !IsRefType(node->output_type(0));
}

void LogDeviceAssignment(const Node* node, bool log_device_placement) {
  // Log placement if log_device_placement is set.
  if (log_device_placement) {
    printf("%s: (%s): %s\n", node->name().c_str(), node->type_string().c_str(),
           node->assigned_device_name().c_str());
    LOG(INFO) << node->name() << ": "
              << "(" << node->type_string() << ")"
              << node->assigned_device_name();
  }
}
// 1.
// 打印:
// code and log:
// https://gist.github.com/shizukanaskytree/3bc7c81cc9490f4a3abae3c7fb26a107

Status AssignAndLog(int assigned_device,  // input
                    Node* node, // input
                    ColocationGraph* colocation_graph, // input
                    bool log_device_placement) { // input

  node->set_assigned_device_name_index(assigned_device);
  // assigned_device
  // 对于 CPU, assigned_device 是 1

  // Constraint the group of node to the assigned device.
  TF_RETURN_IF_ERROR(colocation_graph->LimitToAssignedDevice(*node));
  // ColocationGraph::LimitToAssignedDevice 函数说明:


  LogDeviceAssignment(node, log_device_placement);
  return Status::OK();
}

}  // namespace

Placer::Placer(Graph* graph, const DeviceSet* devices,
               const Device* default_device, bool allow_soft_placement,
               bool log_device_placement)
    : graph_(graph),
      devices_(devices),
      default_device_(default_device),
      allow_soft_placement_(allow_soft_placement),
      log_device_placement_(log_device_placement) {}

Placer::Placer(Graph* graph, const DeviceSet* devices,
               const Device* default_device)
    : Placer(graph, devices, default_device, true, false) {}

Placer::Placer(Graph* graph, const DeviceSet* devices)
    : Placer(graph, devices, nullptr, true, false) {}

Placer::~Placer() {}

// 1.
// 提醒：
// 牢牢记住，虽然这里的函数复杂，但是最后的目的仅仅就是初始化 Graph 里面每个 Node 的 device placement 而已。
Status Placer::Run() {
  if (devices_->devices().empty()) {
    return errors::FailedPrecondition("No devices are registered");
  }

  if (VLOG_IS_ON(3)) {
    DumpGraphToFile("placer_input", *graph_, nullptr);
    for (const Node* node : graph_->op_nodes()) {
      VLOG(3) << "    " << node->name() << ": requested: '"
              << node->requested_device() << "' assigned: '"
              << node->assigned_device_name() << "'";
    }
  }
  // 1.
  // Node::requested_device 函数说明:
  // The device requested by the user.  For the actual assigned device,
  // use assigned_device_name() below.

  // 2.
  // Node::assigned_device_name 函数说明:
  // This gives the device the runtime has assigned this node to.  If
  // you want the device the user requested, use def().device() instead.

  // 3.
  // 打印

  // 3.1 代码
  /*
  import os
  os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
  import tensorflow as tf

  with tf.device('/device:GPU:0'):
    a_gpu = tf.random_normal([2,3], name='a_gpu')
    b_gpu = tf.random_normal([3,2], name='b_gpu')
    c_gpu = tf.matmul(a_gpu, b_gpu, name='c_gpu')

  with tf.device('/cpu:0'):
    a_cpu = tf.random_normal([2,3], name='a_cpu')
    b_cpu = tf.random_normal([3,2], name='b_cpu')
    c_cpu = tf.matmul(a_cpu, b_cpu, name='c_cpu')

  with tf.Session(
      config=tf.ConfigProto(
        log_device_placement=True,
        inter_op_parallelism_threads=1)) as sess:
    # iteration 1
    o = sess.run(c_gpu)
    # iteration 2
    #sess.run(a_cpu.assign(a_gpu))

  print('All Done!')

  // 3.2

  logging:

  2019-09-30 00:35:39.009058: I tensorflow/core/util/dump_graph.cc:106] Dumped GraphDef to /tmp/placer_input.pbtxt
  a_gpu/shape: requested: '/device:GPU:0' assigned: ''
  a_gpu/mean: requested: '/device:GPU:0' assigned: ''
  a_gpu/stddev: requested: '/device:GPU:0' assigned: ''
  a_gpu/RandomStandardNormal: requested: '/device:GPU:0' assigned: ''
  a_gpu/mul: requested: '/device:GPU:0' assigned: ''
  a_gpu: requested: '/device:GPU:0' assigned: ''
  b_gpu/shape: requested: '/device:GPU:0' assigned: ''
  b_gpu/mean: requested: '/device:GPU:0' assigned: ''
  b_gpu/stddev: requested: '/device:GPU:0' assigned: ''
  b_gpu/RandomStandardNormal: requested: '/device:GPU:0' assigned: ''
  b_gpu/mul: requested: '/device:GPU:0' assigned: ''
  b_gpu: requested: '/device:GPU:0' assigned: ''
  c_gpu: requested: '/device:GPU:0' assigned: ''
  a_cpu/shape: requested: '/device:CPU:0' assigned: ''
  a_cpu/mean: requested: '/device:CPU:0' assigned: ''
  a_cpu/stddev: requested: '/device:CPU:0' assigned: ''
  a_cpu/RandomStandardNormal: requested: '/device:CPU:0' assigned: ''
  a_cpu/mul: requested: '/device:CPU:0' assigned: ''
  a_cpu: requested: '/device:CPU:0' assigned: ''
  b_cpu/shape: requested: '/device:CPU:0' assigned: ''
  b_cpu/mean: requested: '/device:CPU:0' assigned: ''
  b_cpu/stddev: requested: '/device:CPU:0' assigned: ''
  b_cpu/RandomStandardNormal: requested: '/device:CPU:0' assigned: ''
  b_cpu/mul: requested: '/device:CPU:0' assigned: ''
  b_cpu: requested: '/device:CPU:0' assigned: ''
  c_cpu: requested: '/device:CPU:0' assigned: ''
  */

  // 4.
  // 打印: 在不设置的情况下
  //
  // 2019-10-23 06:20:11.191876: I tensorflow/core/util/dump_graph.cc:106] Dumped GraphDef to /tmp/placer_input.pbtxt
  // 2019-10-23 06:20:14.986792: I tensorflow/core/common_runtime/placer.cc:107]     a/shape: requested: '' assigned: ''
  // 2019-10-23 06:20:14.986877: I tensorflow/core/common_runtime/placer.cc:107]     a/mean: requested: '' assigned: ''
  // 2019-10-23 06:20:14.986894: I tensorflow/core/common_runtime/placer.cc:107]     a/stddev: requested: '' assigned: ''
  // 2019-10-23 06:20:14.986908: I tensorflow/core/common_runtime/placer.cc:107]     a/RandomStandardNormal: requested: '' assigned: ''
  // 2019-10-23 06:20:14.986922: I tensorflow/core/common_runtime/placer.cc:107]     a/mul: requested: '' assigned: ''
  // 2019-10-23 06:20:14.986936: I tensorflow/core/common_runtime/placer.cc:107]     a: requested: '' assigned: ''
  // 2019-10-23 06:20:14.986949: I tensorflow/core/common_runtime/placer.cc:107]     b/shape: requested: '' assigned: ''
  // 2019-10-23 06:20:14.986963: I tensorflow/core/common_runtime/placer.cc:107]     b/mean: requested: '' assigned: ''
  // 2019-10-23 06:20:14.986976: I tensorflow/core/common_runtime/placer.cc:107]     b/stddev: requested: '' assigned: ''
  // 2019-10-23 06:20:14.986989: I tensorflow/core/common_runtime/placer.cc:107]     b/RandomStandardNormal: requested: '' assigned: ''
  // 2019-10-23 06:20:14.987002: I tensorflow/core/common_runtime/placer.cc:107]     b/mul: requested: '' assigned: ''
  // 2019-10-23 06:20:14.987015: I tensorflow/core/common_runtime/placer.cc:107]     b: requested: '' assigned: ''
  // 2019-10-23 06:20:14.987029: I tensorflow/core/common_runtime/placer.cc:107]     matmul: requested: '' assigned: ''

  ColocationGraph colocation_graph(
                    graph_,
                    devices_,
                    default_device_,
                    allow_soft_placement_,
                    log_device_placement_);
  // 1.
  // class ColocationGraph 构造函数说明:
  // tensorflow/core/common_runtime/colocation_graph.h:168
  //
  // ColocationGraph(const Graph* graph,
  //                 const DeviceSet* device_set,
  //                 const Device* default_device,
  //                 bool allow_soft_placement,
  //                 bool log_device_placement);

  // 2.
  // devices_ 变量说明:
  // 打印
  // p *devices_
  // $7 = {devices_ = std::vector of length 1, capacity 1 = {0x55939ea4c770}, device_by_name_ = std::unordered_map with 2 elements = {["/job:localhost/replica:0/task:0/cpu:0"] = 0x55939ea4c770, ["/job:localhost/replica:0/task:0/device:CPU:0"] = 0x55939ea4c770}, client_device_ = 0x0}



  TF_RETURN_IF_ERROR(colocation_graph.Initialize());
  // 1.
  // ColocationGraph::Initialize 函数说明
  //
  // Status ColocationGraph::Initialize() {
  //   TF_RETURN_IF_ERROR(InitializeMembers());
  //   TF_RETURN_IF_ERROR(ColocateResourceAndRefEdges());
  //   TF_RETURN_IF_ERROR(ColocateAllNodes());
  //   return Status::OK();
  // }


  // For each node, assign a device based on the constraints in the disjoint
  // node set.
  std::vector<Node*> second_pass;
  for (Node* node : graph_->op_nodes()) {
    // The graph may have come pre-populated by the framework with assigned
    // devices (e.g., for stateful placements), so the placer should not try to
    // place nodes that are already placed.
    if (node->has_assigned_device_name()) {
      // 通常是没有的，这个分支不进入

      TF_RETURN_IF_ERROR(colocation_graph.LimitToAssignedDevice(*node));
      LogDeviceAssignment(node, log_device_placement_);
      continue;
    }

    // Heuristic A: prefer to place "generators" with their only
    // consumers.
    //
    // If this is a node with no inputs and one output, we save
    // this for a second pass, so that the consumer's placement
    // is chosen.
    if (IsGeneratorNode(node)) {
      // Returns true if the node has no inputs and produces outputs
      // that are consumed by a single node.
      second_pass.push_back(node);
      // 把只有输出没有输入的生产节点放入到第二次 pass 中去 place.
      continue;
    }

    const std::vector<Device*>* devices; // 临时变量

    // -----------------------------------------------------------------------
    Status status = colocation_graph.GetDevicesForNode(
                                      node,  // input
                                      &devices); // output
    // -----------------------------------------------------------------------
    // 1.
    // GetDevicesForNode 函数说明
    // tensorflow/core/common_runtime/colocation_graph.cc
    // Status ColocationGraph::GetDevicesForNode(
    //     Node* node,  // input
    //     const std::vector<Device*>** possible_devices) { // output
    //
    // 打印 GetDevicesForNode 内得到的结果:
    // 背景信息: p node->DebugString()
    // $30 = "{name:'a/mul' id:6 op device:{} def:{{{node a/mul}} = Mul[T=DT_FLOAT, _device=\"/device:CPU:0\"](a/RandomStandardNormal, a/stddev)}}"
    // 正态分布中的那个乘法，这里还没有对计算图进行简化。
    //
    // p *((**possible_devices)[0])
    // $29 = (tensorflow::GPUCompatibleCPUDevice)
    // 所以下面赋值的

    // 2.
    // 2.1
    // Bug Report:
    //
    // $9 = "{name:'b/RandomStandardNormal'
    // id:11 op device:{}
    // def:{{{node b/RandomStandardNormal}} =
    // RandomStandardNormal[T=DT_INT32, dtype=DT_FLOAT, seed=0, seed2=0,
    //   _device=\"/device:GPU:0\"](b/shape)}}"
    //
    // 2.2
    // Bug 的理由:
    // 如果 python 那边写了 with tf.device('/device:GPU:0'): 但是系统又没有 GPU 的话，
    // 那么就无法成功，不是我的问题。
    //
    // 2.3
    // 错误 message :
    // tensorflow.python.framework.errors_impl.InvalidArgumentError:
    // Cannot assign a device for operation random_normal/RandomStandardNormal:
    // node random_normal/RandomStandardNormal
    // (defined at /tf2/experiment/test_center/cpu_tf/test_cpu.py:16)
    // was explicitly assigned to /device:GPU:0 but available devices are
    // [ /job:localhost/replica:0/task:0/device:CPU:0, /job:localhost/replica:0/task:0/device:XLA_CPU:0 ].
    // Make sure the device specification refers to a valid device.
    //
    // 2.4
    // 设计:
    // 2.4.1. 重新写一个 placer 给 低优先级的用
    // 2.4.2 把 low_priority_executor_state_ 对应的 graph_def 的 _device 这项给强行改成 "" 空的。
    //       这样就能避免修改函数了

    if (!status.ok()) {
      return AttachDef(
          errors::InvalidArgument("Cannot assign a device for operation ",
                                  node->name(), ": ", status.error_message()),
          *node);
    }

    // Returns the first device in sorted devices list so we will always
    // choose the same device.
    //
    // TODO(vrv): Factor this assignment out into a pluggable
    // algorithm, so that Placer is responsible for enforcing
    // preconditions and we can experiment with other algorithms when
    // given a choice of devices. Once we have a better idea of the
    // types of heuristics we want to use and the information needed
    // to perform good placement we can add an interface for this.
    int assigned_device = -1;

    // Heuristic B: If the node only operates on metadata, not data,
    // then it is desirable to place that metadata node with its
    // input.
    if (IsMetadata(node)) {
      // Make sure that the input device type is in the list of supported
      // device types for this node.
      const Node* input = (*node->in_edges().begin())->src();
      // TODO(vrv): if the input is empty, consider postponing this
      // node's assignment to the second pass, so that we handle the
      // case where a metadata node's input comes from a backedge
      // of a loop.
      if (CanAssignToDevice(input->assigned_device_name(), *devices)) {
        assigned_device = input->assigned_device_name_index();
      }
    }

    // Provide the default, if necessary.
    if (assigned_device == -1) {
      // 正式赋值 assigned_device
      // -----------------------------------------------------------------------
      assigned_device = graph_->InternDeviceName(
                                  (*devices)[0]->name()); // input
      // -----------------------------------------------------------------------
      // 1.
      // (*devices)[0]->name() 变量说明
      // 把 possible_devices 的第一个赋值给 assigned_device, 注意，类型为 int
      // devices (possible_devices) 是上面 GetDevicesForNode 所得到的。
      // 打印：
      // $35 = "/job:localhost/replica:0/task:0/device:CPU:0"

      // 2.
      // devices 变量说明:
      // const std::vector<Device*>* devices; // 一个临时变量

      // 2.1
      // p (*devices)[0]->name()
      // $7 = "/job:localhost/replica:0/task:0/device:GPU:0"

      // 2.2
      // p (*devices).size()
      // $11 = 6

      // 2.3
      // p (*devices)
      // $8 = std::vector of length 6, capacity 6 = {0x55ff7b4e01a0, 0x55ff7b4f2f20, 0x55ff79676b10, 0x55ff7b26ff40, 0x55ff7b4bfe50, 0x55ff7b4c9160}

      // 2.4
      // p (*devices)[0]->DebugString() # GeForce RTX 2080 Ti
      // $9 = "name: \"/job:localhost/replica:0/task:0/device:GPU:0\"\ndevice_type: \"GPU\"\nmemory_limit: 10244594074\nlocality {\n  bus_id: 1\n  links {\n  }\n}\nincarnation: 7771573375485167518\nphysical_device_desc: \"device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:03:00.0, compute capability: 7.5\"\n"
      // p (*devices)[1]->DebugString() # GeForce GTX 1080 Ti
      // $10 = "name: \"/job:localhost/replica:0/task:0/device:GPU:1\"\ndevice_type: \"GPU\"\nmemory_limit: 10411278336\nlocality {\n  bus_id: 1\n  links {\n  }\n}\nincarnation: 5155280296900771696\nphysical_device_desc: \"device: 1, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1\"\n"
      // p (*devices)[2]->DebugString()
      // $12 = "name: \"/job:localhost/replica:0/task:0/device:CPU:0\"\ndevice_type: \"CPU\"\nmemory_limit: 268435456\nlocality {\n}\nincarnation: 5645062867885555578\n"
      // p (*devices)[3]->DebugString()
      // $13 = "name: \"/job:localhost/replica:0/task:0/device:XLA_CPU:0\"\ndevice_type: \"XLA_CPU\"\nmemory_limit: 17179869184\nlocality {\n}\nincarnation: 9656328976674275589\nphysical_device_desc: \"device: XLA_CPU device\"\n"
      // p (*devices)[4]->DebugString()
      // $14 = "name: \"/job:localhost/replica:0/task:0/device:XLA_GPU:0\"\ndevice_type: \"XLA_GPU\"\nmemory_limit: 17179869184\nlocality {\n}\nincarnation: 16835346760448377919\nphysical_device_desc: \"device: XLA_GPU device\"\n"
      // p (*devices)[5]->DebugString()
      // $15 = "name: \"/job:localhost/replica:0/task:0/device:XLA_GPU:1\"\ndevice_type: \"XLA_GPU\"\nmemory_limit: 17179869184\nlocality {\n}\nincarnation: 6685459979475867489\nphysical_device_desc: \"device: XLA_GPU device\"\n"

      // 3.
      // Graph::InternDeviceName 函数说明:
      // graph/graph.cc
      // 注意: 返回值是一个 int

    }

    TF_RETURN_IF_ERROR(AssignAndLog(
                         assigned_device, // input, type: int
                         node, // input
                         &colocation_graph, // input
                         log_device_placement_)); // input
    // AssignAndLog 函数说明:
    //
    // Status AssignAndLog(int assigned_device,
    //                     Node* node,
    //                     ColocationGraph* colocation_graph,
    //                     bool log_device_placement)
    //
    // tensorflow/core/common_runtime/placer.cc
  }

  // Perform a second pass assignment for those nodes explicitly
  // skipped during the first pass.
  for (Node* node : second_pass) {
    const std::vector<Device*>* devices;
    Status status = colocation_graph.GetDevicesForNode(node, &devices);
    if (!status.ok()) {
      return AttachDef(
          errors::InvalidArgument("Cannot assign a device for operation ",
                                  node->name(), ": ", status.error_message()),
          *node);
    }

    int assigned_device = -1;

    // Heuristic A application.
    if (IsGeneratorNode(node) && !node->out_edges().empty()) {
      const Node* output = (*node->out_edges().begin())->dst();
      int output_device_name = output->assigned_device_name_index();

      const bool consumers_on_same_device = std::all_of(
          node->out_edges().begin(), node->out_edges().end(),
          [output_device_name](const Edge* e) {
            return e->dst()->assigned_device_name_index() == output_device_name;
          });

      if (consumers_on_same_device &&
          CanAssignToDevice(output->assigned_device_name(), *devices)) {
        assigned_device = output_device_name;
      }
    }

    // Provide the default, if necessary.
    if (assigned_device == -1) {
      assigned_device = graph_->InternDeviceName((*devices)[0]->name());
    }

    TF_RETURN_IF_ERROR(AssignAndLog(assigned_device, node, &colocation_graph,
                                    log_device_placement_));
  }

  if (VLOG_IS_ON(3)) {
    DumpGraphToFile("placer_output", *graph_, nullptr);
  }
  return Status::OK();
}

bool Placer::CanAssignToDevice(const string& candidate_device_name,
                               const std::vector<Device*>& devices) const {
  if (!candidate_device_name.empty()) {
    // 'devices' lists the set of devices that the placer or the user has
    // constrained the operation to.  "candidate_device_name" must
    // refer to a concrete Device that is in the list of 'devices'.
    const Device* other_device =
        devices_->FindDeviceByName(candidate_device_name);
    if (std::find(devices.begin(), devices.end(), other_device) !=
        devices.end()) {
      return true;
    }
  }

  return false;
}

}  // namespace tensorflow
