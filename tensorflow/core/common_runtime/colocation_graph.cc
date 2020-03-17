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

#include "tensorflow/core/common_runtime/colocation_graph.h"

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
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

// We hoist the conversion from C-style string literal to StringPiece here,
// so that we can avoid the many repeated calls to strlen().
const StringPiece kColocationAttrNameStringPiece(kColocationAttrName);
const StringPiece kColocationGroupPrefixStringPiece(kColocationGroupPrefix);


// Returns a list of devices having type in supported_device_types.  The
// returned list is sorted by preferred type (higher numeric type is preferred).
std::vector<Device*> FilterSupportedDevices(
    const std::vector<Device*>& devices, // input
    const PrioritizedDeviceTypeVector& supported_device_types, // input
    const Device* default_device) { // input
  // 1.
  // output 是 retval

  // 2.
  // 被调用是在 :
  // devices = FilterSupportedDevices(
  //   device_set_->devices(), members_[node_root].supported_device_types(),
  //   default_device_);

  // 2.1
  // 调用栈:
  // Thread #1 [python] 14797 [core: 58] (Suspended : Step)
  // 	tensorflow::(anonymous namespace)::FilterSupportedDevices at colocation_graph.cc:58 0x7fba26c52d0a
  // 	tensorflow::ColocationGraph::GetDevicesForNode() at colocation_graph.cc:813 0x7fba26c56214
  // 	tensorflow::Placer::Run() at placer.cc:144 0x7fba26cf145e
  // 	tensorflow::GraphExecutionState::InitBaseGraph() at graph_execution_state.cc:616 0x7fba3723eaea
  // 	tensorflow::GraphExecutionState::MakeForBaseGraph() at graph_execution_state.cc:97 0x7fba3723abc2
  // 	tensorflow::DirectSession::MaybeInitializeExecutionState() at direct_session.cc:1,694 0x7fba3308f659
  // 	tensorflow::DirectSession::ExtendLocked() at direct_session.cc:1,740 0x7fba3308fa41
  // 	tensorflow::DirectSession::Extend() at direct_session.cc:1,733 0x7fba3308f9fc
  // 	tensorflow::SessionRef::Extend() at session_ref.cc:441 0x7fba2eb73833
  // 	tensorflow::ExtendSessionGraphHelper() at c_api.cc:815 0x7fba331137c3
  // 	<...more frames...>

  // 3.
  // devices 输入参数描述:
  // p devices.size()
  // $10 = 6

  // 3.1
  // (gdb) p devices[0]->DebugString()
  // $11 = "name: \"/job:localhost/replica:0/task:0/device:CPU:0\"\ndevice_type: \"CPU\"\nmemory_limit: 268435456\nlocality {\n}\nincarnation: 18378914362316115292\n"
  // (gdb) p devices[1]->DebugString()
  // $12 = "name: \"/job:localhost/replica:0/task:0/device:XLA_CPU:0\"\ndevice_type: \"XLA_CPU\"\nmemory_limit: 17179869184\nlocality {\n}\nincarnation: 2985873009677690439\nphysical_device_desc: \"device: XLA_CPU device\"\n"
  // (gdb) p devices[2]->DebugString()
  // $13 = "name: \"/job:localhost/replica:0/task:0/device:XLA_GPU:0\"\ndevice_type: \"XLA_GPU\"\nmemory_limit: 17179869184\nlocality {\n}\nincarnation: 11431876017459358775\nphysical_device_desc: \"device: XLA_GPU device\"\n"
  // (gdb) p devices[3]->DebugString()
  // $14 = "name: \"/job:localhost/replica:0/task:0/device:XLA_GPU:1\"\ndevice_type: \"XLA_GPU\"\nmemory_limit: 17179869184\nlocality {\n}\nincarnation: 105147883587665474\nphysical_device_desc: \"device: XLA_GPU device\"\n"
  // (gdb) p devices[4]->DebugString()
  // $15 = "name: \"/job:localhost/replica:0/task:0/device:GPU:0\"\ndevice_type: \"GPU\"\nmemory_limit: 10244181198\nlocality {\n  bus_id: 1\n  links {\n  }\n}\nincarnation: 6695519563019833777\nphysical_device_desc: \"device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:03:00.0, compute capability: 7.5\"\n"
  // (gdb) p devices[5]->DebugString()
  // $16 = "name: \"/job:localhost/replica:0/task:0/device:GPU:1\"\ndevice_type: \"GPU\"\nmemory_limit: 10410806478\nlocality {\n  bus_id: 1\n  links {\n  }\n}\nincarnation: 5658237319182308682\nphysical_device_desc: \"device: 1, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1\"\n"
  // device:CPU:0; device:XLA_CPU:0; device:XLA_GPU:0; device:XLA_GPU:1; device:GPU:0; device:GPU:1

  // 4.
  // ptype supported_device_types
  // type = const class absl::InlinedVector<std::pair<tensorflow::DeviceType, int>, 4, std::allocator<std::pair<tensorflow::DeviceType, int> > > [with T = std::pair<tensorflow::DeviceType, int>, A = std::allocator<std::pair<tensorflow::DeviceType, int> >]
  // 是 absl::InlinedVector 类型

  // 4.1
  // 能用来 debug 的 absl::InlinedVector 成员函数:
  // size_type size(void) const;
  // reference operator[](size_type);

  // 4.2
  // (gdb) p supported_device_types.size()
  // $17 = 4

  // 4.3
  // (gdb) p supported_device_types[0]
  // $18 = {first = {type_ = "GPU"}, second = 0}
  // (gdb) p supported_device_types[1]
  // $19 = {first = {type_ = "CPU"}, second = 0}
  // (gdb) p supported_device_types[2]
  // $20 = {first = {type_ = "XLA_CPU"}, second = 0}
  // (gdb) p supported_device_types[3]
  // $21 = {first = {type_ = "XLA_GPU"}, second = 0}

  // 5.
  // ❎default_device 是什么?
  // !!! 千万不要打印! 输入的是空指针!!!
  // 原因在调用这个函数的地方(本文件内)已经写了, 这里不写了.

  Device* filtered_default_device = nullptr;

  std::vector<std::pair<Device*, int32>> prioritized_filtered_devices;

  for (const auto& supported_device_type : supported_device_types) {
    // 1.
    // 依次遍历: GPU, CPU, XLA_CPU, XLA_GPU

    for (Device* device : devices) {
      // 1.
      // 依次遍历:
      // device:CPU:0; device:XLA_CPU:0; device:XLA_GPU:0; device:XLA_GPU:1; device:GPU:0; device:GPU:1

      if (DeviceType(device->attributes().device_type()) ==
          supported_device_type.first) {
        // 1.
        // 如果説

        if (device == default_device) {
          // 1.
          // comment: 这个应该是不可能进入了, 因为 default_device == 0x0
          filtered_default_device = device;
        } else {
          // 进入

          prioritized_filtered_devices.emplace_back(
              device, supported_device_type.second);
          // 1.
          // 所以, 根据 遍历组合
          // 优先级依次从高到低是: GPU, CPU, XLA_CPU, XLA_GPU
          // 并把真实存在的 device 实体加入 prioritized_filtered_devices:
          //   device:CPU:0; device:XLA_CPU:0; device:XLA_GPU:0; device:XLA_GPU:1; device:GPU:0; device:GPU:1

        }
      }
    }
  }
  // 1.1
  // prioritized_filtered_devices 的最终结果是?
  // p prioritized_filtered_devices.size()
  // $3 = 6
  //

  auto device_sort = [](const std::pair<Device*, int32>& a,
                        const std::pair<Device*, int32>& b) {
    if (a.second != b.second) {
      return a.second > b.second;
    }

    auto a_priority =
        DeviceSet::DeviceTypeOrder(DeviceType(a.first->device_type()));
    auto b_priority =
        DeviceSet::DeviceTypeOrder(DeviceType(b.first->device_type()));
    // First sort by prioritized device type (higher is preferred) and
    // then by device name (lexicographically).
    if (a_priority != b_priority) {
      return a_priority > b_priority;
    }
    return StringPiece(a.first->name()) < StringPiece(b.first->name());
  };

  // 1.
  // prioritized_filtered_devices 变量说明:
  // p prioritized_filtered_devices
  // $4 = std::vector of length 6, capacity 8 = {{first = 0x55ff7b4e01a0, second = 0}, {first = 0x55ff7b4f2f20, second = 0}, {first = 0x55ff79676b10, second = 0}, {first = 0x55ff7b26ff40, second = 0}, {first = 0x55ff7b4bfe50, second = 0}, {first = 0x55ff7b4c9160, second = 0}}
  // 对于 2 块 GPU, 1 CPU

  std::sort(prioritized_filtered_devices.begin(),
            prioritized_filtered_devices.end(), device_sort);

  std::vector<Device*> filtered_devices;
  if (filtered_default_device != nullptr) {
    // 未进入
    filtered_devices.emplace_back(filtered_default_device);
  }

  for (const auto& prioritized_filtered_device : prioritized_filtered_devices) {
    filtered_devices.push_back(prioritized_filtered_device.first);
    // 1.
    // 打印部分结果, 第一个加入的是:
    // (gdb) p prioritized_filtered_device.first->DebugString()
    // $5 = "name: \"/job:localhost/replica:0/task:0/device:GPU:0\"\ndevice_type: \"GPU\"\nmemory_limit: 10244181198\nlocality {\n  bus_id: 1\n  links {\n  }\n}\nincarnation: 16511413941568493076\nphysical_device_desc: \"device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:03:00.0, compute capability: 7.5\"\n"

    // 2.
    // 第二个是: "name: \"/job:localhost/replica:0/task:0/device:GPU:1\"\ndevice_type: \"GPU\"\nmemory_limit: 10410806478\nlocality {\n  bus_id: 1\n  links {\n  }\n}\nincarnation: 16526567406605668175\nphysical_device_desc: \"device: 1, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1\"\n"

  }
  return filtered_devices;
  // 1.
  // 最后 filtered_devices 的结果
  //
  // (gdb) p filtered_devices.size()
  // $7 = 6
  // (gdb) p filtered_devices[0]->DebugString()
  // $9 = "name: \"/job:localhost/replica:0/task:0/device:GPU:0\"\ndevice_type: \"GPU\"\nmemory_limit: 10244181198\nlocality {\n  bus_id: 1\n  links {\n  }\n}\nincarnation: 16511413941568493076\nphysical_device_desc: \"device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:03:00.0, compute capability: 7.5\"\n"
  // (gdb) p filtered_devices[1]->DebugString()
  // $10 = "name: \"/job:localhost/replica:0/task:0/device:GPU:1\"\ndevice_type: \"GPU\"\nmemory_limit: 10410806478\nlocality {\n  bus_id: 1\n  links {\n  }\n}\nincarnation: 16526567406605668175\nphysical_device_desc: \"device: 1, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1\"\n"
  // (gdb) p filtered_devices[2]->DebugString()
  // $11 = "name: \"/job:localhost/replica:0/task:0/device:CPU:0\"\ndevice_type: \"CPU\"\nmemory_limit: 268435456\nlocality {\n}\nincarnation: 6029465270020419012\n"
  // (gdb) p filtered_devices[3]->DebugString()
  // $12 = "name: \"/job:localhost/replica:0/task:0/device:XLA_CPU:0\"\ndevice_type: \"XLA_CPU\"\nmemory_limit: 17179869184\nlocality {\n}\nincarnation: 17983607328380520671\nphysical_device_desc: \"device: XLA_CPU device\"\n"
  // (gdb) p filtered_devices[4]->DebugString()
  // $13 = "name: \"/job:localhost/replica:0/task:0/device:XLA_GPU:0\"\ndevice_type: \"XLA_GPU\"\nmemory_limit: 17179869184\nlocality {\n}\nincarnation: 11283929416580422794\nphysical_device_desc: \"device: XLA_GPU device\"\n"
  // (gdb) p filtered_devices[5]->DebugString()
  // $14 = "name: \"/job:localhost/replica:0/task:0/device:XLA_GPU:1\"\ndevice_type: \"XLA_GPU\"\nmemory_limit: 17179869184\nlocality {\n}\nincarnation: 9782071633836128780\nphysical_device_desc: \"device: XLA_GPU device\"\n"
}
// 1.
// 返回到哪? GetDevicesForNode
//
// Thread #1 [python] 15583 [core: 13] (Suspended : Step)
// 	tensorflow::ColocationGraph::GetDevicesForNode() at colocation_graph.cc:817 0x7efc0ac56236
// 	tensorflow::Placer::Run() at placer.cc:144 0x7efc0acf145e
// 	tensorflow::GraphExecutionState::InitBaseGraph() at graph_execution_state.cc:616 0x7efc1b23eaea
// 	tensorflow::GraphExecutionState::MakeForBaseGraph() at graph_execution_state.cc:97 0x7efc1b23abc2
// 	tensorflow::DirectSession::MaybeInitializeExecutionState() at direct_session.cc:1,694 0x7efc1708f659
// 	tensorflow::DirectSession::ExtendLocked() at direct_session.cc:1,740 0x7efc1708fa41
// 	tensorflow::DirectSession::Extend() at direct_session.cc:1,733 0x7efc1708f9fc
// 	tensorflow::SessionRef::Extend() at session_ref.cc:441 0x7efc12b73833
// 	tensorflow::ExtendSessionGraphHelper() at c_api.cc:815 0x7efc171137c3
// 	tensorflow::ExtendSession() at python_api.cc:118 0x7efc12bac2e6
// 	<...more frames...>


// Using absl::StrJoin with lambda does not work in tf-lite builds.
std::vector<string> DevicesToString(const std::vector<Device*> devices) {
  std::vector<string> v;
  v.reserve(devices.size());
  for (Device* d : devices) {
    v.push_back(d->name());
  }
  return v;
}

// Using absl::StrJoin with lambda does not work in tf-lite builds.
std::vector<string> DeviceTypeAndPriorityToString(
    const PrioritizedDeviceTypeVector& devices) {
  std::vector<string> v;
  v.reserve(devices.size());
  for (const std::pair<DeviceType, int32>& device_and_type : devices) {
    v.push_back(DeviceTypeString(device_and_type.first));
  }
  return v;
}

// While Placer can override requested device on ops processing
// resources, i.e. node that take (and potentially return) a resource,
// it must not override requested device on ops generating a resource,
// e.g. VarHandleOp, _Arg. Such ops are currently no-input, single resource/ref
// output nodes.
bool IsResourceGeneratorNode(const Node& node) {
  return node.num_inputs() == 0 && node.num_outputs() == 1 &&
         (IsRefType(node.output_type(0)) || node.output_type(0) == DT_RESOURCE);
}

bool IsExemptFromResourceInputColocation(const Node* node) {
  // Note: Partitioned function calls, which place and partition their
  // function bodies, are exempt from this check: they forward resource and
  // ref inputs to operations that are appropriately placed, instead of
  // dereferencing them.
  const string& op_type = node->op_def().name();
  return op_type == "PartitionedCall" || op_type == "StatefulPartitionedCall";
}

bool HasPriorities(const PrioritizedDeviceTypeVector& device_types) {
  for (const auto& prioritized_device_type : device_types) {
    if (prioritized_device_type.second != 0) return true;
  }
  return false;
}

bool ArePrioritiesSame(const PrioritizedDeviceTypeVector& a_types,
                       const PrioritizedDeviceTypeVector& b_types) {
  if (a_types.size() != b_types.size()) {
    return false;
  }
  for (int i = 0; i < a_types.size(); ++i) {
    if (a_types[i].first != b_types[i].first) {
      return false;
    }
  }
  return true;
}

}  // namespace

// 函数的目的:
// 设定 这个 node 对应的 Member::supported_device_types_
Status Member::SetParentAndSupportedDevices(
    const Node& node, // input
    const std::vector<DeviceType>& types) { // input

  int id = node.id();
  if (id < 0) {
    return errors::Internal("Placer should not be creating a Member for node: ",
                            node.DebugString());
  }
  parent_ = id;
  return SupportedDeviceTypesForNode(types, // input
                                     node.def(), // input
                                     &supported_device_types_); // output
  // SupportedDeviceTypesForNode 函数说明:
  // tensorflow/core/framework/op_kernel.cc
  //
  // Status SupportedDeviceTypesForNode(
  //     const std::vector<DeviceType>& prioritized_types,  // input
  //     const NodeDef& def, // input
  //     PrioritizedDeviceTypeVector* prioritized_device_types) // output
}

Status Member::SetAssignedDeviceName(const string& device_name) {
  if (DeviceNameUtils::HasSomeDetails(requested_device_name_)) {
    return errors::Internal(
        "Setting assigned device name when there is a requested device set "
        "is unsupported");
  }
  if (!DeviceNameUtils::ParseFullName(device_name, &assigned_device_name_)) {
    return errors::Internal("Malformed assigned device '", device_name, "'");
  }
  // Set requested device to assigned_device to maintain the invariant that
  // requested is a specialization of assigned.
  requested_device_name_ = assigned_device_name_;
  return Status::OK();
}

// 功能如其名, 设置 Member::requested_device_name_
Status Member::SetRequestedDeviceName(const Node& node) {

  if (!DeviceNameUtils::ParseFullName(node.requested_device(), // input
                                      &requested_device_name_)) // output
  {
    // ParseFullName 函数说明
    // bool DeviceNameUtils::ParseFullName(StringPiece fullname, // input
    //                                     ParsedName* p) // output
    // tensorflow/core/util/device_name_utils.cc

    // 2.
    // Member::requested_device_name_ 变量说明
    // requested_device_name_: DeviceNameUtils::ParsedName
    // p requested_device_name_
    // $18 = {has_job = false, job = "", has_replica = false, replica = 0, has_task = false, task = 0, has_type = true, type = "CPU", has_id = true, id = 0}


    return errors::InvalidArgument("Malformed device specification '",
                                   node.requested_device(),
                                   "' in node: ", node.DebugString());
  }

  if (DeviceNameUtils::HasSomeDetails(assigned_device_name_)) {
    return errors::Internal(
        "Setting requested device name when there is an assigned device set "
        "is unsupported");
  }

  return Status::OK();
}

Status Member::EnsureCompatibilityAcrossResourceEdge(
    const Node& src, const Member& src_root,
    const Node& dst, /*dst_root is this*/
    bool log_device_placement) {
  if (!DeviceNameUtils::AreCompatibleDevNames(src_root.assigned_device_name_,
                                              assigned_device_name_)) {
    return errors::InvalidArgument(
        "Cannot place the graph because a reference or resource edge "
        "connects colocation groups with incompatible assigned devices: ",
        DeviceNameUtils::ParsedNameToString(src_root.assigned_device_name_),
        " vs ", DeviceNameUtils::ParsedNameToString(assigned_device_name_),
        ". The edge src node is ", src.name(), " , and the dst node is ",
        dst.name());
  }

  if (DeviceNameUtils::AreCompatibleDevNames(src_root.requested_device_name_,
                                             requested_device_name_)) {
    return Status::OK();
  }

  // If we are here, assigned devices are compatible but requested ones are
  // not. We will be overriding the requested device for destination node, but
  // need to preserve the invariant that it will be a specialization of
  // the assigned device.
  if (log_device_placement) {
    LOG(INFO) << "Ignoring device specification "
              << DeviceNameUtils::ParsedNameToString(requested_device_name_)
              << " for node '" << dst.name()
              << "' because the input edge from '" << src.name()
              << "' is a reference connection and already has a device "
                 "field set to "
              << DeviceNameUtils::ParsedNameToString(
                     src_root.requested_device_name_);
  }
  requested_device_name_ = src_root.requested_device_name_;
  DeviceNameUtils::EnsureSpecification(&requested_device_name_,
                                       assigned_device_name_);
  return Status::OK();
}

void Member::Merge(std::vector<Member>* tree, int x_root, int y_root,
                   Member** new_root, Member** old_root, bool dry_run) {
  Member& x_root_member = (*tree)[x_root];
  Member& y_root_member = (*tree)[y_root];

  // Merge the sets by setting the parent pointer of the smaller tree's root
  // node to point to the root of the larger tree. Together with path
  // compression in ColocationGraph::FindRoot, this ensures that we do not
  // experience pathological performance on graphs such as chains.
  int new_root_id, old_root_id;
  if (x_root_member.rank_ < y_root_member.rank_) {
    // The tree rooted at x_root is shallower, so connect it to
    // y_root. The rank of y_root is unchanged because its new
    // child has strictly less rank.
    if (!dry_run) {
      x_root_member.parent_ = y_root;
    }
    new_root_id = y_root;
    old_root_id = x_root;
  } else if (x_root_member.rank_ > y_root_member.rank_) {
    // The tree rooted at y_root is shallower, so connect it to
    // x_root. The rank of x_root is unchanged because its new
    // child has strictly less rank.
    if (!dry_run) {
      y_root_member.parent_ = x_root;
    }
    new_root_id = x_root;
    old_root_id = y_root;
  } else {
    if (!dry_run) {
      // Both trees have the same rank, so break the tie by choosing
      // x_root as the new root.
      y_root_member.parent_ = x_root;
      // Increment the rank of the tree rooted at x_root, because it
      // is now strictly deeper than before.
      ++x_root_member.rank_;
    }
    new_root_id = x_root;
    old_root_id = y_root;
  }

  *new_root = &(*tree)[new_root_id];
  *old_root = &(*tree)[old_root_id];
}

// tree is non-const because we can change some `parent` pointers in some
// members for more efficient future lookups. The vector itself is not
// changed.
int Member::FindRoot(std::vector<Member>* tree, int node_id) {
  Member& member = (*tree)[node_id];
  if (member.parent_ == node_id) {
    // member.parent is the root of this disjoint tree.  Do nothing.
  } else {
    member.parent_ = FindRoot(tree, member.parent_);
  }
  // Now it is guaranteed that member.parent is the root of this disjoint
  // tree.
  return member.parent_;
}

Status Member::MergeDeviceNames(const Member& other,
                                bool allow_soft_placement) {
  // Assuming the "requested is a specialization of assigned" invariant holds
  // for this and `other`, it will hold after the two merges below.
  DeviceNameUtils::ParsedName assigned_device_name_copy = assigned_device_name_;
  TF_RETURN_IF_ERROR(DeviceNameUtils::MergeDevNames(
      &assigned_device_name_copy, other.assigned_device_name_));

  DeviceNameUtils::ParsedName requested_device_name_copy =
      requested_device_name_;
  TF_RETURN_IF_ERROR(DeviceNameUtils::MergeDevNames(
      &requested_device_name_copy, other.requested_device_name_,
      allow_soft_placement));

  // We checked for all errors, now change the devices.
  assigned_device_name_ = assigned_device_name_copy;
  requested_device_name_ = requested_device_name_copy;
  return Status::OK();
}

// Updates this to contain the intersection of the device types in
// this and "other".
bool Member::MergeSupportedDevices(const Member& other) {
  // Generate intersection with priorities.
  // Each vector contains the same device types but with different priorities.
  // The priorities are taken from the corresponding source vector.
  PrioritizedDeviceTypeVector target_intersection;
  PrioritizedDeviceTypeVector other_intersection;
  for (const auto& prioritized_device_type : supported_device_types_) {
    bool found = false;
    for (const auto& other_prioritized_device_type :
         other.supported_device_types_) {
      if (prioritized_device_type.first ==
          other_prioritized_device_type.first) {
        found = true;
        other_intersection.push_back(other_prioritized_device_type);
        break;
      }
    }
    if (found) {
      target_intersection.push_back(prioritized_device_type);
    }
  }

  // Sort the devices by priority order.
  auto device_sort = [](const std::pair<DeviceType, int32>& a,
                        const std::pair<DeviceType, int32>& b) {
    // First look at set priorities.
    if (a.second != b.second) {
      return a.second > b.second;
    }
    // Then fallback to default priorities.
    auto a_priority = DeviceSet::DeviceTypeOrder(a.first);
    auto b_priority = DeviceSet::DeviceTypeOrder(b.first);
    if (a_priority != b_priority) {
      return a_priority > b_priority;
    }
    // Finally just look at the Device type strings.
    return a.first.type_string() < b.first.type_string();
  };

  std::sort(target_intersection.begin(), target_intersection.end(),
            device_sort);
  std::sort(other_intersection.begin(), other_intersection.end(), device_sort);

  PrioritizedDeviceTypeVector result;

  bool is_target_prioritized = HasPriorities(target_intersection);
  bool is_other_prioritized = HasPriorities(other_intersection);
  if (!is_target_prioritized && !is_other_prioritized) {
    // If neither are prioritized then we just return the original i.e. target
    // prioritization.
    result = target_intersection;
  } else if (is_target_prioritized && !is_other_prioritized) {
    // If only one is prioritized, then we respect priorities of that in the
    // intersection.
    result = target_intersection;
  } else if (!is_target_prioritized && is_other_prioritized) {
    result = other_intersection;
  } else {
    // If both have priorities and agree then we go with that. If the
    // prioritization order is different, then we just fallback to the default
    // i.e. what the DeviceTypeOrder suggests. In that case, we also set the
    // merged priorities to 0, so that downstream merges work correctly as well.
    if (ArePrioritiesSame(target_intersection, other_intersection)) {
      result = target_intersection;
    } else {
      for (const auto& prioritized_device : target_intersection) {
        result.push_back(std::make_pair(prioritized_device.first, 0));
      }
      std::sort(result.begin(), result.end(), device_sort);
    }
  }

  if (result.empty()) {
    return false;
  }
  supported_device_types_ = result;
  return true;
}

Status Member::AssignDevice(const Node& node, bool allow_soft_placement) {
  if (node.assigned_device_name_index() == assigned_device_name_index_) {
    return Status::OK();
  }

  DeviceNameUtils::ParsedName parsed;
  DeviceNameUtils::ParseFullName(node.assigned_device_name(), &parsed);
  Status s = DeviceNameUtils::MergeDevNames(&assigned_device_name_, parsed,
                                            allow_soft_placement);
  if (!s.ok()) {
    return errors::Internal(
        "Constraining by assigned device should not cause an error. Original "
        "root's assigned device name: ",
        DeviceNameUtils::ParsedNameToString(assigned_device_name_),
        " node's assigned device name \"", node.assigned_device_name(),
        ". Error: ", s.error_message());
  }
  s = DeviceNameUtils::MergeDevNames(&requested_device_name_, parsed,
                                     allow_soft_placement);
  if (!s.ok()) {
    return errors::Internal(
        "Constraining by assigned device should not cause an error. Original "
        "root's requested device name: \"",
        DeviceNameUtils::ParsedNameToString(requested_device_name_),
        "\", node's assigned device name \"", node.assigned_device_name(),
        "\". Error: ", s.error_message());
  }

  assigned_device_name_index_ = node.assigned_device_name_index();
  // Clear cached possible_devices, if any.
  possible_devices_.clear();
  return Status::OK();
}
string Member::DebugString() {
  return absl::StrCat(
      "Member(assigned_device_name_index_=", assigned_device_name_index_,
      " requested_device_name_=",
      DeviceNameUtils::ParsedNameToString(requested_device_name_),
      " assigned_device_name_=",
      DeviceNameUtils::ParsedNameToString(assigned_device_name_),
      " supported_device_types_=[",
      absl::StrJoin(DeviceTypeAndPriorityToString(supported_device_types_),
                    ", "),
      "] possible_devices_=[",
      absl::StrJoin(DevicesToString(possible_devices_), ", "), "]");
}


ColocationGraph::ColocationGraph(const Graph* graph,
                                 const DeviceSet* device_set,
                                 const Device* default_device,
                                 bool allow_soft_placement,
                                 bool log_device_placement)
    : graph_(graph),
      device_set_(device_set),
      device_types_(device_set->PrioritizedDeviceTypeList()),
      default_device_(default_device),
      allow_soft_placement_(allow_soft_placement),
      log_device_placement_(log_device_placement) {
  members_.resize(graph->num_node_ids());
  // PrioritizedDeviceTypeList 函数说明:
  // tensorflow/core/common_runtime/device_set.cc:74:
  // std::vector<DeviceType> DeviceSet::PrioritizedDeviceTypeList()
  //

}

// Adds each node of the Graph to this ColocationGraph as a singleton.
//
// NOTE: The implementation assumes that the ids of nodes passed to
// this method are dense and zero-based; the memory used will be linear in
// the largest node ID.
// NOTE: If this method returns an error, *this is left in an undefined
// state.
Status ColocationGraph::ColocateAllNodes() {
  // This maps from a colocation group identifier to the 'root' of that
  // colocation group.  Note that the keys in this map are StringPiece; the
  // actual strings are stored under the NodeDef.  The lifetime of this map
  // is limited to this ColocateAllNodes() method, and no part of the
  // NodeDef trees are changed during the lifetime of this method, so using
  // StringPiece as a key is safe.
  //
  // Also, as a further optimization, we remove the "loc:@" prefix from
  // "class" attribute values, when they are used as keys in this table.
  // This allows us to use StringPiece values that refer to substrings of
  // 'string' values stored in NodeDef attribute lists, as well as StringPiece
  // values that refer to 'string' values from NodeDef::name(), without
  // performing any string allocations.
  std::unordered_map<StringPiece, const Node*, StringPieceHasher>
      colocation_group_root;

  for (const Node* node : graph_->op_nodes()) {
    // When adding the node, identify whether it is part of a colocation
    // group.

    // This code is effectively the equivalent of GetNodeAttr() for a string
    // array, but it avoids all internal allocations (the allocation of the
    // backing store of the std::vector<string> as well as the copies of the
    // strings within it).  Instead, we combine the query of the colocation
    // attribute with the calls to ColocateNodeToGroup.
    bool found_spec = false;
    const AttrValue* attr_value =
        node->attrs().Find(kColocationAttrNameStringPiece);
    if (attr_value != nullptr && attr_value->has_list()) {
      for (const string& class_spec : attr_value->list().s()) {
        StringPiece spec(class_spec);
        if (str_util::ConsumePrefix(&spec, kColocationGroupPrefixStringPiece)) {
          found_spec = true;
          TF_RETURN_IF_ERROR(
              ColocateNodeToGroup(&colocation_group_root, node, spec));
        }
      }
    }

    // TODO(iga): Even when the node has a spec, we need to colocate the
    // node to its "name group" because other nodes can still use
    // "loc:@<this_node_name>" in their colocation specs.
    if (!found_spec) {
      // If the node does not specify a colocation group, then use the
      // name of this node as the colocation group.
      TF_RETURN_IF_ERROR(
          ColocateNodeToGroup(&colocation_group_root, node, node->name()));
    }
  }

  return Status::OK();
}

Status ColocationGraph::ColocateResourceOrRefEdge(Node* src, Node* dst) {
  // Colocate `src` and `dst` to maintain the invariant that nodes
  // connected by reference edges are colocated.
  int src_root_id = FindRoot(src->id());
  int dst_root_id = FindRoot(dst->id());
  auto& src_root = members_[src_root_id];
  auto& dst_root = members_[dst_root_id];

  TF_RETURN_IF_ERROR(dst_root.EnsureCompatibilityAcrossResourceEdge(
      *src, src_root, *dst, log_device_placement_));
  Status status = ColocateNodes(*src, src_root_id, *dst, dst_root_id);
  if (!status.ok()) {
    return AttachDef(
        errors::InvalidArgument("Nodes were connected by a "
                                "reference connection (requiring them to "
                                "be on the same device), but the two nodes "
                                "were assigned two different devices: ",
                                status.error_message()),
        *dst);
  }
  return Status::OK();
}

Status ColocationGraph::ColocateResourceAndRefEdges() {
  // Enumerate the constraint edges, and use them to update the disjoint
  // node set.
  // If `node` has an input edge with reference type, add an edge from the
  // source of that edge to `node`.
  for (const Edge* edge : graph_->edges()) {
    if (edge->IsControlEdge()) {
      continue;
    }
    Node* src = edge->src();
    Node* dst = edge->dst();
    DataType input_type = dst->input_type(edge->dst_input());
    if ((input_type == DT_RESOURCE || IsRefType(input_type)) &&
        !IsExemptFromResourceInputColocation(dst)) {
      TF_RETURN_IF_ERROR(ColocateResourceOrRefEdge(src, dst));
    }
  }
  return Status::OK();
}

Status ColocationGraph::Initialize() {
  TF_RETURN_IF_ERROR(InitializeMembers());
  // InitializeMembers 函数说明
  //
  TF_RETURN_IF_ERROR(ColocateResourceAndRefEdges());
  TF_RETURN_IF_ERROR(ColocateAllNodes());
  return Status::OK();
}

Status ColocationGraph::ColocateNodeToGroup(
    std::unordered_map<StringPiece, const Node*, StringPieceHasher>*
        colocation_group_root,
    const Node* node, StringPiece colocation_group) {
  const Node*& root_node = (*colocation_group_root)[colocation_group];
  if (root_node == nullptr) {
    // This is the first node of the colocation group, so
    // designate this node as the 'root' of that colocation group.
    root_node = node;
  } else {
    // Try to colocate the node with the root.  If there is an
    // error, return it.
    Status s = ColocateNodes(*node, *root_node);
    if (!s.ok()) {
      if (!allow_soft_placement_) {
        return AttachDef(s, *node);
      }
      if (log_device_placement_) {
        LOG(INFO) << "Ignoring request to colocate node '" << node->name()
                  << "' with nodes in colocation group '" << colocation_group
                  << "' because soft placement is on and an attempt at doing "
                     "so resulted in the following error: "
                  << AttachDef(s, *node).ToString();
      }
    }
  }
  return Status::OK();
}

// Merge the (possibly disjoint) sets containing nodes "x" and
// "y". Returns OK if the all nodes in the union of these sets can
// be placed on the same device type.
//
// NOTE: If this method returns an error, *this is left in an undefined
// state.
Status ColocationGraph::ColocateNodes(const Node& x, const Node& y) {
  int x_root = FindRoot(x.id());
  int y_root = FindRoot(y.id());
  return ColocateNodes(x, x_root, y, y_root);
}

// This overload of ColocateNodes() allows a caller to provide the root node
// ids for the two nodes. For large graphs, this noticeably reduces the
// graph load time.
Status ColocationGraph::ColocateNodes(const Node& x, int x_root, const Node& y,
                                      int y_root) {
  if (x_root == y_root) {
    return Status::OK();
  }

  Member* new_root_member;
  Member* old_root_member;
  Member::Merge(&members_, x_root, y_root, &new_root_member, &old_root_member,
                /*dry_run=*/true);

  // Merge the partial device specifications, and ensure that they are
  // compatible. NULL options_ is treated as allowing soft placement.
  // If there is an error, nothing is modified.
  // TODO(mrry): Consider enriching the error message by pointing
  // out which nodes have the explicit partial device
  // specifications that caused this conflict.
  Status s = new_root_member->MergeDeviceNames(*old_root_member,
                                               allow_soft_placement_);
  if (!s.ok()) {
    return errors::InvalidArgument(
        "Cannot colocate nodes ",
        errors::FormatColocationNodeForError(x.name()), " and ",
        errors::FormatColocationNodeForError(y.name()), ": ",
        s.error_message());
  }

  // Ensure that the common root has at least one supported device
  // type, by computing the intersection of
  // new_root_member.supported_device_types and
  // old_root_member.supported_device_types.
  if (!new_root_member->MergeSupportedDevices(*old_root_member)) {
    return errors::InvalidArgument(
        "Cannot colocate nodes ",
        errors::FormatColocationNodeForError(x.name()), " and ",
        errors::FormatColocationNodeForError(y.name()),
        " because no device type supports both of those nodes and the "
        "other nodes colocated with them.",
        DebugInfo(x_root), DebugInfo(y_root));
  }

  // All error checks are done, merge the colocation graphs.
  Member::Merge(&members_, x_root, y_root, &new_root_member, &old_root_member,
                /*dry_run=*/false);
  return Status::OK();
}

// Limits the possible devices of `node`'s colocation group to the device
// to which `node` is assigned. This makes sure that all nodes in this
// colocation group will be assigned to the same device. Without this
// explicit restriction, heuristics can choose a different possible device
// for other nodes in the group.
Status ColocationGraph::LimitToAssignedDevice(const Node& node) {
  if (node.assigned_device_name_index() < 0) {
    return errors::Internal(
        "Expected an assigned node as argument to LimitToAssignedDevice but "
        "got: ",
        node.DebugString());
  }
  int root = FindRoot(node.id());
  Member& root_member = members_[root];
  return root_member.AssignDevice(node, allow_soft_placement_);
}


// For the given node, subject to the constraints previously given
// to this ColocationGraph, set its assigned_device_name. Returns OK
// if a satisfying device can be found, otherwise an error.
//
// Note: This method returns a pointer to a field within members_.
// The caller must not use the returned pointer after there is any possibility
// that the members_[i].possible_devices field has been modified.
Status ColocationGraph::GetDevicesForNode(
    Node* node,  // input
    const std::vector<Device*>** possible_devices) { // output
  // 1.
  // case study
  // p node->DebugString()
  // $7 = "{name:'MatMul' id:4 op device:{} def:{{{node MatMul}} = MatMul[T=DT_FLOAT, transpose_a=false, transpose_b=false](x, y)}}"

  // 2.
  // case study
  // (gdb) p node->DebugString()
  // $27 = "{name:'x' id:2 op device:{} def:{{{node x}} = Placeholder[dtype=DT_FLOAT, shape=[2,5]]()}}"

  // 3.
  // string ColocationGraph::DebugInfo(const int node_root)
  // (gdb) p this-> DebugInfo(2)
  //
  // Colocation Debug Info:
  // Colocation group had the following types and devices:
  // Placeholder: GPU CPU XLA_CPU XLA_GPU
  //
  // Colocation members and user-requested devices:
  //   x (Placeholder)

  *possible_devices = nullptr; // output

  const int node_root = FindRoot(node->id());
  // 1.
  // (gdb) p node_root
  // $8 = 4
  // (gdb) p node->id()
  // $9 = 4

  if (!members_[node_root].possible_devices().empty()) {
    // 未进入
    *possible_devices = &members_[node_root].possible_devices();
    return Status::OK();
  }

  // We have not yet computed the possible devices for the
  // colocated node set containing 'node', so we do so now using the
  // constraints on the root node.

  // "devices" will contain the set of feasible placements for the
  // colocated node set containing 'node'.
  std::vector<Device*> devices;

  if (DeviceNameUtils::HasSomeDetails(
          members_[node_root].requested_device_name())) {
    // 1.
    // 👀未进入

    // The root node has a (possibly partial) device
    // specification, so enumerate the physical devices that
    // conform to it.
    device_set_->FindMatchingDevices(
        members_[node_root].requested_device_name(), &devices);

    if (!devices.empty()) {
      // Filter devices into those that are compatible with the root
      // node (and its children).
      devices = FilterSupportedDevices(
          devices, members_[node_root].supported_device_types(),
          default_device_);
    }

    // Perform soft placement if allow_soft_placement_ is set.
    if (devices.empty() && allow_soft_placement_) {
      // The soft_device_name is the same as the node's device name
      // without specifying the device type or ID.
      DeviceNameUtils::ParsedName soft_device_name =
          members_[node_root].requested_device_name();
      soft_device_name.type.clear();
      soft_device_name.has_type = false;
      soft_device_name.has_id = false;
      device_set_->FindMatchingDevices(soft_device_name, &devices);
      if (!devices.empty()) {
        devices = FilterSupportedDevices(
            devices, members_[node_root].supported_device_types(),
            default_device_);
      }
    }

    if (devices.empty()) {
      // Return an error when a physical device that matches an explicit
      // device specification is not found. This ensures that we don't
      // assign a node to GPU when the user wanted to force it on CPU.
      string debug_info = DebugInfo(node_root);

      DeviceNameUtils::ParsedName specified_device_name;
      if (DeviceNameUtils::ParseFullName(node->requested_device(),
                                         &specified_device_name) &&
          specified_device_name ==
              members_[node_root].requested_device_name()) {
        // The specified device and merged set device match, and
        // will appear in the GraphDef (for debugging), so just
        // print the specified device.
        std::vector<Device*> devices_matching_nodedef;
        device_set_->FindMatchingDevices(specified_device_name,
                                         &devices_matching_nodedef);
        if (devices_matching_nodedef.empty()) {
          // Sometimes it is almost impossible to understand the problem
          // without a list of available devices.
          std::vector<string> device_names;
          for (const Device* device : device_set_->devices()) {
            device_names.push_back(device->name());
          }
          std::sort(device_names.begin(), device_names.end());

          string gpu_msg = "";
          if (!IsGoogleCudaEnabled() &&
              str_util::Lowercase(specified_device_name.type) == "gpu") {
            gpu_msg =
                " The requested device appears to be a GPU, but CUDA is not "
                "enabled.";
          }

          return errors::InvalidArgument(
              errors::FormatNodeNameForError(node->name()),
              "was explicitly assigned to ", node->requested_device(),
              " but available devices are [ ",
              str_util::Join(device_names, ", "), " ]. Make sure ",
              "the device specification refers to a valid device.", gpu_msg);
        } else if (specified_device_name.has_type) {
          return errors::InvalidArgument(
              "Could not satisfy explicit device specification '",
              node->requested_device(), "' because no supported kernel for ",
              specified_device_name.type, " devices is available.", debug_info,
              "\nRegistered kernels:\n",
              KernelsRegisteredForOp(node->type_string()));
        } else {
          return errors::InvalidArgument(
              "Could not satisfy explicit device specification '",
              node->requested_device(), debug_info);
        }
      } else {
        // The specified device may be a valid device but the
        // merged set device is different, so print both.
        return errors::InvalidArgument(
            "Could not satisfy explicit device specification '",
            node->requested_device(), "' because the node ",
            errors::FormatColocationNodeForError(node->name()),
            " was colocated with a group of nodes that ",
            "required incompatible device '",
            DeviceNameUtils::ParsedNameToString(
                members_[node_root].requested_device_name()),
            "'", debug_info);
      }
    }
  } else {
    // 进入这个分支:

    // The device is completely unspecified, so enumerate the devices that
    // support all of the nodes in the set.
    // 1.
    // 这个注释很清楚

    if (device_set_->devices().empty()) {
      return errors::Internal("No devices are registered");
    }

    // Returns a list of devices having type in supported_device_types.  The
    // returned list is sorted by preferred type (higher numeric type is preferred).
    devices = FilterSupportedDevices(
        device_set_->devices(), // input
        members_[node_root].supported_device_types(), // input
        default_device_); // input
    // 1.
    // ❎
    // 提醒: ColocationGraph::default_device_ 是空指针, 千万别打印

    // 1.1
    // 因为 👀, 所以起初就没有赋值啊!
    //
    // Thread #1 [python] 15583 [core: 19] (Suspended : Breakpoint)
    // 	tensorflow::(anonymous namespace)::FilterSupportedDevices at colocation_graph.cc:57 0x7efc0ac52cc9
    // 	tensorflow::ColocationGraph::GetDevicesForNode() at colocation_graph.cc:813 0x7efc0ac56214
    // 	tensorflow::Placer::Run() 👀at placer.cc:144 0x7efc0acf145e
    // 	tensorflow::GraphExecutionState::InitBaseGraph() at graph_execution_state.cc:616 0x7efc1b23eaea
    // 	tensorflow::GraphExecutionState::MakeForBaseGraph() at graph_execution_state.cc:97 0x7efc1b23abc2
    // 	tensorflow::DirectSession::MaybeInitializeExecutionState() at direct_session.cc:1,694 0x7efc1708f659
    // 	tensorflow::DirectSession::ExtendLocked() at direct_session.cc:1,740 0x7efc1708fa41
    // 	tensorflow::DirectSession::Extend() at direct_session.cc:1,733 0x7efc1708f9fc
    // 	tensorflow::SessionRef::Extend() at session_ref.cc:441 0x7efc12b73833
    // 	tensorflow::ExtendSessionGraphHelper() at c_api.cc:815 0x7efc171137c3
    // 	<...more frames...>

    // Placer placer(new_graph.get(), device_set_,  /* default_device= */ nullptr, 👀
    //                 session_options_ == nullptr ||
    //                     session_options_->config.allow_soft_placement(),
    //                 session_options_ != nullptr &&
    //                     session_options_->config.log_device_placement());
    //


    if (devices.empty()) {
      return errors::InvalidArgument(
          "Node had no OpKernel registered to support this operation: ",
          "Operation was ", node->type_string(), " and inputs were ",
          DataTypeVectorString(node->input_types()), DebugInfo(node_root));
    }
  }

  // Cache the result of the possible devices for this node group.
  members_[node_root].set_possible_devices(std::move(devices));
  // 1.
  // (gdb) p node_root
  // $16 = 4
  // 这里是以 matmul op 为 study case 的.

  *possible_devices = &members_[node_root].possible_devices();
  // 1.
  // possible_devices 是这个函数的 Output

  return Status::OK();
}
// 1.
// 函数返回到哪里?
// Placer::Run()
//
// Thread #1 [python] 15583 [core: 13] (Suspended : Step)
// 	tensorflow::ColocationGraph::GetDevicesForNode() at colocation_graph.cc:817 0x7efc0ac56236
// 	tensorflow::Placer::Run() at placer.cc:144 0x7efc0acf145e
// 	tensorflow::GraphExecutionState::InitBaseGraph() at graph_execution_state.cc:616 0x7efc1b23eaea
// 	tensorflow::GraphExecutionState::MakeForBaseGraph() at graph_execution_state.cc:97 0x7efc1b23abc2
// 	tensorflow::DirectSession::MaybeInitializeExecutionState() at direct_session.cc:1,694 0x7efc1708f659
// 	tensorflow::DirectSession::ExtendLocked() at direct_session.cc:1,740 0x7efc1708fa41
// 	tensorflow::DirectSession::Extend() at direct_session.cc:1,733 0x7efc1708f9fc
// 	tensorflow::SessionRef::Extend() at session_ref.cc:441 0x7efc12b73833
// 	tensorflow::ExtendSessionGraphHelper() at c_api.cc:815 0x7efc171137c3
// 	tensorflow::ExtendSession() at python_api.cc:118 0x7efc12bac2e6
// 	<...more frames...>


Status ColocationGraph::InitializeMembers() {
  for (Node* node : graph_->op_nodes()) {

    Status status = InitializeMember(
                      *node,  // input
                      &members_[node->id()]); // input
    // 1.
    // members_ 变量说明
    // std::vector<Member> members_;

    // 2.
    // Member 数据结构
    // - parent_: int , default: -1
    // - rank_: int, default: 0
    // - assigned_device_name_index_: int , default: -1
    // - requested_device_name_: DeviceNameUtils::ParsedName
    // - assigned_device_name_: DeviceNameUtils::ParsedName
    // - supported_device_types_: PrioritizedDeviceTypeVector
    // - possible_devices_: std::vector<Device*>

    // 3.
    // &members_[node->id()] 变量说明
    // &members_[node->id()]  <==> Member* member

    // 3.
    // InitializeMember 函数说明：
    // ColocationGraph::InitializeMember(
    //                          const Node& node,  // input
    //                          Member* member) {  // output

    if (!status.ok()) {
      return AttachDef(status, *node);
    }
  }
  return Status::OK();
}

string ColocationGraph::DebugString() {
  std::unordered_set<int> roots;
  std::vector<string> root_strings;
  for (const Node* node : graph_->nodes()) {
    if (!node->IsOp()) {
      continue;
    }
    int node_root = FindRoot(node->id());
    if (roots.count(node_root) == 0) {
      root_strings.push_back(DebugInfo(node_root));
      roots.insert(node_root);
    }
  }
  return absl::StrJoin(root_strings, "\n");
}

// Returns debugging info for the node referred to by 'node_root'.
string ColocationGraph::DebugInfo(const int node_root) {
  string text(
      "\nColocation Debug Info:\n"
      "Colocation group had the following types and devices: ");

  // If this node is part of a colocation group, then we want to
  // collect the mapping of ops to supported devices, so that
  // the user can see why an unsatisfiable placement occurred.

  std::unordered_map<string, string> type_to_devices;
  std::vector<const Node*> colocation_nodes;
  int num_nodes_found = 0;

  for (const Node* node : graph_->nodes()) {
    if (!node->IsOp()) {
      continue;
    }
    int id = node->id();
    if (FindRoot(id) != node_root) {
      continue;
    }
    ++num_nodes_found;
    colocation_nodes.push_back(node);
    const string& op_type = node->type_string();
    string devices_registered;
    for (const auto& device_type : members_[id].supported_device_types()) {
      strings::StrAppend(&devices_registered,
                         DeviceTypeString(device_type.first), " ");
    }

    type_to_devices[op_type] = std::move(devices_registered);
  }

  for (const auto& td : type_to_devices) {
    strings::StrAppend(&text, "\n", td.first, ": ", td.second);
  }
  strings::StrAppend(&text,
                     "\n\nColocation members and user-requested devices:");
  for (const Node* node : colocation_nodes) {
    strings::StrAppend(&text, "\n  ", node->name(), " (", node->type_string(),
                       ") ", node->requested_device());
  }
  strings::StrAppend(&text, "\n");

  if (num_nodes_found <= 0) {
    text.clear();
  }
  return text;
}

Status ColocationGraph::InitializeMemberWithAssignedDevice(
    const string& assigned_device_name, const string& node_type,
    bool must_be_full_name, Member* member) {
  // This node has already been assigned to a device, so we
  // respect this placement, after sanity-checking it.
  // NOTE: Since any assignment must have been performed by
  // the TensorFlow runtime, we consider errors in this branch to
  // be INTERNAL.
  TF_RETURN_IF_ERROR(member->SetAssignedDeviceName(assigned_device_name));
  if (!must_be_full_name) {
    return Status::OK();
  }
  // Since assigned device must be a full specification, do extra checks.
  const Device* assigned_device =
      device_set_->FindDeviceByName(assigned_device_name);
  if (assigned_device == nullptr) {
    return errors::Internal("Assigned device '", assigned_device_name,
                            "' does not match any device");
  }

  for (const auto& d : member->supported_device_types()) {
    if (DeviceType(assigned_device->attributes().device_type()) == d.first) {
      return Status::OK();
    }
  }

  return errors::Internal("Assigned device '", assigned_device_name,
                          "' does not have registered OpKernel support "
                          "for ",
                          node_type);
}

Status ColocationGraph::InitializeMember(
                          const Node& node,  // input
                          Member* member) {  // output
  // Member 数据结构
  // tensorflow/core/common_runtime/colocation_graph.h
  // 综述: 设定 这个 node 对应的 Member::supported_device_types_
  TF_RETURN_IF_ERROR(member->SetParentAndSupportedDevices(node, device_types_));

  if (node.has_assigned_device_name()) {
    // 未进入这个分支
    TF_RETURN_IF_ERROR(InitializeMemberWithAssignedDevice(
        node.assigned_device_name(), node.type_string(), true, member));
  } else {
    // This node has not yet been assigned to a device, so we
    // calculate any constraints due to the set of registered
    // kernels and any (partial) user-provided device specification
    // in the NodeDef.

    // If no kernels are registered for this op type, fail with an error.
    if (member->supported_device_types().empty()) {
      // 未进入这个分支
      std::set<string> registered_device_types;
      for (Device* d : device_set_->devices()) {
        registered_device_types.insert(d->device_type());
      }
      std::vector<string> attr_key_vals;
      for (const auto& it : node.attrs()) {
        const string& name = it.first;
        const AttrValue& attr_value = it.second;
        attr_key_vals.push_back(
            strings::StrCat(name, "=", SummarizeAttrValue(attr_value)));
      }
      return errors::InvalidArgument(
          "No OpKernel was registered to support Op '", node.type_string(),
          "' used by ", errors::FormatNodeNameForError(node.name()),
          "with these attrs: [", str_util::Join(attr_key_vals, ", "),
          "]\n"
          "Registered devices: [",
          str_util::Join(registered_device_types, ", "), "]\n",
          "Registered kernels:\n", KernelsRegisteredForOp(node.type_string()));
    }

    // If the NodeDef contains a device, then we interpret it as a
    // (partial) device specification.
    if (!node.requested_device().empty()) {
      if (IsResourceGeneratorNode(node)) {
        // Treat requested device on resource generating nodes as assigned
        // device so that we don't override it.
        TF_RETURN_IF_ERROR(InitializeMemberWithAssignedDevice(
            node.requested_device(), node.type_string(), false, member));
      } else {
        // The user has specified a device in the NodeDef, try to find a
        // valid device matching their specification in the set of
        // devices.
        // NOTE: The full name may specify a device that is not in
        // n.supported_device_types(), but we check that in AssignDevice().
        TF_RETURN_IF_ERROR(member->SetRequestedDeviceName(node));
        // SetRequestedDeviceName 函数说明:
        // tensorflow/core/common_runtime/colocation_graph.cc
        // 赋值 Member::requested_device_name_
      }
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
