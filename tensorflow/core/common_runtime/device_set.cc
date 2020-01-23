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

#include "tensorflow/core/common_runtime/device_set.h"

#include <set>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

DeviceSet::DeviceSet() {}

DeviceSet::~DeviceSet() {}

void DeviceSet::AddDevice(Device* device) {
  devices_.push_back(device);
  for (const string& name :
       DeviceNameUtils::GetNamesForDeviceMappings(device->parsed_name())) {
    device_by_name_.insert({name, device});
  }
}
// 1.
// low_priority_device_set_ 的 devices_ 插入顺序, 所以提取 CPU 就是 最后一个
//
// +-----------------------------------------------------------------------------------+
// |                       computation_capability_device_map                           |
// |-----------------------------------------------------------------------------------|
// |                                                                                   |
// | +--------------------------+     +-----------+-----------+-----------+-----------+|
// | |computation capability|7.5|---->| Device*   | Device*   | Device*   | Device*   ||
// | +--------------------------+     +-----------+-----------+-----------+-----------+|
// |                                                                                   |
// | +--------------------------+     +-----------+-----------+-----------+-----------+|
// | |computation capability|6.1|---->| Device*   | Device*   | Device*   | Device*   ||
// | +--------------------------+     +-----------+-----------+-----------+-----------+|
// |                                                                                   |
// |      ...                           ...                                            |
// |                                                                                   |
// | +--------------------------+     +-----------+-----------+-----------+-----------+|
// | |computation capability|1.0|---->| Device*   | Device*   | Device*   | Device*   ||
// | +--------------------------+     +-----------+-----------+-----------+-----------+|
// +-----------------------------------------------------------------------------------+


void DeviceSet::FindMatchingDevices(const DeviceNameUtils::ParsedName& spec,
                                    std::vector<Device*>* devices) const {
  // TODO(jeff): If we are going to repeatedly lookup the set of devices
  // for the same spec, maybe we should have a cache of some sort
  devices->clear();
  for (Device* d : devices_) {
    if (DeviceNameUtils::IsCompleteSpecification(spec, d->parsed_name())) {
      devices->push_back(d);
    }
  }
}

Device* DeviceSet::FindDeviceByName(const string& name) const {
  return gtl::FindPtrOrNull(device_by_name_, name);
}

// static
int DeviceSet::DeviceTypeOrder(const DeviceType& d) {
  return DeviceFactory::DevicePriority(d.type_string());
  // DeviceFactory::DevicePriority 函数说明
  // tensorflow/core/common_runtime/device_factory.cc:54:
  //int32 DeviceFactory::DevicePriority(const string& device_type)

  // Search device priority
  // tensorflow/core/common_runtime/device_factory.cc-66-
  // void DeviceFactory::Register(const string& device_type, DeviceFactory* factory,
  // 在这里设置断点就会得到 priority 的结果

}

static bool DeviceTypeComparator(const DeviceType& a, const DeviceType& b) {
  // First sort by prioritized device type (higher is preferred) and
  // then by device name (lexicographically).
  auto a_priority = DeviceSet::DeviceTypeOrder(a);
  auto b_priority = DeviceSet::DeviceTypeOrder(b);
  if (a_priority != b_priority) {
    return a_priority > b_priority;
  }

  return StringPiece(a.type()) < StringPiece(b.type());
}

std::vector<DeviceType> DeviceSet::PrioritizedDeviceTypeList() const {
  std::vector<DeviceType> result;
  std::set<string> seen;
  for (Device* d : devices_) {
    const auto& t = d->device_type();
    if (seen.insert(t).second) {
      result.emplace_back(t);
    }
  }
  std::sort(result.begin(), result.end(), DeviceTypeComparator);
  return result;
}
// 1.
// class DeviceType 数据结构说明:
// tensorflow/core/framework/types.h
//
// 概述:
// A DeviceType is just a string, but we wrap it up in a class to give
// some type checking as we're passing these around
//
// - type_ : string

// 2.
// PrioritizedDeviceTypeList() 打印
// p device_set_.PrioritizedDeviceTypeList()
// $16 = std::vector of length 4, capacity 4 = {{type_ = "GPU"}, {type_ = "CPU"}, {type_ = "XLA_CPU"}, {type_ = "XLA_GPU"}}


}  // namespace tensorflow
