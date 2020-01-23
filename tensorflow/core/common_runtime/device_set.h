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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_SET_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_SET_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

// DeviceSet is a container class for managing the various types of
// devices used by a model.
class DeviceSet {
 public:
  DeviceSet();
  ~DeviceSet();

  // Does not take ownership of 'device'.
  void AddDevice(Device* device);

  // Set the device designated as the "client".  This device
  // must also be registered via AddDevice().
  void set_client_device(Device* device) {
    DCHECK(client_device_ == nullptr);
    client_device_ = device;
  }

  // Returns a pointer to the device designated as the "client".
  Device* client_device() const { return client_device_; }

  // Return the list of devices in this set.
  const std::vector<Device*>& devices() const { return devices_; }

  // Given a DeviceNameUtils::ParsedName (which may have some
  // wildcards for different components), fills "*devices" with all
  // devices in "*this" that match "spec".
  void FindMatchingDevices(const DeviceNameUtils::ParsedName& spec,
                           std::vector<Device*>* devices) const;

  // Finds the device with the given "fullname". Returns nullptr if
  // not found.
  Device* FindDeviceByName(const string& fullname) const;

  // Return the list of unique device types in this set, ordered
  // with more preferable devices earlier.
  std::vector<DeviceType> PrioritizedDeviceTypeList() const;
  // 1.
  // PrioritizedDeviceTypeList() 打印
  // p device_set_.PrioritizedDeviceTypeList()
  // $16 = std::vector of length 4, capacity 4 = {{type_ = "GPU"}, {type_ = "CPU"}, {type_ = "XLA_CPU"}, {type_ = "XLA_GPU"}}


  // An order to sort by device types according to system-determined
  // priority.
  //
  // Higher result implies higher priority.
  static int DeviceTypeOrder(const DeviceType& d);

 private:
  // Not owned.
  std::vector<Device*> devices_;

  // Fullname -> device* for device in devices_.
  std::unordered_map<string, Device*> device_by_name_;

  // client_device_ points to an element of devices_ that we consider
  // to be the client device (in this local process).
  Device* client_device_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceSet);
};
// 1.
// class DeviceSet 数据结构
// 概述:
// DeviceSet is a container class for managing the various types of
// devices used by a model.
//
// tensorflow/core/common_runtime/device_set.h
// - devices_: td::vector<Device*> , not owned
// - device_by_name_: std::unordered_map<string, Device*>
//   Fullname -> device* for device in devices_.
// - client_device_: Device*
//   client_device 指的是 host CPU , 下面打印部分印证。
//   The device designated as the "client".
//   client_device_ points to an element of devices_ that we consider
//   to be the client device (in this local process).
//
// 重要接口:
// PrioritizedDeviceTypeList

// 2.
// 打印
// 2.1
// 只有 CPU , 没有 GPU 的情况:
/*
$2 = {
  devices_ = std::vector of length 2,
  capacity 2 = {
    0x5573b1bf6300,
    0x5573b37f6570
  },
  device_by_name_ = std::unordered_map with 4 elements = {
    ["/job:localhost/replica:0/task:0/xla_cpu:0"] = 0x5573b37f6570,
    ["/job:localhost/replica:0/task:0/device:XLA_CPU:0"] = 0x5573b37f6570,
    ["/job:localhost/replica:0/task:0/device:CPU:0"] = 0x5573b1bf6300,
    ["/job:localhost/replica:0/task:0/cpu:0"] = 0x5573b1bf6300
  },
  client_device_ = 0x5573b1bf6300
}
*/

// 2.2
// 4 个 GPU 都可见
/*
$2 = {
  devices_ = std::vector of length 10,
  capacity 16 = {
    0x5600134775d0,
    0x560015081440,
    0x5600152d7ca0,
    0x5600152e07c0,
    0x5600152e8a90,
    0x5600152f1a80,
    0x5600153086d0,
    0x560015319fc0,
    0x5600153277f0,
    0x5600153350f0
  },
  device_by_name_ = std::unordered_map with 20 elements = {
    ["/job:localhost/replica:0/task:0/gpu:3"] = 0x5600153350f0,
    ["/job:localhost/replica:0/task:0/device:GPU:3"] = 0x5600153350f0,
    ["/job:localhost/replica:0/task:0/device:GPU:2"] = 0x5600153277f0,
    ["/job:localhost/replica:0/task:0/device:XLA_GPU:0"] = 0x5600152d7ca0,
    ["/job:localhost/replica:0/task:0/xla_cpu:0"] = 0x560015081440,
    ["/job:localhost/replica:0/task:0/device:XLA_GPU:3"] = 0x5600152f1a80,
    ["/job:localhost/replica:0/task:0/device:XLA_CPU:0"] = 0x560015081440,
    ["/job:localhost/replica:0/task:0/device:CPU:0"] = 0x5600134775d0,
    ["/job:localhost/replica:0/task:0/cpu:0"] = 0x5600134775d0,
    ["/job:localhost/replica:0/task:0/gpu:2"] = 0x5600153277f0,
    ["/job:localhost/replica:0/task:0/xla_gpu:3"] = 0x5600152f1a80,
    ["/job:localhost/replica:0/task:0/xla_gpu:0"] = 0x5600152d7ca0,
    ["/job:localhost/replica:0/task:0/xla_gpu:1"] = 0x5600152e07c0,
    ["/job:localhost/replica:0/task:0/device:XLA_GPU:1"] = 0x5600152e07c0,
    ["/job:localhost/replica:0/task:0/gpu:1"] = 0x560015319fc0,
    ["/job:localhost/replica:0/task:0/gpu:0"] = 0x5600153086d0,
    ["/job:localhost/replica:0/task:0/device:XLA_GPU:2"] = 0x5600152e8a90,
    ["/job:localhost/replica:0/task:0/xla_gpu:2"] = 0x5600152e8a90,
    ["/job:localhost/replica:0/task:0/device:GPU:1"] = 0x560015319fc0,
    ["/job:localhost/replica:0/task:0/device:GPU:0"] = 0x5600153086d0
  },
  client_device_ = 0x5600134775d0
}
*/

// TODO 提示:
// Device* client_device() const { return client_device_; }
// 我改写的只需要 client_device 就行了。
/*
p device_set_.client_device()
$3 = (tensorflow::GPUCompatibleCPUDevice *) 0x5600134775d0

p device_set_.client_device()->name()
$4 = "/job:localhost/replica:0/task:0/device:CPU:0"
*/

// 3.
// Device 数据结构
// class Device : public DeviceBase
// tensorflow/core/common_runtime/device.h
// 重要接口:
// - Compute
// - ComputeAsync
// - FillContextMap
// - resource_manager
// 成员变量:
// - device_mgr_: DeviceMgr*
// - device_attributes_: const DeviceAttributes
// - parsed_name_: DeviceNameUtils::ParsedName
// - op_seg_: OpSegment
// - rmgr_: ResourceMgr*
// friend class:
// friend class DeviceMgr;

// 3.1
// message DeviceAttributes
// tensorflow/core/framework/device_attributes.proto
// - name: string
// - device_type: string
// - memory_limit: int64
// - locality: DeviceLocality
// - incarnation: fixed64
// - physical_device_desc: string

// 4.
// class DeviceBase 数据结构
// tensorflow/core/framework/device_base.h
// - struct CpuWorkerThreads
//    - num_threads: int , default : 0
//    - workers: thread::ThreadPool*, default : nullptr
// - struct GpuDeviceInfo
//    - stream: stream_executor::Stream*
//    - default_context: DeviceContext*
//    - event_mgr: EventMgr*
//    - gpu_id: int, default: -1
// - env_: Env* const
// - cpu_worker_threads_: CpuWorkerThreads* , default : nullptr
// - gpu_device_info_: GpuDeviceInfo* , default : nullptr
// - device_thread_pool_: thread::ThreadPool* , default : nullptr
// - eigen_cpu_devices_: std::vector<Eigen::ThreadPoolDevice*>
// - eigen_sycl_device_: Eigen::SyclDevice*, default : nullptr

// 5.
// class DeviceContext: public core::RefCounted
// tensorflow/core/framework/device_base.h
// 概述:
// 接口定义
//
// 接口:
// - stream()
// - MaintainLifetimeOnStream
// - CopyCPUTensorToDevice
// - CopyTensorInSameDevice
// - CopyDeviceTensorToCPU
// - ThenExecute

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_SET_H_
