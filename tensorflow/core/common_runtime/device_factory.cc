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

#include "tensorflow/core/common_runtime/device_factory.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

static mutex* get_device_factory_lock() {
  static mutex device_factory_lock(LINKER_INITIALIZED);
  return &device_factory_lock;
}

struct FactoryItem {
  std::unique_ptr<DeviceFactory> factory;
  int priority;
};
// 1.
// 告诉我哪里是 DeviceFactory ?
// tensorflow/core/common_runtime/device_factory.h

// 2.
// DeviceFactory 继承关系
//
// DeviceFactory
// - BaseGPUDeviceFactory
//  - GPUDeviceFactory
// - GPUCompatibleCPUDeviceFactory
// - SYCLDeviceFactory
// - ThreadPoolDeviceFactory
// - XlaCpuDeviceFactory
// - XlaGpuDeviceFactory
// - XlaInterpreterDeviceFactory
// - DummyFactory

std::unordered_map<string, FactoryItem>& device_factories() {
  static std::unordered_map<string, FactoryItem>* factories =
      new std::unordered_map<string, FactoryItem>;
  return *factories;
  // 1.
  // debug print:
  // p *factories
  // $4 = std::unordered_map with 4 elements =
  // {["XLA_CPU"] = {factory = std::unique_ptr<tensorflow::DeviceFactory> = {get() = 0x5602a967e220}, priority = 50},
  //  ["GPU"] = {factory = std::unique_ptr<tensorflow::DeviceFactory> = {get() = 0x5602a9071170}, priority = 210},
  //  ["XLA_GPU"] = {factory = std::unique_ptr<tensorflow::DeviceFactory> = {get() = 0x5602a968dd20}, priority = 50},
  //  ["CPU"] = {factory = std::unique_ptr<tensorflow::DeviceFactory> = {get() = 0x5602a9071200}, priority = 70}}

  // 2.
  // CPU
  // XLA_CPU
  // GPU
  // XLA_GPU

  // 3.
  // 给 2 个 GPU 也是一样的, 如上, 同!
}

}  // namespace

// static
int32 DeviceFactory::DevicePriority(const string& device_type) {
  mutex_lock l(*get_device_factory_lock());
  std::unordered_map<string, FactoryItem>& factories = device_factories();

  auto iter = factories.find(device_type);
  if (iter != factories.end()) {
    return iter->second.priority;
  }

  return -1;
}

// static
void DeviceFactory::Register(const string& device_type, DeviceFactory* factory,
                             int priority) {
  // 1.
  // 具体的话呢
  // 请看最下面的 宏定义, 那里是使用这个函数地方. 我也查出了调用的实例

  mutex_lock l(*get_device_factory_lock());
  std::unique_ptr<DeviceFactory> factory_ptr(factory);
  std::unordered_map<string, FactoryItem>& factories = device_factories();
  auto iter = factories.find(device_type);
  if (iter == factories.end()) {
    factories[device_type] = {std::move(factory_ptr), priority};
  } else {
    if (iter->second.priority < priority) {
      iter->second = {std::move(factory_ptr), priority};
    } else if (iter->second.priority == priority) {
      LOG(FATAL) << "Duplicate registration of device factory for type "
                 << device_type << " with the same priority " << priority;
    }
  }
}

DeviceFactory* DeviceFactory::GetFactory(const string& device_type) {
  mutex_lock l(*get_device_factory_lock());  // could use reader lock
  auto it = device_factories().find(device_type);
  if (it == device_factories().end()) {
    return nullptr;
  }
  return it->second.factory.get();
}

Status DeviceFactory::AddDevices(
    const SessionOptions& options, // input
    const string& name_prefix, // input
    std::vector<std::unique_ptr<Device>>* devices) { // output
  // 1.
  // 调用方:
  // tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc
  // GrpcServer::Init

  // 2.
  // name_prefix 变量说明
  // 在 DirectSessionFactory::NewSession
  // tensorflow/core/common_runtime/direct_session.cc
  // e.g., "/job:localhost/replica:0/task:0"
  //
  // e.g., "/job:worker/replica:0/task:1"

  // CPU first. A CPU device is required.
  auto cpu_factory = GetFactory("CPU");
  if (!cpu_factory) {
    return errors::NotFound(
        "CPU Factory not registered.  Did you link in threadpool_device?");
  }

  size_t init_size = devices->size();
  // 1.
  // init_size
  // init_size == 0

  TF_RETURN_IF_ERROR(cpu_factory->CreateDevices(options, name_prefix, devices));

  if (devices->size() == init_size) {
    return errors::NotFound("No CPU devices are available in this process");
  }

  // Then the rest (including GPU).
  mutex_lock l(*get_device_factory_lock());

  for (auto& p : device_factories()) {
    // 1.
    // device_factories 函数说明
    //

    // 2.
    // 可是他是怎么知道还有其他的 gpu device 的呢?
    // ...

    auto factory = p.second.factory.get();
    if (factory != cpu_factory) {
      TF_RETURN_IF_ERROR(factory->CreateDevices(options, name_prefix, devices));
      // 1.
      // CPU
      // XLA_CPU
      // GPU
      // XLA_GPU

      // 2.
      // tensorflow/core/common_runtime/gpu/gpu_device.cc
      // Status BaseGPUDeviceFactory::CreateDevices(

      // 3.
      // tensorflow/compiler/jit/xla_gpu_device.cc
      // XlaGpuDeviceFactory::CreateDevices

      // tensorflow/compiler/jit/xla_cpu_device.cc
      // XlaCpuDeviceFactory::CreateDevices

      // tensorflow/core/common_runtime/gpu/gpu_device.cc
      // BaseGPUDeviceFactory::CreateDevices

      // tensorflow/core/common_runtime/gpu/gpu_device_factory.cc
      // GPUCompatibleCPUDeviceFactory::CreateDevices

      // 4.
      // Device 继承关系

      // GPUCompatibleCPUDevice - ThreadPoolDevice - LocalDevice - Device
      // CPU

      // - XlaCpuDeviceFactory::CreateDevices <= XLA_CPU
      //  - XlaDevice
      //   - LocalDevice
      //    - Device

      // - CreateGPUDevice
      //  - GPUDevice
      //   - BaseGPUDevice, BaseGPUDeviceFactory, gpu_device.cc
      //    - LocalDevice
      //      - Device

      // 5.
      // p name_prefix
      // $9 = "/job:worker/replica:0/task:1"

    }
  }

  return Status::OK();
}

std::unique_ptr<Device> DeviceFactory::NewDevice(const string& type,
                                                 const SessionOptions& options,
                                                 const string& name_prefix) {
  auto device_factory = GetFactory(type);
  if (!device_factory) {
    return nullptr;
  }
  SessionOptions opt = options;
  (*opt.config.mutable_device_count())[type] = 1;
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(device_factory->CreateDevices(opt, name_prefix, &devices));
  int expected_num_devices = 1;
  auto iter = options.config.device_count().find(type);
  if (iter != options.config.device_count().end()) {
    expected_num_devices = iter->second;
  }
  DCHECK_EQ(devices.size(), static_cast<size_t>(expected_num_devices));
  return std::move(devices[0]);
}

}  // namespace tensorflow
