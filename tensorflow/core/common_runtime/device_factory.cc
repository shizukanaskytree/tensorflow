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

std::unordered_map<string, FactoryItem>& device_factories() {
  static std::unordered_map<string, FactoryItem>* factories =
      new std::unordered_map<string, FactoryItem>;
  return *factories;
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
    const SessionOptions& options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  // CPU first. A CPU device is required.
  auto cpu_factory = GetFactory("CPU");
  if (!cpu_factory) {
    return errors::NotFound(
        "CPU Factory not registered.  Did you link in threadpool_device?");
  }
  size_t init_size = devices->size();
  TF_RETURN_IF_ERROR(cpu_factory->CreateDevices(options, name_prefix, devices));
  if (devices->size() == init_size) {
    return errors::NotFound("No CPU devices are available in this process");
  }

  // Then the rest (including GPU).
  mutex_lock l(*get_device_factory_lock());
  for (auto& p : device_factories()) {
    auto factory = p.second.factory.get();
    if (factory != cpu_factory) {
      TF_RETURN_IF_ERROR(factory->CreateDevices(options, name_prefix, devices));
    }
  }

  return Status::OK();
}

Status DeviceFactory::AddSelectedDevices(
    const SessionOptions& options, const string& name_prefix,
    int selected_dev,
    std::vector<std::unique_ptr<Device>>* devices) {
  // CPU first. A CPU device is required.
  auto cpu_factory = GetFactory("CPU");
  // 涉及 GPU 的都要单独考虑了.
  auto gpu_factory = GetFactory("GPU");
  auto xla_gpu_factory = GetFactory("XLA_GPU");

  if (!cpu_factory) {
    return errors::NotFound(
        "CPU Factory not registered.  Did you link in threadpool_device?");
  }
  size_t init_size = devices->size();
  TF_RETURN_IF_ERROR(cpu_factory->CreateDevices(options, name_prefix, devices));
  if (devices->size() == init_size) {
    return errors::NotFound("No CPU devices are available in this process");
  }

  //if (selected_dev == -1) {
    // CPU only now. 提前返回, 不对不对, 还缺了 
    //
    // /job:ps/replica:0/task:0/device:CPU:0
    // /job:ps/replica:0/task:0/device:XLA_CPU:0
    // /job:ps/replica:0/task:0/device:GPU:0
    // /job:ps/replica:0/task:0/device:GPU:1
    // /job:ps/replica:0/task:0/device:GPU:2
    // /job:ps/replica:0/task:0/device:GPU:3
    // /job:ps/replica:0/task:0/device:XLA_GPU:0
    // /job:ps/replica:0/task:0/device:XLA_GPU:1
    // /job:ps/replica:0/task:0/device:XLA_GPU:2
    // /job:ps/replica:0/task:0/device:XLA_GPU:3 
    //
  //  return Status::OK();
  //}

  VLOG(0) << "selected_dev: " << selected_dev;

  // Then the rest (including GPU).
  mutex_lock l(*get_device_factory_lock());
  for (auto& p : device_factories()) {
    auto factory = p.second.factory.get();
    // gpu is special now.
    
    // 不构建 xla gpu 了, 有问题目前, 不知道为什么
    //if (factory == gpu_factory) {
    if (factory == gpu_factory || factory == xla_gpu_factory) {
      // Pass: 把删选条件放到 CreateSelectedDevices 里面吧.
      TF_RETURN_IF_ERROR(
        factory->CreateSelectedDevices(options, name_prefix, 
          selected_dev, devices));
      // 因为这个函数的变化所以导致我需要大面积重复这个 AddSelectedDevices 函数.
    }
    if (factory != cpu_factory && factory != gpu_factory && 
        factory != xla_gpu_factory) {
      TF_RETURN_IF_ERROR(factory->CreateDevices(options, name_prefix, devices));
    }
  }

  return Status::OK();
}

// 应该说是要分配 Devices 而不是如 AddDevices 那样去找!
// c++ 调整 device visibility, environment var, 然后再
//c// Status DeviceFactory::ChangeVisibleDevices(
//c//   const SessionOptions& options, const string& name_prefix,
//c//   std::vector<std::unique_ptr<Device>>* devices) {
//c//   
//c//   // 调整 visibility 
//c//   // set env var:
//c//   // https://stackoverflow.com/questions/899517/set-local-environment-variables-in-c
//c//   // CPU first. A CPU device is required.
//c// 
//c//   
//c//   auto cpu_factory = GetFactory("CPU");
//c//   if (!cpu_factory) {
//c//     return errors::NotFound(
//c//         "CPU Factory not registered.  Did you link in threadpool_device?");
//c//   }
//c//   size_t init_size = devices->size();
//c//   TF_RETURN_IF_ERROR(cpu_factory->CreateDevices(options, name_prefix, devices));
//c//   if (devices->size() == init_size) {
//c//     return errors::NotFound("No CPU devices are available in this process");
//c//   }
//c// 
//c// 
//c// }

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
