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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/platform/numa.h"

namespace tensorflow {

class GPUDevice : public BaseGPUDevice {
 public:
  /** \brief Construct a GPU device.
   *
   *  \param options SessionOptions has a pointer to Env, target string from
   *                 client perspective, and ConfigProto options for GPU, CPU,
   *                 OS setting.
   *
   *  \param name
   *         Example of device_name string are "/device:GPU:0", "/device:GPU:1".
   *
   *  \param memory_limit Memory limit of GPU.
   *
   *  \param locality
   *         message DeviceLocality contains bus_id, numa_node, links.
   *
   *  \param tf_gpu_id TF logical GPU id.
   *
   *  \param physical_device_desc
   *         Example of physical_device_desc is
   *         "device: 0, name: ..., pci bus id: ... , compute capability: 6.1"
   *
   *  \param gpu_allocator
   *         An abstract interface for allocating and deallocating byte memory
   *
   *  \param cpu_allocator
   *         An abstract interface for allocating and deallocating byte memory
   *
   *  \return a unique pointer to BaseGPUDevice
   */
  GPUDevice(const SessionOptions& options, const string& name,
            Bytes memory_limit, const DeviceLocality& locality,
            TfGpuId tf_gpu_id, const string& physical_device_desc,
            Allocator* gpu_allocator, Allocator* cpu_allocator)
      : BaseGPUDevice(options, name, memory_limit, locality, tf_gpu_id,
                      physical_device_desc, gpu_allocator, cpu_allocator,
                      false /* sync every op */,
                //////////////////////////////
                // 这个 1 表示的是 stream group 的个数
                      1 /* max_streams */) {
                //////////////////////////////
    if (options.config.has_gpu_options()) {
      force_gpu_compatible_ =
          options.config.gpu_options().force_gpu_compatible();
    }
  }

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    CHECK(cpu_allocator_) << "bad place 1";
    if (attr.on_host()) {
      if (attr.gpu_compatible() || force_gpu_compatible_) {
        GPUProcessState* ps = GPUProcessState::singleton();
        return ps->GetGpuHostAllocator(0);
      } else {
        return cpu_allocator_;
        // cpu_allocator_ 变量说明
        // class BaseGPUDevice :: Allocator* cpu_allocator_;  // not owned
        // common_runtime/gpu/gpu_device.h
      }
    } else {
      return gpu_allocator_;
      // 1.
      // gpu_allocator_ 变量说明
      // class BaseGPUDevice :: Allocator* gpu_allocator_;  // not owned
      // common_runtime/gpu/gpu_device.h

      // 2. Allocator 数据结构
      // framework/allocator.h
      // Allocator is an abstract interface for allocating and deallocating
      // device memory.
    }
  }

 private:
  bool force_gpu_compatible_ = false;
};
// 1.
// class GPUDevice 数据结构说明:
// tensorflow/core/common_runtime/gpu/gpu_device_factory.cc
// - force_gpu_compatible_: bool, default value : false

// 2.
// class BaseGPUDevice 数据结构说明:
// tensorflow/core/common_runtime/gpu/gpu_device.h
// class BaseGPUDevice : public LocalDevice
// - gpu_allocator_: Allocator* // not owned
// - cpu_allocator_: Allocator* // not owned
// - executor_: se::StreamExecutor* // not owned
// - scoped_allocator_mgr_: std::unique_ptr<ScopedAllocatorMgr>
// - streams_: gtl::InlinedVector<StreamGroup*, 4>
// - scratch_init_mutex_: mutex
// - scratch_: gtl::InlinedVector<char*, 4>
// - device_contexts_: std::vector<GPUDeviceContext*>
// - gpu_device_info_: GpuDeviceInfo*
// - trace_mu_: mutex
// - tf_gpu_id_: TfGpuId
// - sync_every_op_: const bool, default value : false
// - max_streams_: const int32
// - em_: std::unique_ptr<EventMgr>
// - thread_pool_: std::unique_ptr<thread::ThreadPool>
// - kernel_tracker_: std::unique_ptr<GPUKernelTracker>
// - pending_cap_: int, default value : 0
// - timestamped_allocator_: bool, default value : 0


class GPUDeviceFactory : public BaseGPUDeviceFactory {
 private:

  /** \brief Create a GPU Device for a tf process.
   *
   *  \param options SessionOptions has a pointer to Env, target string from
   *                 client perspective, and ConfigProto options for GPU, CPU,
   *                 OS setting.
   *
   *  \param name
   *         Example of device_name string are "/device:GPU:0", "/device:GPU:1".
   *
   *  \param memory_limit Memory limit of GPU.
   *
   *  \param locality
   *         message DeviceLocality contains bus_id, numa_node, links.
   *
   *  \param tf_gpu_id TF logical GPU id.
   *
   *  \param physical_device_desc
   *         Example of physical_device_desc is
   *         "device: 0, name: ..., pci bus id: ... , compute capability: 6.1"
   *
   *  \param gpu_allocator
   *         An abstract interface for allocating and deallocating byte memory
   *
   *  \param cpu_allocator
   *         An abstract interface for allocating and deallocating byte memory
   *
   *  \return a unique pointer to BaseGPUDevice
   */
  std::unique_ptr<BaseGPUDevice> CreateGPUDevice(
      const SessionOptions& options, const string& name, Bytes memory_limit,
      const DeviceLocality& locality, TfGpuId tf_gpu_id,
      const string& physical_device_desc, Allocator* gpu_allocator,
      Allocator* cpu_allocator) override {
    return absl::make_unique<GPUDevice>(options, name, memory_limit, locality,
                                        tf_gpu_id, physical_device_desc,
                                        gpu_allocator, cpu_allocator);
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("GPU", GPUDeviceFactory, 210);

//------------------------------------------------------------------------------
// A CPUDevice that optimizes for interaction with GPUs in the
// process.
// -----------------------------------------------------------------------------
class GPUCompatibleCPUDevice : public ThreadPoolDevice {
 public:
  GPUCompatibleCPUDevice(const SessionOptions& options, const string& name,
                         Bytes memory_limit, const DeviceLocality& locality,
                         Allocator* allocator)
      : ThreadPoolDevice(options, name, memory_limit, locality, allocator),
        numa_node_(locality.numa_node()) {
    if (options.config.has_gpu_options()) {
      force_gpu_compatible_ =
          options.config.gpu_options().force_gpu_compatible();
    }
  }
  ~GPUCompatibleCPUDevice() override {}

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    GPUProcessState* ps = GPUProcessState::singleton();
    if (attr.gpu_compatible() || force_gpu_compatible_) {
      return ps->GetGpuHostAllocator(numa_node_);
    } else {
      // Call the parent's implementation.
      return ThreadPoolDevice::GetAllocator(attr);
    }
  }

 private:
  bool force_gpu_compatible_ = false;
  int numa_node_ = port::kNUMANoAffinity;
};

// The associated factory.
class GPUCompatibleCPUDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    int n = 1;
    auto iter = options.config.device_count().find("CPU");
    if (iter != options.config.device_count().end()) {
      n = iter->second;
    }
    int num_numa_nodes = options.config.experimental().use_numa_affinity()
                             ? port::NUMANumNodes()
                             : 1;
    for (int i = 0; i < n; i++) {
      string name = strings::StrCat(name_prefix, "/device:CPU:", i);
      int numa_node = i % num_numa_nodes;
      DeviceLocality locality;
      locality.set_numa_node(numa_node);
      devices->push_back(absl::make_unique<GPUCompatibleCPUDevice>(
          options, name, Bytes(256 << 20), DeviceLocality(),
          ProcessState::singleton()->GetCPUAllocator(numa_node)));
    }

    return Status::OK();
  }
};
REGISTER_LOCAL_DEVICE_FACTORY("CPU", GPUCompatibleCPUDeviceFactory, 70);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
