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

#include "tensorflow/core/common_runtime/copy_tensor.h"

#include <atomic>
#include <utility>
#include <vector>
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/util/reffed_status_callback.h"

// 1.
// summary of this file

// 数据结构
// - struct RegistrationInfo

// 函数
// - std::vector<RegistrationInfo>* MutableRegistry()
// -
// - void CopyHostToDevice(const Tensor* input, // input // cpu_tensor*
//                         Allocator* cpu_allocator, // input
//                         Allocator* out_allocator, // input
//                         StringPiece edge_name, // input
//                         Device* dst,  // input
//                         Tensor* output, // output // gpu_tensor*
//                         DeviceContext* recv_dev_context, // input
//                         StatusCallback done)
// -
// -
// -


namespace tensorflow {
namespace {

struct RegistrationInfo {
  RegistrationInfo(DeviceType s, DeviceType r, CopyTensor::CopyFunction cf)
      : sender_device_type(std::move(s)),
        receiver_device_type(std::move(r)),
        copy_function(cf) {}
  DeviceType sender_device_type;
  DeviceType receiver_device_type;
  CopyTensor::CopyFunction copy_function;
};

// We use a vector instead of a map since we expect there to be very
// few registrations.
std::vector<RegistrationInfo>* MutableRegistry() {
  static std::vector<RegistrationInfo>* registry =
      new std::vector<RegistrationInfo>;
  return registry;
}


// CPU->GPU
void CopyHostToDevice(const Tensor* input, // input // cpu_tensor*
                      Allocator* cpu_allocator, // input
                      Allocator* out_allocator, // input
                      StringPiece edge_name, // input
                      Device* dst,  // input
                      Tensor* output, // output // gpu_tensor*
                      DeviceContext* recv_dev_context, // input
                      StatusCallback done) { // input
  // 1.
  // CopyHostToDevice 被调用:
  // CopyTensor::ViaDMA
  // copy_tensor.cc

  // 2.
  // class Allocator 数据结构
  // tensorflow/core/framework/allocator.h
  // - kAllocatorAlignment: static constexpr size_t, default_value:64
  //   Align to 64 byte boundary.
  // 都是接口函数
  // - string Name()
  // - void* AllocateRaw(size_t alignment, size_t num_bytes)
  // - void* AllocateRaw(size_t alignment, size_t num_bytes,
  //                     const AllocationAttributes& allocation_attr)
  // - DeallocateRaw(void* ptr)
  // - template <typename T>
  //   T* Allocate(size_t num_elements)
  // - template <typename T>
  //   T* Allocate(size_t num_elements,
  //               const AllocationAttributes& allocation_attr)
  // - template <typename T>
  //   void Deallocate(T* ptr, size_t num_elements)
  // - bool TracksAllocationSizes()
  // - bool ShouldAllocateEmptyTensors()
  // - size_t RequestedSize(const void* ptr)
  // - size_t AllocatedSize(const void* ptr)
  // - int64 AllocationId(const void* ptr)
  // - size_t AllocatedSizeSlow(const void* ptr)
  // - absl::optional<AllocatorStats> GetStats()
  // - void RunCtor(T* p, size_t n)
  // - void RunStringCtor(string* p, size_t n)
  // - void RunStringDtor(string* p, size_t n)
  // - void RunResourceCtor(ResourceHandle* p, size_t n)
  // - void RunResourceDtor(ResourceHandle* p, size_t n)
  // - void RunVariantCtor(Variant* p, size_t n)
  // - void RunVariantDtor(Variant* p, size_t n)

  // 3.
  // class DeviceContext 数据结构
  // tensorflow/core/framework/device_base.h
  // class DeviceContext: public core::RefCounted
  // 概述:
  // A class that devices can subclass to pass around
  // Device-specific context to OpKernels.
  //
  // 没有成员变量
  //
  // 接口:
  // - stream()
  // - MaintainLifetimeOnStream
  // - CopyCPUTensorToDevice
  // - CopyTensorInSameDevice
  // - CopyDeviceTensorToCPU
  // - ThenExecute


  if (input->dtype() == DT_VARIANT) {
    // 1.
    // DT_VARIANT 类型说明
    // tensorflow/tensorflow/core/framework/types.proto
    // enum DataType
    // 表示的是 Arbitrary C++ data types

    Tensor copy(cpu_allocator, DT_VARIANT, input->shape());
    auto* status_cb = new ReffedStatusCallback(std::move(done));
    core::ScopedUnref status_cb_unref(status_cb);

    auto wrapped_done = [status_cb](const Status& s) {
      status_cb->UpdateStatus(s);
      status_cb->Unref();
    };

    auto copier = std::bind(
        [dst, recv_dev_context, out_allocator, status_cb, cpu_allocator,
         edge_name](StatusCallback wrapped_done_,
                    // Begin unbound arguments
                    const Tensor& from, Tensor* to) {
          if (from.dtype() == DT_VARIANT) {
            status_cb->Ref();
            CopyHostToDevice(&from, cpu_allocator, out_allocator, edge_name,
                             dst, to, recv_dev_context, wrapped_done_);
            return Status::OK();
          } else {
            if (!DMAHelper::CanUseDMA(&from)) {
              Status err = errors::InvalidArgument(
                  "During Variant Host->Device Copy: "
                  "non-DMA-copy attempted of tensor type: ",
                  DataTypeString(from.dtype()));
              status_cb->UpdateStatus(err);
              return err;
            }
            if (status_cb->ok()) {
              status_cb->Ref();
              *to = Tensor(out_allocator, from.dtype(), from.shape());
              recv_dev_context->CopyCPUTensorToDevice(&from, dst, to,
                                                      wrapped_done_);
              return Status::OK();
            } else {
              return status_cb->status();
            }
          }
        },
        std::move(wrapped_done), std::placeholders::_1, std::placeholders::_2);

    const Variant* v = input->flat<Variant>().data();
    Variant* v_out = copy.flat<Variant>().data();
    Status s_copy_init;
    for (int64 i = 0; i < input->NumElements(); ++i) {
      s_copy_init = VariantDeviceCopy(
          VariantDeviceCopyDirection::HOST_TO_DEVICE, v[i], &v_out[i], copier);
      if (!s_copy_init.ok()) {
        status_cb->UpdateStatus(s_copy_init);
        break;
      }
    }

    if (s_copy_init.ok()) {
      *output = std::move(copy);
    }

  } else {

    recv_dev_context->CopyCPUTensorToDevice(
      input,            // input  // const Tensor* cpu_tensor
      dst,              // input  //dst: Device*
      output,           // output // Tensor* device_tensor
      std::move(done)); // input  // StatusCallback done
    // 1.
    // recv_dev_context 变量说明
    // recv_dev_context: DeviceContext*
    // recv_dev_context 这里的真实的类型是 GPUDeviceContext*

    // 2.
    // class DeviceContext 数据结构
    // tensorflow/core/framework/device_base.h
    // class DeviceContext: public core::RefCounted
    // 概述:
    // A class that devices can subclass to pass around
    // Device-specific context to OpKernels.
    //
    // 没有成员变量
    //
    // 接口:
    // - stream()
    // - MaintainLifetimeOnStream
    // - CopyCPUTensorToDevice
    // - CopyTensorInSameDevice
    // - CopyDeviceTensorToCPU
    // - ThenExecute

    // 3.
    // CopyCPUTensorToDevice 函数说明:
    // 因为 recv_dev_context 这里的真实的类型是 GPUDeviceContext* ，所以
    //
    // GPUDeviceContext::CopyCPUTensorToDevice
    // tensorflow/core/common_runtime/gpu/gpu_util_platform_specific.cc
    //
    // 函数接口：
    // void GPUDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
    //                                              Device* device,
    //                                              Tensor* device_tensor,
    //                                              StatusCallback done)

    //
    // GPUDeviceContext
  }

}

void CopyDeviceToHost(const Tensor* input,
                      Allocator* cpu_allocator,
                      Allocator* out_allocator,
                      StringPiece edge_name, // 我应该不会用这个了吧
                      Device* src,
                      Tensor* output,
                      DeviceContext* send_dev_context,
                      StatusCallback done) {
  // 1.
  // CopyHostToDevice 被调用:
  // CopyTensor::ViaDMA
  // copy_tensor.cc

  // 2.
  // class Allocator 数据结构
  // tensorflow/core/framework/allocator.h
  // - kAllocatorAlignment: static constexpr size_t, default_value:64
  //   Align to 64 byte boundary.
  // 都是接口函数
  // - string Name()
  // - void* AllocateRaw(size_t alignment, size_t num_bytes)
  // - void* AllocateRaw(size_t alignment, size_t num_bytes,
  //                     const AllocationAttributes& allocation_attr)
  // - DeallocateRaw(void* ptr)
  // - template <typename T>
  //   T* Allocate(size_t num_elements)
  // - template <typename T>
  //   T* Allocate(size_t num_elements,
  //               const AllocationAttributes& allocation_attr)
  // - template <typename T>
  //   void Deallocate(T* ptr, size_t num_elements)
  // - bool TracksAllocationSizes()
  // - bool ShouldAllocateEmptyTensors()
  // - size_t RequestedSize(const void* ptr)
  // - size_t AllocatedSize(const void* ptr)
  // - int64 AllocationId(const void* ptr)
  // - size_t AllocatedSizeSlow(const void* ptr)
  // - absl::optional<AllocatorStats> GetStats()
  // - void RunCtor(T* p, size_t n)
  // - void RunStringCtor(string* p, size_t n)
  // - void RunStringDtor(string* p, size_t n)
  // - void RunResourceCtor(ResourceHandle* p, size_t n)
  // - void RunResourceDtor(ResourceHandle* p, size_t n)
  // - void RunVariantCtor(Variant* p, size_t n)
  // - void RunVariantDtor(Variant* p, size_t n)

  // 3.
  // class DeviceContext 数据结构
  // tensorflow/core/framework/device_base.h
  // class DeviceContext: public core::RefCounted
  // 概述:
  // A class that devices can subclass to pass around
  // Device-specific context to OpKernels.
  //
  // 没有成员变量
  //
  // 接口:
  // - stream()
  // - MaintainLifetimeOnStream
  // - CopyCPUTensorToDevice
  // - CopyTensorInSameDevice
  // - CopyDeviceTensorToCPU
  // - ThenExecute

  if (input->dtype() == DT_VARIANT) {
    // 1.
    // DT_VARIANT 类型说明
    // tensorflow/tensorflow/core/framework/types.proto
    // enum DataType
    // 表示的是 Arbitrary C++ data types

    Tensor copy(cpu_allocator, DT_VARIANT, input->shape());
    auto* status_cb = new ReffedStatusCallback(std::move(done));
    core::ScopedUnref status_cb_unref(status_cb);

    auto wrapped_done = [status_cb](const Status& s) {
      status_cb->UpdateStatus(s);
      status_cb->Unref();
    };

    auto copier = std::bind(
        [edge_name, src, send_dev_context, out_allocator, status_cb,
         cpu_allocator](StatusCallback wrapped_done_,
                        // Begin unbound arguments
                        const Tensor& from, Tensor* to) {
          if (from.dtype() == DT_VARIANT) {
            status_cb->Ref();
            CopyDeviceToHost(&from, cpu_allocator, out_allocator, edge_name,
                             src, to, send_dev_context, wrapped_done_);
            return Status::OK();
          } else {
            if (!DMAHelper::CanUseDMA(&from)) {
              Status err = errors::InvalidArgument(
                  "During Variant Device->Host Copy: "
                  "non-DMA-copy attempted of tensor type: ",
                  DataTypeString(from.dtype()));
              status_cb->UpdateStatus(err);
              return err;
            }
            if (status_cb->ok()) {
              status_cb->Ref();
              *to = Tensor(out_allocator, from.dtype(), from.shape());
              send_dev_context->CopyDeviceTensorToCPU(&from, edge_name, src, to,
                                                      wrapped_done_);
              return Status::OK();
            } else {
              return status_cb->status();
            }
          }
        },
        std::move(wrapped_done), std::placeholders::_1, std::placeholders::_2);

    const Variant* v = input->flat<Variant>().data();

    Variant* v_out = copy.flat<Variant>().data();

    Status s_copy_init;

    for (int64 i = 0; i < input->NumElements(); ++i) {

      s_copy_init = VariantDeviceCopy(
          VariantDeviceCopyDirection::DEVICE_TO_HOST, v[i], &v_out[i], copier);

      if (!s_copy_init.ok()) {
        status_cb->UpdateStatus(s_copy_init);
        break;
      }
    }

    if (s_copy_init.ok()) {
      *output = std::move(copy);
    }

  } else {

    send_dev_context->CopyDeviceTensorToCPU(
                        input, // gpu_tensor
                        edge_name,
                        src,
                        output, // cpu_tensor
                        std::move(done));
    // 1.
    // CopyDeviceTensorToCPU 函数参数说明:
    // void GPUDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
    //                                              StringPiece tensor_name,
    //                                              Device* device,
    //                                              Tensor* cpu_tensor,
    //                                              StatusCallback done)
    // tensorflow/core/common_runtime/gpu/gpu_util_platform_specific.cc

    // 2.
    // send_dev_context 变量说明
    // send_dev_context: DeviceContext*

    // 3.
    // class DeviceContext 数据结构
    // tensorflow/core/framework/device_base.h
    // class DeviceContext: public core::RefCounted
    // 概述:
    // A class that devices can subclass to pass around
    // Device-specific context to OpKernels.
    //
    // 没有成员变量
    //
    // 接口:
    // - stream()
    // - MaintainLifetimeOnStream
    // - CopyCPUTensorToDevice
    // - CopyTensorInSameDevice
    // - CopyDeviceTensorToCPU
    // - ThenExecute


  }
}

void CopyDeviceToDevice(CopyTensor::CopyFunction copy_function,
                        Allocator* cpu_allocator, Allocator* out_allocator,
                        DeviceContext* send_dev_context,
                        DeviceContext* recv_dev_context, Device* src,
                        Device* dst, const AllocatorAttributes src_alloc_attr,
                        const AllocatorAttributes dst_alloc_attr,
                        const Tensor* input, Tensor* output,
                        int dev_to_dev_stream_index, StatusCallback done) {

  if (input->dtype() == DT_VARIANT) {
    Tensor copy(cpu_allocator, DT_VARIANT, input->shape());
    auto* status_cb = new ReffedStatusCallback(std::move(done));
    core::ScopedUnref status_cb_unref(status_cb);

    auto wrapped_done = [status_cb](const Status& s) {
      status_cb->UpdateStatus(s);
      status_cb->Unref();
    };
    auto copier = std::bind(
        [copy_function, src, dst, src_alloc_attr, dst_alloc_attr,
         recv_dev_context, send_dev_context, out_allocator, status_cb,
         dev_to_dev_stream_index](StatusCallback wrapped_done_,
                                  // Begin unbound arguments
                                  const Tensor& from, Tensor* to) {
          if (!DMAHelper::CanUseDMA(&from)) {
            Status err = errors::InvalidArgument(
                "During Variant Device->Device Copy: "
                "non-DMA-copy attempted of tensor type: ",
                DataTypeString(from.dtype()));
            status_cb->UpdateStatus(err);
            return err;
          }
          if (status_cb->ok()) {
            status_cb->Ref();
            *to = Tensor(out_allocator, from.dtype(), from.shape());
            copy_function(send_dev_context, recv_dev_context, src, dst,
                          src_alloc_attr, dst_alloc_attr, &from, to,
                          dev_to_dev_stream_index, std::move(wrapped_done_));
            return Status::OK();
          } else {
            return status_cb->status();
          }
        },
        std::move(wrapped_done), std::placeholders::_1, std::placeholders::_2);

    const Variant* v = input->flat<Variant>().data();
    Variant* v_out = copy.flat<Variant>().data();
    Status s_copy_init;
    for (int64 i = 0; i < input->NumElements(); ++i) {
      s_copy_init =
          VariantDeviceCopy(VariantDeviceCopyDirection::DEVICE_TO_DEVICE, v[i],
                            &v_out[i], copier);
      if (!s_copy_init.ok()) {
        status_cb->UpdateStatus(s_copy_init);
        break;
      }
    }
    if (s_copy_init.ok()) {
      *output = std::move(copy);
    }
  } else {
    copy_function(send_dev_context, recv_dev_context, src, dst, src_alloc_attr,
                  dst_alloc_attr, input, output, dev_to_dev_stream_index,
                  std::move(done));
  }
}

}  // namespace


// 1.
// QQQ.是怎么使用的？

// static
void CopyTensor::ViaDMA(StringPiece edge_name,           // input
                        DeviceContext* send_dev_context, // input
                        DeviceContext* recv_dev_context, // input
                        Device* src, // input
                        Device* dst, // input
                        const AllocatorAttributes src_alloc_attr, // input
                        const AllocatorAttributes dst_alloc_attr, // input
                        const Tensor* input, // gpu_tensor if CPU=>GPU // input
                        Tensor* output, // cpu_tensor if GPU=>CPU    // output
                        int dev_to_dev_stream_index, // input
                        StatusCallback done) { // input
  // 1.
  // 提醒: 有的输入在下面函数中并没有被用到
  // 在某些函数中没有被用到的话，就肯定是 input 了

  tracing::ScopedAnnotation annotation(edge_name);
  VLOG(1) << "Copy " << edge_name;

  // 自动确定方向
  const DeviceType src_device_type(
      src_alloc_attr.on_host() ? DEVICE_CPU : src->attributes().device_type());
  // 1.
  // set_on_host 和 on_host
  // src_alloc_attr.on_host() 意义:
  // 说明这个 device memory 是 CPU 的，不是 GPU 的，
  // 所以，不能在要创建 GPU memory 的时候 set_on_host(true);

  const DeviceType dst_device_type(
      dst_alloc_attr.on_host() ? DEVICE_CPU : dst->attributes().device_type());

  const bool non_cpu_src = src_device_type != DeviceType(DEVICE_CPU);

  const bool non_cpu_dst = dst_device_type != DeviceType(DEVICE_CPU);

  // TODO(phawkins): choose an allocator optimal for both the src and dst
  // devices, not just the src device.
  AllocatorAttributes host_alloc_attrs;

  host_alloc_attrs.set_gpu_compatible(true);
  host_alloc_attrs.set_on_host(true);
  // 1.
  // AllocatorAttributes::set_on_host(true) 设置说明
  // 这个应该是指定创建 cpu memory 的，不是 GPU memory

  // 1.1
  //

  // 1.2
  // logging 和实验
  // https://docs.google.com/document/d/1kVcuUZ-hBpy1M4AeqQS_pVfmCDGxliJJzK6gLta62Hk/edit#
  // search : 2. TransferCPU2GPUAllocationCreate BugReport

  // 1.1

  Allocator* cpu_allocator = src->GetAllocator(host_alloc_attrs);
  Allocator* out_allocator = dst->GetAllocator(dst_alloc_attr);

  // E.g., gpu -> gpu
  if (non_cpu_src && non_cpu_dst) {
    // Device to device copy.  Look through registry for an appropriate
    // CopyFunction.
    std::vector<RegistrationInfo>* registry = MutableRegistry();
    for (const RegistrationInfo& ri : *registry) {
      if (ri.sender_device_type == src_device_type &&
          ri.receiver_device_type == dst_device_type) {
        CopyDeviceToDevice(ri.copy_function, cpu_allocator, out_allocator,
                           send_dev_context, recv_dev_context, src, dst,
                           src_alloc_attr, dst_alloc_attr, input, output,
                           dev_to_dev_stream_index, std::move(done));
        return;
      }
    }

    // Fall back to copying via the host.
    VLOG(1) << "No function registered to copy from devices of type "
            << src_device_type.type() << " to devices of type "
            << dst_device_type.type()
            << ". Falling back to copying via the host.";

    Tensor* cpu_tensor =
        new Tensor(cpu_allocator, input->dtype(), input->shape());
    // 1.
    // cpu_tensor 的 cpu_allocator 说明:
    //

    std::function<void(const Status&)> delete_and_done = std::bind(
        [cpu_tensor](StatusCallback done_,
                     // Begin unbound arguments.
                     const Status& status) {
          delete cpu_tensor;
          done_(status);
        },
        std::move(done), std::placeholders::_1);
    std::function<void(const Status&)> then_copy_to_other_device = std::bind(
        [delete_and_done, recv_dev_context, cpu_tensor, cpu_allocator,
         out_allocator, edge_name, dst, output](StatusCallback delete_and_done_,
                                                // Begin unbound arguments.
                                                Status status) {
          if (!status.ok()) {
            delete_and_done_(status);
            return;
          }
          CopyHostToDevice(cpu_tensor, cpu_allocator, out_allocator, edge_name,
                           dst, output, recv_dev_context,
                           std::move(delete_and_done_));
        },
        std::move(delete_and_done), std::placeholders::_1);
    CopyDeviceToHost(input, cpu_allocator, out_allocator, edge_name, src,
                     cpu_tensor, send_dev_context,
                     std::move(then_copy_to_other_device));
    return;
  }

  // E.g., gpu -> cpu
  if (non_cpu_src && !non_cpu_dst) {
    // Device to host copy.
    CopyDeviceToHost(input, cpu_allocator, out_allocator, edge_name, src,
                     output, send_dev_context, std::move(done));
    return;
  }

  // E.g., cpu -> gpu
  if (!non_cpu_src && non_cpu_dst) {
    // Host to Device copy.
    CopyHostToDevice(input,           // input // cpu tensor*
                     cpu_allocator,   // input
                     out_allocator,   // input
                     edge_name,       // input
                     dst,             // input
                     output,          // output // gpu tensor*
                     recv_dev_context, // input
                     std::move(done)); // input

    // CopyHostToDevice 函数说明:
    // tensorflow/core/common_runtime/copy_tensor.cc

    return;
  }

  // cpu -> cpu
  CHECK(!non_cpu_src && !non_cpu_dst);
  *output = *input;
  done(Status::OK());
}

// static
Status CopyTensor::Register(DeviceType sender_device_type,
                            DeviceType receiver_device_type,
                            CopyFunction copy_function) {
  std::vector<RegistrationInfo>* registry = MutableRegistry();
  registry->emplace_back(sender_device_type, receiver_device_type,
                         copy_function);
  return Status::OK();
}

namespace {

// The following registrations enable a DT_VARIANT tensor element that contains
// a wrapped `tensorflow::Tensor` to be copied between devices.
static Status WrappedTensorDeviceCopy(
    const Tensor& from, Tensor* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
  if (from.dtype() == DT_VARIANT) {
    // TODO(b/116349787): Implement support for nested variants.
    return errors::Unimplemented(
        "Support for copying nested variants to device has not yet been "
        "implemented.");
  } else if (DMAHelper::CanUseDMA(&from)) {
    TF_RETURN_IF_ERROR(copy(from, to));
  } else {
    *to = from;
  }

  return Status::OK();
}

#define REGISTER_WRAPPED_TENSOR_COPY(DIRECTION)         \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      Tensor, DIRECTION, WrappedTensorDeviceCopy)

REGISTER_WRAPPED_TENSOR_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_WRAPPED_TENSOR_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_WRAPPED_TENSOR_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

}  // namespace

}  // namespace tensorflow
