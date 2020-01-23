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

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

// CPU->GPU
void GPUDeviceContext::CopyCPUTensorToDevice(
  const Tensor* cpu_tensor,   // input  // const Tensor* cpu_tensor
  Device* device,             // input  //dst: Device*
  Tensor* device_tensor,      // output // Tensor* device_tensor
  StatusCallback done) const  // input  // StatusCallback done
{

  GPUUtil::CopyCPUTensorToGPU(cpu_tensor, // input  // const Tensor* cpu_tensor
                              this, // input // GPUDeviceContext*
                              device, // input  //dst: Device*
                              device_tensor, // output // Tensor* device_tensor
                              done);  // input  // StatusCallback done
  // 1.
  // GPUUtil::CopyCPUTensorToGPU 函数说明
  // tensorflow/core/common_runtime/gpu/gpu_util.cc
  //
  // 函数接口
  // void GPUUtil::CopyCPUTensorToGPU(const Tensor* cpu_tensor,
  //                                  const DeviceContext* device_context,
  //                                  Device* gpu_device,
  //                                  Tensor* gpu_tensor,
  //                                  StatusCallback done)
}

// GPU->CPU
void GPUDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor, // input
                                             StringPiece tensor_name, // input , 无效的输入
                                             Device* device,
                                             Tensor* cpu_tensor, // output
                                             StatusCallback done) {
  GPUUtil::CopyGPUTensorToCPU(device, // input
                              this, // input
                              device_tensor, // input
                              cpu_tensor,
                              done);
  // 1.
  // GPUUtil::CopyGPUTensorToCPU 函数说明:
  // tensorflow/core/common_runtime/gpu/gpu_util.cc
  //
  // void GPUUtil::CopyGPUTensorToCPU(Device* gpu_device,
  //                                  const DeviceContext* device_context,
  //                                  const Tensor* gpu_tensor,
  //                                  Tensor* cpu_tensor,
  //                                  StatusCallback done)

  // 2.
  // QQQ. cpu_tensor 在输入时是怎么准备的? 是什么样的?
  // AAA.
  // 参考:
  // IntraProcessRendezvous::SameWorkerRecvDone
  // tensorflow/core/common_runtime/rendezvous_mgr.cc
  // keywords:
  // - out_allocator
  // - copy
  // -

}


void GPUDeviceContext::CopyTensorInSameDevice(const Tensor* input_tensor,
                                              Device* device,
                                              Tensor* output_tensor,
                                              StatusCallback done) const {
  GPUUtil::CopyGPUTensorToSameGPU(device, this, input_tensor, output_tensor,
                                  done);
}


Status GPUDeviceContext::ThenExecute(Device* device, se::Stream* stream,
                                     std::function<void()> func) {
  const DeviceBase::GpuDeviceInfo* gpu_info =
      device->tensorflow_gpu_device_info();
  gpu_info->event_mgr->ThenExecute(stream, func);
  return Status::OK();
}

}  // namespace tensorflow
