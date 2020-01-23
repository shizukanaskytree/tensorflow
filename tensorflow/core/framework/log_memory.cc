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

#include "tensorflow/core/framework/log_memory.h"

#include "tensorflow/core/framework/log_memory.pb_text.h"
#include "tensorflow/core/framework/log_memory.pb.h"

namespace tensorflow {

const string LogMemory::kLogMemoryLabel = "__LOG_MEMORY__";

bool LogMemory::IsEnabled() { return VLOG_IS_ON(1); }

namespace {

// Write the proto entry to LOG(INFO).
template <typename T>
void OutputToLog(const T& proto) {
  string type_name = proto.GetTypeName();
  const size_t index = type_name.find_last_of(".");
  if (index != string::npos) type_name = type_name.substr(index + 1);
  LOG(INFO) << LogMemory::kLogMemoryLabel << " " << type_name << " { "
            << ProtoShortDebugString(proto) << " }";
}
// 1.
// OutputToLog 函数说明:
// void OutputToLog(const T& proto)
// tensorflow/core/framework/log_memory.cc
// 核心操作 : ProtoShortDebugString(proto)


}  // namespace

void LogMemory::RecordStep(const int64 step_id, const string& handle) {
  MemoryLogStep step;
  step.set_step_id(step_id);
  step.set_handle(handle);
  OutputToLog(step);
}

void LogMemory::RecordTensorAllocation(const string& kernel_name,
                                       const int64 step_id,
                                       const Tensor& tensor) {
  MemoryLogTensorAllocation allocation;
  // 1.
  // message MemoryLogTensorAllocation 数据结构
  // tensorflow/core/framework/log_memory.proto
  // - step_id: int64
  // - kernel_name: string
  // - tensor: TensorDescription

  // 1.2.
  // message TensorDescription 数据结构
  // tensorflow/core/framework/tensor_description.proto
  // - dtype: DataType
  //   Data type of tensor elements
  // - shape: TensorShapeProto
  //   Shape of the tensor.
  // - allocation_description: AllocationDescription
  //   Information about the size and allocator used for the data

  // 1.3.
  // tensorflow/core/framework/allocation_description.proto
  // message AllocationDescription 数据结构
  // tensorflow/core/framework/allocation_description.proto
  // - requested_bytes: int64
  //   Total number of bytes requested
  // - allocated_bytes: int64
  //   Total number of bytes allocated if known
  // - allocator_name: string
  //   Name of the allocator used
  // - allocation_id: int64
  //   Identifier of the allocated buffer if known
  // - has_single_reference: bool
  //   Set if this tensor only has one remaining reference
  // - ptr: uint64
  //   Address of the allocation.

  // 2.
  // 使用的例子
  // 2.1
  // LogMemory::RecordTensorAllocation("Unknown (with attributes)", LogMemory::UNKNOWN_STEP_ID, *this);
  // tensorflow/core/framework/tensor.cc
  // 2.2
  // LogMemory::RecordTensorAllocation("Unknown (from Proto)", LogMemory::UNKNOWN_STEP_ID, *this);
  // tensorflow/core/framework/tensor.cc
  // 2.3
  // LogMemory::RecordTensorAllocation(def_->name(), LogMemory::OP_KERNEL_CONSTRUCTION_STEP_ID, new_temp);
  // tensorflow/core/framework/op_kernel.cc
  // 2.4
  // LogMemory::RecordTensorAllocation(params_->op_kernel->name(), params_->step_id, new_tensor);
  // tensorflow/core/framework/op_kernel.cc

  allocation.set_step_id(step_id);
  allocation.set_kernel_name(kernel_name);
  tensor.FillDescription(allocation.mutable_tensor());


  OutputToLog(allocation);
  // 1.
  // OutputToLog 函数说明:
  // void OutputToLog(const T& proto)
  // tensorflow/core/framework/log_memory.cc
  // 核心操作 : ProtoShortDebugString(proto)
}

void LogMemory::RecordTensorDeallocation(const int64 allocation_id,
                                         const string& allocator_name) {
  MemoryLogTensorDeallocation deallocation;
  deallocation.set_allocation_id(allocation_id);
  deallocation.set_allocator_name(allocator_name);
  OutputToLog(deallocation);
}

void LogMemory::RecordTensorOutput(const string& kernel_name,
                                   const int64 step_id, const int index,
                                   const Tensor& tensor) {
  MemoryLogTensorOutput output;
  output.set_step_id(step_id);
  output.set_kernel_name(kernel_name);
  output.set_index(index);
  tensor.FillDescription(output.mutable_tensor());
  OutputToLog(output);
}

void LogMemory::RecordRawAllocation(const string& operation,
                                    const int64 step_id, size_t num_bytes,
                                    void* ptr, Allocator* allocator) {
  MemoryLogRawAllocation allocation;
  allocation.set_step_id(step_id);
  allocation.set_operation(operation);
  allocation.set_num_bytes(static_cast<int64>(num_bytes));
  allocation.set_ptr(reinterpret_cast<uintptr_t>(ptr));
  allocation.set_allocation_id(allocator->AllocationId(ptr));
  allocation.set_allocator_name(allocator->Name());
  OutputToLog(allocation);
}

void LogMemory::RecordRawDeallocation(const string& operation,
                                      const int64 step_id, void* ptr,
                                      Allocator* allocator, bool deferred) {
  MemoryLogRawDeallocation deallocation;
  deallocation.set_step_id(step_id);
  deallocation.set_operation(operation);
  deallocation.set_allocation_id(allocator->AllocationId(ptr));
  deallocation.set_allocator_name(allocator->Name());
  deallocation.set_deferred(deferred);
  OutputToLog(deallocation);
}

}  // namespace tensorflow
