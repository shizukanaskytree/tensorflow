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

#include "tensorflow/core/framework/memory_types.h"

#include <utility>

#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {
// Returns the largest endpoint of anything in the name_map.
int GetTotal(const NameRangeMap& name_map) {
  int total = 0;
  for (const auto& item : name_map) {
    total = std::max(total, item.second.second);
  }
  return total;
}

// Fills memory_types for either input or output, setting everything
// to DEVICE_MEMORY except those args in host_memory_args.  Removes
// elements of host_memory_args that were used.
void MemoryTypesHelper(const NameRangeMap& name_map, // input
                       std::vector<string>* host_memory_args, // input
                       MemoryTypeVector* memory_types) { // output
  // Update args that have been marked as in "HOST_MEMORY".
  size_t keep = 0;

  for (size_t i = 0; i < host_memory_args->size(); ++i) {
    auto iter = name_map.find((*host_memory_args)[i]);
    if (iter != name_map.end()) {
      for (int j = iter->second.first; j < iter->second.second; ++j) {
        (*memory_types)[j] = HOST_MEMORY;
      }
    } else {
      // (*host_memory_args)[i] not found, save it for the next pass.
      if (i > keep) (*host_memory_args)[keep] = (*host_memory_args)[i];
      ++keep;
    }
  }
  host_memory_args->resize(keep);
}

bool IsFunctionCallOp(const string& op_type) {
  return op_type == "SymbolicGradient" || op_type == "PartitionedCall" ||
         op_type == "StatefulPartitionedCall" || op_type == "While";
}

}  // namespace

// 规则:
// - HOST_MEMORY
//    * DT_INT32
//    * DT_STRING
//    * DT_STRING_REF
//    * DT_RESOURCE
// - DEVICE_MEMORY
//    * Other DT_*
MemoryType MTypeFromDType(const DataType dtype) {
  // DataTypeAlwaysOnHost 函数说明:
  // ./tensorflow/core/framework/types.cc:214:
  // bool DataTypeAlwaysOnHost(DataType dt)
  // if the data type is DT_STRING, DT_STRING_REF, DT_RESOURCE, THEN this data type is always on host.
  // other datatype excluding DT_INT32 will not be always on host.
  return (dtype == DT_INT32 || DataTypeAlwaysOnHost(dtype)) ? HOST_MEMORY
                                                            : DEVICE_MEMORY;
}


// 目的: 根据 device type 决定 memory type
Status MemoryTypesForNode(const OpRegistryInterface* op_registry, // input
                          const DeviceType& device_type, // input
                          const NodeDef& ndef, // input
                          MemoryTypeVector* inp_mtypes,  // output
                          MemoryTypeVector* out_mtypes) {  // output
  /*
  打印:

  p device_type
  $62 = (const tensorflow::DeviceType &) @0x7fff3d4dad80: {type_ = "CPU"} # 看这里

  p ndef.DebugString()
  $63 = "name: \"y/shape\"\nop: \"Const\"\ndevice: \"/job:localhost/replica:0/task:0/device:CPU:0\"\nattr {\n  key: \"dtype\"\n  value {\n    type: DT_INT32\n  }\n}\nattr {\n  key: \"value\"\n  value {\n    tensor {\n      dtype: DT_INT32\n      tensor_shape {\n        dim {\n          size: 2\n        }\n      }\n      tensor_content: \"\\036\\000\\000\\000\\024\\000\\000\\000\"\n    }\n  }\n}\n"

  ===

  name: "y/shape"
  op: "Const"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\036\000\000\000\024\000\000\000"
      }
    }
  }

  ===

  p ndef.DebugString()

  name: "x/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "x/shape"
  device: "/job:localhost/replica:0/task:0/device:GPU:0"   # 看这里
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }

  */

  // OpDef 数据结构:
  // tensorflow/core/framework/op_def.proto

  // Look up the Op registered for this op name.
  const OpDef* op_def;

  // LookUpOpDef 函数说明:
  // tensorflow/core/framework/op.cc:37
  // Status OpRegistryInterface::LookUpOpDef(const string& op_type_name, // input
  //                                         const OpDef** op_def) // output

  // ndef.op(): string
  //    // The operation name.  There may be custom parameters in attrs.
  //    // Op names starting with an underscore are reserved for internal use.
  //    /// For example, op can be "Const"
  //    string op = 2;

  // Op<
  //   name=MatMul;
  //   signature=a:T, b:T ->product:T;         //map, : 作为 pair 分割，, 作为 pair 直接的分割
  //   attr=transpose_a:bool, default=false;
  //   attr=transpose_b:bool, default=false;
  //   attr=T:type,
  //        allowed=[DT_BFLOAT16,
  //                 DT_HALF,
  //                 DT_FLOAT,
  //                 DT_DOUBLE,
  //                 DT_INT32,
  //                 DT_INT64,
  //                 DT_COMPLEX64,
  //                 DT_COMPLEX128]>

  // ndef.op(): string
  // op_def: const OpDef**
  TF_RETURN_IF_ERROR(
    op_registry->LookUpOpDef(
      ndef.op(), // input
      &op_def)   // output
    );

  /*
  p ndef.op()
  $6 = "Const"

  p op_def->DebugString()
  name: "Const"
  output_arg {
    name: "output"
    type_attr: "dtype"
  }
  attr {
    name: "value"
    type: "tensor"
  }
  attr {
    name: "dtype"
    type: "type"
  }
  */

  // message KernelDef 数据结构:
  // tensorflow/core/framework/kernel_def.proto
  // - op: string, op name
  // - device_type : string
  //    * Type of device this kernel runs on.
  // - constraint: AttrConstraint (name, allowed_values)
  // - host_memory_arg: string
  // - label: string
  // - priority: int32
  //    * Prioritization of kernel amongst different devices.
  // Look up the Kernel registered for this node def.
  const KernelDef* kdef = nullptr;

  // FindKernelDef 函数说明:
  // tensorflow/core/framework/op_kernel.cc
  Status status =
      FindKernelDef(
        device_type, // input
        ndef, // input
        &kdef, // output
        nullptr /* kernel_class_name */);
  /*
  打印: y/shape node

  p kdef->DebugString()
  op: "Const"
  device_type: "CPU"
  */

  // **For functions (which have no KernelDef) and their gradients, we can only
  // best-effort derive the memory type from the data type.** **For now, we assume
  // int32 is always on host memory and other types are always on device memory.**
  DataTypeVector inp_dtypes;
  DataTypeVector out_dtypes;

  // InOutTypesForNode 函数说明:
  // framework/node_def_util.cc
  // 函数返回数据类型
  TF_RETURN_IF_ERROR(
      InOutTypesForNode(
        ndef,    // input
        *op_def,  // input
        &inp_dtypes, // output
        &out_dtypes));  // output


  inp_mtypes->clear();
  out_mtypes->clear();


  // **For functions (which have no KernelDef) and their gradients, we can only
  // best-effort derive the memory type from the data type.** **For now, we assume
  // int32 is always on host memory and other types are always on device memory.**
  // TODO(zhifengc,phawkins): We should do type inference over function bodies
  // to derive the correct input/output memory types. We should also split
  // host-memory and non host-memory arguments into separate type lists.

  // 这个分支没有进入，对于 "y/shape" 节点

  // IsFunctionCallOp 函数说明:
  // tensorflow/core/framework/memory_types.cc:63:
  // bool IsFunctionCallOp(const string& op_type)
  // return true if input is
  //  * SymbolicGradient
  //  * PartitionedCall
  //  * StatefulPartitionedCall
  //  * While
  if (!status.ok() || IsFunctionCallOp(ndef.op())) {

    // MTypeFromDType 函数说明:
    // tensorflow/core/framework/memory_types.cc:70
    // MemoryType MTypeFromDType(const DataType dtype)
    // 规则:
    // - HOST_MEMORY
    //    * DT_INT32
    //    * DT_STRING
    //    * DT_STRING_REF
    //    * DT_RESOURCE
    // - DEVICE_MEMORY
    //    * Other DT_*
    for (const auto& t : inp_dtypes) inp_mtypes->push_back(MTypeFromDType(t));
    for (const auto& t : out_dtypes) out_mtypes->push_back(MTypeFromDType(t));
    return Status::OK();
  }

  // NameRangeMap 类型说明:
  // typedef gtl::FlatMap<StringPiece, std::pair<int, int>, hash<StringPiece>> NameRangeMap

  // Gets the input/output names and their corresponding endpoint ranges.
  NameRangeMap inp_names;
  NameRangeMap out_names;

  // NameRangesForNode 函数说明 :
  // tensorflow/core/framework/node_def_util.cc
  // Status NameRangesForNode(const NodeDef& node_def, const OpDef& op_def,
  //                          NameRangeMap* inputs, NameRangeMap* outputs)
  //
  TF_RETURN_IF_ERROR(NameRangesForNode(
    ndef, // input
    *op_def, // input
    &inp_names, // output
    &out_names)); // output

  // Now that we know the size, fill with the default 'DEVICE_MEMORY'.
  // 我大概的理解是 对于 op node 这个 function 函数的 inputs / outputs 统计一下个数
  // 因为每个 input / output 都对应了 一个 node ，所以都要初始化 memory type
  inp_mtypes->resize(GetTotal(inp_names), DEVICE_MEMORY);
  out_mtypes->resize(GetTotal(out_names), DEVICE_MEMORY);

  // Fills in host memory types based on the kernel def.
  const auto& from_proto = kdef->host_memory_arg();

  std::vector<string> host_memory_args(from_proto.begin(), from_proto.end());

  // MemoryTypesHelper
  MemoryTypesHelper(inp_names, &host_memory_args, inp_mtypes);
  MemoryTypesHelper(out_names, &host_memory_args, out_mtypes);

  if (!host_memory_args.empty()) {
    return errors::InvalidArgument(
        "HostMemory args '", str_util::Join(host_memory_args, "', '"),
        "' not found in OpDef: ", SummarizeOpDef(*op_def));
  }
  CHECK_LE(inp_mtypes->size(), inp_dtypes.size());
  CHECK_LE(out_mtypes->size(), out_dtypes.size());

  // Mark e.g. all resource and string types as host memory.
  for (int i = 0; i < inp_mtypes->size(); ++i) {
    if (DataTypeAlwaysOnHost(inp_dtypes[i])) {
      (*inp_mtypes)[i] = HOST_MEMORY;
    }
  }
  for (int i = 0; i < out_mtypes->size(); ++i) {
    if (DataTypeAlwaysOnHost(out_dtypes[i])) {
      (*out_mtypes)[i] = HOST_MEMORY;
    }
  }

  std::vector<int32> hostmem_attr;
  if (GetNodeAttr(ndef, "_input_hostmem", &hostmem_attr).ok()) {
    for (int32 i : hostmem_attr) {
      if (0 <= i && i < inp_mtypes->size()) {
        (*inp_mtypes)[i] = HOST_MEMORY;
      }
    }
  }
  if (GetNodeAttr(ndef, "_output_hostmem", &hostmem_attr).ok()) {
    for (int32 i : hostmem_attr) {
      if (0 <= i && i < out_mtypes->size()) {
        (*out_mtypes)[i] = HOST_MEMORY;
      }
    }
  }

  return Status::OK();
}

}  // namespace tensorflow
