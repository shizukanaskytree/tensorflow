/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Our general strategy for preventing conflicts between concurrent
// reads and writes of resource variables is to:
// * For read operations, we:
//   - acquire the variable's mutex (in "shared" mode);
//   - make a (shallow) copy of the Tensor object, which increments
//     the reference count on the variable's TensorBuffer;
//   - release the variable's mutex;
//   - use the copy of the Tensor object to do the read.
// * For write operations, we:
//   - acquire the variable's mutex (in "exclusive" mode);
//   - check the reference count of variable's TensorBuffer and
//     if it is >1, make a deep copy of the variable's Tensor;
//   - mutate the variable's Tensor;
//   - and release the variable's mutex.
// This allows several read operations to all use the same
// TensorBuffer without needing to copy. When it comes time to write
// it will only make a copy if there is an outstanding read using the
// buffer. Write operations are serialized by the variable's mutex.
//
// For sparse operations (scatter, gather, sparse optimizer updates),
// we need to avoid copies, since there may not be enough memory for
// to copies of the whole tensor. To support this, we make two
// modifications to the above strategy:
// * For sparse reads (gather), we hold the variable's mutex (still in
//   "shared" mode) for the duration of the whole read. This means
//   that as long as you only do sparse read operations no write will
//   see the reference count >1.
// * For sparse write operations where the user explicitly specifies
//   that they want to perform the write without locks held
//   (use_locking=false), we never copy even if the variable's
//   reference count is >1.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include <memory>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/gather_functor.h"
#include "tensorflow/core/kernels/resource_variable_ops.h"
#include "tensorflow/core/kernels/scatter_functor.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

REGISTER_RESOURCE_HANDLE_KERNEL(Var);
REGISTER_KERNEL_BUILDER(Name("_VarHandlesOp").Device(DEVICE_CPU),
                        ResourceHandlesOp<Var>);

ReadVariableOp::ReadVariableOp(OpKernelConstruction* c) : OpKernel(c) {
  OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
}

namespace {

Status CopyVariable(int output_idx, OpKernelContext* ctx, const Tensor* t) {
  Tensor* output;
  Notification n;
  Status status;
  AllocatorAttributes attr;
  if (t->dtype() == DT_VARIANT) {
    attr.set_on_host(true);
  }
  TF_RETURN_IF_ERROR(
      ctx->allocate_output(output_idx, t->shape(), &output, attr));
  if (t->dtype() == DT_VARIANT) {
    output->flat<Variant>() = t->flat<Variant>();
  } else if (ctx->op_device_context() != nullptr) {
    // TODO(apassos): remove the down_cast by just returning Device* from
    // OpKernelContext
    Device* device = static_cast<Device*>(ctx->device());
    ctx->op_device_context()->CopyTensorInSameDevice(
        t, device, output, [&n, &status](const Status& s) {
          status = s;
          n.Notify();
        });
    n.WaitForNotification();
    return status;
  } else {
    switch (t->dtype()) {
#define HANDLER(type)                       \
  case DataTypeToEnum<type>::value:         \
    output->flat<type>() = t->flat<type>(); \
    break;
      TF_CALL_ALL_TYPES(HANDLER);
#undef HANDLER
      default:
        return errors::Internal("Unsupported dtype", t->dtype());
    }
  }
  return Status::OK();
}

}  // namespace

void ReadVariableOp::Compute(OpKernelContext* ctx) {
  // 1.
  // ctx: OpKernelContext*
  //
  // class OpKernelContext
  // tensorflow/core/framework/op_kernel.h
  // * typedef std::pair<Allocator*, TrackingAllocator*> WrappedAllocator;
  // - struct Params
  //    * step_id: int64
  //    * op_kernel: OpKernel*
  //      ========================================================
  //      调试访问:  p ctx->op_kernel().def().DebugString()   # 重要
  //      ========================================================
  //    * device: DeviceBase*
  //    * eigen_gpu_device: PerOpGpuDevice*
  //    * track_allocations: bool
  //    * log_memory: bool
  //    * record_tensor_accesses: bool
  //    * output_attr_array: const AllocatorAttributes*
  //    * resource_manager: ResourceMgr*  # 这个和 tensor 是怎么联系起来的？
  //    * step_container: ScopedStepContainer*
  //      Per-step resources accessible by this op kernel invocation should be stored in this container.
  //    * rendezvous: Rendezvous*
  //      Mechanism used by this op kernel invocation to communicate with computations running on other devices.
  //    * collective_executor: CollectiveExecutor*
  //      Mechanism for executing a collective op that needs to coordinate with parallel instances running on other devices.
  //    * session_state: SessionState*
  //      The session state for this op.
  //    * session_handle: string
  //      Unique session identifier. Can be empty.
  //    * tensor_store: TensorStore*  # 留意这个
  //      The tensor store for this op.
  //    * cancellation_manager: CancellationManager*
  //    * inputs: const gtl::InlinedVector<TensorValue, 4>*  # 关注一下
  //    * is_input_dead: bool
  //    * input_alloc_attrs: const gtl::InlinedVector<AllocatorAttributes, 4>*
  //    * input_device_contexts: const gtl::InlinedVector<DeviceContext*, 4>*
  //    * op_device_context: DeviceContext*
  //    * frame_iter: FrameAndIter
  //      Control-flow op supports.
  //    * call_frame: CallFrameInterface*
  //    * function_library: FunctionLibraryRuntime*
  //    * runner: std::function<void(std::function<void()>)>*
  //    * stats_collector: StepStatsCollectorInterface*
  //    * graph_collector: GraphCollector*
  //    * slice_reader_cache: checkpoint::TensorSliceReaderCacheWrapper*
  //    * forward_from_array: const int*
  //    * inc_num_deferred_ops_function: std::function<void()>
  //    * dec_num_deferred_ops_function: std::function<void()>
  // - params_: Params*
  // - status_: Status
  // - wrapped_allocators_: gtl::InlinedVector<WrappedAllocator, 4>
  // - outputs_: gtl::InlinedVector<TensorValue, 4>
  // - referenced_tensors_: ManualConstructor<UniqueTensorReferences>
  // - temp_memory_allocated_: int64
  // - persistent_memory_allocated_: int64
  // - temp_tensor_buffer_and_size_: std::unique_ptr<gtl::InlinedVector<std::pair<const void*, int64>, 2>>
  // - persistent_alloc_ids_: std::unique_ptr<gtl::InlinedVector<int64, 2>>

  // 1.1
  // class TensorStore 数据结构
  // tensorflow/core/framework/session_state.h
  // The tensor store remembers the tensors we choose to keep for the
  // current run call. It is available to every op kernel.

  // 1.2
  // TensorValue 数据结构
  // tensorflow/core/framework/op_kernel.h:513:struct TensorValue
  // 概述:
  // Holds a tensor or tensor reference. For tensor references, we need
  // a mutex to prevent concurrent access to the tensor.
  //
  // - tensor: Tensor*
  // - mutex_if_ref: mutex*

  // 1.3
  // class OpKernel 数据结构
  // - kInitialCostEstimateCycles: static const uint64 , default_value: 100 * 1000 * 1000
  // - kOpIsExpensiveThresholdCycles: static const uint64, default_value: 5000
  // - kCostDecay: static const uint64, default_value: 10
  //
  // - def_: const std::unique_ptr<const NodeDef>
  // - input_types_: const DataTypeVector
  // - input_memory_types_: const MemoryTypeVector
  // - output_types_: const DataTypeVector
  // - output_memory_types_: const MemoryTypeVector
  // - graph_def_version_: const int
  // - is_internal_: const bool
  // - input_name_map_: NameRangeMap
  // - output_name_map_: NameRangeMap
  // - expensive_: bool
  // - cost_estimate_: std::atomic_uint_fast64_t
  //
  // - DebugString 等价的有
  // def().DebugString()

  Var* variable = nullptr;
  const ResourceHandle& handle = HandleFromInput(ctx, 0);
  const auto status = LookupResource(ctx, handle, &variable);
  // 1.
  // LookupResource
  // tensorflow/core/framework/resource_mgr.h
  //
  // template <typename T, bool use_dynamic_cast>
  // Status LookupResource(OpKernelContext* ctx,     // 为什么查找还要 OpKernelContext ? ==> 因为 需要 ctx->resource_manager() 得到 resource_mgr
  //                       const ResourceHandle& p,
  //                       T** value) {
  //   TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));
  //   return ctx->resource_manager()->Lookup<T, use_dynamic_cast>(p.container(),
  //                                                               p.name(),
  //                                                               value);
  // }

  // 2.
  // class ResourceHandle 数据结构
  // tensorflow/core/framework/resource_handle.h
  //
  // 概述:
  // Class representing a handle to a tensorflow resource.
  //
  // - device_: string
  // - container_: string
  // - name_: string
  // - hash_code_: uint64
  // - maybe_type_name_: string

  // 3.
  // 为什么查找还要 OpKernelContext ? ==> 因为 需要 ctx->resource_manager() 得到 resource_mgr

  OP_REQUIRES(ctx, status.ok(),
              errors::FailedPrecondition(
                  "Error while reading resource variable ", handle.name(),
                  " from Container: ", handle.container(),
                  ". This could mean that the variable was uninitialized. ",
                  status.ToString()));

  core::ScopedUnref s(variable);
  // 1.
  // class ScopedUnref
  // tensorflow/core/lib/core/refcount.h
  // 核心功能：
  // out of scope 后 variable 自动 Unref()


  // We're acquiring a reference to the underlying buffer while
  // holding a shared lock to guarantee ordering of reads and
  // writes.
  tf_shared_lock ml(*variable->mu());
  const Tensor* t = variable->tensor();

  OP_REQUIRES(ctx, dtype_ == t->dtype(),
              errors::InvalidArgument(
                  "Trying to read variable with wrong dtype. Expected ",
                  DataTypeString(dtype_), " got ", DataTypeString(t->dtype())));
  if (variable->copy_on_read_mode.load()) {
    // copy_on_read_mode 函数说明
    // core/framework/resource_var.h
    //
    // class Var::copy_on_read_mode
    // Also fake-guarded by mu_. Should be set to True whenever any sparse
    // operation uses the variable. Once this is true no tensor is allowed to
    // alias the memory of the variable, and we always copy the variable on
    // reads. This allows sparse operations to happen with only a shared lock if
    // so desired.
    // std::atomic<bool> copy_on_read_mode{false};

    OP_REQUIRES_OK(ctx, CopyVariable(0, ctx, t));
  } else {
    // ===进入的是这个分支===
    ctx->set_output(0, *t);
  }
}

ReadVariablesOp::ReadVariablesOp(OpKernelConstruction* c) : OpKernel(c) {
  int n;
  OP_REQUIRES_OK(c, c->GetAttr("N", &n));
  OP_REQUIRES_OK(c, c->GetAttr("dtypes", &dtypes_));
  OP_REQUIRES(c, n == dtypes_.size(),
              errors::InvalidArgument(
                  "Mismatched number of arguments to ReadVariablesOp (", n,
                  " vs. ", dtypes_.size(), ")"));
}

void ReadVariablesOp::Compute(OpKernelContext* ctx) {
  std::vector<std::unique_ptr<Var, core::RefCountDeleter>> variables(
      dtypes_.size());
  std::vector<const ResourceHandle*> handles(dtypes_.size());
  for (size_t i = 0; i < dtypes_.size(); ++i) {
    handles[i] = &HandleFromInput(ctx, i);
  }

  OP_REQUIRES_OK(ctx, LookupResources(ctx, handles, &variables));

  std::vector<string> uninitialized_vars;
  for (int64 i = 0; i < variables.size(); i++) {
    if (variables[i] == nullptr) {
      uninitialized_vars.push_back(handles[i]->name());
    }
  }

  OP_REQUIRES(
      ctx, uninitialized_vars.empty(),
      errors::InvalidArgument("In ReadVariableOp the following variables were "
                              "found uninitialized: ",
                              absl::StrJoin(uninitialized_vars, ", ")));

  for (size_t i = 0; i < dtypes_.size(); ++i) {
    // We're acquiring a reference to the underlying buffer while
    // holding a shared lock to guarantee ordering of reads and
    // writes.
    tf_shared_lock ml(*variables[i]->mu());
    OP_REQUIRES(ctx, dtypes_[i] == variables[i]->tensor()->dtype(),
                errors::InvalidArgument(
                    "Trying to read variable ", handles[i]->name(),
                    " from Container: ", handles[i]->container(),
                    " with wrong dtype. Expected ", DataTypeString(dtypes_[i]),
                    " got ", DataTypeString(variables[i]->tensor()->dtype())));
    if (variables[i]->copy_on_read_mode.load()) {
      OP_REQUIRES_OK(ctx, CopyVariable(i, ctx, variables[i]->tensor()));
    } else {
      const Tensor& t = *variables[i]->tensor();
      ctx->set_output(i, t);
    }
  }
}

REGISTER_KERNEL_BUILDER(Name("ReadVariableOp").Device(DEVICE_CPU),
                        ReadVariableOp);
REGISTER_KERNEL_BUILDER(Name("_ReadVariablesOp").Device(DEVICE_CPU),
                        ReadVariablesOp);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("ReadVariableOp").Device(DEVICE_GPU).HostMemory("resource"),
    ReadVariableOp);
REGISTER_KERNEL_BUILDER(
    Name("_ReadVariablesOp").Device(DEVICE_GPU).HostMemory("resources"),
    ReadVariablesOp);

#define REGISTER_GPU_KERNELS(type)                             \
  namespace functor {                                          \
  template <>                                                  \
  void DenseUpdate<GPUDevice, type, ASSIGN>::operator()(       \
      const GPUDevice& d, typename TTypes<type>::Flat lhs,     \
      typename TTypes<type>::ConstFlat rhs);                   \
  extern template struct DenseUpdate<GPUDevice, type, ASSIGN>; \
  }                                                            \
  REGISTER_KERNEL_BUILDER(Name("VarHandleOp")                  \
                              .Device(DEVICE_GPU)              \
                              .HostMemory("resource")          \
                              .TypeConstraint<type>("dtype"),  \
                          ResourceHandleOp<Var>)
TF_CALL_GPU_ALL_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_int64(REGISTER_GPU_KERNELS);
TF_CALL_variant(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS

REGISTER_KERNEL_BUILDER(Name("_VarHandlesOp")
                            .Device(DEVICE_GPU)
                            .HostMemory("resources")
                            .TypeConstraint("dtypes",
                                            {DT_INT64, DT_COMPLEX64,
                                             DT_COMPLEX128, DT_HALF, DT_FLOAT,
                                             DT_DOUBLE, DT_BOOL, DT_VARIANT}),
                        ResourceHandlesOp<Var>);

#endif  // GOOGLE_CUDA

template <typename T>
class VariableShapeOp : public OpKernel {
 public:
  explicit VariableShapeOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    Var* variable = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &variable));
    core::ScopedUnref s(variable);
    variable->mu()->lock_shared();
    TensorShape shape = variable->tensor()->shape();
    variable->mu()->unlock_shared();
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {shape.dims()}, &output));
    for (int i = 0; i < shape.dims(); ++i) {
      output->flat<T>()(i) = shape.dim_size(i);
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("VariableShape").Device(DEVICE_CPU).TypeConstraint<int32>("out_type"),
    VariableShapeOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("VariableShape").Device(DEVICE_CPU).TypeConstraint<int64>("out_type"),
    VariableShapeOp<int64>);

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("VariableShape")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("out_type")
                            .HostMemory("output")
                            .HostMemory("input"),
                        VariableShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("VariableShape")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int64>("out_type")
                            .HostMemory("output")
                            .HostMemory("input"),
                        VariableShapeOp<int64>);

#endif  // GOOGLE_CUDA

DestroyResourceOp::DestroyResourceOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr("ignore_lookup_error", &ignore_lookup_error_));
}

void DestroyResourceOp::Compute(OpKernelContext* ctx) {
  const ResourceHandle& p = HandleFromInput(ctx, 0);
  Status status = DeleteResource(ctx, p);
  if (ignore_lookup_error_ && errors::IsNotFound(status)) {
    return;
  }
  OP_REQUIRES_OK(ctx, status);
}

REGISTER_KERNEL_BUILDER(Name("DestroyResourceOp").Device(DEVICE_CPU),
                        DestroyResourceOp);
REGISTER_KERNEL_BUILDER(
    Name("DestroyResourceOp").Device(DEVICE_GPU).HostMemory("resource"),
    DestroyResourceOp);

template <typename Device, typename T>
class AssignVariableOp : public OpKernel {
 public:
  explicit AssignVariableOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    if (!c->GetAttr("_grappler_relax_allocator_constraints",
                    &relax_constraints_)
             .ok()) {
      relax_constraints_ = false;
    }
  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, dtype_ == context->input(1).dtype(),
                errors::InvalidArgument(
                    "Variable and value dtypes don't match; respectively, ",
                    DataTypeString(dtype_), " and ",
                    DataTypeString(context->input(1).dtype())));
    Var* variable = nullptr;
    const Tensor& value = context->input(1);
    // Note: every resource-variable-manipulating op assumes copy-on-write
    // semantics, and creates a copy of the variable's Tensor if its refcount is
    // bigger than 1 when we try to modify it. This means we never need to copy
    // the original tensor for AssignVariableOp; even if there are other live
    // users of it we know none can modify it so this is always safe (even in
    // esoteric cases where the same tensor is used to initialize multiple
    // variables or the tensor is a constant this is safe, as future writes will
    // trigger copies).
    OP_REQUIRES_OK(context, LookupOrCreateResource<Var>(
                                context, HandleFromInput(context, 0), &variable,
                                [this, &value](Var** ptr) {
                                  *ptr = new Var(dtype_);
                                  *(*ptr)->tensor() = value;
                                  (*ptr)->is_initialized = true;
                                  return Status::OK();
                                }));
    core::ScopedUnref s(variable);
    mutex_lock ml(*variable->mu());
    OP_REQUIRES(context, variable->tensor()->dtype() == dtype_,
                errors::InvalidArgument(
                    "Trying to assign variable with wrong dtype. Expected ",
                    DataTypeString(variable->tensor()->dtype()), " got ",
                    DataTypeString(dtype_)));
    if (variable->copy_on_read_mode.load()) {
      PersistentTensor unused;
      Tensor* tmp;
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      attr.set_nic_compatible(true);
      OP_REQUIRES_OK(context,
                     context->allocate_persistent(value.dtype(), value.shape(),
                                                  &unused, &tmp, attr));
      functor::DenseUpdate<Device, T, ASSIGN> copy_functor;
      copy_functor(context->eigen_device<Device>(), tmp->flat<T>(),
                   value.flat<T>());
      *variable->tensor() = *tmp;
    } else {
      *variable->tensor() = value;
    }
    variable->is_initialized = true;
  }

 private:
  DataType dtype_;
  bool relax_constraints_;
};

template <typename Device>
class AssignVariableOp<Device, Variant> : public OpKernel {
 public:
  explicit AssignVariableOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES(c, dtype_ == DT_VARIANT,
                errors::Internal("Variant kernel called with dtype: ",
                                 DataTypeString(dtype_)));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& value = context->input(1);
    Var* variable = nullptr;
    OP_REQUIRES_OK(context, LookupOrCreateResource<Var>(
                                context, HandleFromInput(context, 0), &variable,
                                [](Var** ptr) {
                                  // Created on host.
                                  *ptr = new Var(DT_VARIANT);
                                  return Status::OK();
                                }));
    core::ScopedUnref s(variable);

    // For purposes of forwarding DT_VARIANT, we want the least
    // restrictive attr; we already know the input is on host.
    AllocatorAttributes attr;

    // Copying is unnecessary if we are the last user of the value
    // tensor, we can just adopt the input tensor's buffer instead.
    // Note that Variant objects themselves always reside on host.
    //
    // We nevertheless want to signal to the runtime that the tensor
    // should reside in memory of the associated device, as Variant
    // tensors may be marked as sitting on either CPU or GPU.  This
    // helps to elide one or more copies.
    std::unique_ptr<Tensor> input_alias = context->forward_input(
        1, OpKernelContext::Params::kNoReservation /*output_index*/, DT_VARIANT,
        value.shape(),
        DEVICE_MEMORY /* HOST_MEMORY is only reserved for special cases */,
        attr);

    mutex_lock ml(*variable->mu());
    OP_REQUIRES(context, variable->tensor()->dtype() == DT_VARIANT,
                errors::InvalidArgument(
                    "Trying to assign variable with wrong dtype. Expected ",
                    DataTypeString(variable->tensor()->dtype()), " got ",
                    DataTypeString(DT_VARIANT)));
    variable->is_initialized = true;
    *variable->tensor() = Tensor(DT_VARIANT, value.shape());

    if (input_alias) {
      *variable->tensor() = *input_alias;
      return;
    }

    // Need to copy, but maybe we can re-use variable's buffer?
    if (!variable->tensor()->RefCountIsOne() ||
        !variable->tensor()->shape().IsSameSize(value.shape())) {
      PersistentTensor unused;
      Tensor* tmp;
      // Allocation of DT_VARIANT is always on host.
      attr.set_on_host(true);
      OP_REQUIRES_OK(context,
                     context->allocate_persistent(DT_VARIANT, value.shape(),
                                                  &unused, &tmp, attr));
      *variable->tensor() = *tmp;
    }

    const auto elements_in = value.flat<Variant>();
    auto elements_out = variable->tensor()->flat<Variant>();
    for (int64 i = 0; i < elements_in.size(); ++i) {
      elements_out(i) = elements_in(i);
    }
  }

 private:
  DataType dtype_;
};

#define REGISTER_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("AssignVariableOp")            \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("dtype"), \
                          AssignVariableOp<Eigen::ThreadPoolDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNELS(type)                           \
  REGISTER_KERNEL_BUILDER(Name("AssignVariableOp")           \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("resource"),       \
                          AssignVariableOp<GPUDevice, type>);

TF_CALL_GPU_ALL_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_int64(REGISTER_GPU_KERNELS);
TF_CALL_variant(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA

template <typename Device, typename T, DenseUpdateType Op>
class AssignUpdateVariableOp : public OpKernel {
 public:
  explicit AssignUpdateVariableOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* context) override {
    Var* variable = nullptr;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &variable));
    core::ScopedUnref s(variable);

    const Tensor& value = context->input(1);
    // TODO(apassos): We could possibly avoid the copy done by
    // PrepareToUpdateVariable() for commutative operations like Op ==
    // ADD if value's refcount was 1.
    mutex_lock ml(*variable->mu());
    Tensor* var_tensor = variable->tensor();
    OP_REQUIRES(context, var_tensor->shape().IsSameSize(value.shape()),
                errors::InvalidArgument("Cannot update variable with shape ",
                                        var_tensor->shape().DebugString(),
                                        " using a Tensor with shape ",
                                        value.shape().DebugString(),
                                        ", shapes must be equal."));
    OP_REQUIRES_OK(
        context, PrepareToUpdateVariable<Device, T>(
                     context, var_tensor, variable->copy_on_read_mode.load()));
    functor::DenseUpdate<Device, T, Op> update_functor;
    update_functor(context->eigen_device<Device>(), var_tensor->flat<T>(),
                   value.flat<T>());
  }
};

#define REGISTER_KERNELS(type)                                     \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("AssignAddVariableOp")                                  \
          .Device(DEVICE_CPU)                                      \
          .TypeConstraint<type>("dtype"),                          \
      AssignUpdateVariableOp<Eigen::ThreadPoolDevice, type, ADD>); \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("AssignSubVariableOp")                                  \
          .Device(DEVICE_CPU)                                      \
          .TypeConstraint<type>("dtype"),                          \
      AssignUpdateVariableOp<Eigen::ThreadPoolDevice, type, SUB>);

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(Name("AssignAddVariableOp")                    \
                              .Device(DEVICE_GPU)                        \
                              .HostMemory("resource")                    \
                              .TypeConstraint<type>("dtype"),            \
                          AssignUpdateVariableOp<GPUDevice, type, ADD>); \
  REGISTER_KERNEL_BUILDER(Name("AssignSubVariableOp")                    \
                              .Device(DEVICE_GPU)                        \
                              .HostMemory("resource")                    \
                              .TypeConstraint<type>("dtype"),            \
                          AssignUpdateVariableOp<GPUDevice, type, SUB>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_int64(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA

class VarIsInitializedOp : public OpKernel {
 public:
  explicit VarIsInitializedOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* context) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    auto output_tensor = output->tensor<bool, 0>();
    Var* variable = nullptr;
    Status s = LookupResource(context, HandleFromInput(context, 0), &variable);
    if (!s.ok()) {
      output_tensor() = false;
      return;
    }
    core::ScopedUnref su(variable);
    mutex_lock ml(*variable->mu());
    output_tensor() = variable->is_initialized;
  }
};

REGISTER_KERNEL_BUILDER(Name("VarIsInitializedOp").Device(DEVICE_CPU),
                        VarIsInitializedOp);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("VarIsInitializedOp")
                            .Device(DEVICE_GPU)
                            .HostMemory("resource")
                            .HostMemory("is_initialized"),
                        IsResourceInitialized<Var>);
#endif  // GOOGLE_CUDA

template <typename Device, typename T, typename Index>
class ResourceGatherOp : public OpKernel {
 private:
  int32 batch_dims_ = 0;

  // Add the batch offset derrived from params to each batch of indices.
  // Example: batch_dims = 1, indices = [[0, 1, 2], [0, 1, 2]]
  // If indexing into a params dimension of size 4, then the indices will become
  // [0, 1, 2, 4, 5, 6]
  void AddBatchOffsets(Tensor* indices, const Tensor& params) {
    int64 batch_size = 1;  // The size of all batch dimensions.
    for (int idx = 0; idx < batch_dims_; ++idx) {
      batch_size *= params.dim_size(idx);
    }

    auto indices_flat = indices->flat<Index>();
    int64 const index_inner_size = indices->NumElements() / batch_size;
    int64 const batch_offset = params.dim_size(batch_dims_);
    for (int64 batch_idx = 0, dest_idx = 0; batch_idx < batch_size;
         ++batch_idx) {
      for (int64 idx = 0; idx < index_inner_size; ++idx) {
        indices_flat(dest_idx++) += batch_offset * batch_idx;
      }
    }
  }

 public:
  explicit ResourceGatherOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("batch_dims", &batch_dims_));
  }

  void Compute(OpKernelContext* c) override {
    Var* v = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
    core::ScopedUnref su(v);
    OP_REQUIRES_OK(c, EnsureSparseVariableAccess<Device, T>(c, v));
    // NOTE: We hold the lock for the whole gather operation instead
    // of increasing the reference count of v->tensor() to avoid a
    // situation where a write to the same variable will see a
    // reference count greater than one and make a copy of the
    // (potentially very large) tensor buffer.
    tf_shared_lock ml(*v->mu());
    const Tensor& params = *v->tensor();
    const Tensor& indices = c->input(1);
    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    // Check that we have enough index space
    const int64 N = indices.NumElements();
    OP_REQUIRES(
        c, params.dim_size(0) <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[0] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", params.dim_size(0), " > ",
                                std::numeric_limits<Index>::max()));

    // The result shape is params.shape[:batch_dims] +
    // indices.shape[batch_dims:] + params.shape[batch_dims+1:].
    TensorShape result_shape;
    for (int i = 0; i < batch_dims_; ++i) {
      result_shape.AddDim(params.dim_size(i));
    }
    for (int i = batch_dims_; i < indices.dims(); ++i) {
      result_shape.AddDim(indices.dim_size(i));
    }
    for (int i = batch_dims_ + 1; i < params.dims(); ++i) {
      result_shape.AddDim(params.dim_size(i));
    }

    Tensor* out = nullptr;
    Tensor tmp;
    if (params.dtype() == DT_VARIANT) {
      tmp = Tensor(DT_VARIANT, result_shape);
      c->set_output(0, tmp);
      out = &tmp;
    } else {
      OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));
    }

    if (N > 0) {
      Tensor tmp_indices;

      // Points to the original or updated (if batch_dims is set) indices.
      const Tensor* op_indices = &indices;
      if (batch_dims_ > 0) {
        OP_REQUIRES_OK(c, c->allocate_temp(indices.dtype(), indices.shape(),
                                           &tmp_indices));
        functor::DenseUpdate<Device, Index, ASSIGN> copy_functor;
        copy_functor(c->eigen_device<Device>(), tmp_indices.flat<Index>(),
                     indices.flat<Index>());

        AddBatchOffsets(&tmp_indices, params);
        op_indices = &tmp_indices;
      }

      int64 gather_dim_size = 1;
      for (int idx = 0; idx <= batch_dims_; ++idx) {
        gather_dim_size *= params.dim_size(idx);
      }
      int64 inner_size = 1;
      for (int i = batch_dims_ + 1; i < params.dims(); ++i) {
        inner_size *= params.dim_size(i);
      }
      auto params_flat = params.shaped<T, 3>({1, gather_dim_size, inner_size});
      const auto indices_flat = op_indices->flat<Index>();
      auto out_flat = out->shaped<T, 3>({1, N, out->NumElements() / N});

      functor::GatherFunctor<Device, T, Index> functor;
      int64 bad_i = functor(c, params_flat, indices_flat, out_flat);

      OP_REQUIRES(
          c, bad_i < 0,
          errors::InvalidArgument(
              "indices", SliceDebugString(indices.shape(), bad_i), " = ",
              indices_flat(bad_i), " is not in [0, ", params.dim_size(0), ")"));
    }
  }
};

#define REGISTER_GATHER_FULL(dev, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("ResourceGather")                       \
                              .Device(DEVICE_##dev)                    \
                              .HostMemory("resource")                  \
                              .TypeConstraint<type>("dtype")           \
                              .TypeConstraint<index_type>("Tindices"), \
                          ResourceGatherOp<dev##Device, type, index_type>)

#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, type, int32);      \
  REGISTER_GATHER_FULL(dev, type, int64)

#define REGISTER_GATHER_CPU(type) REGISTER_GATHER_ALL_INDICES(CPU, type)

// Registration of the CPU implementations.
TF_CALL_ALL_TYPES(REGISTER_GATHER_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_CPU);

// Registers GPU kernels.
#if GOOGLE_CUDA
#define REGISTER_GATHER_GPU(type) REGISTER_GATHER_ALL_INDICES(GPU, type)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GATHER_GPU);

// Variant objects themselves sit on CPU, even if they contain data
// pointing to a device.
REGISTER_KERNEL_BUILDER(Name("ResourceGather")
                            .Device(DEVICE_GPU)
                            .HostMemory("resource")
                            .HostMemory("indices")
                            .TypeConstraint<Variant>("dtype")
                            .TypeConstraint<int32>("Tindices"),
                        ResourceGatherOp<GPUDevice, Variant, int32>)
REGISTER_KERNEL_BUILDER(Name("ResourceGather")
                            .Device(DEVICE_GPU)
                            .HostMemory("resource")
                            .HostMemory("indices")
                            .TypeConstraint<Variant>("dtype")
                            .TypeConstraint<int64>("Tindices"),
                        ResourceGatherOp<GPUDevice, Variant, int64>)

#endif  // GOOGLE_CUDA

#undef REGISTER_GATHER_CPU
#undef REGISTER_GATHER_GPU
#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
class ResourceScatterUpdateOp : public OpKernel {
 public:
  explicit ResourceScatterUpdateOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    Var* v = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
    core::ScopedUnref unref_v(v);
    OP_REQUIRES_OK(c, EnsureSparseVariableAccess<Device, T>(c, v));
    tf_shared_lock ml(*v->mu());
    Tensor* params = v->tensor();
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);

    // Check that we have enough index space
    const int64 N_big = indices.NumElements();
    OP_REQUIRES(
        c, N_big <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("indices has too many elements for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", N_big, " > ",
                                std::numeric_limits<Index>::max()));
    const Index N = static_cast<Index>(N_big);
    OP_REQUIRES(
        c, params->dim_size(0) <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[0] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", params->dim_size(0), " > ",
                                std::numeric_limits<Index>::max()));

    if (N > 0) {
      auto indices_flat = indices.flat<Index>();
      auto params_flat = params->flat_outer_dims<T>();
      if (TensorShapeUtils::IsScalar(updates.shape())) {
        const auto update = updates.scalar<T>();

        functor::ScatterScalarFunctor<Device, T, Index, op> functor;
        const Index bad_i = functor(c, c->template eigen_device<Device>(),
                                    params_flat, update, indices_flat);
        OP_REQUIRES(c, bad_i < 0,
                    errors::InvalidArgument(
                        "indices", SliceDebugString(indices.shape(), bad_i),
                        " = ", indices_flat(bad_i), " is not in [0, ",
                        params->dim_size(0), ")"));
      } else {
        int64 num_updates = updates.NumElements();
        OP_REQUIRES(c, num_updates % N == 0,
                    errors::InvalidArgument(
                        "shape of indices (", indices.shape().DebugString(),
                        ") is not compatible with the shape of updates (",
                        updates.shape().DebugString(), ")"));
        auto updates_flat = updates.shaped<T, 2>({N, num_updates / N});

        functor::ScatterFunctor<Device, T, Index, op> functor;
        const Index bad_i = functor(c, c->template eigen_device<Device>(),
                                    params_flat, updates_flat, indices_flat);
        OP_REQUIRES(c, bad_i < 0,
                    errors::InvalidArgument(
                        "indices", SliceDebugString(indices.shape(), bad_i),
                        " = ", indices_flat(bad_i), " is not in [0, ",
                        params->dim_size(0), ")"));
      }
    }
  }
};

#define REGISTER_SCATTER_KERNEL_INDEX(type, index_type, dev, name, op) \
  REGISTER_KERNEL_BUILDER(                                             \
      Name(name)                                                       \
          .Device(DEVICE_##dev)                                        \
          .HostMemory("resource")                                      \
          .TypeConstraint<type>("dtype")                               \
          .TypeConstraint<index_type>("Tindices"),                     \
      ResourceScatterUpdateOp<dev##Device, type, index_type, op>)

#define REGISTER_SCATTER_KERNEL(type, dev, name, op)         \
  REGISTER_SCATTER_KERNEL_INDEX(type, int32, dev, name, op); \
  REGISTER_SCATTER_KERNEL_INDEX(type, int64, dev, name, op);

#define REGISTER_SCATTER_ARITHMETIC(type, dev)                \
  REGISTER_SCATTER_KERNEL(type, dev, "ResourceScatterAdd",    \
                          scatter_op::UpdateOp::ADD);         \
  REGISTER_SCATTER_KERNEL(type, dev, "ResourceScatterSub",    \
                          scatter_op::UpdateOp::SUB);         \
  REGISTER_SCATTER_KERNEL(type, dev, "ResourceScatterMul",    \
                          scatter_op::UpdateOp::MUL);         \
  REGISTER_SCATTER_KERNEL(type, dev, "ResourceScatterDiv",    \
                          scatter_op::UpdateOp::DIV);         \
  REGISTER_SCATTER_KERNEL(type, dev, "ResourceScatterUpdate", \
                          scatter_op::UpdateOp::ASSIGN);
#define REGISTER_SCATTER_MINMAX(type, dev)                 \
  REGISTER_SCATTER_KERNEL(type, dev, "ResourceScatterMin", \
                          scatter_op::UpdateOp::MIN);      \
  REGISTER_SCATTER_KERNEL(type, dev, "ResourceScatterMax", \
                          scatter_op::UpdateOp::MAX);

// Registers CPU kernels.
#define REGISTER_SCATTER_ARITHMETIC_CPU(type) \
  REGISTER_SCATTER_ARITHMETIC(type, CPU);
#define REGISTER_SCATTER_MINMAX_CPU(type) REGISTER_SCATTER_MINMAX(type, CPU);

TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ARITHMETIC_CPU);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_SCATTER_MINMAX_CPU);

REGISTER_SCATTER_KERNEL(string, CPU, "ResourceScatterUpdate",
                        scatter_op::UpdateOp::ASSIGN);
REGISTER_SCATTER_KERNEL(bool, CPU, "ResourceScatterUpdate",
                        scatter_op::UpdateOp::ASSIGN);
REGISTER_SCATTER_KERNEL(Variant, CPU, "ResourceScatterUpdate",
                        scatter_op::UpdateOp::ASSIGN);

// Registers GPU kernels.
#if GOOGLE_CUDA
#define REGISTER_SCATTER_ARITHMETIC_GPU(type) \
  REGISTER_SCATTER_ARITHMETIC(type, GPU);
#define REGISTER_SCATTER_MINMAX_GPU(type) REGISTER_SCATTER_MINMAX(type, GPU);

#define REGISTER_SCATTER_UPDATE_GPU(type) REGISTER_SCATTER_UPDATE(type, GPU);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_ARITHMETIC_GPU);
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SCATTER_MINMAX_GPU);

REGISTER_KERNEL_BUILDER(Name("ResourceScatterUpdate")
                            .Device(DEVICE_GPU)
                            .HostMemory("resource")
                            .HostMemory("indices")
                            .TypeConstraint<Variant>("dtype")
                            .TypeConstraint<int32>("Tindices"),
                        ResourceScatterUpdateOp<GPUDevice, Variant, int32,
                                                scatter_op::UpdateOp::ASSIGN>)
REGISTER_KERNEL_BUILDER(Name("ResourceScatterUpdate")
                            .Device(DEVICE_GPU)
                            .HostMemory("resource")
                            .TypeConstraint<bool>("dtype")
                            .TypeConstraint<int32>("Tindices"),
                        ResourceScatterUpdateOp<GPUDevice, bool, int32,
                                                scatter_op::UpdateOp::ASSIGN>)
REGISTER_KERNEL_BUILDER(Name("ResourceScatterUpdate")
                            .Device(DEVICE_GPU)
                            .HostMemory("resource")
                            .HostMemory("indices")
                            .TypeConstraint<Variant>("dtype")
                            .TypeConstraint<int64>("Tindices"),
                        ResourceScatterUpdateOp<GPUDevice, Variant, int64,
                                                scatter_op::UpdateOp::ASSIGN>)

#endif  // GOOGLE_CUDA

#undef REGISTER_SCATTER_ARITHMETIC
#undef REGISTER_SCATTER_ARITHMETIC_CPU
#undef REGISTER_SCATTER_MINMAX
#undef REGISTER_SCATTER_MINMAX_CPU
#undef REGISTER_SCATTER_KERNEL
#undef REGISTER_SCATTER_KERNEL_INDEX

}  // namespace tensorflow
