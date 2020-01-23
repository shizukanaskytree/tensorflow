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

#define EIGEN_USE_THREADS
#include "tensorflow/core/kernels/variable_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Resource stored by variables in the resource manager
// (legacy, ref-style version).
class LegacyVar : public ResourceBase {
  // 1.
  // 参考 :
  // class Var
  // class Var : public ResourceBase
  // tensorflow/core/framework/resource_var.h

 public:
  explicit LegacyVar(DataType dtype) : tensor_(dtype) {}
  // Not copyable or movable.
  LegacyVar(const LegacyVar&) = delete;
  LegacyVar& operator=(const LegacyVar&) = delete;

  mutex* mu() { return &mu_; }

  // =======================================================================
  Tensor* tensor() { return &tensor_; }
  // =======================================================================
  // 1.
  // tensor_ 变量说明:
  // tensor_: Tensor 就是 VariableOp 真真正正的存的 persistent 值.
  //
  // 比如:
  // 2019-10-04 20:25:10.277766:
  // I tensorflow/core/common_runtime/executor.cc:888]
  // Process node: 2 step 3 {{node W}} = VariableV2[_class=["loc:@assign_W"],
  // container="", dtype=DT_FLOAT, shape=[1], shared_name="",
  // _device="/job:localhost/replica:0/task:0/device:CPU:0"]()
  // device: /job:localhost/replica:0/task:0/device:CPU:0


  string DebugString() const override {
    return strings::StrCat(DataTypeString(tensor_.dtype()), "/",
                           tensor_.shape().DebugString());
  }
  // 1.
  // DebugString 函数用途
  // 在 tensorflow/core/framework/resource_mgr.cc 中的
  // Line l{
  //         &container,
  //         port::Demangle(type),
  //         &resource,
  //         q.second->DebugString()  // ResourceBase*
  //       };
  // 里面用到了

  // 2.
  // 打印:
  // "float/[25,15]"

 private:
  mutex mu_;
  // -----------------------------------------------------------------------
  Tensor tensor_;
  // -----------------------------------------------------------------------
  // 1.
  // tensor_ 说明:
  // 这个 tensor_ 的构造是在 VariableOp::Compute 内:
  // auto creator = [this](LegacyVar** var) { // output
  //   *var = new LegacyVar(dtype_); // 构造，作为输出
  //   (*var)->tensor()->set_shape(shape_);
  //   return Status::OK();
  // };
  // 1.
  // creator 等价
  // Status creator(LegacyVar** var)

  ~LegacyVar() override {}
};
// 1.
// class LegacyVar 数据结构
// class LegacyVar : public ResourceBase
// tensorflow/core/kernels/variable_ops.cc
// - tensor_: Tensor
// - mu_: mutex

// 2.
// class Var 数据结构
// class Var : public ResourceBase
// tensorflow/core/framework/resource_var.h
// - tensor_: Tensor
// - is_initialized: bool, default_value: false
// - copy_on_read_mode: std::atomic<bool>, default_value: false


VariableOp::VariableOp(OpKernelConstruction* context) : OpKernel(context) {
  // class OpKernelConstruction 数据结构
  // tensorflow/core/framework/op_kernel.h:264
  // 被构造是在 CreateOpKernel, tensorflow/core/framework/op_kernel.cc

  OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));

  dtype_ = RemoveRefType(context->output_type(0));
}


// -----------------------------------------------------------------------

void VariableOp::Compute(OpKernelContext* ctx) {
  mutex_lock l(init_mu_);

  if (!initialized_) {
    OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), // 初始化 cinfo_ 的 ResourceMgr*
                                    def(),
                                    true /* use name() */));
    // 1.
    // VariableOp::cinfo_ 变量说明:
    // cinfo_: ContainerInfo

    // 2.
    // class ContainerInfo 数据结构
    // tensorflow/core/framework/resource_mgr.h
    // - rmgr: ResourceMgr*, default: nullptr
    // - container_: string
    // - name_: string
    // - resource_is_private_to_kernel_: bool, default: false

    // 3.
    // ContainerInfo::Init 函数说明:
    // tensorflow/core/framework/resource_mgr.h
    // tensorflow/core/framework/resource_mgr.cc
    // Status Init(ResourceMgr* rmgr,
    //             const NodeDef& ndef,
    //             bool use_node_name_as_default);

    initialized_ = true;
  }

  // =======================================================================
  // 如果这个变量已经初始化过了，这次会取出在 resource_manager 里面的值
  // =======================================================================

  // 函数体
  auto creator = [this](LegacyVar** var) { // output
    *var = new LegacyVar(dtype_); // 构造，作为输出
    (*var)->tensor()->set_shape(shape_);
    return Status::OK();
  };
  // 1.
  // creator 等价
  // Status creator(LegacyVar** var)

  LegacyVar* var;

  OP_REQUIRES_OK(ctx,
                 cinfo_.resource_manager()->LookupOrCreate<LegacyVar>(
                                              cinfo_.container(), // input
                                              cinfo_.name(), // input
                                              &var, // output
                                              creator)); // input
  // 1.
  // cinfo_ 变量说明
  // class VariableOp : public OpKernel 类内的 成员变量
  // ContainerInfo cinfo_ GUARDED_BY(init_mu_);

  // 2.
  // class ContainerInfo 数据结构
  // tensorflow/core/framework/resource_mgr.h
  // - rmgr: ResourceMgr*, default: nullptr
  // - container_: string
  // - name_: string
  // - resource_is_private_to_kernel_: bool, default: false

  // 3.
  // cinfo_.resource_manager() 函数说明
  // ResourceMgr* resource_manager() const { return rmgr_; }
  // tensorflow/core/framework/resource_mgr.h
  // 函数返回 rmgr_: ResourceMgr*

  // 4.
  // LookupOrCreate 函数说明:
  //
  // template <typename T, bool use_dynamic_cast>
  // Status ResourceMgr::LookupOrCreate(const string& container,
  //                                    const string& name,
  //                                    T** resource,
  //                                    std::function<Status(T**)> creator)
  //
  // tensorflow/core/framework/resource_mgr.h

  // 5.
  // 打印
  // cinfo_.container(): "localhost"
  // cinfo_.name(): "W"
  //


  // Output a reference to our tensor, so it may be updated.
  //
  // As long as the resource manager hasn't been cleared the ref we return
  // here is valid because it owns a ref on var.
  ctx->set_output_ref(0, // input
                      var->mu(),  // input
                      var->tensor()); // input
  // 1.
  // ctx 变量说明:
  // ctx: OpKernelContext*
  // ======================================================================
  // 目的:
  // ctx 的 output_[0] 会指向 此 Op 内的 Tensor 值，这个 op compute 主要就是干这个，取值而已。
  // ======================================================================

  // 2.
  // set_output_ref 函数说明：
  // 概述:
  // 把 LegacyVar::tensor_ 的指针赋值给 OpKernelContext::outputs_, 类型是 gtl::InlinedVector<TensorValue, 4> outputs_;
  //
  // tensorflow/core/framework/op_kernel.cc:821:
  // void OpKernelContext::set_output_ref(
  //                         int index,
  //                         mutex* mu,
  //                         Tensor* tensor_for_ref)

  // 3.
  // QQQ. 为什么是硬编码 0 作为 index？
  // AAA. 因为它知道只有一个吧。
  // 2019-10-01 21:21:58.624203: I tensorflow/core/common_runtime/executor.cc:888] Process node: 3 step 1 {{node v4}} = VariableV2[_class=["loc:@v4/Assign"], container="", dtype=DT_FLOAT, shape=[25,15], shared_name="", _device="/job:localhost/replica:0/task:0/device:GPU:0"]() device: /job:localhost/replica:0/task:0/device:GPU:0

  // 4.
  // var->tensor() 变量说明:
  // 打印
  // p tensor_for_ref->DebugString() # 说明，我是进入 set_output_ref 后打印的。
  // $28 = "Tensor<type: float shape: [25,15] values: uninitialized Tensor of 375 elements of type 1>"

  if (ctx->track_allocations() && var->tensor()->IsInitialized()) {
    ctx->record_persistent_memory_allocation(
      var->tensor()->AllocatedBytes());
  }

  var->Unref();
}
/*
结束时我又打印了一次，这次不为空了。
p ctx->resource_manager()->DebugString()
$31 = "localhost            | N10tensorflow9LegacyVarE                 | v4                                       | float/[25,15]"

逐项对应

line.container->c_str(),
line.type.c_str(),
line.resource->c_str(),
line.detail.c_str()));

tensorflow/core/framework/resource_mgr.cc
*/


class TemporaryVariableOp : public OpKernel {
 public:
  explicit TemporaryVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
    // Variable name defaults to op name if not specified explicitly.
    if (var_name_.empty()) var_name_ = name();
  }

  void Compute(OpKernelContext* context) override {
    Status s;
    ResourceMgr* rm = context->resource_manager();
    OP_REQUIRES(context, rm, errors::Internal("No per-step resource manager."));
    auto* tmp_var = new TmpVar;
    OP_REQUIRES(context, tmp_var,
                errors::ResourceExhausted("Could not allocate TmpVar."));
    tmp_var->name = var_name_;
    s = context->allocate_temp(dtype_, shape_, &tmp_var->val);
    if (!s.ok()) tmp_var->Unref();
    OP_REQUIRES_OK(context, s);
    OP_REQUIRES_OK(context, rm->Create(context->step_container()->name(),
                                       var_name_, tmp_var));
    context->set_output_ref(0, &tmp_var->mu, &tmp_var->val);
    if (context->track_allocations()) {
      context->record_persistent_memory_allocation(
          tmp_var->val.AllocatedBytes());
    }
  }

 private:
  // Refcounted temporary variable resource.
  friend class DestroyTemporaryVariableOp;
  struct TmpVar : public ResourceBase {
    mutex mu;
    Tensor val;
    string name;
    string DebugString() const override { return name; }
    ~TmpVar() override { VLOG(3) << "TmpVar " << name << " deleted"; }
  };

  TensorShape shape_;
  DataType dtype_;
  string var_name_;
};

class DestroyTemporaryVariableOp : public OpKernel {
 public:
  explicit DestroyTemporaryVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES(context, IsRefType(context->input_type(0)),
                errors::InvalidArgument("lhs input needs to be a ref type"));
    OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
    OP_REQUIRES(context, !var_name_.empty(),
                errors::InvalidArgument("Missing var_name attribute"));
  }

  void Compute(OpKernelContext* context) override {
    // NOTE(pbar): All other mutators of the Tensor Ref *must* have completed
    // their execution before this DestroyTemporaryVariable op executes.
    // This is typically achieved using control dependencies.
    CHECK(IsRefType(context->input_dtype(0)));
    Tensor tmpvar = context->mutable_input(0, false);
    context->set_output(0, tmpvar);
    ResourceMgr* rm = context->resource_manager();
    OP_REQUIRES(context, rm, errors::Internal("No per-step resource manager."));
    OP_REQUIRES_OK(context, rm->Delete<TemporaryVariableOp::TmpVar>(
                                context->step_container()->name(), var_name_));
    if (context->track_allocations()) {
      context->record_persistent_memory_allocation(
          -static_cast<int64>(tmpvar.AllocatedBytes()));
    }
  }

 private:
  string var_name_;
};

class IsVariableInitializedOp : public OpKernel {
 public:
  explicit IsVariableInitializedOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Get a mutable input tensor of the Ref input.
    const Tensor& input_tensor = context->mutable_input(0, false);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    auto output_tensor = output->tensor<bool, 0>();
    bool result = input_tensor.IsInitialized();
    output_tensor() = result;
  }
};

REGISTER_KERNEL_BUILDER(Name("Variable").Device(DEVICE_CPU), VariableOp);
REGISTER_KERNEL_BUILDER(Name("VariableV2").Device(DEVICE_CPU), VariableOp);
REGISTER_KERNEL_BUILDER(Name("TemporaryVariable").Device(DEVICE_CPU),
                        TemporaryVariableOp);
REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable").Device(DEVICE_CPU),
                        DestroyTemporaryVariableOp);
REGISTER_KERNEL_BUILDER(Name("IsVariableInitialized").Device(DEVICE_CPU),
                        IsVariableInitializedOp);

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("Variable").Device(DEVICE_SYCL).TypeConstraint<type>("dtype"),   \
      VariableOp);                                                          \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("VariableV2").Device(DEVICE_SYCL).TypeConstraint<type>("dtype"), \
      VariableOp);                                                          \
  REGISTER_KERNEL_BUILDER(Name("TemporaryVariable")                         \
                              .Device(DEVICE_SYCL)                          \
                              .TypeConstraint<type>("dtype"),               \
                          TemporaryVariableOp);                             \
  REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable")                  \
                              .Device(DEVICE_SYCL)                          \
                              .TypeConstraint<type>("T"),                   \
                          DestroyTemporaryVariableOp);                      \
  REGISTER_KERNEL_BUILDER(Name("IsVariableInitialized")                     \
                              .Device(DEVICE_SYCL)                          \
                              .TypeConstraint<type>("dtype")                \
                              .HostMemory("is_initialized"),                \
                          IsVariableInitializedOp);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SYCL_KERNEL);
#undef REGISTER_SYCL_KERNEL
#endif  // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
// Only register 'Variable' on GPU for the subset of types also supported by
// 'Assign' (see dense_update_ops.cc.)
#define REGISTER_GPU_KERNELS(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Variable").Device(DEVICE_GPU).TypeConstraint<type>("dtype"),   \
      VariableOp);                                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("VariableV2").Device(DEVICE_GPU).TypeConstraint<type>("dtype"), \
      VariableOp);                                                         \
  REGISTER_KERNEL_BUILDER(Name("TemporaryVariable")                        \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("dtype"),              \
                          TemporaryVariableOp);                            \
  REGISTER_KERNEL_BUILDER(Name("DestroyTemporaryVariable")                 \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("T"),                  \
                          DestroyTemporaryVariableOp);                     \
  REGISTER_KERNEL_BUILDER(Name("IsVariableInitialized")                    \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("dtype")               \
                              .HostMemory("is_initialized"),               \
                          IsVariableInitializedOp);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_int64(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
