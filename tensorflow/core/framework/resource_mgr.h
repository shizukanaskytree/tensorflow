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

#ifndef TENSORFLOW_CORE_FRAMEWORK_RESOURCE_MGR_H_
#define TENSORFLOW_CORE_FRAMEWORK_RESOURCE_MGR_H_

#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// A ResourceMgr instance keeps track of named and typed resources
// grouped into containers.
//
// Each resource must be represented as a sub-class of ResourceBase,
// which is reference counted explicitly.  Each named resource is
// registered with ResourceMgr under a named "container" name. At any
// time, there is at most one instance of a resource given the container
// name, the resource type and the resource name.
//
// All resources for a given container can be dropped by one call of
// Cleanup().
//
// E.g.,
//   struct MyVar : public ResourceBase {
//     mutex mu;
//     Tensor val;
//   }
//
//   ResourceMgr rm;
//
//   // Create a var.
//   MyVar* my_var = new MyVar;
//   my_var->val = Tensor(DT_FLOAT, my_shape);
//   my_var->val.flat<float>().setZeros();   // 0 initialized.
//   ctx->SetStatus(rm.Create("my_container", "my_name", my_var));
//
//   // Create 函数说明
//   // tensorflow/core/framework/resource_mgr.h
//   //
//   // Status ResourceMgr::Create(
//   //                       const string& container,
//   //                       const string& name,
//   //                       T* resource)
//
//   // += a variable.
//   MyVar* my_var = nullptr;
//   Status s = rm.Lookup("my_container", "my_name", &my_var);
//   if (s.ok()) {
//     my_var->val.flat<float>() += grad;
//   }
//   my_var->Unref();   // Or use ScopedUnref().
//   ctx->SetStatus(s);

// 2.
// 使用的例子：
// tensorflow/core/common_runtime/device.cc:30:
// rmgr_ = new ResourceMgr(parsed_name_.job);

// 3.
// tensorflow/core/common_runtime/executor.cc:829:
// params.resource_manager = device->resource_manager();
class ResourceBase : public core::RefCounted {
 public:
  // Returns a debug string for *this.
  virtual string DebugString() const = 0;

  // Returns memory used by this resource.
  virtual int64 MemoryUsed() const { return 0; }
};
// 1.
// class ResourceBase 数据结构
// tensorflow/core/framework/resource_mgr.h
// 只有两个接口
// - DebugString()
//   Returns a debug string for *this.
// - MemoryUsed()
//   Returns memory used by this resource.

// Container used for per-step resources.
class ScopedStepContainer {
 public:
  // step_id: the unique ID of this step. Doesn't have to be sequential, just
  // has to be unique.
  // cleanup: callback to delete a container of this name.
  // prefix: optional string prefix to disambiguate step containers.
  ScopedStepContainer(const int64 step_id,
                      std::function<void(const string&)> cleanup)
      : name_(strings::StrCat("__per_step_", step_id)), cleanup_(cleanup) {}

  ScopedStepContainer(const int64 step_id,
                      std::function<void(const string&)> cleanup,
                      const string& prefix)
      : name_(strings::StrCat("__", prefix, "_per_step_", step_id)),
        cleanup_(cleanup) {}

  ~ScopedStepContainer() { cleanup_(name_); }

  const string& name() const { return name_; }

 private:
  const string name_;
  const std::function<void(const string&)> cleanup_;
};

// 1.
// +------------------------------------------------------------------------------------+
// |                                   containers_                                      |
// |------------------------------------------------------------------------------------|
// |                                                                                    |
// |                   +-----------------------------------------------------------+    |
// | container +------>|Container*                                                 |    |
// | (string)          |-----------------------------------------------------------|    |
// |                   |      +--------------------------------------------+       |    |
// |                   |      |(type.hash_code(), var_name) : ResourceBase*|       |    |
// |                   |      |   (unit64)        (string)                 |       |    |
// |                   |      +--------------------------------------------+       |    |
// |                   |                                                           |    |
// |                   |      +--------------------------------------------+       |    |
// |                   |      |(type.hash_code(), var_name) : ResourceBase*|       |    |
// |                   |      |   (unit64)        (string)                 |       |    |
// |                   |      +--------------------------------------------+       |    |
// |                   |                                                           |    |
// |                   |      +--------------------------------------------+       |    |
// |                   |      |(type.hash_code(), var_name) : ResourceBase*|       |    |
// |                   |      |   (unit64)        (string)                 |       |    |
// |                   |      +--------------------------------------------+       |    |
// |                   |                                                           |    |
// |                   +-----------------------------------------------------------+    |
// |                                                                                    |
// |                   +-----------------------------------------------------------+    |
// | container +------>|Container*                                                 |    |
// | (string)          |-----------------------------------------------------------|    |
// |                   |                                                           |    |
// |                   |                                                           |    |
// |                   |                                                           |    |
// |                   |                                                           |    |

// 2.
// 例子打印
// https://gist.github.com/shizukanaskytree/e7f6534f9d3a7ae533854b71d7100a52
// https://docs.google.com/document/d/1GXMiZe3c4IkE4IwmD9NYYOAdEeFiV727GaZTP44NO1U/edit


// 非常重要，ResourceMgr 和 persistent tensor , stateful op 相关.
class ResourceMgr {
 public:
  // 1.
  ResourceMgr();

  // 2.
  explicit ResourceMgr(const string& default_container);

  // 3.
  ~ResourceMgr();

  // 4.
  // Returns the default container name for *this.
  const string& default_container() const { return default_container_; }

  // Creates a resource "name" in the "container".  The caller transfers
  // the ownership of one ref on "resource" to *this
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr.
  // 5.
  template <typename T>
  Status Create(const string& container, const string& name,
                T* resource) TF_MUST_USE_RESULT;

  // If "container" has a resource "name", returns it in "*resource" and
  // the caller takes the ownership of one ref on "*resource".
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr
  // 6.
  template <typename T, bool use_dynamic_cast = false>
  Status Lookup(const string& container,
                const string& name,
                T** resource) const TF_MUST_USE_RESULT;

  // Similar to Lookup, but looks up multiple resources at once, with only a
  // single lock acquisition.  If containers_and_names[i] is uninitialized
  // then this function does not modify resources[i].
  // 7.
  template <typename T, bool use_dynamic_cast = false>
  Status LookupMany(
    absl::Span<std::pair<const string*,const string*> const> containers_and_names,
    std::vector<std::unique_ptr<T, core::RefCountDeleter>>* resources) const TF_MUST_USE_RESULT;

  // If "container" has a resource "name", returns it in
  // "*resource". Otherwise, invokes creator() to create the resource.
  // The caller takes the ownership of one ref on "*resource".
  //
  // WARNING: creator() must not call any methods on ResourceMgr during its
  // execution, because a non-reentrant lock is held during the creator() call
  // in order to guarantee atomicity of LookupOrCreate().
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr
  // 8.
  template <typename T, bool use_dynamic_cast = false>
  Status LookupOrCreate(const string& container,
                        const string& name,
                        T** resource,
                        std::function<Status(T**)> creator) TF_MUST_USE_RESULT;

  // Deletes the resource "name" from the "container".
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // 9.
  template <typename T>
  Status Delete(const string& container, const string& name) TF_MUST_USE_RESULT;

  // 10.
  // Deletes the resource pointed by "handle".
  Status Delete(const ResourceHandle& handle) TF_MUST_USE_RESULT;

  // 11.
  // Deletes all resources from the "container" and removes the container.
  Status Cleanup(const string& container) TF_MUST_USE_RESULT;
  // 1.
  // Cleanup 函数说明:
  // tensorflow/core/framework/resource_mgr.cc

  // 12.
  // Deletes all resources in all containers.
  void Clear();

  // 13.
  // Returns a text description for all resources.
  string DebugString() const;

 private:

  // 14.
  typedef std::pair<uint64, string> Key;

  // 15.
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      return Hash64(k.second.data(), k.second.size(), k.first);
    }
  };

  // 16.
  struct KeyEqual {
    bool operator()(const Key& x, const Key& y) const {
      return (x.second == y.second) && (x.first == y.first);
    }
  };

  // 17.
  typedef std::unordered_map<Key, ResourceBase*, KeyHash, KeyEqual> Container;

  // 18.
  const string default_container_;

  // 19.
  mutable mutex mu_;

  // 20.
  // -----------------------------------------------------------------------
  std::unordered_map<string, Container*> containers_ GUARDED_BY(mu_);
  // -----------------------------------------------------------------------
  //                    |        |
  //       container name
  // 1.
  // Container 数据结构
  // tensorflow/core/framework/resource_mgr.h:193
  // typedef std::unordered_map<Key, ResourceBase*, KeyHash, KeyEqual> Container;
  //
  // 概述:
  // Container 里面有 Key 和 指向这个 resource 的指针

  // 2.
  // ResourceBase 数据结构
  // tensorflow/core/framework/resource_mgr.h:77:
  // class ResourceBase : public core::RefCounted
  //  - DebugString()
  //  - MemoryUsed()
  //
  // 说明:
  // 具体的数据结构会继承这个类，实现自己对应的构造函数。

  // 3.
  // Key 类型说明
  // typedef std::pair<uint64, string> Key;
  // - uint64 是 type.hash_code()
  // - name 是 variable name, 比如 "v4"
  // 实例:
  // {type.hash_code(), name}, resource}

  // 21.
  template <typename T, bool use_dynamic_cast = false>
  Status LookupInternal(const string& container, const string& name,
                        T** resource) const
      SHARED_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;

  // 22.
  Status DoCreate(const string& container, TypeIndex type, const string& name,
                  ResourceBase* resource)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;

  // 23.
  Status DoLookup(const string& container, TypeIndex type, const string& name,
                  ResourceBase** resource) const
      SHARED_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;

  // 24.
  Status DoDelete(const string& container, uint64 type_hash_code,
                  const string& resource_name,
                  const string& type_name) TF_MUST_USE_RESULT;
  // 25.
  Status DoDelete(const string& container, TypeIndex type,
                  const string& resource_name) TF_MUST_USE_RESULT;

  // 26.
  // Inserts the type name for 'hash_code' into the hash_code to type name map.
  Status InsertDebugTypeName(uint64 hash_code, const string& type_name)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;

  // Returns the type name for the 'hash_code'.
  // Returns "<unknown>" if a resource with such a type was never inserted into
  // the container.
  // 27.
  const char* DebugTypeName(uint64 hash_code) const
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // 28.
  // Map from type hash_code to type name.
  std::unordered_map<uint64, string> debug_type_names_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(ResourceMgr);
};
// 1.
// class ResourceMgr 数据结构
// * typedef std::pair<uint64, string> Key;
// * struct KeyHash functor
//    - std::size_t operator()(const Key& k)
//      return Hash64(k.second.data(), k.second.size(), k.first);
// * struct KeyEqual functor
//    - bool operator()(const Key& x, const Key& y)
//      return (x.second == y.second) && (x.first == y.first);
// * typedef std::unordered_map<Key, ResourceBase*, KeyHash, KeyEqual> Container;
// - default_container_: const string
// - mu_: mutable mutex
// - containers_: std::unordered_map<string, Container*>
// - debug_type_names_: std::unordered_map<uint64, string>

// 2.
// TODO:
// Bug Report:
// https://gist.github.com/shizukanaskytree/8702bb542e14b7f43de3c8551306f1e3


// Makes a resource handle with the specified type for a given container /
// name.
ResourceHandle MakeResourceHandle(OpKernelContext* ctx, const string& container,
                                  const string& name,
                                  const TypeIndex& type_index);

template <typename T>
ResourceHandle MakeResourceHandle(OpKernelContext* ctx, const string& container,
                                  const string& name) {
  return MakeResourceHandle(ctx, container, name, MakeTypeIndex<T>());
}

Status MakeResourceHandleToOutput(OpKernelContext* context, int output_index,
                                  const string& container, const string& name,
                                  const TypeIndex& type_index);

template <typename T>
ResourceHandle MakePerStepResourceHandle(OpKernelContext* ctx,
                                         const string& name);

// Returns a resource handle from a numbered op input.
const ResourceHandle& HandleFromInput(OpKernelContext* ctx, int input);
Status HandleFromInput(OpKernelContext* ctx, StringPiece input,
                       ResourceHandle* handle);

// Create a resource pointed by a given resource handle.
//
// If successful, the caller transfers the ownership of one ref on `resource` to
// `ctx->resource_mgr()`.
template <typename T>
Status CreateResource(OpKernelContext* ctx, const ResourceHandle& p, T* value);

// Looks up a resource pointed by a given resource handle.
//
// If the lookup is successful, the caller takes the ownership of one ref on
// `*value`, and must call its `Unref()` method when it has finished using it.
template <typename T, bool use_dynamic_cast = false>
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p, T** value);

// Looks up multiple resources pointed by a sequence of resource handles.  If
// p[i] is uninitialized then values[i] is unmodified.
template <typename T>
Status LookupResources(
    OpKernelContext* ctx, absl::Span<ResourceHandle const> p,
    std::vector<std::unique_ptr<T, core::RefCountDeleter>>* values);

// Looks up or creates a resource.
//
// If successful, the caller takes the ownership of one ref on `*value`, and
// must call its `Unref()` method when it has finished using it. If the
// `creator` is invoked, its reference on the created resource is transferred
// to `ctx->resource_mgr()`.
template <typename T>
Status LookupOrCreateResource(OpKernelContext* ctx, const ResourceHandle& p,
                              T** value, std::function<Status(T**)> creator);

// Destroys a resource pointed by a given resource handle.
template <typename T>
Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p);

// Same as above, but uses the hash code of the type directly.
// The type name information will be missing in the debug output when the
// resource is not present in the container.
Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p);

// Policy helper to decide which container/shared_name to use for a
// stateful kernel that accesses shared resource.
class ContainerInfo {
 public:
  // Analyze the node attribute of 'ndef' and decides the container and
  // resource name the kernel should use for accessing the shared
  // resource.
  //
  // 'ndef' is expected to have node attribute "container" and
  // "shared_name". Returns non-OK if they are not provided or they are
  // invalid.
  //
  // The policy is as following:
  // * If the attribute "container" is non-empty, it is used as is.
  //   Otherwise, uses the resource manager's default container.
  // * If the attribute "shared_name" is non-empty, it is used as is.
  //   Otherwise, if "use_node_name_as_default" is true, the kernel's
  //   node name is used as the resource name. Otherwise, a string
  //   unique to this process is used.
  Status Init(ResourceMgr* rmgr,
              const NodeDef& ndef,
              bool use_node_name_as_default);

  Status Init(ResourceMgr* rmgr,
              const NodeDef& ndef) {
    return Init(rmgr, ndef, false);
  }

  // The policy decides that the kernel should access the resource in
  // resource_manager(), the resource is in the container() and its
  // name is name().  If resource_is_private_to_kernel() is true, the
  // kernel should delete the resource when the kernel is deleted.
  ResourceMgr* resource_manager() const { return rmgr_; }
  const string& container() const { return container_; }
  const string& name() const { return name_; }
  bool resource_is_private_to_kernel() const {
    return resource_is_private_to_kernel_;
  }

  // Returns a readable string for *this.
  string DebugString() const;

 private:
  ResourceMgr* rmgr_ = nullptr;
  string container_;
  string name_;
  bool resource_is_private_to_kernel_ = false;
};
// class ContainerInfo 数据结构
// tensorflow/core/framework/resource_mgr.h
// - rmgr: ResourceMgr*, default: nullptr
// - container_: string
// - name_: string
// - resource_is_private_to_kernel_: bool, default: false


// Helper for kernels to obtain 'resource' from the
// ctx->resource_manager().
//
// "input_name" specifies the kernel's ref input which gives a string
// tensor with two elements, which specifies the container and
// resource name.
//
// Returns OK if the resource is found and transfers one ref of
// *resource to the caller. Otherwise, returns an error.
template <typename T>
Status GetResourceFromContext(OpKernelContext* ctx, const string& input_name,
                              T** resource);

// Utility op kernel to check if a handle to resource type T is initialized.
template <typename T>
class IsResourceInitialized : public OpKernel {
 public:
  explicit IsResourceInitialized(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override;
};

// Registers an op which produces just a resource handle to a resource of the
// specified type. The type will be a part of the generated op name.
// TODO(apassos): figure out how to get non-cpu-allocated tensors to work
// through constant folding so this doesn't have to be marked as stateful.
#define REGISTER_RESOURCE_HANDLE_OP(Type) \
  REGISTER_OP(#Type "HandleOp")           \
      .Attr("container: string = ''")     \
      .Attr("shared_name: string = ''")   \
      .Output("resource: resource")       \
      .SetIsStateful()                    \
      .SetShapeFn(tensorflow::shape_inference::ScalarShape)

// Utility op kernel to produce a handle to a resource of type T.
template <typename T>
class ResourceHandleOp : public OpKernel {
 public:
  explicit ResourceHandleOp(OpKernelConstruction* context);

  void Compute(OpKernelContext* ctx) override;

  bool IsExpensive() override { return false; }

 private:
  string container_;
  string name_;
  mutex mutex_;
  Tensor resource_;
  std::atomic<bool> initialized_{false};
};

// Utility op kernel to produce a handle to a resource of type T.
template <typename T>
class ResourceHandlesOp : public OpKernel {
 public:
  explicit ResourceHandlesOp(OpKernelConstruction* context);

  void Compute(OpKernelContext* ctx) override;

  bool IsExpensive() override { return false; }

 private:
  std::vector<string> containers_;
  std::vector<string> names_;
  mutex mutex_;
  std::vector<Tensor> resources_;
  std::atomic<bool> initialized_{false};
};

Status ResourceHandlesShape(shape_inference::InferenceContext* c);

// Registers a kernel for an op which produces a handle to a resource of the
// specified type.
#define REGISTER_RESOURCE_HANDLE_KERNEL(Type)                        \
  REGISTER_KERNEL_BUILDER(Name(#Type "HandleOp").Device(DEVICE_CPU), \
                          ResourceHandleOp<Type>)

// Implementation details below.

template <typename T>
void CheckDeriveFromResourceBase() {
  static_assert(std::is_base_of<ResourceBase, T>::value,
                "T must derive from ResourceBase");
}

template <typename T>
Status ResourceMgr::Create(
                      const string& container,
                      const string& name,
                      T* resource) {
  CheckDeriveFromResourceBase<T>();
  CHECK(resource != nullptr);
  mutex_lock l(mu_);

  return DoCreate(container, MakeTypeIndex<T>(), name, resource);
  // DoCreate 函数说明
  // tensorflow/core/framework/resource_mgr.cc
}


// =======================================================================
template <typename T, bool use_dynamic_cast>
Status ResourceMgr::Lookup(const string& container,
                           const string& name,
                           T** resource) const {
// =======================================================================
  // 1.
  // 打印:
  // name: "training/SGD/SGD/update_predictions/kernel/ResourceApplyKerasMomentum/ReadVariableOp_1"
  //   名字不一样！
  // op: "ReadVariableOp"
  // input: "training/SGD/momentum"  # 一样的
  // device: "/job:localhost/replica:0/task:0/device:GPU:0"
  // attr {
  //   key: "dtype"
  //   value {
  //     type: DT_FLOAT
  //   }
  // }
  // experimental_debug_info {
  //   original_node_names: "training/SGD/SGD/update_predictions/kernel/ResourceApplyKerasMomentum/ReadVariableOp_1"
  // }
  //
  // name: "training/SGD/SGD/update_predictions/bias/ResourceApplyKerasMomentum/ReadVariableOp_1"
  //   名字不一样！
  // op: "ReadVariableOp"
  // input: "training/SGD/momentum" # 一样的
  // device: "/job:localhost/replica:0/task:0/device:GPU:0"
  // attr {
  //   key: "dtype"
  //   value {
  //     type: DT_FLOAT
  //   }
  // }
  // experimental_debug_info {
  //   original_node_names: "training/SGD/SGD/update_predictions/bias/ResourceApplyKerasMomentum/ReadVariableOp_1"
  // }

  // 2.
  // message NodeDef::input 说明:
  //
  // Each input is "node:src_output" with "node" being a string name and
  // "src_output" indicating which output tensor to use from "node". If
  // "src_output" is 0 the ":0" suffix can be omitted.  Regular inputs
  // may optionally be followed by control inputs that have the format
  // "^node".
  // 所以，control edge 都是 带 "^" 的。
  // repeated string input = 3;

  CheckDeriveFromResourceBase<T>();
  tf_shared_lock l(mu_);
  // 1.
  // class SCOPED_LOCKABLE tf_shared_lock
  // tensorflow/core/platform/default/mutex.h

  // 2.
  // QQQ. 为什么需要锁?
  // AAA. 因为并行地访问 std::unordered_map 要带锁

  return LookupInternal<T, use_dynamic_cast>(container, name, resource);
  // 1.
  // LookupInternal 函数说明
  // framework/resource_mgr.h
}

template <typename T, bool use_dynamic_cast>
Status ResourceMgr::LookupMany(
    absl::Span<std::pair<const string*, const string*> const>
        containers_and_names,
    std::vector<std::unique_ptr<T, core::RefCountDeleter>>* resources) const {
  CheckDeriveFromResourceBase<T>();
  tf_shared_lock l(mu_);
  resources->resize(containers_and_names.size());
  for (size_t i = 0; i < containers_and_names.size(); ++i) {
    T* resource;
    Status s = LookupInternal<T, use_dynamic_cast>(
        *containers_and_names[i].first, *containers_and_names[i].second,
        &resource);
    if (s.ok()) {
      (*resources)[i].reset(resource);
    }
  }
  return Status::OK();
}

// Simple wrapper to allow conditional dynamic / static casts.
template <typename T, bool use_dynamic_cast>
struct TypeCastFunctor {
  static T* Cast(ResourceBase* r) { return static_cast<T*>(r); }
};

template <typename T>
struct TypeCastFunctor<T, true> {
  static T* Cast(ResourceBase* r) { return dynamic_cast<T*>(r); }
};


template <typename T, bool use_dynamic_cast>
Status ResourceMgr::LookupInternal(const string& container,
                                   const string& name,
                                   T** resource) const {
  ResourceBase* found = nullptr;

  Status s = DoLookup(container,
                      MakeTypeIndex<T>(),
                      name,
                      &found);
  // 1.
  // DoLookup 函数说明:
  // tensorflow/core/framework/resource_mgr.cc
  //
  // Status ResourceMgr::DoLookup(const string& container,
  //                              TypeIndex type,
  //                              const string& name,
  //                              ResourceBase** resource) const

  if (s.ok()) {
    // It's safe to down cast 'found' to T* since
    // typeid(T).hash_code() is part of the map key.
    *resource = TypeCastFunctor<T, use_dynamic_cast>::Cast(found);
  }
  return s;
}


template <typename T, bool use_dynamic_cast>
Status ResourceMgr::LookupOrCreate(const string& container, // input
                                   const string& name, // input
                                   T** resource, // output
                                   std::function<Status(T**)> creator) { // input
  // 1.
  // T 类型说明:
  // 对于 tensorflow/core/kernels/variable_ops.h
  // ptype T
  // type = class tensorflow::LegacyVar : public tensorflow::ResourceBase
  //
  // 打印
  // ptype T
  // type = class tensorflow::LegacyVar : public tensorflow::ResourceBase {
  //   private:
  //     tensorflow::mutex mu_;
  //     tensorflow::Tensor tensor_;
  //   public:
  //     LegacyVar(tensorflow::DataType);
  //     LegacyVar(const tensorflow::LegacyVar &);
  //     tensorflow::LegacyVar & operator=(const tensorflow::LegacyVar &);
  //     tensorflow::mutex * mu(void);
  //     tensorflow::Tensor * tensor(void);
  //     virtual std::string DebugString(void) const;
  //   private:
  //     ~LegacyVar();
  // }

  // 2.
  // creator 说明
  // Status creator(T**)
  // 函数体在 VariableOp::Compute, tensorflow/core/kernels/variable_ops.cc
  //
  //  auto creator = [this](LegacyVar** var) {
  //    *var = new LegacyVar(dtype_); // DataType dtype_;
  //    (*var)->tensor()->set_shape(shape_);
  //    return Status::OK();
  //  };

  // 3.
  // use_dynamic_cast 变量说明
  // p use_dynamic_cast
  // $16 = false


  CheckDeriveFromResourceBase<T>();
  // 1.
  // CheckDeriveFromResourceBase 函数概述:
  // 检查 T 这个 class 类型必须是继承自 class ResourceBase
  // "T must derive from ResourceBase"

  *resource = nullptr;

  // 1.
  // QQQ. 为什么要查找两次, LookupInternal ？
  // AAA. 使用的锁不一样，tf_shared_lock, mutex_lock

  Status s;
  {
    tf_shared_lock l(mu_);
    s = LookupInternal<T, use_dynamic_cast>(container, name, resource);
    // 1.
    // LookupInternal 函数说明:
    // tensorflow/core/framework/resource_mgr.h
    if (s.ok()) return s;
  }

  mutex_lock l(mu_); // 🔐要用

  s = LookupInternal<T, use_dynamic_cast>(container, name, resource);
  if (s.ok()) return s;

  TF_RETURN_IF_ERROR(creator(resource));
  // 1.
  // creator 函数定义在:
  // tensorflow/core/kernels/variable_ops.cc
  //
  // 目的:
  // 构造 LegacyVar 对象:
  //
  // auto creator = [this](LegacyVar** var) { // output
  //   *var = new LegacyVar(dtype_); // 构造，作为输出
  //   (*var)->tensor()->set_shape(shape_);
  //   return Status::OK();
  // };
  //
  // creator 等价
  // Status creator(T**)

  s = DoCreate(container, // input
               MakeTypeIndex<T>(), // input
               name, // input
               *resource); // input
  // 1.
  // DoCreate 函数说明:
  // tensorflow/core/framework/resource_mgr.cc
  //
  // 概述:
  // 把 *resource 这个指针放入 ResourceMgr d
  // ::
  //
  // ResourceMgr::DoCreate(const string& container,
  //                              TypeIndex type,
  //                              const string& name,
  //                              ResourceBase* resource)
  //
  // 打印
  // container: "localhost"
  // name: "v4",
  //  对应 v4 = tf.get_variable("v4", initializer = tf.random_normal([25,15],name='v4_random'))

  // 2.
  // TypeIndex 类型说明
  // typedef std::type_index TypeIndex;
  // framework/type_index.h
  // 用于记录 The class type_info holds implementation-specific information about a type, including the name of the type and means to compare two types for equality or collating order.
  // 例子:
  // https://en.cppreference.com/w/cpp/types/type_index

  // 3.
  // MakeTypeIndex 函数说明:
  // tensorflow/core/framework/type_index.h
  // inline TypeIndex MakeTypeIndex()

  if (!s.ok()) {
    return errors::Internal("LookupOrCreate failed unexpectedly");
  }

  // -----------------------------------------------------------------------
  (*resource)->Ref();
  // -----------------------------------------------------------------------
  // 1.
  // (*resource)->Ref() 说明:
  // 创造后就要增加引用，因为日后还要靠 (*resource)->Unref() 来释放资源

  return s;
}
// 1.
// 这个函数的感悟:
// 大函数切割成小函数可以有效地理清逻辑，使得代码好写很多很多。

template <typename T>
Status ResourceMgr::Delete(const string& container, const string& name) {
  CheckDeriveFromResourceBase<T>();
  return DoDelete(container, MakeTypeIndex<T>(), name);
}

template <typename T>
Status GetResourceFromContext(OpKernelContext* ctx, const string& input_name,
                              T** resource) {
  DataType dtype;
  TF_RETURN_IF_ERROR(ctx->input_dtype(input_name, &dtype));
  if (dtype == DT_RESOURCE) {
    const Tensor* handle;
    TF_RETURN_IF_ERROR(ctx->input(input_name, &handle));
    return LookupResource(ctx, handle->scalar<ResourceHandle>()(), resource);
  }
  string container;
  string shared_name;
  {
    mutex* mu;
    TF_RETURN_IF_ERROR(ctx->input_ref_mutex(input_name, &mu));
    mutex_lock l(*mu);
    Tensor tensor;
    TF_RETURN_IF_ERROR(ctx->mutable_input(input_name, &tensor, true));
    if (tensor.NumElements() != 2) {
      return errors::InvalidArgument(
          "Resource handle must have 2 elements, but had shape: ",
          tensor.shape().DebugString());
    }
    container = tensor.flat<string>()(0);
    shared_name = tensor.flat<string>()(1);
  }
  return ctx->resource_manager()->Lookup(container, shared_name, resource);
}

template <typename T>
ResourceHandle MakePerStepResourceHandle(OpKernelContext* ctx,
                                         const string& name) {
  return MakeResourceHandle<T>(ctx, ctx->step_container()->name(), name);
}

namespace internal {

Status ValidateDevice(OpKernelContext* ctx, const ResourceHandle& p);


template <typename T>
Status ValidateDeviceAndType(OpKernelContext* ctx,
                             const ResourceHandle& p) {
  // 1.
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


  TF_RETURN_IF_ERROR(internal::ValidateDevice(ctx, p));

  auto type_index = MakeTypeIndex<T>();
  // 1.
  // type_index 数据结构
  // 打印:
  // ptype type_index
  // type = struct std::type_index {
  //   private:
  //     const std::type_info *_M_target;
  //   public:
  //     type_index(const std::type_info &);
  //     bool operator==(const std::type_index &) const;
  //     bool operator!=(const std::type_index &) const;
  //     bool operator<(const std::type_index &) const;
  //     bool operator<=(const std::type_index &) const;
  //     bool operator>(const std::type_index &) const;
  //     bool operator>=(const std::type_index &) const;
  //     std::size_t hash_code(void) const; // 下面用到
  //     const char * name(void) const;
  // }

  // 2.
  // p type_index
  // $3 = {_M_target = 0x7f1fa3b64620 <typeinfo for tensorflow::Var>}

  if (type_index.hash_code() != p.hash_code()) {
    // 1.
    // type_index.hash_code() 函数说明:
    // struct std::type_index::hash_code()
    //
    // std::size_t hash_code(void) const;

    // 2.
    // p.hash_code() 函数说明:
    //
    // Hash code for the type of the resource. Is only valid in the same device
    // and in the same execution.
    // uint64 hash_code() const { return hash_code_; }
    // void set_hash_code(uint64 hash_code) { hash_code_ = hash_code; }

    return errors::InvalidArgument(
        "Trying to access resource using the wrong type. Expected ",
        p.maybe_type_name(), " got ", type_index.name());
  }
  return Status::OK();
}

}  // namespace internal

template <typename T>
Status CreateResource(OpKernelContext* ctx, const ResourceHandle& p, T* value) {
  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));
  return ctx->resource_manager()->Create(p.container(), p.name(), value);
}



template <typename T, bool use_dynamic_cast>
Status LookupResource(OpKernelContext* ctx,     // input
                      const ResourceHandle& p,  // input
                      T** value) {              // output

  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, // input
                                                        p)); // input

  return ctx->resource_manager()->Lookup<T, use_dynamic_cast>(p.container(), // input
                                                              p.name(), // input
                                                              value); // output
  // Lookup: framework/resource_mgr.h

  // 1.
  // T 打印实例
  // ptype T
  // type = class tensorflow::Var : public tensorflow::ResourceBase {
  //   public:
  //     bool is_initialized;
  //     std::atomic<bool> copy_on_read_mode;
  //   private:
  //     tensorflow::mutex mu_;
  //     tensorflow::Tensor tensor_;
  //
  //   public:
  //     Var(tensorflow::DataType);
  //   private:
  //     Var(const tensorflow::Var &);
  //   public:
  //     tensorflow::mutex * mu(void);
  //     tensorflow::Tensor * tensor(void);
  //     virtual std::string DebugString(void) const;
  //   private:
  //     ~Var();
  //     void operator=(const tensorflow::Var &);
  // }

  // 2.
  // use_dynamic_cast 变量说明
  // 打印: false

  // 3.
  // p 变量说明
  // p: const ResourceHandle

  // 4.
  // p 打印
  // $7 = (const tensorflow::ResourceHandle & ) @0x7f1c1c0024c0: {
  //   static ANONYMOUS_NAME = < optimized out > ,
  //   device_ = "/job:localhost/replica:0/task:0/device:GPU:0",
  //   container_ = "localhost",
  //   name_ = "training/SGD/momentum",
  //   hash_code_ = 17120488210734619367,
  //   maybe_type_name_ = "N10tensorflow3VarE"
  // }

  // 5.
  // p.container()
  // "localhost"

  // 6.
  // p.name()
  // "training/SGD/momentum"
  // 高优先级的图的节点名字是 : (wxf)
  // https://gist.github.com/shizukanaskytree/39fd4c6758a2ef2012270ed5a99097a6
  //
  // 低优先级的图:
  // https://gist.github.com/shizukanaskytree/896ff69ca98a394249c8994cd85175be
  //
  // name: "training/SGD/SGD/update_predictions/kernel/ResourceApplyKerasMomentum/ReadVariableOp_1"
  // op: "ReadVariableOp"
  // input: "training/SGD/momentum"
  // device: "/job:localhost/replica:0/task:0/device:GPU:0"
  // attr {
  //   key: "dtype"
  //   value {
  //     type: DT_FLOAT
  //   }
  // }
  // experimental_debug_info {
  //   original_node_names: "training/SGD/SGD/update_predictions/kernel/ResourceApplyKerasMomentum/ReadVariableOp_1"
  // }

  // 6.2 TODO
  // 低优先级的 name 是不是一样呢?


  // 7.
  // value: T**

  // 8.
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

  // 8.1
  // class TensorStore 数据结构
  // tensorflow/core/framework/session_state.h
  // The tensor store remembers the tensors we choose to keep for the
  // current run call. It is available to every op kernel.


  // 8.2
  // TensorValue 数据结构
  // tensorflow/core/framework/op_kernel.h:513:struct TensorValue
  // 概述:
  // Holds a tensor or tensor reference. For tensor references, we need
  // a mutex to prevent concurrent access to the tensor.
  //
  // - tensor: Tensor*
  // - mutex_if_ref: mutex*

  // 8.3
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

  // =======================================================================

  // 9.
  //
  // 对于 iterator 的情况

  // 9.1
  // T 类型说明:
  // ptype T
  // type = class tensorflow::data::IteratorResource : public tensorflow::ResourceBase {
  //   private:
  //     tensorflow::data::UnboundedThreadPool unbounded_thread_pool_;
  //     tensorflow::mutex mu_;
  //     std::unique_ptr<tensorflow::DeviceMgr> device_mgr_;
  //     std::shared_ptr<tensorflow::data::IteratorResource::State> iterator_state_;
  //     const tensorflow::DataTypeVector output_dtypes_;
  //     std::vector<tensorflow::PartialTensorShape> output_shapes_;
  //
  //   public:
  //     IteratorResource(tensorflow::Env *, const tensorflow::DataTypeVector &, const std::vector<tensorflow::PartialTensorShape> &, int, std::unique_ptr<tensorflow::DeviceMgr>, std::unique_ptr<tensorflow::FunctionLibraryDefinition>, std::unique_ptr<tensorflow::ProcessFunctionLibraryRuntime>, tensorflow::FunctionLibraryRuntime *);
  //     tensorflow::Status GetNext(tensorflow::data::IteratorContext *, std::vector<tensorflow::Tensor> *, bool *);
  //     tensorflow::Status GetNext(tensorflow::data::IteratorContext &&, std::vector<tensorflow::Tensor> *, bool *);
  //     tensorflow::Status Save(tensorflow::data::SerializationContext *, tensorflow::data::IteratorStateWriter *);
  //     tensorflow::Status Restore(tensorflow::OpKernelContext *, tensorflow::data::IteratorStateReader *);
  //     tensorflow::Status AddLibrary(const tensorflow::FunctionLibraryDefinition &);
  //     tensorflow::Status SetIteratorFromDataset(tensorflow::OpKernelContext *, tensorflow::data::DatasetBase *);
  //     virtual std::string DebugString(void) const;
  //     const tensorflow::DataTypeVector & output_dtypes(void) const;
  //     const std::vector<tensorflow::PartialTensorShape> & output_shapes(void) const;
  // }

  // 9.1.1
  // class IteratorResource 数据结构
  // class IteratorResource : public ResourceBase
  // tensorflow/core/kernels/data/iterator_ops.cc
  // * struct State
  // - unbounded_thread_pool_: UnboundedThreadPool
  // - device_mgr_: const std::unique_ptr<DeviceMgr>
  //   =======================================================================
  // - iterator_state_: std::shared_ptr<State> # IteratorResource 的核心
  //   =======================================================================
  // - output_dtypes_: const DataTypeVector
  // - output_shapes_: const std::vector<PartialTensorShape>

  // 9.1.2.
  // ResourceBase 数据结构
  // class ResourceBase : public core::RefCounted
  // tensorflow/core/framework/resource_mgr.h
  // 只有两个接口
  // - DebugString()
  //   Returns a debug string for *this.
  // - MemoryUsed()
  //   Returns memory used by this resource.

  // 9.1.3.
  // class IteratorResource::struct State 数据结构
  // - flib_def: std::shared_ptr<FunctionLibraryDefinition>
  // - pflr: std::shared_ptr<ProcessFunctionLibraryRuntime>
  // - lib: FunctionLibraryRuntime*
  // - function_handle_cache: std::unique_ptr<FunctionHandleCache>
  // - resource_mgr: ResourceMgr
  // - iterator: std::unique_ptr<IteratorBase>

  // 9.1.4.
  // class IteratorBase 数据结构
  // tensorflow/core/framework/dataset.h
  //
  // 概述:
  // 主要是和 iterator 相关的接口函数
  //
  // - cleanup_fns_: std::vector<std::function<void()>>
  // - node_: model::Node*


  // 9.2
  // p 类型说明:
  // p p
  // $23 = (const tensorflow::ResourceHandle & ) @0x7f1c14015740: {
  //   static ANONYMOUS_NAME = < optimized out > ,
  //   device_ = "/job:localhost/replica:0/task:0/device:CPU:0",
  //   container_ = "localhost", # 这个
  //   name_ = "_0_IteratorV2",  # 这个
  //   hash_code_ = 4245265473542939912,
  //   maybe_type_name_ = "N10tensorflow4data16IteratorResourceE"
  // }

  // 9.3
  // p ctx->op_kernel().def().DebugString()
  //
  // name: "IteratorGetNext"
  // op: "IteratorGetNext"
  // input: "IteratorV2"
  // device: "/job:localhost/replica:0/task:0/device:CPU:0"
  // attr {
  //   key: "output_shapes"
  //   value {
  //     list {
  //       shape {
  //         dim {
  //           size: 32
  //         }
  //         dim {
  //           size: 3
  //         }
  //         dim {
  //           size: 224
  //         }
  //         dim {
  //           size: 224
  //         }
  //       }
  //       shape {
  //         dim {
  //           size: 32
  //         }
  //         dim {
  //           size: 1
  //         }
  //       }
  //     }
  //   }
  // }
  // attr {
  //   key: "output_types"
  //   value {
  //     list {
  //       type: DT_FLOAT
  //       type: DT_FLOAT
  //     }
  //   }
  // }
  // experimental_debug_info {
  //   original_node_names: "IteratorGetNext"
  // }

  // 9.4
  // p use_dynamic_cast
  // $25 = false
}



template <typename T>
Status LookupResources(
    OpKernelContext* ctx, absl::Span<ResourceHandle const* const> p,
    std::vector<std::unique_ptr<T, core::RefCountDeleter>>* values) {
  std::vector<std::pair<const string*, const string*>> containers_and_names(
      p.size());
  for (size_t i = 0; i < p.size(); ++i) {
    TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, *p[i]));
    containers_and_names[i] = {&p[i]->container(), &p[i]->name()};
  }
  return ctx->resource_manager()->LookupMany(containers_and_names, values);
}

template <typename T>
Status LookupOrCreateResource(OpKernelContext* ctx, const ResourceHandle& p,
                              T** value, std::function<Status(T**)> creator) {

  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));

  return ctx->resource_manager()->LookupOrCreate(p.container(),
                                                 p.name(),
                                                 value,
                                                 creator);
}

template <typename T>
Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p) {
  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));
  return ctx->resource_manager()->Delete<T>(p.container(), p.name());
}

Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p);

template <typename T>
void IsResourceInitialized<T>::Compute(OpKernelContext* ctx) {
  Tensor* output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &output));
  T* object;
  bool found;
  if (LookupResource(ctx, HandleFromInput(ctx, 0), &object).ok()) {
    found = true;
    object->Unref();
  } else {
    found = false;
  }

  output->flat<bool>()(0) = found;
}

template <typename T>
ResourceHandleOp<T>::ResourceHandleOp(OpKernelConstruction* context)
    : OpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("container", &container_));
  OP_REQUIRES_OK(context, context->GetAttr("shared_name", &name_));
}

template <typename T>
void ResourceHandleOp<T>::Compute(OpKernelContext* ctx) {
  if (name_ == ResourceHandle::ANONYMOUS_NAME) {
    AllocatorAttributes attr;
    attr.set_on_host(true);
    Tensor handle;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &handle, attr));
    handle.scalar<ResourceHandle>()() =
        MakeResourceHandle<T>(ctx, container_, name_);
    ctx->set_output(0, handle);
  } else {
    if (!initialized_.load()) {
      mutex_lock ml(mutex_);
      // Checking again to see if another thread has initialized the resource.
      if (!initialized_.load()) {
        AllocatorAttributes attr;
        attr.set_on_host(true);
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                               &resource_, attr));
        resource_.scalar<ResourceHandle>()() =
            MakeResourceHandle<T>(ctx, container_, name_);
        initialized_.store(true);
      }
    }
    ctx->set_output(0, resource_);
  }
}

template <typename T>
ResourceHandlesOp<T>::ResourceHandlesOp(OpKernelConstruction* context)
    : OpKernel(context) {
  int n;
  OP_REQUIRES_OK(context, context->GetAttr("N", &n));
  OP_REQUIRES_OK(context, context->GetAttr("containers", &containers_));
  OP_REQUIRES_OK(context, context->GetAttr("shared_names", &names_));
  OP_REQUIRES(
      context, containers_.size() == n,
      errors::InvalidArgument("Number of containers (", containers_.size(),
                              ") must be equal to N (", n, ")"));
  OP_REQUIRES(context, names_.size() == n,
              errors::InvalidArgument("Number of names (", containers_.size(),
                                      ") must be equal to N (", n, ")"));
  resources_.resize(n);
}

template <typename T>
void ResourceHandlesOp<T>::Compute(OpKernelContext* ctx) {
  if (!initialized_.load()) {
    mutex_lock ml(mutex_);
    // Checking again to see if another thread has initialized the resource.
    if (!initialized_.load()) {
      AllocatorAttributes attr;
      attr.set_on_host(true);
      for (size_t i = 0; i < resources_.size(); ++i) {
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                               &resources_[i], attr));
        ResourceHandle h =
            MakeResourceHandle<T>(ctx, containers_[i], names_[i]);
        resources_[i].template scalar<ResourceHandle>()() = h;
      }
      initialized_.store(true);
    }
  }
  for (size_t i = 0; i < resources_.size(); ++i) {
    ctx->set_output(i, resources_[i]);
  }
}

}  //  end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_RESOURCE_MGR_H_
