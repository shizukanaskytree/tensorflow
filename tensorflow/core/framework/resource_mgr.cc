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

#include <atomic>

#include "tensorflow/core/framework/resource_mgr.h"

#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/demangle.h"

namespace tensorflow {

// Used to generate unique names for anonymous variables
static std::atomic<int64> current_id_;

ResourceHandle MakeResourceHandle(OpKernelContext* ctx, const string& container,
                                  const string& name,
                                  const TypeIndex& type_index) {
  ResourceHandle result;
  result.set_device(ctx->device()->attributes().name());
  string actual_container;
  if (!container.empty()) {
    actual_container = container;
  } else {
    actual_container = ctx->resource_manager()->default_container();
  }
  result.set_container(actual_container);
  if (name == ResourceHandle::ANONYMOUS_NAME) {
    result.set_name(strings::StrCat("_AnonymousVar", current_id_.fetch_add(1)));
  } else {
    result.set_name(name);
  }
  result.set_hash_code(type_index.hash_code());
  result.set_maybe_type_name(type_index.name());
  return result;
}

Status MakeResourceHandleToOutput(OpKernelContext* context, int output_index,
                                  const string& container, const string& name,
                                  const TypeIndex& type_index) {
  Tensor* handle;
  TF_RETURN_IF_ERROR(
      context->allocate_output(output_index, TensorShape({}), &handle));
  handle->scalar<ResourceHandle>()() =
      MakeResourceHandle(context, container, name, type_index);
  return Status::OK();
}

namespace internal {

Status ValidateDevice(OpKernelContext* ctx, // input
                      const ResourceHandle& p) { // input
  // 1.
  // ResourceHandle 数据结构
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

  if (ctx->device()->attributes().name() != p.device()) {
    // 1.
    // ctx 变量说明:
    // ctx: OpKernelContext*

    // 2.
    // class OpKernelContext
    // * typedef std::pair<Allocator*, TrackingAllocator*> WrappedAllocator;
    // - struct Params
    //    * step_id: int64
    //    * op_kernel: OpKernel*
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

    // 3.
    // class DeviceBase 数据结构
    // tensorflow/core/framework/device_base.h
    // - struct CpuWorkerThreads
    //    - num_threads: int , default : 0
    //    - workers: thread::ThreadPool*, default : nullptr
    // - struct GpuDeviceInfo
    //    - stream: stream_executor::Stream*
    //    - default_context: DeviceContext*
    //    - event_mgr: EventMgr*
    //    - gpu_id: int, default: -1
    // - env_: Env* const
    // - cpu_worker_threads_: CpuWorkerThreads* , default : nullptr
    // - gpu_device_info_: GpuDeviceInfo* , default : nullptr
    // - device_thread_pool_: thread::ThreadPool* , default : nullptr
    // - eigen_cpu_devices_: std::vector<Eigen::ThreadPoolDevice*>
    // - eigen_sycl_device_: Eigen::SyclDevice*, default : nullptr

    // 4.
    // attributes() 函数说明
    // virtual const DeviceAttributes& attributes() const;

    // 5.
    // message DeviceAttributes 数据结构
    // tensorflow/core/framework/device_attributes.proto
    // - name: string
    // - device_type: string
    // - memory_limit: int64
    // - locality: DeviceLocality
    // - incarnation: fixed64
    // - physical_device_desc: string

    return errors::InvalidArgument(
        "Trying to access resource ", p.name(), " located in device ",
        p.device(), " from device ", ctx->device()->attributes().name());
  }
  return Status::OK();
}

}  // end namespace internal

Status ResourceMgr::InsertDebugTypeName(uint64 hash_code,
                                        const string& type_name) {
  auto iter = debug_type_names_.emplace(hash_code, type_name);
  if (iter.first->second != type_name) {
    return errors::AlreadyExists("Duplicate hash code found for type ",
                                 type_name);
  }
  return Status::OK();
}

const char* ResourceMgr::DebugTypeName(uint64 hash_code) const {
  auto type_name_iter = debug_type_names_.find(hash_code);
  if (type_name_iter == debug_type_names_.end()) {
    return "<unknown>";
  } else {
    return type_name_iter->second.c_str();
  }
}

ResourceMgr::ResourceMgr() : default_container_("localhost") {}

ResourceMgr::ResourceMgr(const string& default_container)
    : default_container_(default_container) {}

ResourceMgr::~ResourceMgr() { Clear(); }

void ResourceMgr::Clear() {
  mutex_lock l(mu_);
  for (const auto& p : containers_) {
    for (const auto& q : *p.second) {
      q.second->Unref();
    }
    delete p.second;
  }
  containers_.clear();
}

string ResourceMgr::DebugString() const {

  mutex_lock l(mu_);
  struct Line {
    const string* container;
    const string type;
    const string* resource;
    const string detail;
  };

  std::vector<Line> lines;

  for (const auto& p : containers_) {
    // 1
    // ResourceMgr::containers_ 变量说明:
    // containers_ : std::unordered_map<string, Container*>

    // 1.1
    // p 变量说明:
    // p: std::pair<string, Container*>

    // 2.
    // Container 数据结构
    // typedef std::unordered_map<Key, ResourceBase*, KeyHash, KeyEqual> Container;
    // tensorflow/core/framework/resource_mgr.h:193

    // 3.
    // ResourceBase 数据结构
    // tensorflow/core/framework/resource_mgr.h:77:
    // class ResourceBase : public core::RefCounted
    //  - DebugString()
    //  - MemoryUsed()

    const string& container = p.first;
    // container 变量说明
    // container: string
    // container: contain name

    for (const auto& q : *p.second) {
      // 1.
      // p.second 变量说明:
      // p.second: Container*
      // 所以 q : Container

      // 2.
      // Container 数据结构
      // tensorflow/core/framework/resource_mgr.h:193
      // typedef std::unordered_map<Key, ResourceBase*, KeyHash, KeyEqual> Container;
      //                            ---  -------------
      //                        q.first  q.second
      //                        key      detail

      // 3.
      // Key 类型说明
      // typedef std::pair<uint64, string> Key;
      //                   ------  ------
      //                key.first  key.second
      //                type name  resource name/变量名
      const Key& key = q.first;

      const char* type = DebugTypeName(key.first);
      const string& resource = key.second;
      // Key 类型的 second

      Line l{
              &container,               // "localhost"
              port::Demangle(type),     // "N10tensorflow9LegacyVarE"
              &resource,                // "v1" 或者 "v2" "v3"
              q.second->DebugString()  // detail
            };
      // 1.
      // q.second->DebugString() 函数说明:
      // 对于 Variable 而言, 这个函数对应:
      // tensorflow/core/kernels/variable_ops.cc
      // class LegacyVar : public ResourceBase :: DebugString()

      // 2.
      // 打印
      // "float/[25,15]"

      // 3.
      // 打印:
      // p devices_[6]->resource_manager()->DebugString()
      // "localhost | N10tensorflow9LegacyVarE | v1 | float/[25,35]\nlocalhost | N10tensorflow9LegacyVarE | v2 | float/[30,20]\nlocalhost | N10tensorflow9LegacyVarE | v3 | float/[10,40]\nlocalhost | N10tensorflow9LegacyVarE | v4 | float/[25,15]"
      //
      // 排版后
      // localhost | N10tensorflow9LegacyVarE | v1 | float/[25,35]
      // localhost | N10tensorflow9LegacyVarE | v2 | float/[30,20]
      // localhost | N10tensorflow9LegacyVarE | v3 | float/[10,40]
      // localhost | N10tensorflow9LegacyVarE | v4 | float/[25,15]

      lines.push_back(l);
    }
  }

  std::vector<string> text;

  text.reserve(lines.size());

  for (const Line& line : lines) {
    text.push_back(strings::Printf(
        "%-20s | %-40s | %-40s | %-s",
        line.container->c_str(),
        line.type.c_str(),
        line.resource->c_str(),
        line.detail.c_str()));
  }
  // 打印
  // 对应于 variable_ops, tensorflow/core/kernels/variable_ops.cc
  // $31 = "localhost            | N10tensorflow9LegacyVarE                 | v4                                       | float/[25,15]"

  // 打印:
  // p devices_[6]->resource_manager()->DebugString()
  // "localhost | N10tensorflow9LegacyVarE | v1 | float/[25,35]\nlocalhost | N10tensorflow9LegacyVarE | v2 | float/[30,20]\nlocalhost | N10tensorflow9LegacyVarE | v3 | float/[10,40]\nlocalhost | N10tensorflow9LegacyVarE | v4 | float/[25,15]"
  //
  // 排版后
  // localhost | N10tensorflow9LegacyVarE | v1 | float/[25,35]
  // localhost | N10tensorflow9LegacyVarE | v2 | float/[30,20]
  // localhost | N10tensorflow9LegacyVarE | v3 | float/[10,40]
  // localhost | N10tensorflow9LegacyVarE | v4 | float/[25,15]

  std::sort(text.begin(), text.end());

  return str_util::Join(text, "\n");
}

// 用这个迁移
Status ResourceMgr::DoCreate(const string& container, // input
                             TypeIndex type, // input
                             const string& name, // input
                             ResourceBase* resource) { // input
  // 1.
  // resource 变量说明
  // resource: ResourceBase*
  // 实际的变量都是继承了 ResourceBase 的类型，他们有自己的构造函数

  // 首先，构造 containers_
  Container** b = &containers_[container];
  // 1.
  // b 变量说明:
  // b: Container**

  // 2.
  // Container 数据结构:
  // typedef std::unordered_map<Key, ResourceBase*, KeyHash, KeyEqual> Container;

  // 3.
  // ResourceMgr::containers_ 变量说明:
  // std::unordered_map<string, Container*> containers_
  //                      |
  //                container name

  // 4.
  // QQQ. 如果 container 对应的没有找到怎么办?
  // AAA.

  if (*b == nullptr) {
    *b = new Container;
    // 1.
    // Container 数据结构:
    // typedef std::unordered_map<Key, ResourceBase*, KeyHash, KeyEqual> Container;
    //
    // 概述:
    // Container 里面有 Key 和 指向这个 resource 的指针

    // 2.
    // Key 数据结构
    // typedef std::pair<uint64, string> Key;
  }

  if ((*b)->insert( {
                       {type.hash_code(), name}, // Key
                       resource                  // value
                    }
                  ).second){ // .second 的意图是取出 resource 看是否不为 nullptr
    // 1.
    // Key 类型说明
    // typedef std::pair<uint64, string> Key;
    // - uint64 是 type.hash_code()
    // - name 是 variable name, 比如 "v4"

    // 2.
    // 打印
    // p name
    // $24 = "v4"

    // 3.
    // resource 说明
    // resource: ResourceBase*
    // 比如说:
    // class LegacyVar : public ResourceBase
    // tensorflow/core/kernels/variable_ops.cc

    // 3.
    // TypeIndex 是 std::type_index
    // 用途: The type_index class is a wrapper class around a std::type_info object, that can be used as index in associative and unordered associative containers.
    // 概述:用于管理类型的名称
    // 例子: https://en.cppreference.com/w/cpp/types/type_index

    TF_RETURN_IF_ERROR(InsertDebugTypeName(type.hash_code(), type.name()));
    // 正常处理后退出
    return Status::OK();
  }

  // 异常处理
  resource->Unref();
  return errors::AlreadyExists("Resource ", container, "/", name, "/",
                               type.name());
}

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

Status ResourceMgr::DoLookup(const string& container,
                             TypeIndex type,
                             const string& name,
                             ResourceBase** resource) const {

  const Container* b = gtl::FindPtrOrNull(containers_, container);
  // 1.
  // Container 数据结构
  // tensorflow/core/framework/resource_mgr.h:193
  // typedef std::unordered_map<Key, ResourceBase*, KeyHash, KeyEqual> Container;
  //
  // 概述:
  // Container 里面有 Key 和 指向这个 resource 的指针

  if (b == nullptr) {
    return errors::NotFound("Container ", container,
                            " does not exist. (Could not find resource: ",
                            container, "/", name, ")");
  }

  // r 代表 resource
  auto r = gtl::FindPtrOrNull(*b, {type.hash_code(), name});
  // 1.
  // type.hash_code() : uint64

  // 2.
  // tensorflow/core/framework/type_index.h:41:class TypeIndex

  // 3.
  // r 变量说明
  // r: ResourceBase*
  // 真实类型:
  // class Var : public ResourceBase
  // tensorflow/core/framework/resource_var.h

  // 3.1
  // p name
  // $3 = "_0_IteratorV2"
  // 打印
  // ptype r
  // type = /* real type = tensorflow::data::IteratorResource * */
  // class tensorflow::ResourceBase : public tensorflow::core::RefCounted {
  //   public:
  //     virtual std::string DebugString(void) const;
  //     virtual tensorflow::int64 MemoryUsed(void) const;
  // } *

  // 4.
  // class Var 数据结构
  // class Var : public ResourceBase
  // tensorflow/core/framework/resource_var.h
  // - tensor_: Tensor
  // - is_initialized: bool, default_value: false
  // - copy_on_read_mode: std::atomic<bool>, default_value: false

  // 5.1
  // class Tensor 数据结构
  // tensorflow/core/framework/tensor.h
  // - shape_: TensorShape
  // - buf_: TensorBuffer*

  // 5.2.
  // class TensorBuffer 数据结构
  // tensorflow/core/framework/tensor.h
  // - 简述
  // - Interface to access the raw ref-counted data buffer.
  //
  // - class TensorBuffer : public core::RefCounted
  // - data_: void* const
  //   data_ points to a memory region of size() bytes.

  // 5.3.
  // class TensorShape 数据结构
  // tensorflow/core/framework/tensor_shape.h
  //
  // 简述
  // Represents the shape of a Tensor.
  //
  // A tensor's shape is denoted by its number of dimensions and a size for each
  // dimension.  For example, a Tensor represented by a 3 x 4 matrix would have
  // a shape of 2-D, [3,4].
  //
  // If you know the exact shape of your Tensor when you create the TensorShape
  // object, you can specify it then, or you can create a TensorShape with
  // zero dimensions and one element, and call AddDim() to add dimensions later.
  //
  // class TensorShape : public TensorShapeBase<TensorShape>
  // - 没有成员变量，只有成员函数

  // 5.4.
  // class TensorShapeBase 数据结构
  // tensorflow/core/framework/tensor_shape.h
  // class TensorShapeBase : public TensorShapeRep
  // - 没有成员变量，只有成员函数

  // 5.5.
  // class TensorShapeRep 数据结构
  // tensorflow/core/framework/tensor_shape.h
  // - num_elements_: int64
  // - u_: union
  //   * buf[16] : uint8
  //   * unused_aligner: Rep64*

  if (r == nullptr) {
    return errors::NotFound("Resource ", container, "/", name, "/", type.name(),
                            " does not exist.");
  }
  *resource = const_cast<ResourceBase*>(r); // r 代表 resource
  // 1.
  // ResourceBase** resource 说明:
  // 所以 *resource 是 ResourceBase*
  // 所以 *resource 和 r 是同级指针！类型都是 ResourceBase*

  (*resource)->Ref();
  return Status::OK();
}

Status ResourceMgr::DoDelete(const string& container,
                             uint64 type_hash_code,
                             const string& resource_name,
                             const string& type_name) {
  ResourceBase* base = nullptr;
  // 1.
  // base 变量说明:
  // 这个是要删除的对象
  {
    mutex_lock l(mu_);
    Container* b = gtl::FindPtrOrNull(containers_, container);
    // 1.
    // b: unordered_map<...>*

    if (b == nullptr) {
      return errors::NotFound("Container ", container, " does not exist.");
    }

    auto iter = b->find({type_hash_code, resource_name});
    if (iter == b->end()) {
      return errors::NotFound("Resource ", container, "/", resource_name, "/",
                              type_name, " does not exist.");
    }
    base = iter->second;
    // 1.
    // base 变量说明:
    // base: ResourceBase*
    // 找到了，这个要删除的对象

    b->erase(iter);
    // 1.
    //
  }
  CHECK(base != nullptr);
  base->Unref();
  // 1.
  // Unref() 函数说明:
  // tensorflow/core/lib/core/refcount.h
  // Decrements reference count by one.  If the count remains
  // positive, returns false.  When the count reaches zero, returns
  // true and deletes this, in which case the caller must not access
  // the object afterward.

  return Status::OK();
}

Status ResourceMgr::DoDelete(const string& container, TypeIndex type,
                             const string& resource_name) {
  return DoDelete(container, type.hash_code(), resource_name, type.name());
}

Status ResourceMgr::Delete(const ResourceHandle& handle) {
  return DoDelete(handle.container(), handle.hash_code(), handle.name(),
                  "<unknown>");
}


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

Status ResourceMgr::Cleanup(const string& container) {

  {
    tf_shared_lock l(mu_);
    // 需要锁🔐查找 container

    if (!gtl::FindOrNull(containers_, container)) {
      // Nothing to cleanup.
      return Status::OK();
    }
  }

  Container* b = nullptr;

  {
    mutex_lock l(mu_);
    // 需要锁🔐查找 container

    auto iter = containers_.find(container);
    if (iter == containers_.end()) {
      // Nothing to cleanup, it's OK (concurrent cleanup).
      return Status::OK();
    }

    b = iter->second;
    // 1.
    // b 说明:
    // b: Container*

    // 把 container --> Container* 这个 unordered_map instance 给删了
    containers_.erase(iter);
  }

  CHECK(b != nullptr);
  // 1.
  // b 指针的理解:
  // 可以认为 b 指针是 那个 Container instance 头部的一点点 a bit , 如果这“一点点”存在，则表示整个 instance 存在。

  for (const auto& p : *b) {
    p.second->Unref();
  }

  delete b;

  return Status::OK();
}

static bool IsValidContainerName(StringPiece s) {
  using ::tensorflow::strings::Scanner;
  return Scanner(s)
      .One(Scanner::LETTER_DIGIT_DOT)
      .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH)
      .Eos()
      .GetResult();
}

Status ContainerInfo::Init(ResourceMgr* rmgr,
                           const NodeDef& ndef,
                           bool use_node_name_as_default) {
  CHECK(rmgr);
  rmgr_ = rmgr;
  string attr_container;

  TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "container", &attr_container));

  if (!attr_container.empty() && !IsValidContainerName(attr_container)) {
    return errors::InvalidArgument("container contains invalid characters: ",
                                   attr_container);
  }

  string attr_shared_name;

  TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "shared_name", &attr_shared_name));

  if (!attr_shared_name.empty() && (attr_shared_name[0] == '_')) {
    return errors::InvalidArgument("shared_name cannot start with '_':",
                                   attr_shared_name);
  }

  if (!attr_container.empty()) {
    container_ = attr_container;
  } else {
    container_ = rmgr_->default_container();
  }

  if (!attr_shared_name.empty()) {
    name_ = attr_shared_name;
  } else if (use_node_name_as_default) {
    name_ = ndef.name();
  } else {
    resource_is_private_to_kernel_ = true;
    static std::atomic<int64> counter(0);
    name_ = strings::StrCat("_", counter.fetch_add(1), "_", ndef.name());
  }

  return Status::OK();
}

string ContainerInfo::DebugString() const {
  return strings::StrCat("[", container(), ",", name(), ",",
                         resource_is_private_to_kernel() ? "private" : "public",
                         "]");
}

const ResourceHandle& HandleFromInput(OpKernelContext* ctx, int input) {
  return ctx->input(input).flat<ResourceHandle>()(0);
}

Status HandleFromInput(OpKernelContext* ctx, StringPiece input,
                       ResourceHandle* handle) {
  const Tensor* tensor;
  TF_RETURN_IF_ERROR(ctx->input(input, &tensor));
  *handle = tensor->flat<ResourceHandle>()(0);
  return Status::OK();
}

Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p) {
  TF_RETURN_IF_ERROR(internal::ValidateDevice(ctx, p));
  return ctx->resource_manager()->Delete(p);
}

Status ResourceHandlesShape(shape_inference::InferenceContext* c) {
  int n;
  TF_RETURN_IF_ERROR(c->GetAttr("N", &n));
  for (int i = 0; i < n; ++i) {
    c->set_output(i, c->Scalar());
  }
  return Status::OK();
}

}  //  end namespace tensorflow
