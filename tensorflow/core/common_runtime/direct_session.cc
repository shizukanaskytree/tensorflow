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

#include "tensorflow/core/common_runtime/direct_session.h"

#include <atomic>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/collective_executor_mgr.h"
#include "tensorflow/core/common_runtime/collective_param_resolver_local.h"
#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/debugger_state_interface.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb_text.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/run_handler.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/device_tracer.h" // class DeviceTracer
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

namespace {

auto* direct_session_runs = monitoring::Counter<0>::New(
    "/tensorflow/core/direct_session_runs",
    "The number of times DirectSession::Run() has been called.");

Status NewThreadPoolFromThreadPoolOptions(
    const SessionOptions& options,
    const ThreadPoolOptionProto& thread_pool_options, int pool_number,
    thread::ThreadPool** pool, bool* owned) {
  int32 num_threads = thread_pool_options.num_threads();
  if (num_threads == 0) {
    num_threads = NumInterOpThreadsFromSessionOptions(options);
  }
  const string& name = thread_pool_options.global_name();
  if (name.empty()) {
    // Session-local threadpool.
    VLOG(1) << "Direct session inter op parallelism threads for pool "
            << pool_number << ": " << num_threads;
    *pool = new thread::ThreadPool(
        options.env, strings::StrCat("Compute", pool_number), num_threads);
    *owned = true;
    return Status::OK();
  }

  // Global, named threadpool.
  typedef std::pair<int32, thread::ThreadPool*> MapValue;
  static std::map<string, MapValue>* global_pool_map =
      new std::map<string, MapValue>;
  static mutex* mu = new mutex();
  mutex_lock l(*mu);
  MapValue* mvalue = &(*global_pool_map)[name];
  if (mvalue->second == nullptr) {
    mvalue->first = thread_pool_options.num_threads();
    mvalue->second = new thread::ThreadPool(
        options.env, strings::StrCat("Compute", pool_number), num_threads);
  } else {
    if (mvalue->first != thread_pool_options.num_threads()) {
      return errors::InvalidArgument(
          "Pool ", name,
          " configured previously with num_threads=", mvalue->first,
          "; cannot re-configure with num_threads=",
          thread_pool_options.num_threads());
    }
  }
  *owned = false;
  *pool = mvalue->second;
  return Status::OK();
}


thread::ThreadPool* GlobalThreadPool(const SessionOptions& options) {
  /// Static Variables inside Functions
  /// Static variables when used inside function are initialized only once,
  /// and then they hold there value even through function calls. These static
  /// variables are stored on static storage area , not in stack.
  static thread::ThreadPool* const thread_pool =
      NewThreadPoolFromSessionOptions(options);
  return thread_pool;
}

// TODO(vrv): Figure out how to unify the many different functions
// that generate RendezvousKey, since many of them have to be
// consistent with each other.
string GetRendezvousKey(const string& tensor_name,
                        const DeviceAttributes& device_info,
                        const FrameAndIter& frame_iter) {
  return strings::StrCat(device_info.name(), ";",
                         strings::FpToString(device_info.incarnation()), ";",
                         device_info.name(), ";", tensor_name, ";",
                         frame_iter.frame_id, ":", frame_iter.iter_id);
}

}  // namespace


// 构造 DirectSession 的起点
class DirectSessionFactory : public SessionFactory {
 public:
  DirectSessionFactory() {}

  bool AcceptsOptions(const SessionOptions& options) override {
    return options.target.empty();
  }

  // 构造 DirectSession 的起点函数:
  Status NewSession(const SessionOptions& options,
                    Session** out_session) override {
    // Must do this before the CPU allocator is created.
    if (options.config.graph_options().build_cost_model() > 0) {
      EnableCPUAllocatorFullStats(true);
    }

    std::vector<std::unique_ptr<Device>> devices;

    TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices));
    // 1.
    // devices 打印
    // p devices[0]->DebugString()
    // $2 = "name: \"/job:localhost/replica:0/task:0/device:CPU:0\"\ndevice_type: \"CPU\"\nmemory_limit: 268435456\nlocality {\n}\nincarnation: 7904290016610734523\n"
    // p devices[1]->DebugString()
    // $3 = "name: \"/job:localhost/replica:0/task:0/device:XLA_CPU:0\"\ndevice_type: \"XLA_CPU\"\nmemory_limit: 17179869184\nlocality {\n}\nincarnation: 15932437085108527271\nphysical_device_desc: \"device: XLA_CPU device\"\n"
    // p devices[2]->DebugString()
    // $4 = "name: \"/job:localhost/replica:0/task:0/device:XLA_GPU:0\"\ndevice_type: \"XLA_GPU\"\nmemory_limit: 17179869184\nlocality {\n}\nincarnation: 5993004511160472241\nphysical_device_desc: \"device: XLA_GPU device\"\n"
    // p devices[3]->DebugString()
    // $5 = "name: \"/job:localhost/replica:0/task:0/device:XLA_GPU:1\"\ndevice_type: \"XLA_GPU\"\nmemory_limit: 17179869184\nlocality {\n}\nincarnation: 8896626939223060817\nphysical_device_desc: \"device: XLA_GPU device\"\n"
    // p devices[4]->DebugString() # GeForce RTX 2080 Ti
    // $6 = "name: \"/job:localhost/replica:0/task:0/device:GPU:0\"\ndevice_type: \"GPU\"\nmemory_limit: 10244594074\nlocality {\n  bus_id: 1\n  links {\n  }\n}\nincarnation: 8237163111145486887\nphysical_device_desc: \"device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:03:00.0, compute capability: 7.5\"\n"
    // p devices[5]->DebugString() # GeForce GTX 1080 Ti
    // $7 = "name: \"/job:localhost/replica:0/task:0/device:GPU:1\"\ndevice_type: \"GPU\"\nmemory_limit: 10411278336\nlocality {\n  bus_id: 1\n  links {\n  }\n}\nincarnation: 15954655773881420709\nphysical_device_desc: \"device: 1, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1\"\n"

    // 1.1 评论
    // 还没有把 GeForce RTX 2080 Ti 排序调整成最高的一个

    DirectSession* session =
        new DirectSession(options, new DeviceMgr(std::move(devices)), this);

    {
      mutex_lock l(sessions_lock_);
      sessions_.push_back(session);
    }

    *out_session = session;

    return Status::OK();
  }

  Status Reset(const SessionOptions& options,
               const std::vector<string>& containers) override {
    std::vector<DirectSession*> sessions_to_reset;
    {
      mutex_lock l(sessions_lock_);
      // We create a copy to ensure that we don't have a deadlock when
      // session->Close calls the DirectSessionFactory.Deregister, which
      // acquires sessions_lock_.
      std::swap(sessions_to_reset, sessions_);
    }
    Status s;
    for (auto session : sessions_to_reset) {
      s.Update(session->Reset(containers));
    }
    // TODO(suharshs): Change the Reset behavior of all SessionFactories so that
    // it doesn't close the sessions?
    for (auto session : sessions_to_reset) {
      s.Update(session->Close());
    }
    return s;
  }

  void Deregister(const DirectSession* session) {
    mutex_lock l(sessions_lock_);
    sessions_.erase(std::remove(sessions_.begin(), sessions_.end(), session),
                    sessions_.end());
  }

 private:
  mutex sessions_lock_;
  std::vector<DirectSession*> sessions_ GUARDED_BY(sessions_lock_);
};

class DirectSessionRegistrar {
 public:
  DirectSessionRegistrar() {
    SessionFactory::Register("DIRECT_SESSION", new DirectSessionFactory());
  }
};
static DirectSessionRegistrar registrar;

std::atomic_int_fast64_t DirectSession::step_id_counter_(1);

// NOTE: On Android with a single device, there is never
// a risk of an OpKernel blocking indefinitely:
//
// 1) No operations do I/O that depends on other simultaneous kernels,
//
// 2) Recv nodes always complete immediately: The inputs are sent into
//    the local rendezvous before we start the executor, so the
//    corresponding recvs will not block.
//
// Based on these assumptions, we can use the same thread pool for
// both "non-blocking" and "blocking" OpKernels on Android.
//
// This may change down the road when we add support for multiple
// devices that run concurrently, in which case we will need to
// revisit this decision.
void DirectSession::SchedClosure(thread::ThreadPool* pool,
                                 std::function<void()> c) {
// TODO(sanjay): Get rid of __ANDROID__ path
#ifdef __ANDROID__
  // On Android, there is no implementation of ThreadPool that takes
  // std::function, only Closure, which we cannot easily convert.
  //
  // Instead, we just run the function in-line, which is currently
  // safe given the reasoning above.
  c();
#else
  if (pool != nullptr) {

    //////////////////////////////////////////////////////////////////////////
    pool->Schedule(std::move(c));
    //////////////////////////////////////////////////////////////////////////

  } else {
    c();
  }
#endif  // __ANDROID__
}

static RunHandlerPool* GetOrCreateRunHandlerPool(
    const SessionOptions& options) {
  static RunHandlerPool* pool =
      new RunHandlerPool(NumInterOpThreadsFromSessionOptions(options));
  return pool;
}

bool DirectSession::ShouldUseRunHandlerPool(
    const RunOptions& run_options) const {
  if (options_.config.use_per_session_threads()) return false;
  if (options_.config.session_inter_op_thread_pool_size() > 0 &&
      run_options.inter_op_thread_pool() > 0)
    return false;
  // Only use RunHandlerPool when:
  // a. Single global thread pool is used for inter-op parallelism.
  // b. When multiple inter_op_thread_pool(s) are created, use it only while
  // running sessions on the default inter_op_thread_pool=0. Typically,
  // servo-team uses inter_op_thread_pool > 0 for model loading.
  // TODO(crk): Revisit whether we'd want to create one (static) RunHandlerPool
  // per entry in session_inter_op_thread_pool() in the future.
  return true;
}


DirectSession::DirectSession(const SessionOptions& options,
                             const DeviceMgr* device_mgr,
                             DirectSessionFactory* const factory)
    : options_(options),
      device_mgr_(device_mgr),
      factory_(factory),
      cancellation_manager_(new CancellationManager()),
      operation_timeout_in_ms_(options_.config.operation_timeout_in_ms()) {

  // SessionOptions 变量说明:
  // tensorflow/core/public/session_options.h:28:struct SessionOptions
  // 核心的使用是 sess.run 的 tf.ConfigProto 的设置

  // This option is experimental - it may be replaced with a different mechanism
  // in the future.
  //
  // Configures session thread pools. If this is configured, then RunOptions for
  // a Run call can select the thread pool to use.
  //
  // The intended use is for when some session invocations need to run in a
  // background pool limited to a small number of threads:
  // - For example, a session may be configured to have one large pool (for
  // regular compute) and one small pool (for periodic, low priority work);
  // using the small pool is currently the mechanism for limiting the inter-op
  // parallelism of the low priority work.  Note that it does not limit the
  // parallelism of work spawned by a single op kernel implementation.
  // - Using this setting is normally not needed in training, but may help some
  // serving use cases.
  // - It is also generally recommended to set the global_name field of this
  // proto, to avoid creating multiple large pools. It is typically better to
  // run the non-low-priority work, even across sessions, in a single large
  // pool.
  // repeated ThreadPoolOptionProto session_inter_op_thread_pool = 12;

  // protobuf message int32 默认值是 0 所以第三个分支
  const int thread_pool_size =
      options_.config.session_inter_op_thread_pool_size();

  // 跟踪一下，究竟是几个 thread_pool_size ？
  if (thread_pool_size > 0) {
    for (int i = 0; i < thread_pool_size; ++i) {
      thread::ThreadPool* pool = nullptr;
      bool owned = false;
      init_error_.Update(NewThreadPoolFromThreadPoolOptions(
          options_, options_.config.session_inter_op_thread_pool(i), i, &pool,
          &owned));
      thread_pools_.emplace_back(pool, owned);
    }
  } else if (options_.config.use_per_session_threads()) {

    thread_pools_.emplace_back(NewThreadPoolFromSessionOptions(options_),
                               true /* owned */);
  } else {
    // 这个:
    // High priority threadpool or default threadpool
    // ------------------------------------------------------------------------
    thread_pools_.emplace_back(GlobalThreadPool(options), false /* owned */);
    // ------------------------------------------------------------------------
  }

  // The default value of sync_on_finish will be flipped soon and this
  // environment variable will be removed as well.
  const Status status =
      ReadBoolFromEnvVar("TF_SYNC_ON_FINISH", true, &sync_on_finish_);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
  }

  session_handle_ =
      strings::StrCat("direct", strings::FpToString(random::New64()));

  int devices_added = 0;

  if (options.config.log_device_placement()) {
    const string mapping_str = device_mgr_->DeviceMappingString();
    if (mapping_str.empty()) {
      printf("Device mapping: no known devices.\n");
    } else {
      printf("Device mapping:\n%s", mapping_str.c_str());
    }
    LOG(INFO) << "Device mapping:\n" << mapping_str;
  }
  // 1.
  // 打印:
  // 2019-10-23 03:21:33.418319: I tensorflow/core/common_runtime/direct_session.cc:341] Device mapping:
  // /job:localhost/replica:0/task:0/device:XLA_CPU:0 -> device: XLA_CPU device
  // /job:localhost/replica:0/task:0/device:XLA_GPU:0 -> device: XLA_GPU device
  // /job:localhost/replica:0/task:0/device:XLA_GPU:1 -> device: XLA_GPU device
  // /job:localhost/replica:0/task:0/device:XLA_GPU:2 -> device: XLA_GPU device
  // /job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:82:00.0, compute capability: 7.5
  // /job:localhost/replica:0/task:0/device:GPU:1 -> device: 1, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1
  // /job:localhost/replica:0/task:0/device:GPU:2 -> device: 2, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1

  // 1.1
  // log and code
  // https://gist.github.com/shizukanaskytree/3bc7c81cc9490f4a3abae3c7fb26a107

  // 2.
  // 2.1
  // class DeviceMgr 数据结构说明:
  // tensorflow/core/common_runtime/device_mgr.h
  // - devices_: const std::vector<std::unique_ptr<Device>>
  // - device_map_: std::unordered_map<StringPiece, Device*, StringPieceHasher>
  // - name_backing_store_: core::Arena
  // - device_type_counts_: std::unordered_map<string, int>

  // 2.2
  // message DeviceAttributes 数据结构
  // tensorflow/core/framework/device_attributes.proto
  // - name: string
  // - device_type: string
  // - memory_limit: int64
  // - locality: DeviceLocality
  // - incarnation: fixed64
  // - physical_device_desc: string

  // 2.3
  // message DeviceLocality 数据结构说明:
  // - bus_id: int32
  // - numa_node: int32
  // - links: LocalLinks

  // 2.4
  // message LocalLinks 数据结构说明:
  // - link: repeated InterconnectLink , i.e., vector of InterconnectLink instance

  // 2.5
  // message InterconnectLink 数据结构说明:
  // - device_id: int32
  // - type: string
  // - strength: int32


  // wxf: 重要!
  // https://github.com/shizukanaskytree/tensorflow/commit/768e064fe54786de11ebe03932b026aa60a353a1#diff-c6591be80b797cc1fdd04bc76be82060
  // 我在这里增加了 low_priority_device_set_
  for (auto d : device_mgr_->ListDevices()) {
    // 1. ListDevices() 函数说明:
    //
    // p device_mgr_->ListDevices()
    // $8 = std::vector of length 6, capacity 6 = {0x56382fa0fb10, 0x563831609560, 0x563831859490, 0x5638318627a0, 0x5638318797e0, 0x56383188c560}

    // p device_mgr_->ListDevices()[0]->DebugString() # CPU
    // $9 = "name: \"/job:localhost/replica:0/task:0/device:CPU:0\"\ndevice_type: \"CPU\"\nmemory_limit: 268435456\nlocality {\n}\nincarnation: 7904290016610734523\n"

    // p device_mgr_->ListDevices()[1]->DebugString()
    // $10 = "name: \"/job:localhost/replica:0/task:0/device:XLA_CPU:0\"\ndevice_type: \"XLA_CPU\"\nmemory_limit: 17179869184\nlocality {\n}\nincarnation: 15932437085108527271\nphysical_device_desc: \"device: XLA_CPU device\"\n"

    // p device_mgr_->ListDevices()[2]->DebugString()
    // $11 = "name: \"/job:localhost/replica:0/task:0/device:XLA_GPU:0\"\ndevice_type: \"XLA_GPU\"\nmemory_limit: 17179869184\nlocality {\n}\nincarnation: 5993004511160472241\nphysical_device_desc: \"device: XLA_GPU device\"\n"

    // p device_mgr_->ListDevices()[3]->DebugString()
    // $12 = "name: \"/job:localhost/replica:0/task:0/device:XLA_GPU:1\"\ndevice_type: \"XLA_GPU\"\nmemory_limit: 17179869184\nlocality {\n}\nincarnation: 8896626939223060817\nphysical_device_desc: \"device: XLA_GPU device\"\n"

    // p device_mgr_->ListDevices()[4]->DebugString() # GeForce RTX 2080 Ti
    // $13 = "name: \"/job:localhost/replica:0/task:0/device:GPU:0\"\ndevice_type: \"GPU\"\nmemory_limit: 10244594074\nlocality {\n  bus_id: 1\n  links {\n  }\n}\nincarnation: 8237163111145486887\nphysical_device_desc: \"device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:03:00.0, compute capability: 7.5\"\n"

    // p device_mgr_->ListDevices()[5]->DebugString() # GeForce GTX 1080 Ti
    // $14 = "name: \"/job:localhost/replica:0/task:0/device:GPU:1\"\ndevice_type: \"GPU\"\nmemory_limit: 10411278336\nlocality {\n  bus_id: 1\n  links {\n  }\n}\nincarnation: 15954655773881420709\nphysical_device_desc: \"device: 1, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1\"\n"

    // 1.1 用 c++ regex, if it has found the computation capability

    // 2.
    // 这个顺序没变，相较于 构造 DeviceMgr 时。

    // 3.
    // d 的 real type
    // tensorflow::GPUDevice *
    // - 继承自 class BaseGPUDevice,
    //  - 继承自 class LocalDevice, tensorflow/core/common_runtime/local_device.h
    //    - 继承自 class Device, tensorflow/core/common_runtime/device.h
    //      - 继承自 class DeviceBase, tensorflow/core/framework/device_base.h

    // tensorflow::XlaDevice *
    // -

    // 4.
    // class GPUDevice 类型说明:
    // gpu_device_factory.cc

    devices_.push_back(d);
    device_set_.AddDevice(d);
    // 1.
    // PrioritizedDeviceTypeList() 打印
    // p device_set_.PrioritizedDeviceTypeList()
    // $16 = std::vector of length 4, capacity 4 = {{type_ = "GPU"}, {type_ = "CPU"}, {type_ = "XLA_CPU"}, {type_ = "XLA_GPU"}}

    d->op_segment()->AddHold(session_handle_);
    // 1.
    // Device::op_segment() 函数说明:
    // tensorflow/core/common_runtime/device.h:172
    // Returns the op segment of this device.  The caller can reuse op
    // kernels registered for the same session running on this device.
    // OpSegment* Device::op_segment() { return &op_seg_; }

    // 2.
    // Device::op_seg_ 变量说明
    // tensorflow/core/common_runtime/device.h:215:  OpSegment op_seg_;
    // op_seg_ maps session handle and op name to OpKernel objects.

    // 3.
    // OpSegment 类型说明
    // tensorflow/core/framework/op_segment.h:42:class OpSegment
    // OpSegment keeps track of OpKernels registered for sessions running
    // on a device.
    //
    // The implementation maintains a two-level map. The 1st level maps
    // session handle to the map of registered OpKernels. The 2nd level
    // map maps node names to instantiated OpKernel objects.
    //
    // Each 2-nd level map is reference-counted and the caller can call
    // AddHold to obtain a reference on all kernels of a session and
    // ensure these kernels are alive until a corresponding RemoveHold is
    // called on the same session.
    /*
    +--------------------------------------------------------------+
    |                      +-------------+                         |
    | session handle+-------->  Item     |                         |
    |                      |-------------|                         |
    |                      |             |   +-------------------+ |
    |                      | name_kernel+--->|op name+-->OpKernel| |
    |                      |             |   +-------------------+ |
    |                      |             |   +-------------------+ |
    |                      +-------------+   |op name+-->OpKernel| |
    |                                        +-------------------+ |
    |                                        +-------------------+ |
    |                                        |op name+-->OpKernel| |
    |                                        +-------------------+ |
    |                                                              |
    |                                                              |
    |                                                              |
    |                      +-------------+                         |
    | session handle+-------->  Item     |                         |
    |                      |-------------|                         |
    |                      |             |   +-------------------+ |
    |                      | name_kernel+--->|op name+-->OpKernel| |
                           |             |   +-------------------+
                           |             |   +-------------------+
                           +-------------+   |op name+-->OpKernel|
                                             +-------------------+
                                             +-------------------+
                                             |op name+-->OpKernel|
                                             +-------------------+
    */

    // 4.
    // OpSegment::AddHold 函数说明
    // void AddHold(const string& session_handle);
    // A hold can be placed on a session, preventing all its kernels
    // from being deleted.


    // The first device added is special: it is the 'client device' (a
    // CPU device) from which we feed and fetch Tensors.
    if (devices_added == 0) {
      device_set_.set_client_device(d);
    }
    ++devices_added;
  }

} // DirectSession 构造函数结束

// 1.
// Bug report: https://gist.github.com/shizukanaskytree/bd44411cde16cdba3145d5d947ed1272
// https://docs.google.com/document/d/1Nc1uwm7PgEWKQQTvlzlchDLQ1_HqLXO6irWnEVW7I6g/edit

// 1.1
// Solution:
//
DirectSession::~DirectSession() {
  if (!closed_) Close().IgnoreError();

  // 没有进入这个分支
  for (auto& it : partial_runs_) {
    it.second.reset(nullptr);
  }

  for (auto& it : executors_) {
    // 1.
    // DirectSession::executors_ 变量说明
    // std::unordered_map<string, std::shared_ptr<ExecutorsAndKeys>> # 重要

    // 2.
    // 我的设计, CPU-Only 和 GPU 的都往这里塞了。

    it.second.reset();
    // 1.
    // it.second 变量说明:
    // it.second: std::shared_ptr<ExecutorsAndKeys>
    // 这个的含义是 string key 对应的 executor

    // 2.
    // struct ExecutorsAndKeys 数据结构
    // 综述:An ExecutorsAndKeys is created for a given set of feeds/fetches.
    // 变量说明:
    //   ================================================
    // - items: std::vector<PerPartitionExecutorsAndLib>  # 最重要
    //   ================================================
    // - step_count: std::atomic_int_fast64_t
    //    - the number of times this graph is executed.
    // - graph: std::unique_ptr<Graph>
    //    - the entire graph being executed.
    // - name_to_node: NameNodeMap
    //    - maps node name to node.
    //      We keep 'graph' and 'name_to_node' only in the case of partial runs.
    //      Each item in 'items' is the executor for a partition of the graph
    //      bundled with its dependent library runtime.
    // - input_name_to_index: std::unordered_map<string, size_t>
    // - input_name_to_rendezvous_key: std::unordered_map<string, string>
    // - output_name_to_index: std::unordered_map<string, size_t>
    // - output_name_to_rendezvous_key: std::unordered_map<string, string>
    // - input_types: DataTypeVector
    //    - the rendezvous keys for the feeds
    // - output_types: DataTypeVector
    //    - rendezvous keys for the fetches.
    // - callable_options: CallableOptions
    // - collective_graph_key: int64, default: BuildGraphOptions::kNoCollectiveGraphKey

  }

  callables_.clear();
  // 1.
  // DirectSession::callables_ 变量说明:
  //  - callables_: std::unordered_map<int64, Callable>

  // 2.
  // DirectSession:: struct Callable 数据结构
  // - executors_and_keys: std::shared_ptr<ExecutorsAndKeys>
  // - function_info: std::shared_ptr<FunctionInfo>

  // 3.
  // 我新增的 std::unordered_map<int64, Callable> low_priority_callables_


  for (auto d : device_mgr_->ListDevices()) {
    d->op_segment()->RemoveHold(session_handle_);
  }

  for (auto d : device_mgr_->ListDevices()) {
    d->ClearResourceMgr();
  }

  functions_.clear();

  delete cancellation_manager_;

  for (const auto& p_and_owned : thread_pools_) {
    if (p_and_owned.second) delete p_and_owned.first;
  }

  execution_state_.reset(nullptr);
  // 1.
  // execution_state_ 变量说明:
  // DirectSession::execution_state_: std::unique_ptr<GraphExecutionState>

  // 2.
  // 我新增的 std::unique_ptr<GraphExecutionState> low_priority_execution_state_

  flib_def_.reset(nullptr);
}


// 我看这个函数的目的是:
// 我想弄明白 DirectSession::execution_state_ 是怎么被初始化的，里面的图是哪来的
// 我这下子明白了，其实是把 TF_Session 内的 Graph 转换成 GraphDef (仅仅因为函数接口需要吧)
// 然后接着 GraphDef 内的信息把 DirectSession::execution_state_ 给初始化了。

// QQQ. 请问，DirectSession::execution_state_ 日后被用在了那里?
// AAA.
Status DirectSession::MaybeInitializeExecutionState(
    const GraphDef& graph, // input
    bool* out_already_initialized) { // output

  // If already initialized, do nothing.
  if (flib_def_ && execution_state_) {
    *out_already_initialized = true;
    // -----------------
    return Status::OK();
    // -----------------
  }

  // Set up the per-session execution state.
  // NOTE(mrry): The function library created here will be used for
  // all subsequent extensions of the graph.

  // 1.
  // QQQ. flib_def_ 这个数据结构究竟是干嘛的?
  // AAA.
  flib_def_.reset(
      new FunctionLibraryDefinition(OpRegistry::Global(), graph.library()));
  // 1.
  // flib_def_ 变量说明:
  // flib_def_: std::unique_ptr<FunctionLibraryDefinition>
  //
  // 概述:
  // The function library, before any rewrites or optimizations have been
  // performed. In particular, CreateGraphs() may need to modify the function
  // library; it copies and modifies the function library.

  // 2.
  // FunctionLibraryDefinition 数据结构
  // tensorflow/core/framework/function.h:313:
  // class FunctionLibraryDefinition : public OpRegistryInterface
  // - struct FunctionDefAndOpRegistration
  // - default_registry_: const OpRegistryInterface* const
  // - function_defs_: gtl::FlatMap<string, std::unique_ptr<FunctionDefAndOpRegistration>>
  // - func_grad_: gtl::FlatMap<string, string>

  // 3.
  // struct FunctionDefAndOpRegistration 数据结构 ( within class FunctionLibraryDefinition)
  //
  // FunctionDefAndOpRegistration 数据结构
  // struct FunctionDefAndOpRegistration {
  //   FunctionDef fdef;
  //   OpRegistrationData op_registration_data;
  // }
  //
  // 例子
  // string: name of the function
  // second: function definition
  // example: https://gist.github.com/shizukanaskytree/8660a751d41a25db1848ec1be20dfd1e

  // 4.
  // FunctionDef 数据结构
  // tensorflow/core/framework/function.proto
  // - signature: OpDef
  //   The definition of the function's name, arguments, return values, attrs etc.
  // - attr: map<string, AttrValue> attr
  //   Attributes specific to this function definition.
  // - node_def: repeated NodeDef
  //   In both of the following fields, there is the need to specify an
  //   output that is used as either the input to another node (in
  //   `node_def`) or as a return value of the function (in `ret`).
  // - ret: map<string, string>
  //   A mapping from the output arg names from `signature` to the
  //   outputs from `node_def` that should be returned by the function.
  // - control_ret: map<string, string>
  //   A mapping from control output names from `signature` to node names in
  //   `node_def` which should be control outputs of this function.

  // 5.
  // struct OpRegistrationData 数据结构
  // tensorflow/core/framework/op_def_builder.h
  // - OpDef op_def;
  // - OpShapeInferenceFn shape_inference_fn;
  // - bool is_function_op = false;

  // 6.
  // OpDef 数据结构
  // tensorflow/core/framework/op_def.proto


  GraphExecutionStateOptions options;
  // 1.
  // GraphExecutionStateOptions 数据结构
  // tensorflow/core/common_runtime/graph_execution_state.h:41:struct GraphExecutionStateOptions
  // - device_set: const DeviceSet*
  // - session_options: const SessionOptions*
  // - session_handle: string
  //   Unique session identifier. Can be empty.
  // - stateful_placements
  //   A map from node name to device name, representing the unchangeable
  //   placement of stateful nodes.

  options.device_set = &device_set_;
  // 1.
  // device_set_ 变量说明:
  // DeviceSet device_set_;

  // 2.
  // class DeviceSet 数据结构
  // tensorflow/core/common_runtime/device_set.h
  // - devices_: td::vector<Device*> , not owned
  // - device_by_name_: std::unordered_map<string, Device*>
  //   Fullname -> device* for device in devices_.
  // - client_device_: Device*
  //   The device designated as the "client".
  //   client_device_ points to an element of devices_ that we consider
  //   to be the client device (in this local process).
  //
  // 打印
  // 只有 CPU , GPU 不可见下
  /*
  $2 = {
    devices_ = std::vector of length 2,
    capacity 2 = {
      0x5573b1bf6300,
      0x5573b37f6570
    },
    device_by_name_ = std::unordered_map with 4 elements = {
      ["/job:localhost/replica:0/task:0/xla_cpu:0"] = 0x5573b37f6570,
      ["/job:localhost/replica:0/task:0/device:XLA_CPU:0"] = 0x5573b37f6570,
      ["/job:localhost/replica:0/task:0/device:CPU:0"] = 0x5573b1bf6300,
      ["/job:localhost/replica:0/task:0/cpu:0"] = 0x5573b1bf6300
    },
    client_device_ = 0x5573b1bf6300
  }
  */
  //
  // 4 个 GPU 可见下
  /*
  $2 = {
    devices_ = std::vector of length 10,
    capacity 16 = {
      0x5600134775d0,
      0x560015081440,
      0x5600152d7ca0,
      0x5600152e07c0,
      0x5600152e8a90,
      0x5600152f1a80,
      0x5600153086d0,
      0x560015319fc0,
      0x5600153277f0,
      0x5600153350f0
    },
    device_by_name_ = std::unordered_map with 20 elements = {
      ["/job:localhost/replica:0/task:0/gpu:3"] = 0x5600153350f0,
      ["/job:localhost/replica:0/task:0/device:GPU:3"] = 0x5600153350f0,
      ["/job:localhost/replica:0/task:0/device:GPU:2"] = 0x5600153277f0,
      ["/job:localhost/replica:0/task:0/device:XLA_GPU:0"] = 0x5600152d7ca0,
      ["/job:localhost/replica:0/task:0/xla_cpu:0"] = 0x560015081440,
      ["/job:localhost/replica:0/task:0/device:XLA_GPU:3"] = 0x5600152f1a80,
      ["/job:localhost/replica:0/task:0/device:XLA_CPU:0"] = 0x560015081440,
      ["/job:localhost/replica:0/task:0/device:CPU:0"] = 0x5600134775d0,
      ["/job:localhost/replica:0/task:0/cpu:0"] = 0x5600134775d0,
      ["/job:localhost/replica:0/task:0/gpu:2"] = 0x5600153277f0,
      ["/job:localhost/replica:0/task:0/xla_gpu:3"] = 0x5600152f1a80,
      ["/job:localhost/replica:0/task:0/xla_gpu:0"] = 0x5600152d7ca0,
      ["/job:localhost/replica:0/task:0/xla_gpu:1"] = 0x5600152e07c0,
      ["/job:localhost/replica:0/task:0/device:XLA_GPU:1"] = 0x5600152e07c0,
      ["/job:localhost/replica:0/task:0/gpu:1"] = 0x560015319fc0,
      ["/job:localhost/replica:0/task:0/gpu:0"] = 0x5600153086d0,
      ["/job:localhost/replica:0/task:0/device:XLA_GPU:2"] = 0x5600152e8a90,
      ["/job:localhost/replica:0/task:0/xla_gpu:2"] = 0x5600152e8a90,
      ["/job:localhost/replica:0/task:0/device:GPU:1"] = 0x560015319fc0,
      ["/job:localhost/replica:0/task:0/device:GPU:0"] = 0x5600153086d0
    },
    client_device_ = 0x5600134775d0
  }
  */

  // TODO 提示:
  // Device* client_device() const { return client_device_; }
  // 我改写的只需要 client_device 就行了。
  /*
  p device_set_.client_device()
  $3 = (tensorflow::GPUCompatibleCPUDevice *) 0x5600134775d0

  p device_set_.client_device()->name()
  $4 = "/job:localhost/replica:0/task:0/device:CPU:0"
  */

  // 3.
  // Device 数据结构
  // class Device : public DeviceBase
  // tensorflow/core/common_runtime/device.h
  // 重要接口
  // - Compute
  // - ComputeAsync
  // - FillContextMap
  // - resource_manager
  // 成员变量:
  // - device_mgr_: DeviceMgr*
  // - device_attributes_: const DeviceAttributes
  // - parsed_name_: DeviceNameUtils::ParsedName
  // - op_seg_: OpSegment
  // - rmgr_: ResourceMgr*

  // 3.1
  // message DeviceAttributes
  // tensorflow/core/framework/device_attributes.proto
  // - name: string
  // - device_type: string
  // - memory_limit: int64
  // - locality: DeviceLocality
  // - incarnation: fixed64
  // - physical_device_desc: string

  // 4.
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

  // 5.
  // class DeviceContext: public core::RefCounted
  // tensorflow/core/framework/device_base.h
  // 概述:
  // 接口定义
  //
  // 接口:
  // - stream()
  // - MaintainLifetimeOnStream
  // - CopyCPUTensorToDevice
  // - CopyTensorInSameDevice
  // - CopyDeviceTensorToCPU
  // - ThenExecute

  options.session_options = &options_;
  // 1.
  // options_ 变量说明:
  // options_: const SessionOptions

  // 2.
  // SessionOptions 数据结构
  // tensorflow/core/public/session_options.h:28:struct SessionOptions
  // - config: ConfigProto
  //   sess.run Configuration options.
  // - target: string
  // - env: Env*

  options.session_handle = session_handle_;
  // 1.
  // session_handle_ 变量说明:
  // Unique session identifier.
  // session_handle_: string

  // TODO(mrry,suharshs): We explicitly copy `graph` so that
  // `MakeForBaseGraph()` can take ownership of its
  // contents. Previously this happened implicitly in calls to the
  // `GraphExecutionState`. Other sessions call
  // `MakeForBaseGraph` in such a way that we can destructively read
  // the passed-in `GraphDef`. In principle we could do the same here,
  // with a wider refactoring; we might revise the direct session so
  // that it copies the graph fewer times.

  GraphDef temp(graph);
  // temp 变量说明:
  // 最后这个 temp 一直传到 构造 GraphExecutionState::GraphExecutionState 时，
  // 会使用 Swap 把这个 temp 自动换成空的变量，这就是 GraphExecutionState 构造函数内 Swap 的意图。

  TF_RETURN_IF_ERROR(
    /*static*/ GraphExecutionState::MakeForBaseGraph(
      &temp,   // input
      options, // input
      &execution_state_)); // output
  // 1.
  // GraphExecutionState::MakeForBaseGraph 函数说明
  //
  // Creates a new `GraphExecutionState` for the given
  // `graph_def`, which represents the entire graph for a session.
  //
  // N.B. This method uses `GraphDef::Swap()` and leaves `graph_def`
  // in an undefined state. If it is necessary to use `*graph_def`
  // after this call, make an explicit copy of the graph before
  // calling this method.

  // 2.
  // execution_state_ 变量说明:
  // DirectSession::execution_state_: std::unique_ptr<GraphExecutionState>

  // 3.
  // GraphExecutionState 数据结构
  // tensorflow/core/common_runtime/graph_execution_state.cc
  // - graph_: Graph*
  //    - The dataflow graph owned by this object.
  // - rewrite_metadata_: std::unique_ptr<subgraph::RewriteGraphMetadata>
  //    - `rewrite_metadata_` is only set for GraphExecutionState objects created by `MakeForPrunedGraph()`.
  // - flib_def_: std::unique_ptr<FunctionLibraryDefinition>
  //    - 'flib_def_' is initialized from the initial graph def's library, and may be updated by a graph optimization pass.
  // - node_name_to_cost_id_map_: NodeNameToCostIdMap
  //    - Map from name to Node for the full graph in placed_.
  // - session_handle_: string
  //    -  Unique session identifier. Can be empty.
  // - session_options_: const SessionOptions*
  // - device_set_: const DeviceSet*
  // - original_graph_def_: GraphDef
  // - stateful_placements_: std::unordered_map<string, string>
  //    - Map of placed stateful nodes, i.e. nodes for which is_stateful()
  //    - is true, such as "params" and "queue" nodes.  Once placed these
  //    - nodes can not be moved to a different device.  Maps node names to
  //    - device names.
  //
  // 数据结构说明
  // GraphExecutionState is responsible for generating an
  // executable ClientGraph from the original GraphDef that specifies
  // the complete graph and from BuildGraphOptions which specifies
  // input/output nodes.
  //
  // An executable Graph differs from a GraphDef by being Placed,
  // meaning that each Node is assigned to a single Device in the
  // available set.
  //
  // When GraphExecutionState is first constructed it instantiates
  // a full Graph from the provided GraphDef, and places it, using only
  // the static device assignments from the GraphDef.  Nodes without are
  // currently placed in a very naive way.  Since stateful Nodes cannot
  // be moved after initial placement, it is important that stateful
  // Nodes get sensible initial device assignments in the graph
  // definition.
  //
  // Subsequently, GraphExecutionState generates a SimpleClientGraph on
  // demand, which is a sub-graph of the latest placement of the full
  // Graph.  MasterSession uses such a ClientGraph to execute one or
  // more similar client requests.

  graph_created_ = true;
  *out_already_initialized = false;
  return Status::OK();
}


// 没有被用过的函数
Status DirectSession::Create(const GraphDef& graph) {
  TF_RETURN_IF_ERROR(init_error_);
  if (graph.node_size() > 0) {
    mutex_lock l(graph_state_lock_);
    if (graph_created_) {
      return errors::AlreadyExists(
          "A Graph has already been created for this session.");
    }
    return ExtendLocked(graph);
  }
  return Status::OK();
}

Status DirectSession::Extend(const GraphDef& graph) { // input
  TF_RETURN_IF_ERROR(CheckNotClosed());
  mutex_lock l(graph_state_lock_);
  return ExtendLocked(graph);
}

/** \brief According to the graph, update the Session internal attributes.
 *
 *  \param[in] graph: const GraphDef& ;
 *
 *  \details Update two attributes of the DirectSession object.
 *   - DirectSession::flib_def_: std::unique_ptr<FunctionLibraryDefinition>
 *   - DirectSession::execution_state_: std::unique_ptr<GraphExecutionState>
 */
Status DirectSession::ExtendLocked(const GraphDef& graph) { // input
  // 猜想
  // 之所以要有 GraphDef 是因为 python 和 c++ 直接的交互需要 protobuf
  // class Graph 只属于 c++ 不可能原封不动地和 python 交互。

  bool already_initialized;
  // If this is the first call, we can initialize the execution state
  // with `graph` and do not need to call `Extend()`.
  TF_RETURN_IF_ERROR(
      MaybeInitializeExecutionState(
        graph,  // input
        &already_initialized)); // output

  // 1.
  // Bug Report
  if (already_initialized) {
    // already_initialized 变量说明:
    // 第一次不会进入, 因为 MaybeInitializeExecutionState 内结尾处把它 设置为 false 了
    // 如果有新节点加入就会进入。
    TF_RETURN_IF_ERROR(flib_def_->AddLibrary(graph.library()));

    std::unique_ptr<GraphExecutionState> state;

    TF_RETURN_IF_ERROR(
      execution_state_->Extend(
                          graph,    // input
                          &state)); // output
    // 1.
    // execution_state_ 变量说明:
    // DirectSession::execution_state_: std::unique_ptr<GraphExecutionState>

    // 2.
    // GraphExecutionState 数据结构
    // tensorflow/core/common_runtime/graph_execution_state.cc
    // - graph_: Graph*
    //    - The dataflow graph owned by this object.
    // - rewrite_metadata_: std::unique_ptr<subgraph::RewriteGraphMetadata>
    //    - `rewrite_metadata_` is only set for GraphExecutionState objects created by `MakeForPrunedGraph()`.
    // - flib_def_: std::unique_ptr<FunctionLibraryDefinition>
    //    - 'flib_def_' is initialized from the initial graph def's library, and may be updated by a graph optimization pass.
    // - node_name_to_cost_id_map_: NodeNameToCostIdMap
    //    - Map from name to Node for the full graph in placed_.
    // - session_handle_: string
    //    -  Unique session identifier. Can be empty.
    // - session_options_: const SessionOptions*
    // - device_set_: const DeviceSet*
    // - original_graph_def_: GraphDef
    // - stateful_placements_: std::unordered_map<string, string>
    //    - Map of placed stateful nodes, i.e. nodes for which is_stateful()
    //    - is true, such as "params" and "queue" nodes.  Once placed these
    //    - nodes can not be moved to a different device.  Maps node names to
    //    - device names.
    //
    // 数据结构说明
    // GraphExecutionState is responsible for generating an
    // executable ClientGraph from the original GraphDef that specifies
    // the complete graph and from BuildGraphOptions which specifies
    // input/output nodes.
    //
    // An executable Graph differs from a GraphDef by being Placed,
    // meaning that each Node is assigned to a single Device in the
    // available set.
    //
    // When GraphExecutionState is first constructed it instantiates
    // a full Graph from the provided GraphDef, and places it, using only
    // the static device assignments from the GraphDef.  Nodes without are
    // currently placed in a very naive way.  Since stateful Nodes cannot
    // be moved after initial placement, it is important that stateful
    // Nodes get sensible initial device assignments in the graph
    // definition.
    //
    // Subsequently, GraphExecutionState generates a SimpleClientGraph on
    // demand, which is a sub-graph of the latest placement of the full
    // Graph.  MasterSession uses such a ClientGraph to execute one or
    // more similar client requests.

    // 3.
    // GraphExecutionState::Extend 函数说明:
    // tensorflow/core/common_runtime/graph_execution_state.cc

    execution_state_.swap(state);
  }

  return Status::OK();
}

Status DirectSession::Run(const NamedTensorList& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs) {
  RunMetadata run_metadata;
  return Run(RunOptions(), inputs, output_names, target_nodes, outputs,
             &run_metadata);
}

Status DirectSession::CreateDebuggerState(
    const CallableOptions& callable_options, int64 global_step,
    int64 session_run_index, int64 executor_step_index,
    std::unique_ptr<DebuggerStateInterface>* debugger_state) {
  TF_RETURN_IF_ERROR(DebuggerStateRegistry::CreateState(
      callable_options.run_options().debug_options(), debugger_state));
  std::vector<string> input_names(callable_options.feed().begin(),
                                  callable_options.feed().end());
  std::vector<string> output_names(callable_options.fetch().begin(),
                                   callable_options.fetch().end());
  std::vector<string> target_names(callable_options.target().begin(),
                                   callable_options.target().end());

  TF_RETURN_IF_ERROR(debugger_state->get()->PublishDebugMetadata(
      global_step, session_run_index, executor_step_index, input_names,
      output_names, target_names));
  return Status::OK();
}

Status DirectSession::DecorateAndPublishGraphForDebug(
    const DebugOptions& debug_options, Graph* graph, Device* device) {
  std::unique_ptr<DebugGraphDecoratorInterface> decorator;
  TF_RETURN_IF_ERROR(
      DebugGraphDecoratorRegistry::CreateDecorator(debug_options, &decorator));

  TF_RETURN_IF_ERROR(decorator->DecorateGraph(graph, device));
  TF_RETURN_IF_ERROR(decorator->PublishGraph(*graph, device->name()));
  return Status::OK();
}



// QQQ. 如果我要适应 RunInternal 的 ExecutorsAndKeys* executors_and_keys 的接口，
//      我该怎么做?

Status DirectSession::RunInternal(int64 step_id,
                                  const RunOptions& run_options,
                                  CallFrameInterface* call_frame,
                                  //-----------------------------------
                                  ExecutorsAndKeys* executors_and_keys,
                                  //-----------------------------------
                                  RunMetadata* run_metadata) {
  // 1.
  // ExecutorsAndKeys 数据结构
  // tensorflow/core/common_runtime/direct_session.h
  //
  // struct ExecutorsAndKeys {
  //   ExecutorsAndKeys() : step_count(0) {}
  //   std::atomic_int_fast64_t step_count;
  //   std::unique_ptr<Graph> graph;
  //   NameNodeMap name_to_node;
  //   // ------------------------------------------------------------------------
  //   // 最重要
  //   std::vector<PerPartitionExecutorsAndLib> items;
  //   // ------------------------------------------------------------------------
  //   std::unordered_map<string, size_t> input_name_to_index;
  //   std::unordered_map<string, string> input_name_to_rendezvous_key;
  //   std::unordered_map<string, size_t> output_name_to_index;
  //   std::unordered_map<string, string> output_name_to_rendezvous_key;
  //   DataTypeVector input_types;
  //   DataTypeVector output_types;
  //   // message type
  //   CallableOptions callable_options;
  //   int64 collective_graph_key = BuildGraphOptions::kNoCollectiveGraphKey;
  // };

  const uint64 start_time_usecs = Env::Default()->NowMicros();

  string session_id_meta = strings::StrCat("SessionRun #id=", step_id, "#");
  // 1.
  // 打印 session_id_meta
  // $2 = "SessionRun #id=1#"

  tracing::ScopedActivity activity(session_id_meta);

  const int64 executor_step_count = executors_and_keys->step_count.fetch_add(1);

  std::unique_ptr<DebuggerStateInterface> debugger_state;

  if (!run_options.debug_options().debug_tensor_watch_opts().empty()) {

    TF_RETURN_IF_ERROR(
        CreateDebuggerState(executors_and_keys->callable_options,
                            run_options.debug_options().global_step(), step_id,
                            executor_step_count, &debugger_state));
  }


  // Create a run state and start execution.
  RunState run_state(step_id, &devices_);
  // 1.
  // struct RunState 数据结构
  // tensorflow/core/common_runtime/direct_session.h
  //
  // 概述
  // For each live partial execution, the session maintains a RunState.
  // 'status' is the current status of this partial execution. **'executor_done'
  // is "notified" when all executors are done.** 'pending_inputs' are the set
  // of pending feeds and 'pending_outputs' are the set of pending fetches.
  //
  // - mu_: mutex
  // - rendez: IntraProcessRendezvous*
  // - collective_executor: std::unique_ptr<CollectiveExecutor::Handle>
  // - collector: std::unique_ptr<StepStatsCollector>
  //
  // -----------------------------------------------------------------------
  // - executors_done: Notification
  //   **'executor_done' is "notified" when all executors are done.**
  //   - executor_done 是分布式执行里面的，单机版的没有涉及到这个
  // -----------------------------------------------------------------------
  //
  // - pending_inputs: std::unordered_map<string, bool>
  //   true if fed
  // - pending_outputs: std::unordered_map<string, bool>
  //   true if fetched
  // - tensor_store: TensorStore
  // - step_container: ScopedStepContainer

  // 2.
  // DirectSession::RunState::RunState 构造函数说明
  // tensorflow/core/common_runtime/direct_session.cc

  // 3.
  // class TensorStore
  // core/framework/session_state.h:58
  //
  // 概述:
  // The tensor store remembers the tensors we choose to keep for the
  // current run call. It is available to every op kernel.
  //
  // * struct TensorAndKey
  //   - tensor: Tensor
  //   - id: int64
  //   - device_name: string
  // - lock_: mutex
  // - tensors_: std::unordered_map<string, TensorAndKey>
  //   The tensors that will be saved to session state when this run completes.
  //   A map from tensor string name to tensor.


  // -----------------------------------------------------------------------
  run_state.rendez = new IntraProcessRendezvous(device_mgr_.get());
  // -----------------------------------------------------------------------

#ifndef __ANDROID__
  // 进入了这个宏定义的分支:

  // Set up for collectives if ExecutorsAndKeys declares a key.
  if (executors_and_keys->collective_graph_key !=
      BuildGraphOptions::kNoCollectiveGraphKey) {
    // 没有进入了这个分支:

    if (run_options.experimental().collective_graph_key() !=
        BuildGraphOptions::kNoCollectiveGraphKey) {
      // If a collective_graph_key was specified in run_options, ensure that it
      // matches what came out of GraphExecutionState::BuildGraph().
      if (run_options.experimental().collective_graph_key() !=
          executors_and_keys->collective_graph_key) {
        return errors::Internal(
            "collective_graph_key in RunOptions ",
            run_options.experimental().collective_graph_key(),
            " should match collective_graph_key from optimized graph ",
            executors_and_keys->collective_graph_key);
      }
    }

    if (!collective_executor_mgr_) {
      std::unique_ptr<DeviceResolverInterface> drl(
          new DeviceResolverLocal(device_mgr_.get()));

      std::unique_ptr<ParamResolverInterface> cprl(
          new CollectiveParamResolverLocal(options_.config, device_mgr_.get(),
                                           drl.get(),
                                           "/job:localhost/replica:0/task:0"));

      collective_executor_mgr_.reset(new CollectiveExecutorMgr(
          options_.config, device_mgr_.get(), std::move(drl), std::move(cprl)));
    }

    run_state.collective_executor.reset(new CollectiveExecutor::Handle(
        collective_executor_mgr_->FindOrCreate(step_id), true /*inherit_ref*/));
  }
#endif

  // Start parallel Executors.
  // executors_and_keys->items 变量说明:
  // std::vector<PerPartitionExecutorsAndLib> items
  const size_t num_executors = executors_and_keys->items.size();

  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors,
      run_state.rendez,

      [&run_state](const Status& ret) {
        {
          mutex_lock l(run_state.mu_);
          run_state.status.Update(ret);
        }
        run_state.executors_done.Notify();

        // 1.
        // condition_variable 作用:
        // condition_variable is used in combination with a std::mutex to facilitate inter-thread communication.

      }
  );

  Executor::Args args;
  // 1.
  // Executor::Args 数据结构
  //  struct Args {
  //    int64 step_id = 0;
  //    Rendezvous* rendezvous = nullptr;
  //    StepStatsCollectorInterface* stats_collector = nullptr;
  //    CallFrameInterface* call_frame = nullptr;
  //    CancellationManager* cancellation_manager = nullptr;
  //    SessionState* session_state = nullptr;
  //
  //    // Unique session identifier. Can be empty.
  //    string session_handle;
  //    TensorStore* tensor_store = nullptr;
  //    ScopedStepContainer* step_container = nullptr;
  //    CollectiveExecutor* collective_executor = nullptr;
  //
  //    // If true, calls Sync() on the device.
  //    bool sync_on_finish = false;
  //
  //    typedef std::function<void()> Closure;
  //    typedef std::function<void(Closure)> Runner;
  //    Runner runner = nullptr;
  //  };

  args.step_id = step_id;
  args.call_frame = call_frame;
  args.rendezvous = run_state.rendez;
  args.collective_executor =
      (run_state.collective_executor ? run_state.collective_executor->get()
                                     : nullptr);

  // -----------------------------------------------------------------------
  CancellationManager step_cancellation_manager;
  // 1.
  // CancellationManager 构造函数说明:
  // CancellationManager::CancellationManager()
  // : is_cancelling_(false),
  //   is_cancelled_(false),
  //   next_cancellation_token_(0) {}
  //
  // - is_cancelling_ : false
  // - is_cancelled_: false
  // - next_cancellation_token_: 0

  args.cancellation_manager = &step_cancellation_manager;
  // -----------------------------------------------------------------------
  // 1.
  // class CancellationManager 数据结构
  // tensorflow/core/framework/cancellation.h
  // 成员变量
  // - is_cancelling_: bool
  // - is_cancelled_: std::atomic_bool
  // - mu_: mutex
  // - cancelled_notification_: Notification
  // - next_cancellation_token_: CancellationToken
  // - callbacks_: gtl::FlatMap<CancellationToken, CancelCallback>
  //
  // 部分的接口函数
  // - StartCancel()
  //   Run all callbacks associated with this manager.
  // - IsCancelled()
  //   Returns true iff StartCancel() has been called.
  // - Reset()
  //   Resets the cancellation manager to its original pre-cancelled state.
  // - get_cancellation_token()
  //   Returns a token that must be used in calls to RegisterCallback
  //   and DeregisterCallback.
  // - RegisterCallback
  //   Attempts to register the given callback to be invoked when this
  //   manager is cancelled.
  // - ...

  args.session_state = &session_state_;
  args.session_handle = session_handle_;
  args.tensor_store = &run_state.tensor_store;
  args.step_container = &run_state.step_container;
  args.sync_on_finish = sync_on_finish_;

  //////////////////////////////////////////////////////////////////////////
  // 这里决定了是否启动 traing 和收集数据
  // stats_collector 是开关。
  //////////////////////////////////////////////////////////////////////////
  const bool do_trace = (run_options.trace_level() > RunOptions::NO_TRACE);
  //////////////////////////////////////////////////////////////////////////

  bool update_cost_model = false;

  if (options_.config.graph_options().build_cost_model() > 0) {
    // 如果没开 build_cost_model 的话就不进入！

    const int64 build_cost_model_every =
        options_.config.graph_options().build_cost_model();

    const int64 build_cost_model_after =
        options_.config.graph_options().build_cost_model_after();

    int64 measure_step_count = executor_step_count - build_cost_model_after;

    if (measure_step_count >= 0) {
      update_cost_model =
          ((measure_step_count + 1) % build_cost_model_every == 0);
    }
  }

  //////////////////////////////////////////////////////////////////////////
  // 这里决定了是否启动 traing 和收集数据
  // stats_collector 是开关。
  //////////////////////////////////////////////////////////////////////////
  if (do_trace || update_cost_model ||
      run_options.report_tensor_allocations_upon_oom()) {
    // 如果没开 build_cost_model 的话就不进入！

    run_state.collector.reset(
        new StepStatsCollector(run_metadata->mutable_step_stats()));
    args.stats_collector = run_state.collector.get();
  }
  //////////////////////////////////////////////////////////////////////////

  // -----------------------------------------------------------------------
  // tracer 是用来 profiling GPU cuda kernel/memcpy 的！！！
  std::unique_ptr<DeviceTracer> tracer;
  // -----------------------------------------------------------------------

  if (run_options.trace_level() >= RunOptions::HARDWARE_TRACE) {

    // -----------------------------------------------------------------------
    // tracer 是用来 profiling GPU cuda kernel/memcpy 的！！！
    tracer = CreateDeviceTracer();
    // -----------------------------------------------------------------------

    // tracer may be NULL on platforms without accelerators.
    if (tracer) {
      Status s = tracer->Start();
      if (!s.ok()) {
        run_state.executors_done.Notify();
        delete barrier;
        return s;
      }
    }
  }

  // 异常处理，可以不看
  if (run_options.inter_op_thread_pool() < -1 ||
      run_options.inter_op_thread_pool() >=
          static_cast<int32>(thread_pools_.size())) {

    run_state.executors_done.Notify();
    delete barrier;
    return errors::InvalidArgument("Invalid inter_op_thread_pool: ",
                                   run_options.inter_op_thread_pool());
  }

  // Register this step with session's cancellation manager, so that
  // `Session::Close()` will cancel the step.
  const CancellationToken cancellation_token =
      cancellation_manager_->get_cancellation_token();
  // 1.
  // CancellationToken 数据结构
  // typedef int64 CancellationToken;
  // tensorflow/core/framework/cancellation.h

  // 2.
  // CancellationManager::get_cancellation_token 函数说明:
  // tensorflow/core/framework/cancellation.cc
  // return next_cancellation_token_++;

  // 2.1
  // return next_cancellation_token_++; 说明和注意！
  // return 后 ++ .
  // 所以返回的是 0, 但是 next_cancellation_token_ 已经是 1！

  // 3.
  // cancellation_manager_ 变量说明:
  // cancellation_manager_: CancellationManager*

  // 4.
  // class CancellationManager 数据结构
  // tensorflow/core/framework/cancellation.h
  // 成员变量
  // - is_cancelling_: bool
  // - is_cancelled_: std::atomic_bool
  // - mu_: mutex
  // - cancelled_notification_: Notification
  // - next_cancellation_token_: CancellationToken
  // - callbacks_: gtl::FlatMap<CancellationToken, CancelCallback>
  //
  // 部分的接口函数
  // - StartCancel()
  //   Run all callbacks associated with this manager.
  // - IsCancelled()
  //   Returns true iff StartCancel() has been called.
  // - Reset()
  //   Resets the cancellation manager to its original pre-cancelled state.
  // - get_cancellation_token()
  //   Returns a token that must be used in calls to RegisterCallback
  //   and DeregisterCallback.
  // - RegisterCallback
  //   Attempts to register the given callback to be invoked when this
  //   manager is cancelled.
  // - ...

  // 5.
  // QQQ. cancellation_manager_ 是什么时候被构造的?
  // AAA.
  // 在 DirectSession::DirectSession, cancellation_manager_(new CancellationManager())

  // 5.1
  // CancellationManager 构造函数
  // CancellationManager::CancellationManager()
  //     : is_cancelling_(false),
  //       is_cancelled_(false),
  //       next_cancellation_token_(0) {}

  const bool already_cancelled = !cancellation_manager_->RegisterCallback(
      cancellation_token, // 第一次进入时值为 0
      [&step_cancellation_manager]() {
        step_cancellation_manager.StartCancel();
        // 1.
        // step_cancellation_manager 变量说明:
        // class CancellationManager

        // 2.
        // void CancellationManager::StartCancel()
        // tensorflow/core/framework/cancellation.cc
        //
        // 概述:
        // 执行注册的 CancelCallback lambda function
      });
  // 1.
  // RegisterCallback 函数说明:
  // tensorflow/core/framework/cancellation.cc
  //
  // Attempts to register the given callback to be invoked when this
  // manager is cancelled. Returns **true** if the callback was
  // registered; returns **false** if this manager was already cancelled,
  // and the callback was not registered.
  //
  // **If this method returns false, it is the caller's responsibility
  // to perform any cancellation cleanup.**
  //
  // This method is tricky to use correctly.
  //
  // bool CancellationManager::RegisterCallback(CancellationToken token, // int 类型
  //                                            CancelCallback callback)
  //

  // 2.
  // already_cancelled 打印
  // p already_cancelled
  // $2 = false

  if (already_cancelled) {
    // NOTE(mrry): If we don't explicitly notify
    // `run_state.executors_done`, the RunState destructor would
    // block on this notification.
    run_state.executors_done.Notify();
    // 1.
    // run_state.executors_done 变量说明:
    // struct RunState::executors_done: Notification
    // tensorflow/core/platform/default/notification.h
    //
    // 1.1
    // run_state.executors_done.Notify(); 主要功能
    // 唤醒 main thread 继续执行
    // tf main thread              Other threads
    //       +                   +  +  +        +
    //       |                   |  |  |        |
    //       |                   |  |  | ...... |
    //       |                   |  |  |        |
    //       |                   v  v  v        v
    //       v                  ------------------
    //   ---------                     执行任务
    // main thread 卡在这里
    //
    // run_state.executors_done.Notify() 后继续执行
    // run_state 的 executors_done Notification 变量 notify_all 后让 main thread 继续执行。

    // 2.
    // How to use a cv?
    // 2.1 p_cond_wait(cv, m), signal
    //
    // 2.2 Condition variables are used with a mutex and with a loop (to check a condition).
    // 2.3 Condition variables allow a set of threads to sleep until tickled! You can tickle one thread or all threads that are sleeping. If you only wake one thread then the operating system will decide which thread to wake up. You don't wake threads directly instead you 'signal' the condition variable, which then will wake up one (or all) threads that are sleeping inside the condition variable.

    // 3.
    // class Notification
    // - cv_ : condition_variable
    //   signaled when notified_ becomes non-zero
    // - notified_: std::atomic<bool>
    //   mutations under mu_
    // - mu_: mutex
    //
    // 接口函数
    // - Notify()
    // - HasBeenNotified()
    // - WaitForNotification()

    // 4.
    // QQQ. Notify 唤醒了谁？
    // AAA.
    // 线程停止了 WaitForNotification 这里，所以是 唤醒了所有的卡在那里的线程
    // https://docs.google.com/document/d/1IF_r6yGBnNkgklq0GOslw4pzt6DuFhZgk2zuuJr1ndg/edit#heading=h.dg2e1t53gi2y

    delete barrier;
    return errors::Cancelled("Run call was cancelled");
  }

  // 看来这个是要根据用户自己选的，首先是可以创建多个 threadpool 的，但是如果用户不主动设置
  // 那么这个就是用的 0 ，因为 int32 inter_op_thread_pool 默认值就是 0.
  // 注意这里, 这里因为我们使用的是 GlobalThreadPool, 所以默认这个 vector 的第一个
  // 就是那个 threadpool, i.e., thread_pools_[0]
  thread::ThreadPool* pool =
      run_options.inter_op_thread_pool() >= 0
          ? thread_pools_[run_options.inter_op_thread_pool()].first  // 结果是这个
          : nullptr;
  // 1.
  // inter_op_thread_pool 变量说明:
  // tensorflow/core/protobuf/config.proto
  //
  // The thread pool to use, if session_inter_op_thread_pool is configured.
  // To use the caller thread set this to -1 - this uses the caller thread
  // to execute Session::Run() and thus avoids a context switch. Using the
  // caller thread to execute Session::Run() should be done ONLY for simple
  // graphs, where the overhead of an additional context switch is
  // comparable with the overhead of Session::Run().
  // int32 inter_op_thread_pool = 3;
  //
  // 所以这里如果用户不设置的话，inter_op_thread_pool 默认值就是 0 了

  // 2.
  // run_options 变量说明
  // run_options: message RunOptions
  // Options for a single Run() call.

  if (pool == nullptr) {
    // We allow using the caller thread only when having a single executor
    // specified.
    if (executors_and_keys->items.size() > 1) {
      pool = thread_pools_[0].first;
    } else {
      VLOG(1) << "Executing Session::Run() synchronously!";
    }
  }

  std::unique_ptr<RunHandler> handler;

  if (ShouldUseRunHandlerPool(run_options) &&
      run_options.experimental().use_run_handler_pool()) {
    // 未进入这个分支:
    VLOG(1) << "Using RunHandler to scheduler inter-op closures.";
    handler = GetOrCreateRunHandlerPool(options_)->Get();
  }

  auto* handler_ptr = handler.get();
  // 1.
  // handler_ptr type: RunHandler

  Executor::Args::Runner default_runner = nullptr;

  if (pool == nullptr) {
    // 未进入这个分支
    default_runner = [](Executor::Args::Closure c) { c(); };
  } else if (handler_ptr != nullptr) {
    // 未进入这个分支
    default_runner = [handler_ptr](Executor::Args::Closure c) {
      handler_ptr->ScheduleInterOpClosure(std::move(c));
    };
  } else {
    // 进入这个分支！
    // -----------------------------------------------------------------------
    default_runner = [this, pool](Executor::Args::Closure c) {
      SchedClosure(pool, std::move(c));
    };
    // -----------------------------------------------------------------------
  }

  // 这里要执行 executor 了
  for (const auto& item : executors_and_keys->items) {
    // TODO(azaks): support partial run.
    // TODO(azaks): if the device picks its own threadpool, we need to assign
    //     less threads to the main compute pool by default.
    thread::ThreadPool* device_thread_pool =
        item.device->tensorflow_device_thread_pool();

    // TODO(crk): Investigate usage of RunHandlerPool when using device specific
    // thread pool(s).
    if (!device_thread_pool) {
      // 注意: 进入的是这个分支！！！

      args.runner = default_runner;

    } else {
      // 未进入这个分支:
      args.runner = [this, device_thread_pool](Executor::Args::Closure c) {
        SchedClosure(device_thread_pool, std::move(c));
      };

    }


    // 异步执行 executor
    // 重要
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    item.executor->RunAsync(args, barrier->Get());
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // 1.
    // item.executor 变量说明:
    // DirectSession::PerPartitionExecutorsAndLib::executor : std::unique_ptr<Executor>

    // 1.1
    // Executor 实际类型:
    // Executor real type is ExecutorImpl

    // 2.
    // RunAsync 函数说明:
    // ExecutorImpl::RunAsync
    // tensorflow/core/common_runtime/executor.cc

    // 3.
    // class ExecutorImpl 数据结构
    // tensorflow/core/common_runtime/executor.cc:412
    // class ExecutorImpl : public Executor

    // 4.
    // args 变量说明:
    // defined above : Executor::Args args;
    // 1.
    // Executor::Args 数据结构
    //  struct Args {
    //    int64 step_id = 0;
    //    Rendezvous* rendezvous = nullptr;
    //    StepStatsCollectorInterface* stats_collector = nullptr;
    //    CallFrameInterface* call_frame = nullptr;
    //    CancellationManager* cancellation_manager = nullptr;
    //    SessionState* session_state = nullptr;
    //
    //    // Unique session identifier. Can be empty.
    //    string session_handle;
    //    TensorStore* tensor_store = nullptr;
    //    ScopedStepContainer* step_container = nullptr;
    //    CollectiveExecutor* collective_executor = nullptr;
    //
    //    // If true, calls Sync() on the device.
    //    bool sync_on_finish = false;
    //
    //    typedef std::function<void()> Closure;
    //    typedef std::function<void(Closure)> Runner;
    //    Runner runner = nullptr;
    //  };

    // 5.
    // barrier->Get() 函数说明:
    // Returns a closure that Executors must call when they are done
    // computing, passing the status of their execution as an argument.
    //
    // 定义:
    // executor.h
    // StatusCallback Get() {
    //   return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1);
    // }

    // 5.1
    // StatusCallback 数据结构
    // tensorflow/core/common_runtime/executor.h:150:
    // typedef std::function<void(const Status&)> StatusCallback;

    // 6.
    // barrier 变量说明:
    // ExecutorBarrier* barrier

    // 7.
    // class ExecutorBarrier 数据结构
    // tensorflow/core/common_runtime/executor.h
    //
    // 概述:
    // A class to help run multiple executors in parallel and wait until
    // all of them are complete.
    //
    // ExecutorBarrier deletes itself after the function returned by Get()
    // is called.
    //
    // - rendez_: Rendezvous* , default value : nullptr
    // - done_cb_: StatusCallback, default value : nullptr
    // - mu_: mutable mutex
    // - pending_: int
    // - status_group_: StatusGroup
    //
    // 重要接口:
    // - WhenDone(const Status& s)
    // - StatusCallback Get()
  }

  // tf main thread              Other threads
  //       +                   +  +  +        +
  //       |                   |  |  |        |
  //       |                   |  |  | ...... |
  //       |                   |  |  |        |
  //       |                   v  v  v        v
  //       v
  //  WaitForNotification   ExecutorImpl::RunAsync

  WaitForNotification(&run_state,  // input
                      &step_cancellation_manager,  // input
                      run_options.timeout_in_ms() > 0  // input
                          ? run_options.timeout_in_ms()
                          : operation_timeout_in_ms_);
  // 1.
  // WaitForNotification 函数接口说明:
  // tensorflow/core/common_runtime/direct_session.cc
  //
  // void DirectSession::WaitForNotification(RunState* run_state, // input
  //                                         CancellationManager* cm, // input
  //                                         int64 timeout_in_ms) // input
  //


  // =======================================================================
  if (!cancellation_manager_->DeregisterCallback(cancellation_token)) {
  // =======================================================================
  // 1.
  // DeregisterCallback 函数说明:
  // 1.1
  // 我调了 3 天的 bug, 奇怪的，不能删除上面这句！否则 core dump

  // 1.2
  // 只能注释掉下面的，因为 被 Cancelled 不能当做是 errors status 来看待。

  // 1.3
  // tensorflow/core/framework/cancellation.h
  // bool DeregisterCallback(CancellationToken token);
  //
  // Deregister the callback that, when registered, was associated
  // with the given cancellation token. Returns true iff the callback
  // was deregistered and will not be invoked; otherwise returns false
  // after the callback has been invoked, blocking if necessary.
  //
  // NOTE(mrry): This method may block if cancellation is in progress.
  // The caller of this method must not hold any mutexes that are required
  // to invoke any cancellation callback that has been registered with this
  // cancellation manager.

  // 1.4
  // 这个函数的功能好像就是等待或者 return true or false 以表征这步是否被 cancelled.

    // The step has been cancelled: make sure we don't attempt to receive the
    // outputs as this would make it block forever.
    mutex_lock l(run_state.mu_);
    run_state.status.Update(errors::Cancelled("Run call was cancelled"));

  }


  /// 上述的所有 op node 都跑完了，所以可以开始收集数据了。
  if (tracer) {
    TF_RETURN_IF_ERROR(tracer->Stop());
    TF_RETURN_IF_ERROR(tracer->Collect(run_state.collector.get()));
  }

  {
    mutex_lock l(run_state.mu_);
    TF_RETURN_IF_ERROR(run_state.status);
  }

  // Save the output tensors of this run we choose to keep.
  if (!run_state.tensor_store.empty()) {
    // 1.
    // run_state 变量说明
    // run_state : struct RunState

    // 2.
    // struct RunState 数据结构
    // tensorflow/core/common_runtime/direct_session.h
    //
    // 概述
    // For each live partial execution, the session maintains a RunState.
    // 'status' is the current status of this partial execution. **'executor_done'
    // is "notified" when all executors are done.** 'pending_inputs' are the set
    // of pending feeds and 'pending_outputs' are the set of pending fetches.
    //
    // - mu_: mutex
    // - rendez: IntraProcessRendezvous*
    // - collective_executor: std::unique_ptr<CollectiveExecutor::Handle>
    // - collector: std::unique_ptr<StepStatsCollector>
    // - executors_done: Notification
    //   **'executor_done' is "notified" when all executors are done.**
    //   - executor_done 是分布式执行里面的，单机版的没有涉及到这个
    // - pending_inputs: std::unordered_map<string, bool>
    //   true if fed
    // - pending_outputs: std::unordered_map<string, bool>
    //   true if fetched
    // -----------------------------------------------------------------------
    // - tensor_store: TensorStore
    // -----------------------------------------------------------------------
    // - step_container: ScopedStepContainer

    // 3.
    // tensor_store 变量说明:
    // tensor_store: TensorStore

    // 4.
    // class TensorStore
    // core/framework/session_state.h:58
    //
    // 概述:
    // The tensor store remembers the tensors we choose to keep for the
    // current run call. It is available to every op kernel.
    //
    // * struct TensorAndKey
    //   - tensor: Tensor
    //   - id: int64
    //   - device_name: string
    // - lock_: mutex
    // - tensors_: std::unordered_map<string, TensorAndKey>
    //   The tensors that will be saved to session state when this run completes.
    //   A map from tensor string name to tensor.


    TF_RETURN_IF_ERROR(
      run_state.tensor_store.SaveTensors(
        {executors_and_keys->callable_options.fetch().begin(),
         executors_and_keys->callable_options.fetch().end()},
        &session_state_));
  }

  if (run_state.collector) {
    run_state.collector->Finalize();
  }

  // Build and return the cost model as instructed.
  if (update_cost_model) {
    // Build the cost model
    std::unordered_map<string, const Graph*> device_to_graph;
    for (const PerPartitionExecutorsAndLib& partition :
         executors_and_keys->items) {
      const Graph* graph = partition.graph;
      const string device = partition.flib->device()->name();
      device_to_graph[device] = graph;
    }

    mutex_lock l(executor_lock_);
    run_state.collector->BuildCostModel(&cost_model_manager_, device_to_graph);

    // annotate stats onto cost graph.
    CostGraphDef* cost_graph = run_metadata->mutable_cost_graph();
    for (const auto& item : executors_and_keys->items) {
      TF_RETURN_IF_ERROR(
          cost_model_manager_.AddToCostGraphDef(item.graph, cost_graph));
    }
  }

  // If requested via RunOptions, output the partition graphs.
  if (run_options.output_partition_graphs()) {
    protobuf::RepeatedPtrField<GraphDef>* partition_graph_defs =
        run_metadata->mutable_partition_graphs();
    for (const PerPartitionExecutorsAndLib& exec_and_lib :
         executors_and_keys->items) {
      GraphDef* partition_graph_def = partition_graph_defs->Add();
      exec_and_lib.graph->ToGraphDef(partition_graph_def);
    }
  }
  metrics::UpdateGraphExecTime(Env::Default()->NowMicros() - start_time_usecs);

  return Status::OK();
}


////////////////////////////////////////////////////////////////////////


Status DirectSession::Run(const RunOptions& run_options, // input

                          const NamedTensorList& inputs, // input

                          const std::vector<string>& output_names, // input
                          const std::vector<string>& target_nodes, // input
                          std::vector<Tensor>* outputs, // output of the Executor Result
                          RunMetadata* run_metadata) {// output
  // 1.
  // NamedTensorList 数据结构
  // tensorflow/core/common_runtime/direct_session.h:68:
  // typedef std::vector<std::pair<string, Tensor>> NamedTensorList;
  // NamedTensorList 人话: A list/vector of tensor_name and tensor pair.

  // 2.
  // RunOptions 数据结构
  // tensorflow/core/protobuf/config.proto:454
  // message RunOptions
  // Options for a single Run() call.
  //
  // message RunOptions {
  //   // TODO(pbar) Turn this into a TraceOptions proto which allows
  //   // tracing to be controlled in a more orthogonal manner?
  //   enum TraceLevel {
  //     NO_TRACE = 0;
  //     SOFTWARE_TRACE = 1;
  //     HARDWARE_TRACE = 2;
  //     FULL_TRACE = 3;
  //   }
  //   TraceLevel trace_level = 1;
  //
  //   // Time to wait for operation to complete in milliseconds.
  //   int64 timeout_in_ms = 2;
  //
  //   // The thread pool to use, if session_inter_op_thread_pool is configured.
  //   // To use the caller thread set this to -1 - this uses the caller thread
  //   // to execute Session::Run() and thus avoids a context switch. Using the
  //   // caller thread to execute Session::Run() should be done ONLY for simple
  //   // graphs, where the overhead of an additional context switch is
  //   // comparable with the overhead of Session::Run().
  //   int32 inter_op_thread_pool = 3;
  //
  //   // Whether the partition graph(s) executed by the executor(s) should be
  //   // outputted via RunMetadata.
  //   bool output_partition_graphs = 5;
  //
  //   // EXPERIMENTAL.  Options used to initialize DebuggerState, if enabled.
  //   DebugOptions debug_options = 6;
  //
  //   // When enabled, causes tensor allocation information to be included in
  //   // the error message when the Run() call fails because the allocator ran
  //   // out of memory (OOM).
  //   //
  //   // Enabling this option can slow down the Run() call.
  //   bool report_tensor_allocations_upon_oom = 7;
  //
  //   // Everything inside Experimental is subject to change and is not subject
  //   // to API stability guarantees in
  //   // https://www.tensorflow.org/guide/version_compat.
  //   message Experimental {
  //     // If non-zero, declares that this graph is going to use collective
  //     // ops and must synchronize step_ids with any other graph with this
  //     // same group_key value (in a distributed computation where tasks
  //     // run disjoint graphs).
  //     int64 collective_graph_key = 1;
  //     // If true, then operations (using the inter-op pool) across all
  //     // session::run() calls will be centrally scheduled, optimizing for (median
  //     // and tail) latency.
  //     // Consider using this option for CPU-bound workloads like inference.
  //     bool use_run_handler_pool = 2;
  //   };
  //
  //   Experimental experimental = 8;
  //
  //   reserved 4;
  // }

  // 3.
  // RunMetadata 数据结构
  // tensorflow/core/protobuf/config.proto


  TF_RETURN_IF_ERROR(CheckNotClosed());
  TF_RETURN_IF_ERROR(CheckGraphCreated("Run()"));

  direct_session_runs->GetCell()->IncrementBy(1);

  // Extract the inputs names for this run of the session.
  std::vector<string> input_tensor_names;

  input_tensor_names.reserve(inputs.size());

  for (const auto& it : inputs) {
    input_tensor_names.push_back(it.first);
  }

  // 逻辑是: 下面对这个 executors_and_keys 变量进行初始化后放入 RunInternal 内执行

  ExecutorsAndKeys* executors_and_keys;
  // 1.
  // struct ExecutorsAndKeys 数据结构:
  // tensorflow/core/common_runtime/direct_session.h:155:
  // struct ExecutorsAndKeys
  //    struct ExecutorsAndKeys {
  //
  //      ExecutorsAndKeys() : step_count(0) {}
  //
  //      std::atomic_int_fast64_t step_count;
  //
  //      std::unique_ptr<Graph> graph;
  //
  //      NameNodeMap name_to_node;
  //
  //      std::vector<PerPartitionExecutorsAndLib> items;
  //
  //      std::unordered_map<string, size_t> input_name_to_index;
  //
  //      std::unordered_map<string, string> input_name_to_rendezvous_key;
  //
  //      std::unordered_map<string, size_t> output_name_to_index;
  //
  //      std::unordered_map<string, string> output_name_to_rendezvous_key;
  //
  //      DataTypeVector input_types;
  //      DataTypeVector output_types;
  //
  //      // message type
  //      CallableOptions callable_options;
  //
  //      int64 collective_graph_key = BuildGraphOptions::kNoCollectiveGraphKey;
  //    };


  // Debug 相关的
  RunStateArgs run_state_args(run_options.debug_options());
  run_state_args.collective_graph_key =
      run_options.experimental().collective_graph_key();

  // -----------------------------------------------------------------------
  // Check if we already have an executor for these arguments.
  TF_RETURN_IF_ERROR(
    GetOrCreateExecutors(
      input_tensor_names, // input
      output_names, // input
      target_nodes, // input

      &executors_and_keys, // output, 主要的构造对象

      &run_state_args)); // input and output
  // -----------------------------------------------------------------------


  {
    mutex_lock l(collective_graph_key_lock_);
    collective_graph_key_ = executors_and_keys->collective_graph_key;
  }

  // Configure a call frame for the step, which we use to feed and
  // fetch values to and from the executors.
  FunctionCallFrame call_frame(
    executors_and_keys->input_types,  // arg_types: DataTypeSlice
    executors_and_keys->output_types); // ret_types: DataTypeSlice
  // 1.
  // class FunctionCallFrame 数据结构
  // tensorflow/core/framework/function.h:276:
  //
  // 综述:
  // 定义 Executor 输入，输出
  //
  // class FunctionCallFrame : public CallFrameInterface
  // Represents a function call frame. I.e., the data structure used to
  // pass arguments to a function and retrieve its results.
  // - arg_types_: DataTypeVector
  // - ret_types_: DataTypeVector
  // - args_: gtl::InlinedVector<Tensor, 4>
  // - rets_: gtl::InlinedVector<Retval, 4>
  //    * struct Retval
  //      + has_val: bool
  //      + val: Tensor

  // 2.
  // DataTypeVector 数据结构
  // framework/types.h
  // typedef gtl::InlinedVector<DataType, 4> DataTypeVector;

  // 3.
  // DataType 数据结构
  //


  gtl::InlinedVector<Tensor, 4> feed_args(inputs.size());

  for (const auto& it : inputs) {

    if (it.second.dtype() == DT_RESOURCE) {

      Tensor tensor_from_handle;

      TF_RETURN_IF_ERROR(
          ResourceHandleToInputTensor(
            it.second,
            &tensor_from_handle));

      feed_args[executors_and_keys->input_name_to_index[it.first]] =
          tensor_from_handle;

    } else {

      feed_args[executors_and_keys->input_name_to_index[it.first]] = it.second;

    }
  }

  const Status s = call_frame.SetArgs(feed_args);
  // 1.
  // FunctionCallFrame::SetArgs 函数说明:
  // tensorflow/core/framework/function.cc
  //

  if (errors::IsInternal(s)) {
    return errors::InvalidArgument(s.error_message());
  } else if (!s.ok()) {
    return s;
  }

  const int64 step_id = step_id_counter_.fetch_add(1);

  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(step_id, run_state_args.handle);
  }


  // 重要:
  // -----------------------------------------------------------------------
  TF_RETURN_IF_ERROR(
    RunInternal(
      step_id,     // input
      run_options, // input
      &call_frame, // input and output(output when it gets retval from call_frame)
      executors_and_keys, // input
      run_metadata));     // output
  // -----------------------------------------------------------------------
  // RunInternal 函数说明:
  // tensorflow/core/common_runtime/direct_session.cc

  // -----------------------------------------------------------------------
  // Receive outputs.
  if (outputs) {

    std::vector<Tensor> sorted_outputs; // output

    const Status s = call_frame.ConsumeRetvals(
                       &sorted_outputs,  // output
                       /* allow_dead_tensors = */ false); // input

    // 1.
    // ConsumeRetvals 函数说明:
    // tensorflow/core/framework/function.cc
    // Moves the return values from the frame to rets.
    // If allow_dead_tensors is false it will fail if any of the retvals do not have a value.
    //
    // Status ConsumeRetvals(
    //          std::vector<Tensor>* rets,
    //          bool allow_dead_tensors);

    // 2.
    // allow_dead_tensors 说明:
    //


    // 异常处理
    if (errors::IsInternal(s)) {
      return errors::InvalidArgument(s.error_message());
    } else if (!s.ok()) {
      return s;
    }

    const bool unique_outputs =
        output_names.size() == executors_and_keys->output_name_to_index.size();
    // first_indices[i] = j implies that j is the smallest value for which
    // output_names[i] == output_names[j].
    std::vector<int> first_indices;
    if (!unique_outputs) {
      first_indices.resize(output_names.size());
      for (int i = 0; i < output_names.size(); ++i) {
        for (int j = 0; j <= i; ++j) {
          if (output_names[i] == output_names[j]) {
            first_indices[i] = j;
            break;
          }
        }
      }
    }

    outputs->clear();
    outputs->reserve(sorted_outputs.size());
    for (int i = 0; i < output_names.size(); ++i) {
      const string& output_name = output_names[i];
      if (first_indices.empty() || first_indices[i] == i) {
        outputs->emplace_back(
            std::move(sorted_outputs[executors_and_keys
                                         ->output_name_to_index[output_name]]));
      } else {
        outputs->push_back((*outputs)[first_indices[i]]);
      }
    }
  } // receive output 分支结束
  // -----------------------------------------------------------------------

  return Status::OK();
}


////////////////////////////////////////////////////////////////////////


Status DirectSession::PRunSetup(const std::vector<string>& input_names,
                                const std::vector<string>& output_names,
                                const std::vector<string>& target_nodes,
                                string* handle) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  TF_RETURN_IF_ERROR(CheckGraphCreated("PRunSetup()"));

  // RunOptions is not available in PRunSetup, so use thread pool 0.
  thread::ThreadPool* pool = thread_pools_[0].first;

  // Check if we already have an executor for these arguments.
  ExecutorsAndKeys* executors_and_keys;
  // TODO(cais): TFDBG support for partial runs.
  DebugOptions debug_options;
  RunStateArgs run_state_args(debug_options);
  run_state_args.is_partial_run = true;
  TF_RETURN_IF_ERROR(GetOrCreateExecutors(input_names, output_names,
                                          target_nodes, &executors_and_keys,
                                          &run_state_args));

  // Create the run state and save it for future PRun calls.
  Executor::Args args;
  args.step_id = step_id_counter_.fetch_add(1);
  RunState* run_state =
      new RunState(input_names, output_names, args.step_id, &devices_);
  run_state->rendez = new IntraProcessRendezvous(device_mgr_.get());
  {
    mutex_lock l(executor_lock_);
    if (!partial_runs_
             .emplace(run_state_args.handle,
                      std::unique_ptr<RunState>(run_state))
             .second) {
      return errors::Internal("The handle '", run_state_args.handle,
                              "' created for this partial run is not unique.");
    }
  }

  // Start parallel Executors.
  const size_t num_executors = executors_and_keys->items.size();
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors,
      run_state->rendez,

      [run_state](const Status& ret) {
        if (!ret.ok()) {
          mutex_lock l(run_state->mu_);
          run_state->status.Update(ret);
        }
        run_state->executors_done.Notify();
      }
    );

  args.rendezvous = run_state->rendez;
  args.cancellation_manager = cancellation_manager_;
  // Note that Collectives are not supported in partial runs
  // because RunOptions is not passed in so we can't know whether
  // their use is intended.
  args.collective_executor = nullptr;
  args.runner = [this, pool](Executor::Args::Closure c) {
    SchedClosure(pool, std::move(c));
  };
  args.session_state = &session_state_;
  args.session_handle = session_handle_;
  args.tensor_store = &run_state->tensor_store;
  args.step_container = &run_state->step_container;
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(args.step_id, run_state_args.handle);
  }
  args.sync_on_finish = sync_on_finish_;

  if (options_.config.graph_options().build_cost_model()) {
    run_state->collector.reset(new StepStatsCollector(nullptr));
    args.stats_collector = run_state->collector.get();
  }

  for (auto& item : executors_and_keys->items) {
    item.executor->RunAsync(args, barrier->Get());
  }

  *handle = run_state_args.handle;
  return Status::OK();
}

Status DirectSession::PRun(const string& handle, const NamedTensorList& inputs,
                           const std::vector<string>& output_names,
                           std::vector<Tensor>* outputs) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  std::vector<string> parts = str_util::Split(handle, ';');
  const string& key = parts[0];
  // Get the executors for this partial run.
  ExecutorsAndKeys* executors_and_keys;
  RunState* run_state;
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto exc_it = executors_.find(key);
    if (exc_it == executors_.end()) {
      return errors::InvalidArgument(
          "Must run 'setup' before performing partial runs!");
    }
    executors_and_keys = exc_it->second.get();

    auto prun_it = partial_runs_.find(handle);
    if (prun_it == partial_runs_.end()) {
      return errors::InvalidArgument(
          "Must run 'setup' before performing partial runs!");
    }
    run_state = prun_it->second.get();

    // Make sure that this is a new set of feeds that are still pending.
    for (const auto& input : inputs) {
      auto it = run_state->pending_inputs.find(input.first);
      if (it == run_state->pending_inputs.end()) {
        return errors::InvalidArgument(
            "The feed ", input.first,
            " was not specified in partial_run_setup.");
      } else if (it->second) {
        return errors::InvalidArgument("The feed ", input.first,
                                       " has already been fed.");
      }
    }
    // Check that this is a new set of fetches that are still pending.
    for (const auto& output : output_names) {
      auto it = run_state->pending_outputs.find(output);
      if (it == run_state->pending_outputs.end()) {
        return errors::InvalidArgument(
            "The fetch ", output, " was not specified in partial_run_setup.");
      } else if (it->second) {
        return errors::InvalidArgument("The fetch ", output,
                                       " has already been fetched.");
      }
    }
  }

  // Check that this new set of fetches can be computed from all the
  // feeds we have supplied.
  TF_RETURN_IF_ERROR(
      CheckFetch(inputs, output_names, executors_and_keys, run_state));

  // Send inputs.
  Status s = SendPRunInputs(inputs, executors_and_keys, run_state->rendez);

  // Receive outputs.
  if (s.ok()) {
    s = RecvPRunOutputs(output_names, executors_and_keys, run_state, outputs);
  }

  // Save the output tensors of this run we choose to keep.
  if (s.ok()) {
    s = run_state->tensor_store.SaveTensors(output_names, &session_state_);
  }

  {
    mutex_lock l(executor_lock_);
    // Delete the run state if there is an error or all fetches are done.
    bool done = true;
    if (s.ok()) {
      {
        mutex_lock l(run_state->mu_);
        if (!run_state->status.ok()) {
          LOG(WARNING) << "An error unrelated to this prun has been detected. "
                       << run_state->status;
        }
      }
      for (const auto& input : inputs) {
        auto it = run_state->pending_inputs.find(input.first);
        it->second = true;
      }
      for (const auto& name : output_names) {
        auto it = run_state->pending_outputs.find(name);
        it->second = true;
      }
      done = run_state->PendingDone();
    }
    if (done) {
      WaitForNotification(run_state, cancellation_manager_,
                          operation_timeout_in_ms_);
      partial_runs_.erase(handle);
    }
  }

  return s;
}

Status DirectSession::ResourceHandleToInputTensor(const Tensor& resource_tensor,
                                                  Tensor* retrieved_tensor) {
  if (resource_tensor.dtype() != DT_RESOURCE) {
    return errors::InvalidArgument(strings::StrCat(
        "ResourceHandleToInputTensor() received non-DT_RESOURCE Tensor: ",
        resource_tensor.dtype()));
  }

  const ResourceHandle& resource_handle =
      resource_tensor.scalar<ResourceHandle>()();

  if (resource_handle.container() ==
      SessionState::kTensorHandleResourceTypeName) {
    return session_state_.GetTensor(resource_handle.name(), retrieved_tensor);
  } else {
    return errors::InvalidArgument(strings::StrCat(
        "Invalid resource type hash code: ", resource_handle.hash_code(),
        "(name: ", resource_handle.name(),
        " type: ", resource_handle.maybe_type_name(),
        "). Perhaps a resource tensor was being provided as a feed? That is "
        "not currently allowed. Please file an issue at "
        "https://github.com/tensorflow/tensorflow/issues/new, ideally with a "
        "short code snippet that leads to this error message."));
  }
}

Status DirectSession::SendPRunInputs(const NamedTensorList& inputs,
                                     const ExecutorsAndKeys* executors_and_keys,
                                     IntraProcessRendezvous* rendez) {
  Status s;
  Rendezvous::ParsedKey parsed;
  // Insert the input tensors into the local rendezvous by their
  // rendezvous key.
  for (const auto& input : inputs) {
    auto it =
        executors_and_keys->input_name_to_rendezvous_key.find(input.first);
    if (it == executors_and_keys->input_name_to_rendezvous_key.end()) {
      return errors::Internal("'", input.first, "' is not a pre-defined feed.");
    }
    const string& input_key = it->second;

    s = Rendezvous::ParseKey(input_key, &parsed);
    if (!s.ok()) {
      rendez->StartAbort(s);
      return s;
    }

    if (input.second.dtype() == DT_RESOURCE) {
      Tensor tensor_from_handle;
      s = ResourceHandleToInputTensor(input.second, &tensor_from_handle);
      if (s.ok()) {
        s = rendez->Send(parsed, Rendezvous::Args(), tensor_from_handle, false);
      }
    } else {
      s = rendez->Send(parsed, Rendezvous::Args(), input.second, false);
    }

    if (!s.ok()) {
      rendez->StartAbort(s);
      return s;
    }
  }
  return Status::OK();
}

Status DirectSession::RecvPRunOutputs(
    const std::vector<string>& output_names,
    const ExecutorsAndKeys* executors_and_keys, RunState* run_state,
    std::vector<Tensor>* outputs) {
  Status s;
  if (!output_names.empty()) {
    outputs->resize(output_names.size());
  }

  Rendezvous::ParsedKey parsed;
  // Get the outputs from the rendezvous
  for (size_t output_offset = 0; output_offset < output_names.size();
       ++output_offset) {
    const string& output_name = output_names[output_offset];
    auto it =
        executors_and_keys->output_name_to_rendezvous_key.find(output_name);
    if (it == executors_and_keys->output_name_to_rendezvous_key.end()) {
      return errors::Internal("'", output_name,
                              "' is not a pre-defined fetch.");
    }
    const string& output_key = it->second;
    Tensor output_tensor;
    bool is_dead;
    IntraProcessRendezvous* rendez = run_state->rendez;

    s = Rendezvous::ParseKey(output_key, &parsed);
    if (s.ok()) {
      // Fetch data from the Rendezvous.
      s = rendez->Recv(parsed, Rendezvous::Args(), &output_tensor, &is_dead,
                       operation_timeout_in_ms_);
      if (is_dead && s.ok()) {
        s = errors::InvalidArgument("The tensor returned for ", output_name,
                                    " was not valid.");
      }
    }
    if (!s.ok()) {
      rendez->StartAbort(s);
      outputs->clear();
      return s;
    }

    (*outputs)[output_offset] = output_tensor;
  }
  return Status::OK();
}

Status DirectSession::CheckFetch(const NamedTensorList& feeds,
                                 const std::vector<string>& fetches,
                                 const ExecutorsAndKeys* executors_and_keys,
                                 const RunState* run_state) {
  const Graph* graph = executors_and_keys->graph.get();
  const NameNodeMap* name_to_node = &executors_and_keys->name_to_node;

  // Build the set of pending feeds that we haven't seen.
  std::unordered_set<TensorId, TensorId::Hasher> pending_feeds;
  {
    mutex_lock l(executor_lock_);
    for (const auto& input : run_state->pending_inputs) {
      // Skip if the feed has already been fed.
      if (input.second) continue;
      TensorId id(ParseTensorName(input.first));
      auto it = name_to_node->find(id.first);
      if (it == name_to_node->end()) {
        return errors::NotFound("Feed ", input.first, ": not found");
      }
      pending_feeds.insert(id);
    }
  }
  for (const auto& it : feeds) {
    TensorId id(ParseTensorName(it.first));
    pending_feeds.erase(id);
  }

  // Initialize the stack with the fetch nodes.
  std::vector<const Node*> stack;
  for (const string& fetch : fetches) {
    TensorId id(ParseTensorName(fetch));
    auto it = name_to_node->find(id.first);
    if (it == name_to_node->end()) {
      return errors::NotFound("Fetch ", fetch, ": not found");
    }
    stack.push_back(it->second);
  }

  // Any tensor needed for fetches can't be in pending_feeds.
  std::vector<bool> visited(graph->num_node_ids(), false);
  while (!stack.empty()) {
    const Node* n = stack.back();
    stack.pop_back();

    for (const Edge* in_edge : n->in_edges()) {
      const Node* in_node = in_edge->src();
      if (pending_feeds.count({in_node->name(), in_edge->src_output()}) > 0) {
        return errors::InvalidArgument("Fetch ", in_node->name(), ":",
                                       in_edge->src_output(),
                                       " can't be computed from the feeds"
                                       " that have been fed so far.");
      }
      if (!visited[in_node->id()]) {
        visited[in_node->id()] = true;
        stack.push_back(in_node);
      }
    }
  }
  return Status::OK();
}


//////////////////////////////////////////////////////////////////////////
// 这个函数的主要目的是 构造 和 初始化 ExecutorsAndKeys instance ek 的各个 field.
// std::unique_ptr<ExecutorsAndKeys> ek(new ExecutorsAndKeys);

// QQQ. Executors 内的原始的图在哪里？
// AAA.
Status DirectSession::CreateExecutors(
    const CallableOptions& callable_options, // input
    std::unique_ptr<ExecutorsAndKeys>* out_executors_and_keys, // output
    std::unique_ptr<FunctionInfo>* out_func_info, // output
    RunStateArgs* run_state_args) // input and output (from the perspective of Graph* graph within RunStateArgs)
{
  // 1.
  // message CallableOptions 数据结构
  // tensorflow/core/protobuf/config.proto:559:
  // message CallableOptions, feed, fetch, target, feed_devices, fetch_devices ...
  //
  // 打印:
  // p callable_options
  // fetch: "out:0"
  // run_options {
  //   debug_options {
  //   }
  //   experimental {
  //   }
  // }

  // 2.
  // ExecutorsAndKeys 数据结构
  // tensorflow/core/common_runtime/direct_session.h
  //
  // struct ExecutorsAndKeys {
  //   ExecutorsAndKeys() : step_count(0) {}
  //   std::atomic_int_fast64_t step_count;
  //   std::unique_ptr<Graph> graph;
  //   NameNodeMap name_to_node;
  //   // ------------------------------------------------------------------------
  //   // 最重要
  //   std::vector<PerPartitionExecutorsAndLib> items;
  //   // ------------------------------------------------------------------------
  //   std::unordered_map<string, size_t> input_name_to_index;
  //   std::unordered_map<string, string> input_name_to_rendezvous_key;
  //   std::unordered_map<string, size_t> output_name_to_index;
  //   std::unordered_map<string, string> output_name_to_rendezvous_key;
  //   DataTypeVector input_types;
  //   DataTypeVector output_types;
  //   // message type
  //   CallableOptions callable_options;
  //   int64 collective_graph_key = BuildGraphOptions::kNoCollectiveGraphKey;
  // };

  BuildGraphOptions options;
  options.callable_options = callable_options;
  options.use_function_convention = !run_state_args->is_partial_run;
  options.collective_graph_key =
      callable_options.run_options().experimental().collective_graph_key();
  if (options_.config.experimental()
          .collective_deterministic_sequential_execution()) {
    options.collective_order = GraphCollectiveOrder::kEdges;
  } else if (options_.config.experimental().collective_nccl()) {
    options.collective_order = GraphCollectiveOrder::kAttrs;
  }
  // 1.
  // options 变量说明:
  /**
  BuildGraphOptions options 打印:

  Feed endpoints:
  Fetch endpoints: out:0,
  Target nodes:
  collective_order: none
  */

  // 2.
  // BuildGraphOptions 数据结构
  // tensorflow/core/common_runtime/build_graph_options.h:27:
  // struct BuildGraphOptions
  // - callable_options
  // - use_function_convention
  // - collective_graph_key
  // - collective_order
  // - 你可以使用自带的 string DebugString() const; 调试


  // -----------------------------------------------------------------------
  std::unique_ptr<FunctionInfo> func_info(new FunctionInfo);
  // -----------------------------------------------------------------------
  // 1.
  // 技巧[IMPT]:
  // search "func_info->" to see when func_info's fields are initialized.

  // 2.
  // func_info 变量说明:
  // func_info: std::unique_ptr<FunctionInfo>

  // 3.
  // FunctionInfo 数据结构
  // tensorflow/core/common_runtime/direct_session.h:192:  struct FunctionInfo
  // - flib_def: std::unique_ptr<FunctionLibraryDefinition>
  // - proc_flr: std::unique_ptr<ProcessFunctionLibraryRuntime>
  //
  // 概述:
  // A FunctionInfo object is created for every unique set of feeds/fetches.
  // This info could be folded into the ExecutorsAndKeys object but we would
  // like to maintain a deletion order in which the OpKernels (owned by the
  // executor) should be destroyed first, followed by the resources in the
  // device and then followed by the function stuff.
  // TODO(rohanj): Consolidate function library definitions so that we can
  // instantiate only one ProcFLR and lib_def and make this just a member
  // variable and not a vector.
  // 'flib_def' is the function library used.
  // 'proc_flr' is the collection of FunctionLibraryRuntime objects, one per
  // device.

  // 4.
  // class FunctionLibraryDefinition 数据结构
  // tensorflow/core/framework/function.h:313:
  // class FunctionLibraryDefinition : public OpRegistryInterface
  //
  // - struct FunctionDefAndOpRegistration
  // - default_registry_: const OpRegistryInterface* const
  // - function_defs_: gtl::FlatMap<string, std::unique_ptr<FunctionDefAndOpRegistration>>
  // - func_grad_: gtl::FlatMap<string, string>

  // 5.
  // struct FunctionDefAndOpRegistration 数据结构 (within class FunctionLibraryDefinition)
  //
  // - fdef: FunctionDef
  // - op_registration_data: OpRegistrationData
  //
  // 例子
  // string: name of the function
  // second: function definition
  // example: https://gist.github.com/shizukanaskytree/8660a751d41a25db1848ec1be20dfd1e

  // 6.
  // message FunctionDef 数据结构
  // tensorflow/core/framework/function.proto
  // - signature: OpDef
  //   The definition of the function's name, arguments, return values, attrs etc.
  // - attr: map<string, AttrValue> attr
  //   Attributes specific to this function definition.
  // - node_def: repeated NodeDef
  //   In both of the following fields, there is the need to specify an
  //   output that is used as either the input to another node (in
  //   `node_def`) or as a return value of the function (in `ret`).
  // - ret: map<string, string>
  //   A mapping from the output arg names from `signature` to the
  //   outputs from `node_def` that should be returned by the function.
  // - control_ret: map<string, string>
  //   A mapping from control output names from `signature` to node names in
  //   `node_def` which should be control outputs of this function.

  // 7.
  // struct OpRegistrationData 数据结构
  // tensorflow/core/framework/op_def_builder.h
  // - op_def: OpDef
  // - shape_inference_fn: OpShapeInferenceFn
  // - is_function_op: bool, default : false;

  // 8.
  // OpDef 数据结构
  // tensorflow/core/framework/op_def.proto

  // 9.
  // class ProcessFunctionLibraryRuntime 数据结构
  // tensorflow/core/common_runtime/process_function_library_runtime.h
  //
  // 概述:
  // A class that stores all the FunctionLibraryRuntime objects, one per device.
  //
  // - struct ComponentFunctionData
  //    - handle_: FunctionLibraryRuntime::Handle
  //    - arg_indices_: std::vector<int>
  //    - ret_indices_: std::vector<int>
  //    - arg_alloc_attrs_: std::vector<AllocatorAttributes>
  //    - ret_alloc_attrs_: std::vector<AllocatorAttributes>
  // - struct MultiDeviceFunctionData
  //    - num_outputs_: const int
  //    - instantiation_counter_: uint64
  //    - function_name_: const string
  //    - function_key_: const string
  //    - overlay_lib_: FunctionLibraryDefinition
  //    - glue_: std::unordered_map<string, ComponentFunctionData>
  // - class FunctionData
  //    - target_device_: const string
  //    - local_handle_: FunctionLibraryRuntime::LocalHandle
  //    - function_key_: const string
  //    - init_started_: bool
  //    - init_result_: Status
  //    - init_done_: Notification
  // - env_ : Env* const
  // - device_mgr_: const DeviceMgr* const
  // - lib_def_: const FunctionLibraryDefinition*
  // - default_thread_pool_: thread::ThreadPool*
  // - table_: std::unordered_map<string, FunctionLibraryRuntime::Handle>
  // - function_data_: std::unordered_map<FunctionLibraryRuntime::Handle, std::unique_ptr<FunctionData>>
  // - mdevice_data_: std::unordered_map<FunctionLibraryRuntime::Handle,std::unique_ptr<MultiDeviceFunctionData>>
  //     Function data for instantiated multi-device functions.
  // - flr_map_: std::unordered_map<Device*, std::unique_ptr<FunctionLibraryRuntime>>
  // - next_handle_: int
  // - parent_: DistributedFunctionLibraryRuntime* const

  // 10.
  // class FunctionLibraryRuntime
  // tensorflow/core/framework/function.h
  // - struct InstantiateOptions
  //  - 概述:
  //    Instantiate a function with the given "attrs".
  //  - target: string
  //  - is_multi_device_function: bool, default : false
  //  - input_devices: std::vector<string>
  //  - output_devices: std::vector<string>
  //  - overlay_lib: const FunctionLibraryDefinition*
  //  - state_handle: string
  //  - executor_type : string
  //  - create_kernels_eagerly: bool, default: false
  //  - config_proto: ConfigProto
  //  - optimize_graph_fn:
  //      std::function<Status(std::vector<string> /*ret_node_names*/,
  //                           std::vector<string> /*keep_node_names*/,
  //                           FunctionLibraryDefinition*, const DeviceSet&,
  //                           Device* /*cpu_device*/, std::unique_ptr<Graph>*)>
  //  - graph_collector: GraphCollector*
  // - typedef uint64 Handle
  // - struct Options
  //  - step_id: int64
  //  - rendezvous: Rendezvous*
  //  - cancellation_manager: CancellationManager*
  //  - collective_executor: CollectiveExecutor*
  //  - step_container: ScopedStepContainer*
  //  - stats_collector: StepStatsCollectorInterface
  //  - runner: std::function<void(std::function<void()>)>*
  //  - remote_execution: bool
  //  - source_device: string
  //  - args_alloc_attrs: std::vector<AllocatorAttributes>
  //  - rets_alloc_attrs: std::vector<AllocatorAttributes>
  //  - create_rendezvous: bool
  //  - allow_dead_tensors: bool
  // - DoneCallback
  //
  // 核心函数
  // - CreateKernel


  // ek: std::unique_ptr<ExecutorsAndKeys> 的构造和初始化
  // -----------------------------------------------------------------------
  std::unique_ptr<ExecutorsAndKeys> ek(new ExecutorsAndKeys);
  // -----------------------------------------------------------------------
  // 1.
  // struct ExecutorsAndKeys 数据结构
  // tensorflow/core/common_runtime/direct_session.h:155:
  // - step_count : std::atomic_int_fast64_t
  // - graph : std::unique_ptr<Graph>
  // - name_to_node : NameNodeMap
  //   ================================================
  // - items : std::vector<PerPartitionExecutorsAndLib> # 最重要
  //   ================================================
  // - input_name_to_index : std::unordered_map<string, size_t>
  // - input_name_to_rendezvous_key : std::unordered_map<string, string>
  // - output_name_to_index : std::unordered_map<string, size_t>
  // - output_name_to_rendezvous_key : std::unordered_map<string, string>
  // - input_types : DataTypeVector
  // - output_types : DataTypeVector
  // - callable_options : CallableOptions [有被初始化]
  // - collective_graph_key : BuildGraphOptions::kNoCollectiveGraphKey;

  // 2.
  // trick: search "ek->" to see all fields (when they are initialized.)

  ek->callable_options = callable_options;

  // -----------------------------------------------------------------------
  // 这个段落是 CreateGraphs, graphs
  // -----------------------------------------------------------------------
  std::unordered_map<string, std::unique_ptr<Graph>> graphs;
  //                    |                      |
  //                  devices               subgraph
  // [idea]: 构造一个额外的整图，全部放在 CPU 上，交给下面的 CreateGraphs 处理

  // graphs 内的 std::unique_ptr<Graph> 锁指向的图也是在 CreateGraphs 里面构造的
  TF_RETURN_IF_ERROR(
    CreateGraphs(
      options, // input
      &graphs,  // output
      &func_info->flib_def, // output
      run_state_args, // input and output
      &ek->input_types,  // output
      &ek->output_types, // output
      &ek->collective_graph_key) // output
  );
  // -----------------------------------------------------------------------
  // 1.
  // graphs 变量说明
  // graphs: std::unordered_map<string, std::unique_ptr<Graph>>
  // 说明：
  // CreateGraphs 后，则 Graph 里面已添加了 Send Recv RetVal 等节点了。

  // 2.
  // QQQ. DirectSession::CreateGraphs 里面怎么知道我的 device placement的？
  // AAA. DirectSession::CreateGraphs 内使用了 DirectSession::execution_state_ : std::unique_ptr<GraphExecutionState>
  //      这个成员变量在 DirectSession::Extend 调用栈系列里面被构造和初始化，包括了 device placement 也弄好了。


  // 不看这个分支
  if (run_state_args->is_partial_run) {
    ek->graph = std::move(run_state_args->graph);
    std::unordered_set<StringPiece, StringPieceHasher> names;
    for (const string& input : callable_options.feed()) {
      TensorId id(ParseTensorName(input));
      names.emplace(id.first);
    }
    for (const string& output : callable_options.fetch()) {
      TensorId id(ParseTensorName(output));
      names.emplace(id.first);
    }
    for (Node* n : ek->graph->nodes()) {
      if (names.count(n->name()) > 0) {
        ek->name_to_node.insert({n->name(), n});
      }
    }
  }

  // 准备构造
  ek->items.reserve(graphs.size());
  // 1.
  // ek: std::unique_ptr<ExecutorsAndKeys>

  // 2.
  // ExecutorsAndKeys 数据结构
  // struct ExecutorsAndKeys
  //    // 综述:An ExecutorsAndKeys is created for a given set of feeds/fetches.
  //    // 变量说明:
  //    // - 'step_count' is the number of times this graph is executed.
  //    // - 'graph' is the entire graph being executed.
  //    // - 'name_to_node' maps node name to node.
  //    // We keep 'graph' and 'name_to_node' only in the case of partial runs.
  //    // Each item in 'items' is the executor for a partition of the graph
  //    // bundled with its dependent library runtime.
  //    // - 'input_keys' are the rendezvous keys for the feeds and
  //    // - 'output_keys' are rendezvous keys for the fetches.
  //    struct ExecutorsAndKeys {
  //
  //      ExecutorsAndKeys() : step_count(0) {}
  //
  //      std::atomic_int_fast64_t step_count;
  //
  //      std::unique_ptr<Graph> graph;
  //
  //      NameNodeMap name_to_node;
  //
  //      std::vector<PerPartitionExecutorsAndLib> items;
  //
  //      std::unordered_map<string, size_t> input_name_to_index;
  //
  //      std::unordered_map<string, string> input_name_to_rendezvous_key;
  //
  //      std::unordered_map<string, size_t> output_name_to_index;
  //
  //      std::unordered_map<string, string> output_name_to_rendezvous_key;
  //
  //      DataTypeVector input_types;
  //      DataTypeVector output_types;
  //
  //      // message type
  //      CallableOptions callable_options;
  //
  //      int64 collective_graph_key = BuildGraphOptions::kNoCollectiveGraphKey;
  //    };

  // 3.
  // ek->items 变量说明
  // items: std::vector<PerPartitionExecutorsAndLib>

  // 4.
  // PerPartitionExecutorsAndLib 数据结构
  // struct PerPartitionExecutorsAndLib {
  //   Graph* graph = nullptr;                  // not owned.
  //   Device* device = nullptr;                // not owned.
  //   FunctionLibraryRuntime* flib = nullptr;  // not owned.
  //   std::unique_ptr<Executor> executor;
  // };

  // 这个我在 tf.ConfigProto 里面没有设置过。
  const auto& optimizer_opts =
      options_.config.graph_options().optimizer_options();
  // 1.
  // optimizer_opts 变量说明:
  // optimizer_opts: const OptimizerOptions&

  // 2.
  // optimizer_options 变量说明：
  // Options controlling how graph is optimized.
  // OptimizerOptions optimizer_options = 3;

  // 3.
  // OptimizerOptions 数据结构
  // tensorflow/core/protobuf/config.proto
  /*
  // Options passed to the graph optimizer
  message OptimizerOptions {
    // If true, optimize the graph using common subexpression elimination.
    bool do_common_subexpression_elimination = 1;

    // If true, perform constant folding optimization on the graph.
    bool do_constant_folding = 2;

    // Constant folding optimization replaces tensors whose values can be
    // predetermined, with constant nodes. To avoid inserting too large constants,
    // the size of each constant created can be limited. If this value is zero, a
    // default limit of 10 MiB will be applied. If constant folding optimization
    // is disabled, this value is ignored.
    int64 max_folded_constant_in_bytes = 6;

    // If true, perform function inlining on the graph.
    bool do_function_inlining = 4;

    // Optimization level
    enum Level {
      // L1 is the default level.
      // Optimization performed at L1 :
      // 1. Common subexpression elimination
      // 2. Constant folding
      L1 = 0;

      // No optimizations
      L0 = -1;
    }

    // Overall optimization level. The actual optimizations applied will be the
    // logical OR of the flags that this level implies and any flags already set.
    Level opt_level = 3;

    // Control the use of the compiler/jit.  Experimental.
    enum GlobalJitLevel {
      DEFAULT = 0;  // Default setting ("off" now, but later expected to be "on")
      OFF = -1;
      // The following settings turn on compilation, with higher values being
      // more aggressive.  Higher values may reduce opportunities for parallelism
      // and may use more memory.  (At present, there is no distinction, but this
      // is expected to change.)
      ON_1 = 1;
      ON_2 = 2;
    }
    GlobalJitLevel global_jit_level = 5;
  }
  */


  int graph_def_version;
  {
    mutex_lock l(graph_state_lock_);
    graph_def_version =
        execution_state_->original_graph_def().versions().producer();
  }

  // -----------------------------------------------------------------------
  // 功能: Creates FunctionLibraryRuntime objects for each device within DeviceMgr.
  //
  // proc_flr: std::unique_ptr<ProcessFunctionLibraryRuntime>
  func_info->proc_flr.reset(
    // class ProcessFunctionLibraryRuntime
    // tensorflow/core/common_runtime/process_function_library_runtime.h
    new ProcessFunctionLibraryRuntime(
      device_mgr_.get(),
      options_.env,
      graph_def_version,
      func_info->flib_def.get(),
      // 居然用自己的 FunctionInfo::flib_def 去初始化
      // FunctionInfo::proc_flr

      optimizer_opts,
      thread_pools_[0].first)
  );
  // -----------------------------------------------------------------------

  GraphOptimizer optimizer(optimizer_opts);
  // class GraphOptimizer 数据结构
  // tensorflow/core/common_runtime/graph_optimizer.h


  // 重要!
  ////////////////////////////////////////////////////////////////////////
  for (auto iter = graphs.begin(); iter != graphs.end(); ++iter) {
  ////////////////////////////////////////////////////////////////////////
    const string& partition_name = iter->first; // device name: CPU/GPU
    std::unique_ptr<Graph>& partition_graph = iter->second;

    Device* device; // output
    // /cpu:0 是 cpu device
    TF_RETURN_IF_ERROR(device_mgr_->LookupDevice(
      partition_name, // input
      &device)); // output

    ek->items.resize(ek->items.size() + 1);
    // QQQ.为什么多一个?
    // AAA.

    auto* item = &(ek->items.back());
    // 1.
    // item 变量说明
    // item: PerPartitionExecutorsAndLib*

    // 2.
    // PerPartitionExecutorsAndLib 数据结构
    // struct PerPartitionExecutorsAndLib {
    //   Graph* graph = nullptr;                  // not owned. 初始化 2
    //   Device* device = nullptr;                // not owned. 初始化 4
    //   FunctionLibraryRuntime* flib = nullptr;  // not owned. 初始化 1
    //   std::unique_ptr<Executor> executor;      //初始化 3
    // };

    // ----------------------------------------------------------------------
    // lib: FunctionLibraryRuntime*
    // GetFLR is from
    //   FunctionLibraryRuntime* GetFLR(const string& device_name) const;
    //   tensorflow/core/common_runtime/process_function_library_runtime.h
    auto lib = func_info->proc_flr->GetFLR(partition_name);
    // ----------------------------------------------------------------------

    if (lib == nullptr) {
      return errors::Internal("Could not find device: ", partition_name);
    }

    item->flib = lib; // 初始化 1

    // --------------------------------------------------------------------
    LocalExecutorParams params; // 构造 LocalExecutorParams 用于 构造 Executor
    // --------------------------------------------------------------------
    // struct LocalExecutorParams 数据结构
    // tensorflow/core/common_runtime/executor.h
    //
    // - Device* device;
    //
    // // The library runtime support.
    // - FunctionLibraryRuntime* function_library = nullptr;
    //
    // // create_kernel returns an instance of op kernel based on NodeDef.
    // // delete_kernel is called for every kernel used by the executor
    // // when the executor is deleted.
    // - std::function<Status(const NodeDef&, OpKernel**)> create_kernel;
    // - std::function<void(OpKernel*)> delete_kernel;

    params.device = device;
    params.function_library = lib;
    auto opseg = device->op_segment();

    // -------------------------------------------------------------------
    params.create_kernel = [this, lib, opseg](const NodeDef& ndef,
                                              OpKernel** kernel) {
      // NOTE(mrry): We must not share function kernels (implemented
      // using `CallOp`) between subgraphs, because `CallOp::handle_`
      // is tied to a particular subgraph. Even if the function itself
      // is stateful, the `CallOp` that invokes it is not.
      if (!OpSegment::ShouldOwnKernel(lib, ndef.op())) {
        // call FunctionLibraryRuntime::CreateKernel , 纯虚函数
        // 具体的调用还是这个
        // ./tensorflow/core/common_runtime/function.cc:539:
        // Status FunctionLibraryRuntimeImpl::CreateKernel
        return lib->CreateKernel(ndef, kernel);
      }

      // ---

      auto create_fn = [lib, &ndef](OpKernel** kernel) {
        return lib->CreateKernel(ndef, kernel);
      };
      // Kernels created for subgraph nodes need to be cached.  On
      // cache miss, create_fn() is invoked to create a kernel based
      // on the function library here + global op registry.
      return opseg->FindOrCreate(
                      session_handle_,
                      ndef.name(),
                      kernel,
                      create_fn);
    };
    // -------------------------------------------------------------------

    params.delete_kernel = [lib](OpKernel* kernel) {
      if (kernel && !OpSegment::ShouldOwnKernel(lib, kernel->type_string()))
        delete kernel;
    };
    // 1.
    // 析构时使用

    // -------------------------------------------------------------------
    optimizer.Optimize(lib, options_.env, device, &partition_graph,
                       /*shape_map=*/nullptr);
    // 1.
    // optimizer 变量说明:
    // optimizer: GraphOptimizer

    // 2.
    // class GraphOptimizer 数据结构
    // tensorflow/core/grappler/optimizers/graph_optimizer.h


    // 3.
    // Optimize 函数说明
    // tensorflow/core/common_runtime/graph_optimizer.cc:36:void GraphOptimizer::Optimize(



    // TensorFlow Debugger (tfdbg) inserts debug nodes in the graph.
    const DebugOptions& debug_options =
        options.callable_options.run_options().debug_options();
    if (!debug_options.debug_tensor_watch_opts().empty()) {
      TF_RETURN_IF_ERROR(DecorateAndPublishGraphForDebug(
          debug_options, partition_graph.get(), params.device));
    }

    TF_RETURN_IF_ERROR(EnsureMemoryTypes(DeviceType(device->device_type()),
                                         device->name(),
                                         partition_graph.get()));

    // NewLocalExecutor takes ownership of partition_graph.
    item->graph = partition_graph.get(); // 初始化 2
    item->executor = nullptr; // 初始化 3
    item->device = device; // 初始化 4

    auto executor_type = options_.config.experimental().executor_type();
    // 1.
    // executor_type 变量说明:
    // executor_type: const string&

    // 2.
    // string executor_type = 3;
    // tensorflow/core/protobuf/config.proto
    // Which executor to use, the default executor will be used
    // if it is an empty string or "DEFAULT"


    // 2.
    // options_ 变量说明:
    // options_: const SessionOptions
    //


    TF_RETURN_IF_ERROR(
      // -----------------------------------------------------------------------
      NewExecutor(
        executor_type, // input
        params, // input
        std::move(partition_graph), // input
        &item->executor)); // output
      // -----------------------------------------------------------------------
      // 1.
      // NewExecutor 函数说明:
      // Status NewExecutor(const string& executor_type,
      //                    const LocalExecutorParams& params,
      //                    std::unique_ptr<const Graph> graph,
      //                    std::unique_ptr<Executor>* out_executor) {
      //   ExecutorFactory* factory = nullptr;
      //   TF_RETURN_IF_ERROR(ExecutorFactory::GetFactory(executor_type, &factory));
      //   // 构造:
      //   return factory->NewExecutor(params, std::move(graph), out_executor);
      // }

      // 2.
      // executor_type : const string&
      // - DefaultExecutorRegistrar(): "", "DEFAULT"
      // - SingleThreadedExecutorRegistrar: "SINGLE_THREADED_EXECUTOR"

      // 3.
      // params: const LocalExecutorParams&
      // LocalExecutorParams 数据结构
      //

      // 4.
      // partition_graph: std::unique_ptr<const Graph>

      // 5.
      // item->executor: std::unique_ptr<Executor>*
  }

  // Cache the mapping from input/output names to graph elements to
  // avoid recomputing it every time.
  if (!run_state_args->is_partial_run) {
    // For regular `Run()`, we use the function calling convention, and so
    // maintain a mapping from input/output names to
    // argument/return-value ordinal index.
    for (int i = 0; i < callable_options.feed().size(); ++i) {
      const string& input = callable_options.feed(i);
      ek->input_name_to_index[input] = i;
    }
    for (int i = 0; i < callable_options.fetch().size(); ++i) {
      const string& output = callable_options.fetch(i);
      ek->output_name_to_index[output] = i;
    }

  } else {

    // For `PRun()`, we use the rendezvous calling convention, and so
    // maintain a mapping from input/output names to rendezvous keys.
    //
    // We always use the first device as the device name portion of the
    // key, even if we're feeding another graph.
    for (int i = 0; i < callable_options.feed().size(); ++i) {
      const string& input = callable_options.feed(i);
      ek->input_name_to_rendezvous_key[input] = GetRendezvousKey(
          input, device_set_.client_device()->attributes(), FrameAndIter(0, 0));
    }
    for (int i = 0; i < callable_options.fetch().size(); ++i) {
      const string& output = callable_options.fetch(i);
      ek->output_name_to_rendezvous_key[output] =
          GetRendezvousKey(output, device_set_.client_device()->attributes(),
                           FrameAndIter(0, 0));
    }
  }


  *out_executors_and_keys = std::move(ek);
  *out_func_info = std::move(func_info);


  return Status::OK();
}
// DirectSession::CreateExecutors END!


/////////////////////////////////////////////////////////////////////////////
// 关注
// CreateExecutors
Status DirectSession::GetOrCreateExecutors(
    gtl::ArraySlice<string> inputs, // input
    gtl::ArraySlice<string> outputs, // input
    gtl::ArraySlice<string> target_nodes, // input
    ExecutorsAndKeys** executors_and_keys, // output
    RunStateArgs* run_state_args) { // input and output(for Graph graph device etc perspective.)

  // 1.
  // RunStateArgs 数据结构
  // tensorflow/core/common_runtime/direct_session.h:219:
  // struct RunStateArgs {
  //   RunStateArgs(const DebugOptions& options) : debug_options(options) {}
  //
  //   bool is_partial_run = false;
  //   string handle;
  //   std::unique_ptr<Graph> graph;   # 这里面居然包含了图？！！
  //   const DebugOptions& debug_options;
  //   int64 collective_graph_key = BuildGraphOptions::kNoCollectiveGraphKey;
  // };

  /**
  RunStateArgs* run_state_args 比如：

  p *run_state_args
  $10= {
    is_partial_run=false,
    handle="->out:0//0/;0",
    graph= {
      _M_t=std::tuple containing= {
        [1]=0x0,
        [2]= {
          <std::default_delete<tensorflow::Graph>>= {
            <No data fields>
          },
          <No data fields>
        }
      }
    },
    debug_options=@0x7f6e681a2780,
    collective_graph_key=0
  }
  */

  int64 handle_name_counter_value = -1;
  // handle_name_counter_value 函数说明
  // 根据下面这个变量是在 (LogMemory::IsEnabled() || run_state_args->is_partial_run) 时被计数增加的。
  // 提示: 这个变量的域只在这个函数里面，每次进入这个函数都重置为 -1 了。
  //      所以，应该是不重要的。

  if (LogMemory::IsEnabled() || run_state_args->is_partial_run) {
    handle_name_counter_value = handle_name_counter_.fetch_add(1);
    // handle_name_counter_ 变量说明:
    // For generating unique names for this session instance.
    // std::atomic<int64> edge_name_counter_ = {0};
    // std::atomic<int64> handle_name_counter_ = {0};
  }

  string debug_tensor_watches_summary;
  if (!run_state_args->debug_options.debug_tensor_watch_opts().empty()) {
    debug_tensor_watches_summary = SummarizeDebugTensorWatches(
        run_state_args->debug_options.debug_tensor_watch_opts());
  }

  // Since we insert the value under the original key, so the fast path lookup will work
  // if the user uses the same order of inputs, outputs, and targets again.
  // Fast lookup path, no sorting.
  // key =
  // inputs,->outputs,/target_nodes,/is_partial_run/debug_tensor_watches_summary
  const string key = strings::StrCat(
      str_util::Join(inputs, ","),
      "->",
      str_util::Join(outputs, ","),
      "/",
      str_util::Join(target_nodes, ","),
      "/",
      run_state_args->is_partial_run,
      "/",
      debug_tensor_watches_summary);

  /**
  打印:

  p key
  $7 = "->out:0//0/"
  */

  // Set the handle, if it's needed to log memory or for partial run.
  if (handle_name_counter_value >= 0) {
    run_state_args->handle =
        strings::StrCat(key, ";", handle_name_counter_value);
  }

  // --------------------------------------------------------------------
  // Get the executors if they exist
  // --------------------------------------------------------------------
  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto it = executors_.find(key);
    // 1.
    // executors_ 数据结构
    // tensorflow/core/common_runtime/direct_session.h:367:
    // std::unordered_map<string, std::shared_ptr<ExecutorsAndKeys>> executors_
    //                      |
    //                     key
    // 打印
    // p executors_
    // $5 = std::unordered_map with 1 element = {["->out:0//0/"] = std::shared_ptr<tensorflow::DirectSession::ExecutorsAndKeys> (use count 1, weak count 0) = {get() = 0x5643efe6ed30}}


    // 2.
    // it.second 变量说明:
    // it.second 是 std::shared_ptr<ExecutorsAndKeys>

    // 3.
    // struct ExecutorsAndKeys 数据结构
    // tensorflow/core/common_runtime/direct_session.h:155
    // 综述:An ExecutorsAndKeys is created for a given set of feeds/fetches.
    // 变量说明:
    // - items: std::vector<PerPartitionExecutorsAndLib>
    // - step_count: std::atomic_int_fast64_t
    //    - the number of times this graph is executed.
    // - graph: std::unique_ptr<Graph>
    //    - the entire graph being executed.
    // - name_to_node: NameNodeMap
    //    - maps node name to node.
    //      We keep 'graph' and 'name_to_node' only in the case of partial runs.
    //      Each item in 'items' is the executor for a partition of the graph
    //      bundled with its dependent library runtime.
    // - input_name_to_index: std::unordered_map<string, size_t>
    // - input_name_to_rendezvous_key: std::unordered_map<string, string>
    // - output_name_to_index: std::unordered_map<string, size_t>
    // - output_name_to_rendezvous_key: std::unordered_map<string, string>
    // - input_types: DataTypeVector
    //    - the rendezvous keys for the feeds
    // - output_types: DataTypeVector
    //    - rendezvous keys for the fetches.
    // - callable_options: CallableOptions
    // - collective_graph_key: int64, default: BuildGraphOptions::kNoCollectiveGraphKey

    // 4. PerPartitionExecutorsAndLib 数据结构说明:
    // tensorflow/core/common_runtime/direct_session.h:134:
    // struct PerPartitionExecutorsAndLib
    // tensorflow/core/common_runtime/direct_session.h
    // 非常非常的丰富，必须打开看。


    if (it != executors_.end()) {

      *executors_and_keys = it->second.get();
      // executors_and_keys 变量说明:
      // executors_and_keys: ExecutorsAndKeys**

      // ------------------------------
      // 所以，只在第一次时构造 executor, 第二次时，这里肯定就返回了
      return Status::OK();
      // ------------------------------
    }
  }
  // --------------------------------------------------------------------

  // Slow lookup path, the unsorted key missed the cache.
  // Sort the inputs and outputs, and look up with the sorted key in case an
  // earlier call used a different order of inputs and outputs.
  //
  // We could consider some other signature instead of sorting that
  // preserves the same property to avoid the sort in the future.
  std::vector<string> inputs_sorted(inputs.begin(), inputs.end());

  std::sort(inputs_sorted.begin(), inputs_sorted.end());

  std::vector<string> outputs_sorted(outputs.begin(), outputs.end());

  std::sort(outputs_sorted.begin(), outputs_sorted.end());

  std::vector<string> tn_sorted(target_nodes.begin(), target_nodes.end());

  std::sort(tn_sorted.begin(), tn_sorted.end());

  /**
  比如说：

  p inputs
  $2 = {static npos = <optimized out>, ptr_ = 0x0, len_ = 0}

  p outputs
  $3 = {static npos = <optimized out>, ptr_ = 0x564a25e446c0, len_ = 1}

  p outputs[0]
  $5 = "out:0"

  p target_nodes
  $6 = {static npos = <optimized out>, ptr_ = 0x0, len_ = 0}

  p key
  $7 = "->out:0//0/"
  */

  const string sorted_key = strings::StrCat(
      str_util::Join(inputs_sorted, ","),   // 上面的例子，没有 inputs_sorted
      "->",
      str_util::Join(outputs_sorted, ","), // 上面的例子, outputs_sorted 是 out:0
      "/",
      str_util::Join(tn_sorted, ","), // 上面的例子，没有 tn_sorted
      "/",
      run_state_args->is_partial_run, // 上面的例子, is_partial_run=0
      "/",
      debug_tensor_watches_summary);//上面的例子，没有 debug_tensor_watches_summary

  /**
  比如说:

  p sorted_key
  $8 = "->out:0//0/"
  */

  // Set the handle, if its needed to log memory or for partial run.
  if (handle_name_counter_value >= 0) {
    run_state_args->handle =
        strings::StrCat(sorted_key, ";", handle_name_counter_value);
  }

  // --------------------------------------------------------------------
  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);
    auto it = executors_.find(sorted_key);
    if (it != executors_.end()) {
      *executors_and_keys = it->second.get();
      // Insert this under the original key.
      executors_.emplace(key, it->second);
      return Status::OK();
    }
  }
  // --------------------------------------------------------------------

  /////////////////////////////////////////////////////////////////////////
  // Create Executors
  /////////////////////////////////////////////////////////////////////////

  // -----------------------------------------------------------------------
  // 如果没有找到 executors, 则进行 executors 的构造和初始化
  // -----------------------------------------------------------------------
  // Nothing found, so create the executors and store in the cache.
  // The executor_lock_ is intentionally released while executors are
  // being created.
  //
  // CallableOptions:
  // IMPT message defines feed and fetch
  // tensorflow/core/protobuf/config.proto
  // - input
  // - output
  // - target
  // - debug_options
  // - set_collective_graph_key
  CallableOptions callable_options;

  for (const string& input : inputs_sorted) {
    callable_options.add_feed(input);
  }
  for (const string& output : outputs_sorted) {
    callable_options.add_fetch(output);
  }
  for (const string& target : tn_sorted) {
    callable_options.add_target(target);
  }

  *callable_options.mutable_run_options()->mutable_debug_options() =
      run_state_args->debug_options;

  callable_options.mutable_run_options()
      ->mutable_experimental()
      ->set_collective_graph_key(run_state_args->collective_graph_key);

  // -----------------------------------------------------------------------

  /**
  CallableOptions 打印:

  p callable_options

  fetch: "out:0"
  run_options {
    debug_options {
    }
    experimental {
    }
  }

  评价: 大部分都是空的

  ===

  比如说:

  run_state_args: RunStateArgs*

  p *run_state_args

  $14= {
    is_partial_run=false,
    handle="->out:0//0/;0",
    graph= {
      _M_t=std::tuple containing= {
        [1]=0x0,
        [2]= {
          <std::default_delete<tensorflow::Graph>>= {
            <No data fields>
          },
          <No data fields>
        }
      }
    },
    debug_options=@0x7f6e681a2780,
    collective_graph_key=0
  }
  */

  // 重要！！！
  // =======================================================================
  std::unique_ptr<ExecutorsAndKeys> ek;
  // =======================================================================
  // 主要的构造对象

  // struct FunctionInfo 数据结构
  // tensorflow/core/common_runtime/direct_session.h:186:
  // - flib_def: std::unique_ptr<FunctionLibraryDefinition>
  // - proc_flr: std::unique_ptr<ProcessFunctionLibraryRuntime>
  std::unique_ptr<FunctionInfo> func_info;

  TF_RETURN_IF_ERROR(
      CreateExecutors(callable_options, // input
                      &ek,  // output
                      &func_info, // output
                      run_state_args) // input and output
                    );
  // -----------------------------------------------------------------------


  // -----------------------------------------------------------------------
  // 保存构造的 executor
  // -----------------------------------------------------------------------

  // Reacquire the lock, try to insert into the map.
  mutex_lock l(executor_lock_);

  functions_.push_back(std::move(func_info));

  // Another thread may have created the entry before us, in which case we will
  // reuse the already created one.

  // struct ExecutorsAndKeys 数据结构
  // tensorflow/core/common_runtime/direct_session.h:155

  // executors_ 数据结构
  // std::unordered_map<string, std::shared_ptr<ExecutorsAndKeys>> executors_
  //                       |
  //                      key

  auto insert_result = executors_.emplace(
      sorted_key,
      std::shared_ptr<ExecutorsAndKeys>(std::move(ek)));

  // Insert the value under the original key, so the fast path lookup will work
  // if the user uses the same order of inputs, outputs, and targets again.
  executors_.emplace(key, insert_result.first->second);

  *executors_and_keys = insert_result.first->second.get();
  // -----------------------------------------------------------------------

  return Status::OK();
}






// [已经整理]
/**

QQQ. 被谁调用？
AAA. 在 DirectSession::CreateExecutors 里面

QQQ. DirectSession::CreateGraphs 目的是什么？
AAA.

QQQ. DirectSession::CreateGraphs 完成了哪些事情？
AAA.

QQQ. CreateGraphs 最原始的图从哪里输入的？
AAA. 来自 DirectSession::execution_state_ : std::unique_ptr<GraphExecutionState>
     这个成员变量在 DirectSession::Extend 调用栈系列里面被构造和初始化，包括了 device placement 也弄好了。

*/
Status DirectSession::CreateGraphs(
    const BuildGraphOptions& subgraph_options,  // input
    //-----------------------------------------------------------------------
    // string 是表示 GPU 或者 CPU
    std::unordered_map<string, std::unique_ptr<Graph>>* outputs, // output
    //-----------------------------------------------------------------------

    std::unique_ptr<FunctionLibraryDefinition>* flib_def, // output

    // ------------------------------------
    RunStateArgs* run_state_args, // input and output
    // ------------------------------------

    DataTypeVector* input_types, // output
    DataTypeVector* output_types, // output
    int64* collective_graph_key) // output
{
  // 1.
  // subgraph_options 变量说明:
  // subgraph_options 打印:
  //
  // Feed endpoints:
  // Fetch endpoints: out:0,
  // Target nodes:
  // collective_order: none

  // 2.
  // run_state_args 变量说明:
  // run_state_args: RunStateArgs*
  // 打印:
  // p *run_state_args
  // $10= {
  //   is_partial_run=false,
  //   handle="->out:0//0/;0",
  //   graph= {
  //     _M_t=std::tuple containing= {
  //       [1]=0x0,
  //       [2]= {
  //         <std::default_delete<tensorflow::Graph>>= {
  //           <No data fields>
  //         },
  //         <No data fields>
  //       }
  //     }
  //   },
  //   debug_options=@0x7f6e681a2780,
  //   collective_graph_key=0
  // }

  mutex_lock l(graph_state_lock_);

  // -----------------------------------------------------------------------
  std::unique_ptr<ClientGraph> client_graph;
  // -----------------------------------------------------------------------
  // 1.
  // struct ClientGraph 数据结构
  // tensorflow/core/common_runtime/graph_execution_state.h:53:
  // - flib_def: std::unique_ptr<FunctionLibraryDefinition>
  // - graph: Graph
  // - feed_types: DataTypeVector
  // - fetch_types: DataTypeVector
  // - collective_graph_key: int64
  //
  // 概述:
  // A ClientGraph is simply a sub-graph of the full graph as induced by
  // BuildGraphOptions.
  //
  // 这个变量用在 GraphExecutionState::BuildGraph 里面构造了 ClientGraph
  // 的 output, 被这个 client_graph 指针指着

  std::unique_ptr<GraphExecutionState> temp_exec_state_holder;
  // 1.
  // class GraphExecutionState 数据结构
  // tensorflow/core/common_runtime/graph_execution_state.h

  GraphExecutionState* execution_state = nullptr; // 首次定义这个变量

  // -------------------------------------------------------------
  // 没有进入 place_pruned_graph() 这个分支,
  // 进入了 execution_state->BuildGraph 分支
  // -------------------------------------------------------------

  // 这个 if 分支不看
  // p options_.config.graph_options().place_pruned_graph()
  // $20 = false
  if (options_.config.graph_options().place_pruned_graph()) {
    // 没有进入这个分支

    // 1.
    // options_ 变量说明:
    // DirectSession::options_: const SessionOptions options_
    // SessionOptions 数据结构
    // tensorflow/core/public/session_options.h:28: struct SessionOptions
    // - env: Env*
    // - target: string, The TensorFlow runtime to connect to engine
    // - config: message ConfigProto

    // 2.
    // message GraphOptions::place_pruned_graph
    // Only place the subgraphs that are run, rather than the entire graph.
    //
    // This is useful for interactive graph building, where one might
    // produce graphs that cannot be placed during the debugging
    // process.  In particular, it allows the client to continue work in
    // a session after adding a node to a graph whose placement
    // constraints are unsatisfiable.

    // Because we are placing pruned graphs, we need to create a
    // new GraphExecutionState for every new unseen graph,
    // and then place it.
    //
    // tensorflow/core/common_runtime/graph_execution_state.h:41:
    // struct GraphExecutionStateOptions
    //   - device_set: const DeviceSet*
    //   - session_options: const SessionOptions* session_options
    //   - session_handle: string
    //   - stateful_placements : std::unordered_map<string, string>
    //     * A map from node name to device name, representing the unchangeable placement of stateful nodes.

    GraphExecutionStateOptions prune_options;
    prune_options.device_set = &device_set_;
    prune_options.session_options = &options_;
    prune_options.stateful_placements = stateful_placements_;
    prune_options.session_handle = session_handle_;

    // -----------------------------------------------------------------------
    // Execution_state; used when placing the entire graph.
    TF_RETURN_IF_ERROR(
      GraphExecutionState::MakeForPrunedGraph(
        execution_state_->original_graph_def().library(), // input
        prune_options, // input
        execution_state_->original_graph_def(), // input
        subgraph_options, // input
        &temp_exec_state_holder, // output
        &client_graph)); // output
    // -----------------------------------------------------------------------
    // 1.
    // execution_state_ 变量说明:
    // DirectSession::execution_state_: std::unique_ptr<GraphExecutionState>
    // QQQ. execution_state_ 何时初始化的?
    // AAA.
    // tensorflow/core/common_runtime/direct_session.cc
    // Status DirectSession::MaybeInitializeExecutionState
    // tensorflow/core/common_runtime/direct_session.cc
    // GraphExecutionState::MakeForBaseGraph(&temp, options, &execution_state_));
    // tensorflow/core/common_runtime/direct_session.cc-462-
    // Status DirectSession::ExtendLocked(const GraphDef& graph) {
    //  - MaybeInitializeExecutionState(graph, &already_initialized)
    //  - execution_state_->Extend(graph, &state)
    //  - execution_state_.swap(state)

    // 2.
    // class GraphExecutionState 数据结构
    // tensorflow/core/common_runtime/graph_execution_state.h
    // 很丰富，打开看
    // GraphExecutionState is responsible for generating an
    // executable ClientGraph from the original GraphDef that specifies
    // the complete graph and from BuildGraphOptions which specifies
    // input/output nodes.
    //
    // An executable Graph differs from a GraphDef by being Placed,
    // meaning that each Node is assigned to a single Device in the
    // available set.
    // - graph_: Graph*
    // - original_graph_def_: GraphDef
    // - session_handle_: string

    // 2.
    // MakeForPrunedGraph 函数说明:
    // GraphExecutionState::MakeForPrunedGraph 这个是用来构造 fetch 所需要的子图的
    // tensorflow/core/common_runtime/graph_execution_state.cc

    // QQQ. 为什么 DirectSession 也有  execution_state_?
    // AAA. 用于 used when placing **the entire** graph.
    // 在 GraphExecutionState::MakeForBaseGraph(&temp, options, &execution_state_)); 里面被初始化吧
    // 或者是在 DirectSession::ExtendLocked 里面的
    //   execution_state_->Extend(graph, &state);
    //   execution_state_.swap(state); 被初始化吧

    execution_state = temp_exec_state_holder.get();

  } else {
    // 最先是进入了这个分支

    execution_state = execution_state_.get();
    // 1.
    // execution_state_ 变量说明:
    // DirectSession::execution_state_ : std::unique_ptr<GraphExecutionState>
    // 包含了 device placement 信息了。

    // 2.
    // GraphExecutionState 数据结构
    // tensorflow/core/common_runtime/graph_execution_state.h

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // 使用 DirectSession::low_priority_execution_state_ 构造 BuildGraph
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // -----------------------------------------------------------------------
    TF_RETURN_IF_ERROR(
        // BuildGraph 用来构造 client graph 的。
        execution_state->BuildGraph(
          subgraph_options, // input
          &client_graph)); // output : std::unique_ptr<ClientGraph>*
    // -----------------------------------------------------------------------
  }
  // 到目前为止，没有初始化 op kernel

  *collective_graph_key = client_graph->collective_graph_key;

  // 异常处理部分，可以不看
  if (subgraph_options.callable_options.feed_size() !=
      client_graph->feed_types.size()) {
    return errors::Internal(
        "Graph pruning failed: requested number of feed endpoints = ",
        subgraph_options.callable_options.feed_size(),
        " versus number of pruned feed endpoints = ",
        client_graph->feed_types.size());
  }
  // 异常处理部分，可以不看
  if (subgraph_options.callable_options.fetch_size() !=
      client_graph->fetch_types.size()) {
    return errors::Internal(
        "Graph pruning failed: requested number of fetch endpoints = ",
        subgraph_options.callable_options.fetch_size(),
        " versus number of pruned fetch endpoints = ",
        client_graph->fetch_types.size());
  }

  auto current_stateful_placements = execution_state->GetStatefulPlacements();
  // 1.
  // execution_state->GetStatefulPlacements() 说明:
  // GraphExecutionState::stateful_placements_: std::unordered_map<string, string>
  //                                            含义  Maps node names to device names
  // Once placed these nodes can not be moved to a different device.

  // 2.
  // 打印:
  // p stateful_placements_
  //   $5 = std::unordered_map with 4 elements = {
  //   ["y/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:CPU:0",
  //   ["x/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:GPU:0",
  //   ["a/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:CPU:0",
  //   ["b/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:GPU:0"
  // }

  // Update our current state based on the execution_state's
  // placements.  **If there are any mismatches for a node,
  // we should fail, as this should never happen.**
  for (auto placement_pair : current_stateful_placements) {

    const string& node_name = placement_pair.first;  // node name
    const string& placement = placement_pair.second; // device name

    auto iter = stateful_placements_.find(node_name);
    // 1.
    // stateful_placements_ 变量说明:
    // DirectSession::stateful_placements_ :
    //   std::unordered_map<std::string, std::string>

    // 2.
    // 打印:
    // p stateful_placements_
    // $7 = std::unordered_map with 0 elements
    // 起初是没有值在里面的
    // 第一次赋值应该是在下面

    // 3.
    // Bug Report:
    // 我的设计，在第二次 CPU 图进入时是这样的，导致 mismatch bug
    // p stateful_placements_
    // $4 = std::unordered_map with 4 elements = {["y/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:CPU:0", ["x/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:GPU:0", ["a/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:CPU:0", ["b/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:GPU:0"}

    // 如果没有在 DirectSession::stateful_placements_  内找到 placement device type 就添加到 DirectSession::stateful_placements_  里面
    if (iter == stateful_placements_.end()) {
      stateful_placements_.insert(std::make_pair(node_name, placement));
    } else if (iter->second != placement) {
      // 否则 DirectSession::stateful_placements_ 和
      //     GraphExecutionState::stateful_placements_
      // 不匹配则报错
      return errors::Internal(
          "Stateful placement mismatch. "
          "Current assignment of ",
          node_name, " to ", iter->second, " does not match ", placement);
    }
  }

  stateful_placements_ = execution_state->GetStatefulPlacements();
  // 1.
  // QQQ. 奇怪的重复赋值
  // AAA. 我想，上面的主要目的是 坚持是否有 mismatch 的情况
  /*
  赋值前的 stateful_placements_
  p stateful_placements_
  $8 = std::unordered_map with 4 elements = {["b/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:GPU:0", ["a/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:CPU:0", ["y/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:CPU:0", ["x/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:GPU:0"}

  赋值后的 stateful_placements_
  p stateful_placements_
  $9 = std::unordered_map with 4 elements = {["y/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:CPU:0", ["x/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:GPU:0", ["a/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:CPU:0", ["b/RandomStandardNormal"] = "/job:localhost/replica:0/task:0/device:GPU:0"}

  keys的首字母，排了从大到小的序？
  */

  // Remember the graph in run state if this is a partial run.
  if (run_state_args->is_partial_run) { // 这个分支没有进入
    // 初始化 run_state_args->graph
    run_state_args->graph.reset(new Graph(flib_def_.get()));

    CopyGraph(*execution_state->full_graph(), run_state_args->graph.get());
    // 1.
    // execution_state 变量说明:
    // execution_state: GraphExecutionState*

    // 2.
    // class GraphExecutionState 数据结构
    // tensorflow/core/common_runtime/graph_execution_state.h
    // - graph_: Graph*
    // - rewrite_metadata_: std::unique_ptr<subgraph::RewriteGraphMetadata>
    // - flib_def_: std::unique_ptr<FunctionLibraryDefinition>
    // - node_name_to_cost_id_map_: NodeNameToCostIdMap
    // - session_handle_: string
    // - session_options_: const SessionOptions*
    // - device_set_: const DeviceSet*
    // - original_graph_def_: GraphDef
    // - stateful_placements_: std::unordered_map<string, string>
    //
    // GraphExecutionState 功能和目的:
    // GraphExecutionState is responsible for generating an
    // executable ClientGraph from the original GraphDef that specifies
    // the complete graph and from BuildGraphOptions which specifies
    // input/output nodes.
    //
    // An executable Graph differs from a GraphDef by being Placed,
    // meaning that each Node is assigned to a single Device in the
    // available set.
    // When GraphExecutionState is first constructed it instantiates
    // a full Graph from the provided GraphDef, and places it, using only
    // the static device assignments from the GraphDef.  Nodes without are
    // currently placed in a very naive way.  Since stateful Nodes cannot
    // be moved after initial placement, it is important that stateful
    // Nodes get sensible initial device assignments in the graph
    // definition.
    //
    // Subsequently, GraphExecutionState generates a SimpleClientGraph on
    // demand, which is a sub-graph of the latest placement of the full
    // Graph.  MasterSession uses such a ClientGraph to execute one or
    // more similar client requests.

    // 3.
    // CopyGraph 函数说明:
    // tensorflow/core/graph/graph_constructor.cc:1304
    // void CopyGraph(const Graph& src, Graph* dest)
    // 构造了一个新图，复制了大部分重要的信息，包括节点的 device placement.
  }


  ////////////////////////////////////////////////////////////////////////
  // 以上是构造 client graph
  ////////////////////////////////////////////////////////////////////////
  // 以下是 partition graph
  ////////////////////////////////////////////////////////////////////////

  // Partition the graph across devices.
  PartitionOptions popts;
  // 1.
  // PartitionOptions 数据结构
  // tensorflow/core/graph/graph_partition.h:31:
  // struct PartitionOptions
  // - string NodeToLocFunc(Node*) : lambda
  // - string NewNameFunc(string) : lambda
  // - uint64 GetIncarnationFunc(string) : lambda
  // - flib_def: const FunctionLibraryDefinition*
  //     FunctionLibraryDefinition 数据结构
  //     ./tensorflow/core/framework/function.h:313:
  //     class FunctionLibraryDefinition : public OpRegistryInterface
  //     - default_registry_ : const OpRegistryInterface* const
  //     - function_defs_ : gtl::FlatMap<string, std::unique_ptr<FunctionDefAndOpRegistration>>
  //      * FunctionDefAndOpRegistration
  //        + fdef: FunctionDef
  //        + op_registration_data: OpRegistrationData
  //     - func_grad_ : gtl::FlatMap<string, string>
  // - control_flow_added: bool
  // - DataType ShouldCastFunc(const Edge*): lambda
  // - scheduling_for_recvs: bool
  // - need_to_record_start_times: bool
  // - start_times: std::vector<Microseconds>

  // ------------------------------------------------------------------------
  // QQQ. 为什么这个地方 Node to device 已经赋值好了？
  // AAA. Placer 类负责每个节点的 device 初始化, 所以，每个节点此时都有了被赋值的 device
  // ------------------------------------------------------------------------
  popts.node_to_loc = [](const Node* node) {
    return node->assigned_device_name();
  };
  // ------------------------------------------------------------------------

  popts.new_name = [this](const string& prefix) {
    return strings::StrCat(prefix, "/_", edge_name_counter_.fetch_add(1));
  };
  // 1.
  // edge_name_counter_ 变量说明:
  // DirectSession::edge_name_counter_ : std::atomic<int64>, default : 0
  // 说明:
  // For generating unique names for this session instance.

  popts.get_incarnation = [](const string& name) {
    // The direct session does not have changing incarnation numbers.
    // Just return '1'.
    return 1;
  };

  popts.flib_def = &client_graph->graph.flib_def();
  // 1.
  // popts.flib_def : const FunctionLibraryDefinition* 完整的见上面

  popts.control_flow_added = false;

  // -----------------------------------------------------------------------
  std::unordered_map<string, GraphDef> partitions;
  //                    |
  //                 CPU/GPU
  // -----------------------------------------------------------------------

  TF_RETURN_IF_ERROR(
    Partition(popts,  // input
              &client_graph->graph, // input
              &partitions) // output
  );
  // 1.
  // input 变量说明
  // input: &client_graph->graph, 在 execution_state->BuildGraph 里面构造了

  // 2.
  // output 变量说明
  // output: partitions

  // 3.
  // Partition 函数说明
  // tensorflow/core/graph/graph_partition.h:87:
  // Status Partition(const PartitionOptions& opts,
  //                  Graph* input,
  //                  std::unordered_map<string, GraphDef>* partitions);

  std::vector<string> device_names;
  for (auto device : devices_) {
    // Extract the LocalName from the device.
    device_names.push_back(DeviceNameUtils::LocalName(device->name()));
  }

  // Check for valid partitions.
  for (const auto& partition : partitions) {
    // 检查 device
    const string local_partition_name =
        DeviceNameUtils::LocalName(partition.first);

    if (std::count(device_names.begin(), device_names.end(),
                   local_partition_name) == 0) {
      return errors::InvalidArgument(
          "Creating a partition for ", local_partition_name,
          " which doesn't exist in the list of available devices. Available "
          "devices: ",
          str_util::Join(device_names, ","));
    }
  }


  for (const auto& partition : partitions) {

    // 构造了 子图
    std::unique_ptr<Graph> device_graph(
      new Graph(client_graph->flib_def.get()));

    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = true;

    TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(
        device_opts,
        partition.second,
        device_graph.get()));

    outputs->emplace(partition.first, std::move(device_graph));

  }

  GraphOptimizationPassOptions optimization_options;
  optimization_options.session_options = &options_;
  optimization_options.flib_def = client_graph->flib_def.get();
  optimization_options.partition_graphs = outputs;
  TF_RETURN_IF_ERROR(
    OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_PARTITIONING, optimization_options));

  Status s;
  for (auto& partition : *outputs) {
    const string& partition_name = partition.first;
    std::unique_ptr<Graph>* graph = &partition.second;

    // log: https://gist.github.com/shizukanaskytree/b2484cc4da754563f9fb2cc7cc1e8782
    VLOG(2) << "Created " << DebugString(graph->get()) << " for "
            << partition_name;

    // Give the device an opportunity to rewrite its subgraph.
    Device* d;
    s = device_mgr_->LookupDevice(partition_name, &d);
    if (!s.ok()) break;
    s = d->MaybeRewriteGraph(graph);
    if (!s.ok()) {
      break;
    }
  }

  *flib_def = std::move(client_graph->flib_def);
  std::swap(*input_types, client_graph->feed_types);
  std::swap(*output_types, client_graph->fetch_types);

  return s;
} // DirectSession::CreateGraphs END


////////////////////////////////////////////////////////////////////////


::tensorflow::Status DirectSession::ListDevices(
    std::vector<DeviceAttributes>* response) {
  response->clear();
  response->reserve(devices_.size());
  for (Device* d : devices_) {
    const DeviceAttributes& attrs = d->attributes();
    response->emplace_back(attrs);
  }
  return ::tensorflow::Status::OK();
}

::tensorflow::Status DirectSession::Reset(
    const std::vector<string>& containers) {
  device_mgr_->ClearContainers(containers);
  return ::tensorflow::Status::OK();
}

::tensorflow::Status DirectSession::Close() {
  cancellation_manager_->StartCancel();
  {
    mutex_lock l(closed_lock_);
    if (closed_) return ::tensorflow::Status::OK();
    closed_ = true;
  }
  if (factory_ != nullptr) factory_->Deregister(this);
  return ::tensorflow::Status::OK();
}


DirectSession::RunState::RunState(
    const std::vector<string>& pending_input_names,
    const std::vector<string>& pending_output_names,
    int64 step_id,
    const std::vector<Device*>* devices)

    : step_container(
      step_id,
      [devices, step_id](const string& name) {
        for (auto d : *devices) {
          if (!d->resource_manager()->Cleanup(name).ok()) {
            // Do nothing...
          }

          ScopedAllocatorMgr* sam = d->GetScopedAllocatorMgr();
          if (sam) sam->Cleanup(step_id);
        }
      })
{
  // Initially all the feeds and fetches are pending.
  for (auto& name : pending_input_names) {
    pending_inputs[name] = false;
  }
  for (auto& name : pending_output_names) {
    pending_outputs[name] = false;
  }
}


DirectSession::RunState::RunState(
  int64 step_id,
  const std::vector<Device*>* devices)
    : RunState({}, {}, step_id, devices) {}



DirectSession::RunState::~RunState() {
  if (rendez != nullptr) {
    if (!executors_done.HasBeenNotified()) {
      rendez->StartAbort(errors::Cancelled("PRun cancellation"));
      executors_done.WaitForNotification();
    }
    rendez->Unref();
  }
}

bool DirectSession::RunState::PendingDone() const {
  for (const auto& it : pending_inputs) {
    if (!it.second) return false;
  }
  for (const auto& it : pending_outputs) {
    if (!it.second) return false;
  }
  return true;
}


// tf main thread              Other threads
//       +                   +  +  +        +
//       |                   |  |  |        |
//       |                   |  |  | ...... |
//       |                   |  |  |        |
//       |                   v  v  v        v
//       v                  ------------------
//   ---------                     执行任务
// main thread 卡在这里
void DirectSession::WaitForNotification(RunState* run_state, // input
                                        CancellationManager* cm, // input
                                        int64 timeout_in_ms) { // input
  const Status status =
      WaitForNotification(
        &run_state->executors_done,
        timeout_in_ms);
  // 1.
  // WaitForNotification 函数接口说明:
  // ::tensorflow::Status DirectSession::WaitForNotification(
  //     Notification* notification,
  //     int64 timeout_in_ms)
  //
  // tensorflow/core/common_runtime/direct_session.cc

  if (!status.ok()) {
    {
      mutex_lock l(run_state->mu_);
      run_state->status.Update(status);
    }
    cm->StartCancel();
    // We must wait for the executors to complete, because they have borrowed
    // references to `cm` and other per-step state. After this notification, it
    // is safe to clean up the step.
    run_state->executors_done.WaitForNotification();
  }

}

// tf main thread              Other threads
//       +                   +  +  +        +
//       |                   |  |  |        |
//       |                   |  |  | ...... |
//       |                   |  |  |        |
//       |                   v  v  v        v
//       v                  ------------------
//   ---------                     执行任务
// main thread 卡在这里
::tensorflow::Status DirectSession::WaitForNotification(
    Notification* notification,
    int64 timeout_in_ms) {

  if (timeout_in_ms > 0) {
    const int64 timeout_in_us = timeout_in_ms * 1000;

    const bool notified =
        WaitForNotificationWithTimeout(notification, timeout_in_us);

    if (!notified) {
      return Status(error::DEADLINE_EXCEEDED,
                    "Timed out waiting for notification");
    }

  } else {
    // 无限等待
    notification->WaitForNotification();

  }

  return Status::OK();
}



Status DirectSession::MakeCallable(const CallableOptions& callable_options, // input
                                   CallableHandle* out_handle) {  // output, 从 0 开始计数，0，1，2，3 ...
  // 1.
  // callable_options 数据结构
  // callable_options: const CallableOptions&

  // 2.
  // message CallableOptions 数据结构
  // tensorflow/core/protobuf/config.proto:559
  //
  // 功能和目的:
  // Defines a subgraph in another `GraphDef` as a set of feed points and nodes
  // to be fetched or executed.
  //
  // - feed: repeated string
  //    - Tensors to be fed in the callable. Each feed is the name of a tensor.
  // - fetch: repeated string
  //    - Fetches. A list of tensor names. The caller of the callable expects a
  //      tensor to be returned for each fetch[i] (see RunStepResponse.tensor). The
  //      order of specified fetches does not change the execution order.
  // - target: repeated string
  //    - Target Nodes. A list of node names. The named nodes will be run by the
  //      callable but their outputs will not be returned.
  // - run_options: RunOptions
  //    - Options that will be applied to each run.
  // - tensor_connection: repeated TensorConnection
  //    - Tensors to be connected in the callable. Each TensorConnection denotes
  //      a pair of tensors in the graph, between which an edge will be created
  //      in the callable.
  // - feed_devices: map<string, string>
  // - fetch_devices: map<string, string>
  // - fetch_skip_sync: bool

  // 3.
  // CallableHandle 数据结构
  // typedef int64 CallableHandle
  // tensorflow/core/public/session.h
  // tensorflow/cc/client/client_session.h:92:  typedef int64 CallableHandle;


  TF_RETURN_IF_ERROR(CheckNotClosed());
  // 1.
  // CheckNotClosed 函数说明
  // 概述:
  // 检查 closed_ flag 标志位

  TF_RETURN_IF_ERROR(CheckGraphCreated("MakeCallable()"));
  // 1.
  // CheckGraphCreated 函数说明
  // 概述:
  // 检查 graph_created_ flag 标志位

  std::unique_ptr<ExecutorsAndKeys> ek; // 临时变量
  std::unique_ptr<FunctionInfo> func_info; // 临时变量

  RunStateArgs run_state_args(callable_options.run_options().debug_options());
  // 1.
  // struct RunStateArgs 数据结构
  // tensorflow/core/common_runtime/direct_session.h
  // - is_partial_run: bool, default_value : false
  // - handle : string
  // - graph: std::unique_ptr<Graph>
  // - debug_options: const DebugOptions&
  // - collective_graph_key: int64, default_value: BuildGraphOptions::kNoCollectiveGraphKey

  // =======================================================================
  TF_RETURN_IF_ERROR(
      CreateExecutors(callable_options,
                      &ek,
                      &func_info,
                      &run_state_args));
  // =======================================================================

  {
    mutex_lock l(callables_lock_);

    *out_handle = next_callable_handle_++;
    // 1.
    // next_callable_handle_ 变量说明
    // DirectSession::ext_callable_handle_: int64 , initial_value : 0

    callables_[*out_handle] = {std::move(ek), std::move(func_info)};
    // 1.
    // callables_ 变量说明
    // DirectSession::callables_: std::unordered_map<int64, Callable>
    //
    // 提示: *out_handle 为
    // 0
    // 1
    // 2
    // ...

    // 2.
    // DirectSession:: struct Callable 数据结构
    // - executors_and_keys: std::shared_ptr<ExecutorsAndKeys>
    // - function_info: std::shared_ptr<FunctionInfo>

    // 3.
    // ek 变量说明
    // ek: std::unique_ptr<ExecutorsAndKeys> ;
    //
    // struct ExecutorsAndKeys 数据结构
    // 综述:An ExecutorsAndKeys is created for a given set of feeds/fetches.
    // 变量说明:
    //   ================================================
    // - items: std::vector<PerPartitionExecutorsAndLib>  # 最重要
    //   ================================================
    // - step_count: std::atomic_int_fast64_t
    //    - the number of times this graph is executed.
    // - graph: std::unique_ptr<Graph>
    //    - the entire graph being executed.
    // - name_to_node: NameNodeMap
    //    - maps node name to node.
    //      We keep 'graph' and 'name_to_node' only in the case of partial runs.
    //      Each item in 'items' is the executor for a partition of the graph
    //      bundled with its dependent library runtime.
    // - input_name_to_index: std::unordered_map<string, size_t>
    // - input_name_to_rendezvous_key: std::unordered_map<string, string>
    // - output_name_to_index: std::unordered_map<string, size_t>
    // - output_name_to_rendezvous_key: std::unordered_map<string, string>
    // - input_types: DataTypeVector
    //    - the rendezvous keys for the feeds
    // - output_types: DataTypeVector
    //    - rendezvous keys for the fetches.
    // - callable_options: CallableOptions
    // - collective_graph_key: int64, default: BuildGraphOptions::kNoCollectiveGraphKey

    // 4.
    // func_info 变量说明:
    // func_info: std::unique_ptr<FunctionInfo> ;
    //
    // FunctionInfo 数据结构
    // tensorflow/core/common_runtime/direct_session.h:192:  struct FunctionInfo
    // - flib_def: std::unique_ptr<FunctionLibraryDefinition>
    // - proc_flr: std::unique_ptr<ProcessFunctionLibraryRuntime>
    //
    // 概述:
    // A FunctionInfo object is created for every unique set of feeds/fetches.
    // This info could be folded into the ExecutorsAndKeys object but we would
    // like to maintain a deletion order in which the OpKernels (owned by the
    // executor) should be destroyed first, followed by the resources in the
    // device and then followed by the function stuff.
    // TODO(rohanj): Consolidate function library definitions so that we can
    // instantiate only one ProcFLR and lib_def and make this just a member
    // variable and not a vector.
    // 'flib_def' is the function library used.
    // 'proc_flr' is the collection of FunctionLibraryRuntime objects, one per
    // device.
  }

  return Status::OK();
}



class DirectSession::RunCallableCallFrame : public CallFrameInterface {
 public:
  RunCallableCallFrame(DirectSession* session,
                       ExecutorsAndKeys* executors_and_keys,
                       const std::vector<Tensor>* feed_tensors,
                       std::vector<Tensor>* fetch_tensors)
      : session_(session),
        executors_and_keys_(executors_and_keys),
        feed_tensors_(feed_tensors),
        fetch_tensors_(fetch_tensors) {}

  size_t num_args() const override {
    return executors_and_keys_->input_types.size();
  }
  size_t num_retvals() const override {
    return executors_and_keys_->output_types.size();
  }

  Status GetArg(int index,  // input
                Tensor* val)  // output
                const override {
    if (index > feed_tensors_->size()) {
      return errors::Internal("Args index out of bounds: ", index);
    } else if (executors_and_keys_->input_types[index] == DT_RESOURCE) {
      TF_RETURN_IF_ERROR(
          session_->ResourceHandleToInputTensor((*feed_tensors_)[index], val));
    } else {
      // -------------------------------------------------------------
      *val = (*feed_tensors_)[index];
      // -------------------------------------------------------------
    }
    return Status::OK();
  }

  Status SetRetval(int index,         // input
                   const Tensor& val) // input
                   override {
    if (index > fetch_tensors_->size()) {
      return errors::Internal("RetVal index out of bounds: ", index);
    }
    (*fetch_tensors_)[index] = val;
    return Status::OK();
  }

 private:
  DirectSession* const session_;                   // Not owned.
  ExecutorsAndKeys* const executors_and_keys_;     // Not owned.
  const std::vector<Tensor>* const feed_tensors_;  // Not owned.
  std::vector<Tensor>* const fetch_tensors_;       // Not owned.
};
// 1.
// RunCallableCallFrame 数据结构
// tensorflow/core/common_runtime/direct_session.cc:2329:
// class DirectSession::RunCallableCallFrame : public CallFrameInterface
// - session_: DirectSession* const
// - executors_and_keys_: ExecutorsAndKeys* const
// - feed_tensors_: const std::vector<Tensor>*
// - fetch_tensors_: std::vector<Tensor>* const

// 2.
// RunCallableCallFrame 构造函数接口说明
// RunCallableCallFrame(DirectSession* session,
//                      ExecutorsAndKeys* executors_and_keys,
//                      const std::vector<Tensor>* feed_tensors,
//                      std::vector<Tensor>* fetch_tensors)

// -----------------------------------------------------------------------

::tensorflow::Status DirectSession::RunCallable(
    CallableHandle handle, // input, type alias : int64
    const std::vector<Tensor>& feed_tensors,
    std::vector<Tensor>* fetch_tensors,
    RunMetadata* run_metadata) {
  // 1.
  // handle 变量说明：

  // 2.
  // CallableHandle 数据结构
  // typedef int64 CallableHandle
  // tensorflow/core/public/session.h
  // tensorflow/cc/client/client_session.h:92:  typedef int64 CallableHandle;

  TF_RETURN_IF_ERROR(CheckNotClosed());
  TF_RETURN_IF_ERROR(CheckGraphCreated("RunCallable()"));

  direct_session_runs->GetCell()->IncrementBy(1);
  // 1.
  // direct_session_runs 变量说明
  // 计数跑了几次的，不太重要

  // Check if we already have an executor for these arguments.
  std::shared_ptr<ExecutorsAndKeys> executors_and_keys; // 临时变量
  // 1.
  // executors_and_keys 变量说明

  // 2.
  // struct ExecutorsAndKeys 数据结构
  // 综述:An ExecutorsAndKeys is created for a given set of feeds/fetches.
  // 变量说明:
  //   ================================================
  // - items: std::vector<PerPartitionExecutorsAndLib>  # 最重要
  //   ================================================
  // - step_count: std::atomic_int_fast64_t
  //    - the number of times this graph is executed.
  // - graph: std::unique_ptr<Graph>
  //    - the entire graph being executed.
  // - name_to_node: NameNodeMap
  //    - maps node name to node.
  //      We keep 'graph' and 'name_to_node' only in the case of partial runs.
  //      Each item in 'items' is the executor for a partition of the graph
  //      bundled with its dependent library runtime.
  // - input_name_to_index: std::unordered_map<string, size_t>
  // - input_name_to_rendezvous_key: std::unordered_map<string, string>
  // - output_name_to_index: std::unordered_map<string, size_t>
  // - output_name_to_rendezvous_key: std::unordered_map<string, string>
  // - input_types: DataTypeVector
  //    - the rendezvous keys for the feeds
  // - output_types: DataTypeVector
  //    - rendezvous keys for the fetches.
  // - callable_options: CallableOptions
  // - collective_graph_key: int64, default: BuildGraphOptions::kNoCollectiveGraphKey

  // 3.
  // message CallableOptions 数据结构
  // tensorflow/core/protobuf/config.proto:559
  //
  // 功能和目的:
  // Defines a subgraph in another `GraphDef` as a set of feed points and nodes
  // to be fetched or executed.
  //
  // - feed: repeated string
  //    - Tensors to be fed in the callable. Each feed is the name of a tensor.
  // - fetch: repeated string
  //    - Fetches. A list of tensor names. The caller of the callable expects a
  //      tensor to be returned for each fetch[i] (see RunStepResponse.tensor). The
  //      order of specified fetches does not change the execution order.
  // - target: repeated string
  //    - Target Nodes. A list of node names. The named nodes will be run by the
  //      callable but their outputs will not be returned.
  // - run_options: RunOptions
  //    - Options that will be applied to each run.
  // - tensor_connection: repeated TensorConnection
  //    - Tensors to be connected in the callable. Each TensorConnection denotes
  //      a pair of tensors in the graph, between which an edge will be created
  //      in the callable.
  // - feed_devices: map<string, string>
  // - fetch_devices: map<string, string>
  // - fetch_skip_sync: bool

  const int64 step_id = step_id_counter_.fetch_add(1);

  {
    tf_shared_lock l(callables_lock_);
    // 异常检测
    if (handle >= next_callable_handle_) {
      return errors::InvalidArgument("No such callable handle: ", handle);
    }

    // 正常处理
    // =======================================================================
    executors_and_keys = callables_[handle].executors_and_keys;
    // =======================================================================
    // 1.
    // callables_ 变量说明:
    // std::unordered_map<int64, Callable> callables_ GUARDED_BY(callables_lock_);
    //
    // 打印
    // std::unordered_map with 1 element = {
    //   [0] = {
    //     executors_and_keys = std::shared_ptr < tensorflow::DirectSession::ExecutorsAndKeys > (use count 1, weak count 0) = {
    //       get() = 0x55a40d953400
    //     },
    //     function_info = std::shared_ptr < tensorflow::DirectSession::FunctionInfo > (use count 1, weak count 0) = {
    //       get() = 0x55a40d6b8560
    //     }
    //   }
    // }

    // 2.
    // Callable 数据结构说明:
    // DirectSession:: struct Callable
    // - executors_and_keys: std::shared_ptr<ExecutorsAndKeys>
    // - function_info: std::shared_ptr<FunctionInfo>
  }

  if (!executors_and_keys) {
    return errors::InvalidArgument(
        "Attempted to run callable after handle was released: ", handle);
  }

  // NOTE(mrry): Debug options are not currently supported in the
  // callable interface.
  DebugOptions debug_options;
  RunStateArgs run_state_args(debug_options);

  // Configure a call frame for the step, which we use to feed and
  // fetch values to and from the executors.
  if (feed_tensors.size() != executors_and_keys->input_types.size()) {
    return errors::InvalidArgument(
        "Expected ", executors_and_keys->input_types.size(),
        " feed tensors, but got ", feed_tensors.size());
  }

  if (fetch_tensors != nullptr) {
    fetch_tensors->resize(executors_and_keys->output_types.size());
  } else if (!executors_and_keys->output_types.empty()) {
    return errors::InvalidArgument(
        "`fetch_tensors` must be provided when the callable has one or more "
        "outputs.");
  }

  // Configure a call frame for the step, which we use to feed and
  // fetch values to and from the executors.
  // A specialized CallFrame implementation that takes advantage of the
  // optimized RunCallable interface.
  RunCallableCallFrame call_frame(this,
                                  executors_and_keys.get(),
                                  &feed_tensors,
                                  fetch_tensors);
  // 1.
  // RunCallableCallFrame 数据结构
  // tensorflow/core/common_runtime/direct_session.cc:2329:
  // class DirectSession::RunCallableCallFrame : public CallFrameInterface
  // - session_: DirectSession* const
  // - executors_and_keys_: ExecutorsAndKeys* const
  // - feed_tensors_: const std::vector<Tensor>*
  // - fetch_tensors_: std::vector<Tensor>* const

  // 2.
  // RunCallableCallFrame 构造函数接口说明
  // RunCallableCallFrame(DirectSession* session,
  //                      ExecutorsAndKeys* executors_and_keys,
  //                      const std::vector<Tensor>* feed_tensors,
  //                      std::vector<Tensor>* fetch_tensors)

  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(step_id, run_state_args.handle);
  }

  TF_RETURN_IF_ERROR(
      RunInternal(
        step_id,
        executors_and_keys->callable_options.run_options(),
        &call_frame,
        executors_and_keys.get(),
        run_metadata));

  return Status::OK();
}

::tensorflow::Status DirectSession::ReleaseCallable(CallableHandle handle) {
  mutex_lock l(callables_lock_);
  if (handle >= next_callable_handle_) {
    return errors::InvalidArgument("No such callable handle: ", handle);
  }
  callables_.erase(handle);
  return Status::OK();
}

DirectSession::Callable::~Callable() {
  // We must delete the fields in this order, because the destructor
  // of `executors_and_keys` will call into an object owned by
  // `function_info` (in particular, when deleting a kernel, it relies
  // on the `FunctionLibraryRuntime` to know if the kernel is stateful
  // or not).
  executors_and_keys.reset();
  function_info.reset();
}

}  // namespace tensorflow
