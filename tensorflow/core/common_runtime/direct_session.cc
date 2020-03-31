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
#include <iostream>
#include <thread>
#include <unordered_map>
#include <map>
#include <regex>

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
#include "tensorflow/core/common_runtime/gpu/gpu_util.h" // wxf
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb_text.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/resource_mgr.h" // by wxf
#include "tensorflow/core/framework/resource_var.h" // by wxf
#include "tensorflow/core/framework/run_handler.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/kernels/variable_ops.h" // wxf
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/gtl/map_util.h" // wxf
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/device_tracer.h"
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
  static thread::ThreadPool* const thread_pool =
      NewThreadPoolFromSessionOptions(options);
  return thread_pool;
}

// wxf
// Construct a low priority threadpool
thread::LowPriorityThreadPool* GlobalLowPriorityThreadPool(const SessionOptions& options) {
  static thread::LowPriorityThreadPool* const thread_pool =
      NewLowPriorityThreadPoolFromSessionOptions(options);
  return thread_pool;
}
//~wxf

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

class DirectSessionFactory : public SessionFactory {
 public:
  DirectSessionFactory() {}

  bool AcceptsOptions(const SessionOptions& options) override {
    return options.target.empty();
  }

  Status NewSession(const SessionOptions& options,
                    Session** out_session) override {
    // Must do this before the CPU allocator is created.
    if (options.config.graph_options().build_cost_model() > 0) {
      EnableCPUAllocatorFullStats(true);
    }
    std::vector<std::unique_ptr<Device>> devices;
    TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices));

    DirectSession* session =
        new DirectSession(options, new DeviceMgr(std::move(devices)), this);

    {
      mutex_lock l(sessions_lock_);

      // wxf
      // Set the DirectSession member variable direct_session_priority_ based on
      // the user defined priority level number by tf.set_execution_priority 
      // API from the python side.
      session->SetDirectSessionPriority(
    		  tensorflow::tid_execution_priority_map_[std::this_thread::get_id()]);
      // 1.
      // tid_execution_priority_map_ is a map between:
      //   python tid : priority level number
      //~wxf

      sessions_.push_back(session);
    }

    // wxf
    // Add all DirectSession instances to the DirectSessionsManager 
    // direct_sessions_manager_.
    // Fill the mapping for DirectSessionsManager member
    // direct_session_priority_map_: {direct_session_address:priority_level_num}
    // based on the user pre-set priority mapping: {tid : priority_level_num} 
    direct_sessions_manager_->AddDirectSessionAndPriority(&session);
    //~wxf

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
    pool->Schedule(std::move(c));  
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
  const int thread_pool_size =
      options_.config.session_inter_op_thread_pool_size();
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
    // default threadpool or high priority threadpool
    thread_pools_.emplace_back(GlobalThreadPool(options), false /* owned */);
    // wxf:
    // Construct a low priority threadpool
    low_priority_thread_pool_ = GlobalLowPriorityThreadPool(options);
    //~wxf
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

  // wxf: Collect low_priority_device_set_ for class Placer
  auto ComputationLambda = [](const float &lhs, const float &rhs) {
    return lhs > rhs; 
    // 7.5 > 6.1 order
  };
  std::function<bool(const float& lhs, const float& rhs)> CompuLambda;

  computation_capability_device_map_ = 
    new std::map<float, std::vector<Device*>, decltype(CompuLambda)>(ComputationLambda);

  // wxf: sort the computation capability : Device* map by descending order
  // +-----------------------------------------------------------------------------------+
  // |                       computation_capability_device_map_                           |
  // |-----------------------------------------------------------------------------------|
  // |                                                                                   |
  // | +--------------------------+     +-----------+-----------+-----------+-----------+|
  // | |computation capability|7.5|---->| Device*   | Device*   | Device*   | Device*   ||
  // | +--------------------------+     +-----------+-----------+-----------+-----------+|
  // |                                                                                   |
  // | +--------------------------+     +-----------+-----------+-----------+-----------+|
  // | |computation capability|6.1|---->| Device*   | Device*   | Device*   | Device*   ||
  // | +--------------------------+     +-----------+-----------+-----------+-----------+|
  // |                                                                                   |
  // |      ...                           ...                                            |
  // |                                                                                   |
  // | +--------------------------+     +-----------+-----------+-----------+-----------+|
  // | |computation capability|1.0|---->| Device*   | Device*   | Device*   | Device*   ||
  // | +--------------------------+     +-----------+-----------+-----------+-----------+|
  // +-----------------------------------------------------------------------------------+

  for (auto d : device_mgr_->ListDevices()) {
    devices_.push_back(d);
    device_set_.AddDevice(d);
    d->op_segment()->AddHold(session_handle_);

    // Deprecated:
    //std::map<float, std::vector<Device*>, decltype(ComputationLambda)> 
    //  computation_capability_device_map(ComputationLambda);

    // The first device added is special: it is the 'client device' (a
    // CPU device) from which we feed and fetch Tensors.
    if (devices_added == 0) {
      device_set_.set_client_device(d);

      // wxf, used for create low_priority_execution_state_
      // Deprecated. I want to use less powerful GPUs and CPUs
      //low_priority_device_set_.AddDevice(d);
      // wxf, don't forget to set client device, which is used in 
      // GraphExecutionState::PruneGraph
      // ==> LookupDevice
      low_priority_device_set_.set_client_device(d);
      // wxf, CPU is hardcoded to 1.0 as its computation capacity
      std::vector<Device*> ds{d};
      computation_capability_device_map_->emplace(1.0, ds);
    }

    // wxf: Skip the CPU since we have already added it
    if (devices_added != 0) {
      // wxf: after add the CPU, then adding GPUs
      string device_name = d->DebugString();
      std::smatch m;
      float computation_capacity = 0.0; // major_minor
      std::regex e ("compute capability: (\\d\\.\\d)");
      while (std::regex_search (device_name, m, e)) {
        computation_capacity = std::stof(m[1]);
        // iterate the following substring
        device_name = m.suffix().str(); 
      }

      auto search = computation_capability_device_map_->find(computation_capacity);
      if (search == computation_capability_device_map_->end()) {
        // construct a new row in this map
        std::vector<Device*> temp_ds{d};
        computation_capability_device_map_->emplace(computation_capacity, temp_ds);
      } else {
        // add the Device* to the existing one
        (*computation_capability_device_map_)[computation_capacity].push_back(d);
      }
    }

    ++devices_added;

  } // End of for each device

  // wxf:
  // Deprecated! 
  // I choose to keep it and skip the 1st one when adding it to the
  // low_priority_device_set_

  // The 1st device is always the most powerful one. The rest are LPU devices.
  // After getting the sorted computation_capability : Device*
  // we skip the 1st one most powerful GPU and add the rest to the low_priority_device_set_ 
  int skip = 0;
  // add the left Device* to the low_priority_device_set_ 
  for (auto item: *computation_capability_device_map_){
    skip++;
    if (skip == 1) {
      continue;
    }
    for (auto e: item.second){
      // Add the rest  devices to the low_priority_device_set_
      low_priority_device_set_.AddDevice(e);
    }
  }
}

DirectSession::~DirectSession() {
  if (!closed_) Close().IgnoreError();
  for (auto& it : partial_runs_) {
    it.second.reset(nullptr);
  }
  for (auto& it : executors_) {
    it.second.reset();
  }
  callables_.clear();
  // wxf
  low_priority_callables_.clear();

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
  // wxf
  low_priority_execution_state_.reset(nullptr);

  flib_def_.reset(nullptr);

  // wxf
  // Update statistics inside the DirectSessionsManager instace
  direct_sessions_manager_->DeleteDirectSession(this);

  // Once the last high priority DirectSession instance deconstrcut
  // while there is still low priority DirectSession work to do,
  // then wake up the low priority pool threads.
 // if(this->GetDirectSessionPriority() == 2 &&
 //    !direct_sessions_manager_->high_priority_direct_session_count_.load(
 //      std::memory_order_relaxed) && 
 //    direct_sessions_manager_->low_priority_direct_session_count_.load(
 //      std::memory_order_relaxed)){
 //   low_priority_thread_pool_->WakeUpAll(); 
 // }
}

// wxf
// Set the priority of this DirectSession instance.
// Each instance will set the priority right after the DirectSession instance 
// is constructed, no need to set a lock.
void DirectSession::SetDirectSessionPriority(int priority) {
  direct_session_priority_ = priority;
}

// Get the priority of this DirectSession instance.
int DirectSession::GetDirectSessionPriority() {
  return direct_session_priority_;
}

void DirectSession::TransferGPU2CPUStatefulVars() {
  const uint64 start_time_usecs = Env::Default()->NowMicros();

  // GPU resource mgr (src)
  Device* gpu_device = devices_[3];
  ResourceMgr* gpu_resource_mgr = gpu_device->resource_manager();
  // CPU resource mgr (dst)
  Device* cpu_device = devices_[0];
  ResourceMgr* cpu_resource_mgr = cpu_device->resource_manager();

  // for debug ResourceBase real type
  //gpu_resource_mgr->DebugString();
  //cpu_resource_mgr->DebugString();

  std::vector<string> gpu_containers;
  // iterate all GPU stateful variables per container per item
  for (const auto& p: gpu_resource_mgr->Containers()){
    const string& container = p.first;
    gpu_containers.push_back(container);
    for (const auto& q: *p.second){
      const std::pair<uint64, string>& key = q.first;
      const uint64 hash_code = key.first;
      const string& resource_name = key.second;

      // Record what variables are transferred to CPU and it will be used 
      // when it is transferred back
      string resource_var_name = resource_name;
      transferred_resource_names_and_device_type_.insert({resource_var_name, "GPU"});
      
      ResourceBase* resource;
      gpu_resource_mgr->LookupResourceBase(container, hash_code, resource_name, &resource);

      // Right now, I only consider LegacyVar and Var type
      if (dynamic_cast<LegacyVar*>(resource) == nullptr) {
        // not LegacyVar, so is Var
        Var* variable = nullptr;
        variable = TypeCastFunctor<Var, false>::Cast(resource);
        
        // gpu_resource_mgr->LookupTransferVar<Var>(
        //     container, hash_code, resource_name, &variable);
  
        // We're acquiring a reference to the underlying buffer while
        // holding a shared lock to guarantee ordering of reads and
        // writes.
        tf_shared_lock ml(*variable->mu());
        const Tensor* gpu_tensor = variable->tensor();
        // gpu_tensor->DeviceSafeDebugString();
  
        // Transfer from GPU to CPU
        auto creator = [this, gpu_device, cpu_device, gpu_tensor](Var** cpu_variable){
          // 1.
          // creator lambda signature Explanation:
          // std::function<Status(T**)>

          // construct Var cpu_variable
          *cpu_variable = new Var(gpu_tensor->dtype());
          (*cpu_variable)->tensor()->set_shape(gpu_tensor->shape());

          // 1. Construct Tensor of CPU;
          // 2. Init CPU tenors value by tranferred GPU tensor.
          // use CPU allocator to construct the buf_ in the tensor
          AllocatorAttributes attr;
          attr.set_on_host(true);
          attr.set_gpu_compatible(true);
          Allocator* cpu_allocator = cpu_device->GetAllocator(attr);
          Tensor copy(cpu_allocator, gpu_tensor->dtype(), gpu_tensor->shape());
          *((*cpu_variable)->tensor()) = copy;

          // transfer GPU tensor to CPU
          const DeviceContext* device_context = gpu_device->tensorflow_gpu_device_info()->default_context;
          // 1.
          // default_context explanation:  
          // gpu_device_info_->default_context = device_contexts_[0] assigned in common_runtime/gpu/gpu_device.cc

          Tensor* cpu_tensor = (*cpu_variable)->tensor();

          // -----------------------------------------------------------------------
          // logging the tensor allocation memory size for quantitive analysis
          if (cpu_tensor->buf_ != nullptr && cpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferGPU2CPUAllocationCreate", LogMemory::UNKNOWN_STEP_ID, *cpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }
          // -----------------------------------------------------------------------

          GPUUtil::CopyGPUTensorToCPU(
              gpu_device, device_context, gpu_tensor, cpu_tensor, 
              [](const Status& s){
                if (!s.ok()){
                  VLOG(0) << "GPU->CPU MemCpy Fail!";
                }
              });

          return Status::OK();
        };

        Var* cpu_variable = nullptr;

        bool is_init = false; 
        // 1.
        // is_init explanation:
        // is_init means: is cpu_variable created by creator(true) or
        //   it has already existed(false).
        // Once get from LookupOrCreateVar, the value is determined.
        // I just arbitrarily set false to is_init at first.
        
        // Lookup by resource name, if null then create via memcpy
        cpu_resource_mgr->LookupOrCreateVar<Var>(
            container, resource_name, &is_init, &cpu_variable, creator);

        // if is not by init, then we need to update those stateful vars
        if (!is_init){
          // Update by transferring GPU tensor to CPU
          // Preparation
          const DeviceContext* device_context = gpu_device->tensorflow_gpu_device_info()->default_context;
          Tensor* cpu_tensor = cpu_variable->tensor();
          // -----------------------------------------------------------------------
          // logging the tensor allocation memory size for quantitive analysis
          if (cpu_tensor->buf_ != nullptr && cpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferGPU2CPUAllocationUpdate", LogMemory::UNKNOWN_STEP_ID, *cpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }
          // -----------------------------------------------------------------------

          GPUUtil::CopyGPUTensorToCPU(
              gpu_device, device_context, gpu_tensor, cpu_tensor,
              [](const Status& s){
                if (!s.ok()){
                  VLOG(0) << "GPU->CPU MemCpy Fail!";
                }
              });
        } // End of updating CPU stateful variables
  
      } else if (dynamic_cast<Var*>(resource) == nullptr) {
        // not Var, so is LegacyVar
        LegacyVar* variable = nullptr;
        variable = TypeCastFunctor<LegacyVar, false>::Cast(resource);

        // gpu_resource_mgr->LookupTransferVar<Var>(
        //     container, hash_code, resource_name, &variable);
  
        // We're acquiring a reference to the underlying buffer while
        // holding a shared lock to guarantee ordering of reads and
        // writes.
        tf_shared_lock ml(*variable->mu());
        const Tensor* gpu_tensor = variable->tensor();
        // gpu_tensor->DeviceSafeDebugString();

        // Transfer from GPU to CPU
        auto creator = [this, gpu_device, cpu_device, gpu_tensor](LegacyVar** cpu_variable){
          // 1.
          // creator lambda interface Explanation:
          // std::function<Status(T**)>

          // construct Var cpu_variable
          *cpu_variable = new LegacyVar(gpu_tensor->dtype());
          (*cpu_variable)->tensor()->set_shape(gpu_tensor->shape());

          // use CPU allocator to construct the buf_ in the tensor
          AllocatorAttributes attr;
          attr.set_on_host(true);
          attr.set_gpu_compatible(true);
          Allocator* cpu_allocator = cpu_device->GetAllocator(attr);
          Tensor copy(cpu_allocator, gpu_tensor->dtype(), gpu_tensor->shape());
          *((*cpu_variable)->tensor()) = copy;

          // transfer GPU tensor to CPU
          const DeviceContext* device_context = gpu_device->tensorflow_gpu_device_info()->default_context;
          // 1.
          // default_context explanation:  
          // gpu_device_info_->default_context = device_contexts_[0] assigned in common_runtime/gpu/gpu_device.cc

          Tensor* cpu_tensor = (*cpu_variable)->tensor();

          // -----------------------------------------------------------------------
          // logging the tensor allocation memory size for quantitive analysis
          if (cpu_tensor->buf_ != nullptr && cpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferGPU2CPUAllocationCreate", LogMemory::UNKNOWN_STEP_ID, *cpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }
          // -----------------------------------------------------------------------

          GPUUtil::CopyGPUTensorToCPU(
              gpu_device, device_context, gpu_tensor, cpu_tensor, 
              [](const Status& s){
                if (!s.ok()){
                  VLOG(0) << "GPU->CPU MemCpy Fail!";
                }
              });

          return Status::OK();
        };

        LegacyVar* cpu_variable = nullptr;

        bool is_init = false; 
        // 1.
        // is_init explanation:
        // is_init means: is cpu_variable created by creator(true) or
        //   it has already existed(false).
        // Once get from LookupOrCreateVar, the value is determined.
        // I just arbitrarily set false to is_init at first.

        // Lookup by resource name, if null then create via memcpy
        cpu_resource_mgr->LookupOrCreateVar<LegacyVar>(
            container, resource_name, &is_init, &cpu_variable, creator);

        // if is not by init, then we need to update those stateful vars
        if (!is_init){
          // Update by transferring GPU tensor to CPU
          // Preparation
          const DeviceContext* device_context = gpu_device->tensorflow_gpu_device_info()->default_context;
          Tensor* cpu_tensor = cpu_variable->tensor();
          // -----------------------------------------------------------------------
          // logging the tensor allocation memory size for quantitive analysis
          if (cpu_tensor->buf_ != nullptr && cpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferGPU2CPUAllocationUpdate", LogMemory::UNKNOWN_STEP_ID, *cpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }
          // -----------------------------------------------------------------------

          GPUUtil::CopyGPUTensorToCPU(
              gpu_device, device_context, gpu_tensor, cpu_tensor,
              [](const Status& s){
                if (!s.ok()){
                  VLOG(0) << "GPU->CPU MemCpy Fail!";
                }
              });
        } // End of updating CPU stateful variables
      } // if branch end of LegacyVar
    } // End of for loop of each item 
  }// End of for loop of each container

  // Deallocate GPU memory since low priority task executes on CPU, no need to have a copy on GPU.
  //gpu_resource_mgr->Cleanup("localhost"); // test: pass! 2019-10-11 17:51:33
  for (auto& gpu_container: gpu_containers) {
    gpu_resource_mgr->Cleanup(gpu_container);
  }

  uint64 gpu2cpu_eclipsed_time = Env::Default()->NowMicros() - start_time_usecs;
  VLOG(0) << "GPU to CPU transfer stateful vars time eclipsed(micro sec): " << gpu2cpu_eclipsed_time;
}

void DirectSession::TransferCPU2GPUStatefulVars() {
  const uint64 start_time_usecs = Env::Default()->NowMicros();

  // Get CPU Resource Mgr (src)
  Device* cpu_device = devices_[0];
  ResourceMgr* cpu_resource_mgr = cpu_device->resource_manager();
  // Get GPU Resource Mgr (dst)
  Device* gpu_device = devices_[3];
  ResourceMgr* gpu_resource_mgr = gpu_device->resource_manager();

  // Iterate all CPU (src) stateful variables per container per item except
  // "Iterator resource", e.g., "_1_IteratorV2_1"
  for (const auto& p: cpu_resource_mgr->Containers()){
    const string& container = p.first;
    for (const auto& q: *p.second){

      // Skip moving IteratorResource, LookupInterface
    	//if (q.second->DebugString()=="Iterator resource" || 
      //    q.second->DebugString().find("A lookup table") != string::npos) {
      //  continue;
      //}

      const std::pair<uint64, string>& key = q.first;
      const uint64 hash_code = key.first;
      const string& resource_name = key.second;

      // Skip those vars that are not on GPU originally
      string resource_var_name(resource_name);
      auto iter = transferred_resource_names_and_device_type_.find(resource_var_name); 
      if (iter == transferred_resource_names_and_device_type_.end()) {
        continue;
      }

      ResourceBase* resource;
      cpu_resource_mgr->LookupResourceBase(container, hash_code, resource_name, &resource);

      // LegacyVar or Var type?
      if (dynamic_cast<LegacyVar*>(resource) == nullptr) {
        // not LegacyVar, so it is Var
        Var* variable = nullptr;
        // 1.
        // variable Explanation:
        // CPU doesn't clear Var resource, so I don't call variable->Unref();
        // It is good to try ResourceBase* variable = nullptr here and 
        // LookupTransferVar<Var>(...) below.
        //cpu_resource_mgr->LookupTransferVar(
        //    container, hash_code, resource_name, &variable);
        variable = TypeCastFunctor<Var, false>::Cast(resource);

        // We're acquiring a reference to the underlying buffer while
        // holding a shared lock to guarantee ordering of reads and
        // writes.
        tf_shared_lock ml(*variable->mu());
        const Tensor* cpu_tensor = variable->tensor();

        // creator lambda: std::function<Status(T**)>
        auto creator = [this, gpu_device, cpu_device, cpu_tensor](Var** gpu_variable){
          // construct Var gpu_variable according to cpu tensor attribute
          *gpu_variable = new Var(cpu_tensor->dtype());
          (*gpu_variable)->tensor()->set_shape(cpu_tensor->shape());
  
          // use GPU allocator to construct buf_ in the GPU tensor
          AllocatorAttributes attr;
          //attr.set_on_host(true);
          attr.set_gpu_compatible(true);
          Allocator* gpu_allocator = gpu_device->GetAllocator(attr);
          Tensor copy(gpu_allocator, cpu_tensor->dtype(), cpu_tensor->shape());
          // GPU Tensor is not filled with values, which should be transferred from CPU
          *((*gpu_variable)->tensor()) = copy;
  
          // transfer CPU tensor to GPU
          const DeviceContext* device_context = gpu_device->tensorflow_gpu_device_info()->default_context;
          Tensor* gpu_tensor = (*gpu_variable)->tensor();
          // -----------------------------------------------------------------------
          // logging the tensor allocation memory size for quantitive analysis
          if (gpu_tensor->buf_ != nullptr && gpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferCPU2GPUAllocationCreate", LogMemory::UNKNOWN_STEP_ID, *gpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }
          // -----------------------------------------------------------------------
  
          GPUUtil::CopyCPUTensorToGPU(
              cpu_tensor, device_context, gpu_device, gpu_tensor,
              [](const Status& s){
                if (!s.ok()){
                  VLOG(0) << "CPU->GPU MemCpy Fail!";
                }
              });
  
          return Status::OK();
        };
  
        // Lookup by resource name, if null then create via memcpy
        Var* gpu_variable = nullptr;

        bool is_init = false;
        gpu_resource_mgr->LookupOrCreateVar<Var>(
            container, resource_name, &is_init, &gpu_variable, creator);

        if (!is_init){
          const DeviceContext* device_context = gpu_device->tensorflow_gpu_device_info()->default_context;

          Tensor* gpu_tensor = gpu_variable->tensor();
          // -----------------------------------------------------------------------
          // logging the tensor allocation memory size for quantitive analysis
          if (gpu_tensor->buf_ != nullptr && gpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferCPU2GPUAllocationUpdate", LogMemory::UNKNOWN_STEP_ID, *gpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }
          // -----------------------------------------------------------------------

          GPUUtil::CopyCPUTensorToGPU(
              cpu_tensor, device_context, gpu_device, gpu_tensor,
              [](const Status& s){
                if (!s.ok()){
                  VLOG(0) << "CPU->GPU MemCpy Fail!";
                }
              });
        } // End of updating GPU tensors
            
      } else if (dynamic_cast<Var*>(resource) == nullptr) {
        // not Var, so is LegacyVar
        LegacyVar* variable = nullptr;
        variable = TypeCastFunctor<LegacyVar, false>::Cast(resource);

        // We're acquiring a reference to the underlying buffer while
        // holding a shared lock to guarantee ordering of reads and
        // writes.
        tf_shared_lock ml(*variable->mu());
        const Tensor* cpu_tensor = variable->tensor();

        // creator lambda: std::function<Status(T**)>
        auto creator = [this, gpu_device, cpu_device, cpu_tensor](LegacyVar** gpu_variable){
          // construct Var gpu_variable according to cpu tensor attribute
          *gpu_variable = new LegacyVar(cpu_tensor->dtype());
          (*gpu_variable)->tensor()->set_shape(cpu_tensor->shape());
  
          // use GPU allocator to construct buf_ in the GPU tensor
          AllocatorAttributes attr;
          //attr.set_on_host(true);
          attr.set_gpu_compatible(true);
          Allocator* gpu_allocator = gpu_device->GetAllocator(attr);
          Tensor copy(gpu_allocator, cpu_tensor->dtype(), cpu_tensor->shape());
          // GPU Tensor is not filled with values, which should be transferred from CPU
          *((*gpu_variable)->tensor()) = copy;
  
          // transfer CPU tensor to GPU
          const DeviceContext* device_context = gpu_device->tensorflow_gpu_device_info()->default_context;
          Tensor* gpu_tensor = (*gpu_variable)->tensor();
          // -----------------------------------------------------------------------
          // logging the tensor allocation memory size for quantitive analysis
          if (gpu_tensor->buf_ != nullptr && gpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferCPU2GPUAllocationCreate", LogMemory::UNKNOWN_STEP_ID, *gpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }
          // -----------------------------------------------------------------------
  
          GPUUtil::CopyCPUTensorToGPU(
              cpu_tensor, device_context, gpu_device, gpu_tensor,
              [](const Status& s){
                if (!s.ok()){
                  VLOG(0) << "CPU->GPU MemCpy Fail!";
                }
              });
  
          return Status::OK();
        };

        // Lookup by resource name, if null then create via memcpy
        LegacyVar* gpu_variable = nullptr;

        bool is_init = false;
        gpu_resource_mgr->LookupOrCreateVar<LegacyVar>(
            container, resource_name, &is_init, &gpu_variable, creator);

        if (!is_init){
          const DeviceContext* device_context = gpu_device->tensorflow_gpu_device_info()->default_context;

          Tensor* gpu_tensor = gpu_variable->tensor();
          // -----------------------------------------------------------------------
          // logging the tensor allocation memory size for quantitive analysis
          if (gpu_tensor->buf_ != nullptr && gpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferCPU2GPUAllocationUpdate", LogMemory::UNKNOWN_STEP_ID, *gpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }
          // -----------------------------------------------------------------------

          GPUUtil::CopyCPUTensorToGPU(
              cpu_tensor, device_context, gpu_device, gpu_tensor,
              [](const Status& s){
                if (!s.ok()){
                  VLOG(0) << "CPU->GPU MemCpy Fail!";
                }
              });
        } // End of updating GPU tensors
      } // if branch End
    } // End of for of each item
  } // End of for of container

  uint64 cpu2gpu_eclipsed_time = Env::Default()->NowMicros() - start_time_usecs;
  VLOG(0) << "CPU to GPU transfer stateful vars time eclipsed(micro sec): " << cpu2gpu_eclipsed_time;
} 

void DirectSession::TransferHPU2LPUStatefulVars() {
  const uint64 start_time_usecs = Env::Default()->NowMicros();
  
  // HPU resource mgr (src)
  Device* hpu_device = nullptr;
  // LPU resource mgr (dst)
  Device* lpu_device = nullptr;
  
  // Iterate the first 2 : NO.1 HPU, NO.2 LPU
  int i = 0;
  for (auto const& item : *computation_capability_device_map_) {
    // The default device is the first one of the list of the same computation
    // capability.
    if (i == 0) {
      // +--------------------------+     +-----------+-----------+-----------+
      // |computation capability|7.5|---->| Device*<==| Device*   | Device*   |
      // +--------------------------+     +-----------+-----------+-----------+
      hpu_device = item.second[0];
    }

    if (i == 1) {
      // +--------------------------+     +-----------+-----------+-----------+
      // |computation capability|6.1|---->| Device*<==| Device*   | Device*   |
      // +--------------------------+     +-----------+-----------+-----------+
      lpu_device = item.second[0];
    }

    // Iteration stop condition:
    i++;
    if (i == 2) break;
  }

  ResourceMgr* hpu_resource_mgr = hpu_device->resource_manager();
  ResourceMgr* lpu_resource_mgr = lpu_device->resource_manager();
  // for debug ResourceBase real type
  //string debug_hpu_resource = hpu_resource_mgr->DebugString();
  //string debug_lpu_resource = lpu_resource_mgr->DebugString();

  // Used to take down all hpu containers in which resource tensors to be 
  // deallocated in the end.
  std::vector<string> hpu_containers;

  // Iterate all HPU stateful variables per container per item to construct LPU's
  for (const auto& p : hpu_resource_mgr->Containers()) {
    const string& container = p.first;
    hpu_containers.push_back(container);
    for (const auto& q: *p.second) {
      const std::pair<uint64, string>& key = q.first;
      const uint64 hash_code = key.first;
      const string& resource_name = key.second;

      // No need to record the transferred stateful variables for dev_to_dev case.
      // Record what variables are transferred to LPU and it will be used
      // when it is transferred back.
      //string resource_var_name = resource_name;
      //transferred_resource_names_and_device_type_.insert(
      //    {resource_var_name, "HPU"});

      ResourceBase* resource;
      hpu_resource_mgr->LookupResourceBase(container, hash_code, resource_name,
          &resource);

      // Check Var or LegacyVar type
      if (dynamic_cast<LegacyVar*>(resource) == nullptr) {
        // not LegacyVar, so is Var 
        Var* variable = nullptr;
        variable = TypeCastFunctor<Var, false>::Cast(resource);
        // We're acquiring a reference to the underlying buffer while 
        // holding a shared lock to guarantee ordering of reads and 
        // writes.
        tf_shared_lock ml(*variable->mu());
        const Tensor* hpu_tensor = variable->tensor();

        // Transfer from GPU to CPU
        // lambda function of creating variable if the LPU does not construct it.
        auto creator = [this, hpu_device, lpu_device, hpu_tensor](Var** lpu_variable){
          // 1.
          // creator type (i.e. auto real type) lambda signature Explanation:
          // std::function<Status(T**)>

          // construct Var lpu_variable
          *lpu_variable = new Var(hpu_tensor->dtype());
          (*lpu_variable)->tensor()->set_shape(hpu_tensor->shape());

          // 1. Construct LPU Tensor; 2. Initialize Tenor values from HPU
          // use LPU allocator(GPU) to construct the buf_ in the tensor.
          AllocatorAttributes lpu_alloc_attr;
          lpu_alloc_attr.set_on_host(false);
          lpu_alloc_attr.set_gpu_compatible(true);
          Allocator* lpu_allocator = lpu_device->GetAllocator(lpu_alloc_attr); 
          Tensor copy(lpu_allocator, hpu_tensor->dtype(), hpu_tensor->shape());
          *((*lpu_variable)->tensor()) = copy;

          // construct a hpu_attr used in GPUUtil::DeviceToDeviceCopy.
          AllocatorAttributes hpu_alloc_attr;
          hpu_alloc_attr.set_on_host(false);
          hpu_alloc_attr.set_gpu_compatible(true);
          
          // transfer HPU(src) tensor to LPU(dst)
          DeviceContext* hpu_device_context = hpu_device->tensorflow_gpu_device_info()->default_context;
          DeviceContext* lpu_device_context = lpu_device->tensorflow_gpu_device_info()->default_context;
          // 1.
          // default_context explanation:  
          // gpu_device_info_->default_context = device_contexts_[0] assigned in common_runtime/gpu/gpu_device.cc
          
          Tensor* lpu_tensor = (*lpu_variable)->tensor();

          // logging the tensor allocation memory size for quantitive analysis
          if (lpu_tensor->buf_ != nullptr && lpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferHPU2LPUAllocationCreate", LogMemory::UNKNOWN_STEP_ID, *lpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }

          GPUUtil::DeviceToDeviceCopy(
            hpu_device_context, /* ==> */ lpu_device_context,
            hpu_device, /* ==> */ lpu_device,
            hpu_alloc_attr, lpu_alloc_attr,
            hpu_tensor, /* ==> */ lpu_tensor,
            0, 
            [](const Status& s){
              if (!s.ok()){
                VLOG(0) << "HPU->LPU MemCpy Fail!";
              }
            }
          );
          // 1.
          // dev_to_dev_stream_index Explanation:
          // Only 1 stream group is created so the index is 0.
          // Check the log:
          // https://gist.github.com/shizukanaskytree/1c60070597bdf60fdfd59e8177e9c127
          
          // 2.
          // StatusCallback Explanation:
          // signature: std::function<void(const Status&)> StatusCallback
          // defined in tensorflow/core/common_runtime/rendezvous_mgr.h:79

          return Status::OK();
        };

        Var* lpu_variable = nullptr;
        
        bool is_init = false;
        // 1.
        // is_init Explanation:
        // 1. If variables are constructed and initialized in the LookupOrCreateVar, 
        // is_init == true  
        // 2. If not, is_init == false, they've already existed. Only need update. 

        // Lookup by resource name, if null then create via memcpy
        lpu_resource_mgr->LookupOrCreateVar<Var>(
            container, resource_name, &is_init, &lpu_variable, creator);

        // If not by creator:
        // If not by constructed and initialized, then we need to update those 
        // stateful vars.
        if (!is_init){
          // Update vars by transferring tensors from HPU to LPU
          DeviceContext* hpu_device_context = hpu_device->tensorflow_gpu_device_info()->default_context;
          DeviceContext* lpu_device_context = lpu_device->tensorflow_gpu_device_info()->default_context;

          AllocatorAttributes lpu_alloc_attr;
          lpu_alloc_attr.set_on_host(false);
          lpu_alloc_attr.set_gpu_compatible(true);

          AllocatorAttributes hpu_alloc_attr;
          hpu_alloc_attr.set_on_host(false);
          hpu_alloc_attr.set_gpu_compatible(true);

          // since we find it from LookupOrCreateVar, lpu_variable
          Tensor* lpu_tensor = lpu_variable->tensor();

          // logging the tensor allocation memory size for quantitive analysis
          if (lpu_tensor->buf_ != nullptr && lpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferHPU2LPUAllocationUpdate", LogMemory::UNKNOWN_STEP_ID, *lpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }

          GPUUtil::DeviceToDeviceCopy(
            hpu_device_context, /* ==> */ lpu_device_context,
            hpu_device, /* ==> */ lpu_device,
            hpu_alloc_attr, lpu_alloc_attr,
            hpu_tensor, /* ==> */ lpu_tensor,
            0, 
            [](const Status& s){
              if (!s.ok()){
                VLOG(0) << "HPU->LPU MemCpy Fail!";
              }
            }
          ); // End of GPUUtil::DeviceToDeviceCopy
        } // End of updating LPU stateful variables
      
      // if branch Var Branch End.
      } else if (dynamic_cast<Var*>(resource) == nullptr) {
        // not Var, so is LegacyVar
        LegacyVar* variable = nullptr;
        variable = TypeCastFunctor<LegacyVar, false>::Cast(resource);
        // We're acquiring a reference to the underlying buffer while 
        // holding a shared lock to guarantee ordering of reads and 
        // writes.
        tf_shared_lock ml(*variable->mu());
        const Tensor* hpu_tensor = variable->tensor();

        // Transfer from GPU to CPU
        // lambda function of creating variable if the LPU does not construct it.
        auto creator = [this, hpu_device, lpu_device, hpu_tensor](LegacyVar** lpu_variable){
          // 1.
          // creator type (i.e. auto real type) lambda signature Explanation:
          // std::function<Status(T**)>

          // construct LegacyVar lpu_variable
          *lpu_variable = new LegacyVar(hpu_tensor->dtype());
          (*lpu_variable)->tensor()->set_shape(hpu_tensor->shape());

          // 1. Construct LPU Tensor; 2. Initialize Tenor values from HPU
          // use LPU allocator(GPU) to construct the buf_ in the tensor.
          AllocatorAttributes lpu_alloc_attr;
          lpu_alloc_attr.set_on_host(false);
          lpu_alloc_attr.set_gpu_compatible(true);
          Allocator* lpu_allocator = lpu_device->GetAllocator(lpu_alloc_attr); 
          Tensor copy(lpu_allocator, hpu_tensor->dtype(), hpu_tensor->shape());
          *((*lpu_variable)->tensor()) = copy;

          // construct a hpu_attr used in GPUUtil::DeviceToDeviceCopy.
          AllocatorAttributes hpu_alloc_attr;
          hpu_alloc_attr.set_on_host(false);
          hpu_alloc_attr.set_gpu_compatible(true);
          
          // transfer HPU(src) tensor to LPU(dst)
          DeviceContext* hpu_device_context = hpu_device->tensorflow_gpu_device_info()->default_context;
          DeviceContext* lpu_device_context = lpu_device->tensorflow_gpu_device_info()->default_context;
          // 1.
          // default_context explanation:  
          // gpu_device_info_->default_context = device_contexts_[0] assigned in common_runtime/gpu/gpu_device.cc
          
          Tensor* lpu_tensor = (*lpu_variable)->tensor();

          // logging the tensor allocation memory size for quantitive analysis
          if (lpu_tensor->buf_ != nullptr && lpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferHPU2LPUAllocationCreate", LogMemory::UNKNOWN_STEP_ID, *lpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }

          GPUUtil::DeviceToDeviceCopy(
            hpu_device_context, /* ==> */ lpu_device_context,
            hpu_device, /* ==> */ lpu_device,
            hpu_alloc_attr, lpu_alloc_attr,
            hpu_tensor, /* ==> */ lpu_tensor,
            0, 
            [](const Status& s){
              if (!s.ok()){
                VLOG(0) << "HPU->LPU MemCpy Fail!";
              }
            }
          );
          // 1.
          // dev_to_dev_stream_index Explanation:
          // Only 1 stream group is created so the index is 0.
          // Check the log:
          // https://gist.github.com/shizukanaskytree/1c60070597bdf60fdfd59e8177e9c127
          
          // 2.
          // StatusCallback Explanation:
          // signature: std::function<void(const Status&)> StatusCallback
          // defined in tensorflow/core/common_runtime/rendezvous_mgr.h:79

          return Status::OK();
        };

        LegacyVar* lpu_variable = nullptr;
        
        bool is_init = false;
        // 1.
        // is_init Explanation:
        // 1. If variables are constructed and initialized in the LookupOrCreateVar, 
        // is_init == true  
        // 2. If not, is_init == false, they've already existed. Only need update. 

        // Lookup by resource name, if null then create via memcpy
        lpu_resource_mgr->LookupOrCreateVar<LegacyVar>(
            container, resource_name, &is_init, &lpu_variable, creator);

        // If not by creator:
        // If not by constructed and initialized, then we need to update those 
        // stateful vars.
        if (!is_init){
          // Update vars by transferring tensors from HPU to LPU
          DeviceContext* hpu_device_context = hpu_device->tensorflow_gpu_device_info()->default_context;
          DeviceContext* lpu_device_context = lpu_device->tensorflow_gpu_device_info()->default_context;

          AllocatorAttributes lpu_alloc_attr;
          lpu_alloc_attr.set_on_host(false);
          lpu_alloc_attr.set_gpu_compatible(true);

          AllocatorAttributes hpu_alloc_attr;
          hpu_alloc_attr.set_on_host(false);
          hpu_alloc_attr.set_gpu_compatible(true);

          // since we find it from LookupOrCreateVar, lpu_variable
          Tensor* lpu_tensor = lpu_variable->tensor();

          // logging the tensor allocation memory size for quantitive analysis
          if (lpu_tensor->buf_ != nullptr && lpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferHPU2LPUAllocationUpdate", LogMemory::UNKNOWN_STEP_ID, *lpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }

          GPUUtil::DeviceToDeviceCopy(
            hpu_device_context, /* ==> */ lpu_device_context,
            hpu_device, /* ==> */ lpu_device,
            hpu_alloc_attr, lpu_alloc_attr,
            hpu_tensor, /* ==> */ lpu_tensor,
            0, 
            [](const Status& s){
              if (!s.ok()){
                VLOG(0) << "HPU->LPU MemCpy Fail!";
              }
            }
          ); // End of GPUUtil::DeviceToDeviceCopy
        } // End of updating LPU stateful variables
      } // if branch Legacy Branch End
      
    } // End of for loop of each item
  } // End of for loop of each container

  // Deallocate HPU memory after HPU ==> LPU
  for (auto& hpu_container: hpu_containers) {
    hpu_resource_mgr->Cleanup(hpu_container);
  }

  uint64 hpu2lpu_eclipsed_time = Env::Default()->NowMicros() - start_time_usecs;
  VLOG(0) << "HPU to LPU transfer stateful vars time eclipsed(micro sec): " << hpu2lpu_eclipsed_time;
}

void DirectSession::TransferLPU2HPUStatefulVars() {
  const uint64 start_time_usecs = Env::Default()->NowMicros();

  // HPU resource mgr (src)
  Device* hpu_device = nullptr;
  // LPU resource mgr (dst)
  Device* lpu_device = nullptr;

  // Iterate the first 2 : NO.1 HPU, NO.2 LPU
  int i = 0;
  for (auto const& item : *computation_capability_device_map_) {
    // The default device is the first one of the list of the same computation
    // capability.
    if (i == 0) {
      // +--------------------------+     +-----------+-----------+-----------+
      // |computation capability|7.5|---->| Device*<==| Device*   | Device*   |
      // +--------------------------+     +-----------+-----------+-----------+
      hpu_device = item.second[0];
    }

    if (i == 1) {
      // +--------------------------+     +-----------+-----------+-----------+
      // |computation capability|6.1|---->| Device*<==| Device*   | Device*   |
      // +--------------------------+     +-----------+-----------+-----------+
      lpu_device = item.second[0];
    }

    // Iteration stop condition:
    i++;
    if (i == 2) break;
  }

  ResourceMgr* hpu_resource_mgr = hpu_device->resource_manager();
  ResourceMgr* lpu_resource_mgr = lpu_device->resource_manager();

  // for debug ResourceBase real type
  //string debug_hpu_resource = hpu_resource_mgr->DebugString();
  //string debug_lpu_resource = lpu_resource_mgr->DebugString();
  
  // Used to take down all lpu containers in which resource tensors to be 
  // deallocated in the end.
  std::vector<string> lpu_containers;

  // Iterate all LPU stateful variables per container per item to construct HPU's
  for (const auto& p : lpu_resource_mgr->Containers()) {
    const string& container = p.first;
    lpu_containers.push_back(container);
    for (const auto& q: *p.second) {
      const std::pair<uint64, string>& key = q.first;
      const uint64 hash_code = key.first;
      const string& resource_name = key.second;

      // No need to record the transferred stateful variables for dev_to_dev case.
      ResourceBase* resource;
      lpu_resource_mgr->LookupResourceBase(container, hash_code, resource_name,
          &resource);

      // Check Var or LegacyVar type
      if (dynamic_cast<LegacyVar*>(resource) == nullptr) {
        // not LegacyVar, so is Var
        Var* variable = nullptr;
        variable = TypeCastFunctor<Var, false>::Cast(resource);

        // We're acquiring a reference to the underlying buffer while 
        // holding a shared lock to guarantee ordering of reads and 
        // writes.
        tf_shared_lock ml(*variable->mu());
        const Tensor* lpu_tensor = variable->tensor();

        // Transfer from GPU to CPU
        // lambda function of creating variable if the LPU does not construct it.
        auto creator = [this, hpu_device, lpu_device, lpu_tensor](Var** hpu_variable){
          // 1.
          // creator type (i.e. auto real type) lambda signature Explanation:
          // std::function<Status(T**)>

          // construct Var lpu_variable
          *hpu_variable = new Var(lpu_tensor->dtype());
          (*hpu_variable)->tensor()->set_shape(lpu_tensor->shape());

          // 1. Construct HPU Tensor; 2. Initialize Tenor values from LPU
          // use HPU allocator(GPU) to construct the buf_ in the tensor.
          AllocatorAttributes hpu_alloc_attr;
          hpu_alloc_attr.set_on_host(false);
          hpu_alloc_attr.set_gpu_compatible(true);
          Allocator* hpu_allocator = hpu_device->GetAllocator(hpu_alloc_attr); 
          Tensor copy(hpu_allocator, lpu_tensor->dtype(), lpu_tensor->shape());
          *((*hpu_variable)->tensor()) = copy;

          // construct a lpu_attr used in GPUUtil::DeviceToDeviceCopy.
          AllocatorAttributes lpu_alloc_attr;
          lpu_alloc_attr.set_on_host(false);
          lpu_alloc_attr.set_gpu_compatible(true);
          
          // transfer LPU(src) tensor to HPU(dst)
          DeviceContext* hpu_device_context = hpu_device->tensorflow_gpu_device_info()->default_context;
          DeviceContext* lpu_device_context = lpu_device->tensorflow_gpu_device_info()->default_context;
          // 1.
          // default_context explanation:  
          // gpu_device_info_->default_context = device_contexts_[0] assigned in common_runtime/gpu/gpu_device.cc
          
          Tensor* hpu_tensor = (*hpu_variable)->tensor();

          // logging the tensor allocation memory size for quantitive analysis
          if (hpu_tensor->buf_ != nullptr && hpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferLPU2HPUAllocationCreate", LogMemory::UNKNOWN_STEP_ID, *hpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }

          GPUUtil::DeviceToDeviceCopy(
            lpu_device_context, /* ==> */ hpu_device_context,
            lpu_device, /* ==> */ hpu_device,
            lpu_alloc_attr, hpu_alloc_attr,
            lpu_tensor, /* ==> */ hpu_tensor,
            0, 
            [](const Status& s){
              if (!s.ok()){
                VLOG(0) << "HPU->LPU MemCpy Fail!";
              }
            }
          );
          // 1.
          // dev_to_dev_stream_index Explanation:
          // Only 1 stream group is created so the index is 0.
          // Check the log:
          // https://gist.github.com/shizukanaskytree/1c60070597bdf60fdfd59e8177e9c127
          
          // 2.
          // StatusCallback Explanation:
          // signature: std::function<void(const Status&)> StatusCallback
          // defined in tensorflow/core/common_runtime/rendezvous_mgr.h:79

          return Status::OK();
        };

        Var* hpu_variable = nullptr;
        
        bool is_init = false;
        // 1.
        // is_init Explanation:
        // 1. If variables are constructed and initialized in the LookupOrCreateVar, 
        // is_init == true  
        // 2. If not, is_init == false, they've already existed. Only need update. 

        // Lookup by resource name, if null then create via memcpy
        hpu_resource_mgr->LookupOrCreateVar<Var>(
            container, resource_name, &is_init, &hpu_variable, creator);

        // If not by creator:
        // If not by constructed and initialized, then we need to update those 
        // stateful vars.
        if (!is_init){
          // Update vars by transferring tensors from LPU to HPU
          DeviceContext* hpu_device_context = hpu_device->tensorflow_gpu_device_info()->default_context;
          DeviceContext* lpu_device_context = lpu_device->tensorflow_gpu_device_info()->default_context;

          AllocatorAttributes lpu_alloc_attr;
          lpu_alloc_attr.set_on_host(false);
          lpu_alloc_attr.set_gpu_compatible(true);

          AllocatorAttributes hpu_alloc_attr;
          hpu_alloc_attr.set_on_host(false);
          hpu_alloc_attr.set_gpu_compatible(true);

          // since we find it from LookupOrCreateVar, hpu_variable
          Tensor* hpu_tensor = hpu_variable->tensor();

          // logging the tensor allocation memory size for quantitive analysis
          if (hpu_tensor->buf_ != nullptr && hpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferLPU2HPUAllocationUpdate", LogMemory::UNKNOWN_STEP_ID, *hpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }

          GPUUtil::DeviceToDeviceCopy(
            lpu_device_context, /* ==> */ hpu_device_context,
            lpu_device, /* ==> */ hpu_device,
            lpu_alloc_attr, hpu_alloc_attr,
            lpu_tensor, /* ==> */ hpu_tensor,
            0, 
            [](const Status& s){
              if (!s.ok()){
                VLOG(0) << "HPU->LPU MemCpy Fail!";
              }
            }
          ); // End of GPUUtil::DeviceToDeviceCopy
        } // End of updating HPU stateful variables

      // if branch Var Branch End.
      } else if (dynamic_cast<Var*>(resource) == nullptr) {
        // not Var, so is LegacyVar
        LegacyVar* variable = nullptr;
        variable = TypeCastFunctor<LegacyVar, false>::Cast(resource);

        // We're acquiring a reference to the underlying buffer while 
        // holding a shared lock to guarantee ordering of reads and 
        // writes.
        tf_shared_lock ml(*variable->mu());
        const Tensor* lpu_tensor = variable->tensor();

        // Transfer from GPU to CPU
        // lambda function of creating variable if the LPU does not construct it.
        auto creator = [this, hpu_device, lpu_device, lpu_tensor](LegacyVar** hpu_variable){
          // 1.
          // creator type (i.e. auto real type) lambda signature Explanation:
          // std::function<Status(T**)>

          // construct LegacyVar lpu_variable
          *hpu_variable = new LegacyVar(lpu_tensor->dtype());
          (*hpu_variable)->tensor()->set_shape(lpu_tensor->shape());

          // 1. Construct HPU Tensor; 2. Initialize Tenor values from LPU
          // use HPU allocator(GPU) to construct the buf_ in the tensor.
          AllocatorAttributes hpu_alloc_attr;
          hpu_alloc_attr.set_on_host(false);
          hpu_alloc_attr.set_gpu_compatible(true);
          Allocator* hpu_allocator = hpu_device->GetAllocator(hpu_alloc_attr); 
          Tensor copy(hpu_allocator, lpu_tensor->dtype(), lpu_tensor->shape());
          *((*hpu_variable)->tensor()) = copy;

          // construct a lpu_attr used in GPUUtil::DeviceToDeviceCopy.
          AllocatorAttributes lpu_alloc_attr;
          lpu_alloc_attr.set_on_host(false);
          lpu_alloc_attr.set_gpu_compatible(true);
          
          // transfer LPU(src) tensor to HPU(dst)
          DeviceContext* hpu_device_context = hpu_device->tensorflow_gpu_device_info()->default_context;
          DeviceContext* lpu_device_context = lpu_device->tensorflow_gpu_device_info()->default_context;
          // 1.
          // default_context explanation:  
          // gpu_device_info_->default_context = device_contexts_[0] assigned in common_runtime/gpu/gpu_device.cc
          
          Tensor* hpu_tensor = (*hpu_variable)->tensor();

          // logging the tensor allocation memory size for quantitive analysis
          if (hpu_tensor->buf_ != nullptr && hpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferLPU2HPUAllocationCreate", LogMemory::UNKNOWN_STEP_ID, *hpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }

          GPUUtil::DeviceToDeviceCopy(
            lpu_device_context, /* ==> */ hpu_device_context,
            lpu_device, /* ==> */ hpu_device,
            lpu_alloc_attr, hpu_alloc_attr,
            lpu_tensor, /* ==> */ hpu_tensor,
            0, 
            [](const Status& s){
              if (!s.ok()){
                VLOG(0) << "HPU->LPU MemCpy Fail!";
              }
            }
          );
          // 1.
          // dev_to_dev_stream_index Explanation:
          // Only 1 stream group is created so the index is 0.
          // Check the log:
          // https://gist.github.com/shizukanaskytree/1c60070597bdf60fdfd59e8177e9c127
          
          // 2.
          // StatusCallback Explanation:
          // signature: std::function<void(const Status&)> StatusCallback
          // defined in tensorflow/core/common_runtime/rendezvous_mgr.h:79

          return Status::OK();
        };

        LegacyVar* hpu_variable = nullptr;
        
        bool is_init = false;
        // 1.
        // is_init Explanation:
        // 1. If variables are constructed and initialized in the LookupOrCreateVar, 
        // is_init == true  
        // 2. If not, is_init == false, they've already existed. Only need update. 

        // Lookup by resource name, if null then create via memcpy
        hpu_resource_mgr->LookupOrCreateVar<LegacyVar>(
            container, resource_name, &is_init, &hpu_variable, creator);

        // If not by creator:
        // If not by constructed and initialized, then we need to update those 
        // stateful vars.
        if (!is_init){
          // Update vars by transferring tensors from LPU to HPU
          DeviceContext* hpu_device_context = hpu_device->tensorflow_gpu_device_info()->default_context;
          DeviceContext* lpu_device_context = lpu_device->tensorflow_gpu_device_info()->default_context;

          AllocatorAttributes lpu_alloc_attr;
          lpu_alloc_attr.set_on_host(false);
          lpu_alloc_attr.set_gpu_compatible(true);

          AllocatorAttributes hpu_alloc_attr;
          hpu_alloc_attr.set_on_host(false);
          hpu_alloc_attr.set_gpu_compatible(true);

          // since we find it from LookupOrCreateVar, hpu_variable
          Tensor* hpu_tensor = hpu_variable->tensor();

          // logging the tensor allocation memory size for quantitive analysis
          if (hpu_tensor->buf_ != nullptr && hpu_tensor->buf_->data() != nullptr && LogMemory::IsEnabled()) {
            LogMemory::RecordTensorAllocation("TransferLPU2HPUAllocationUpdate", LogMemory::UNKNOWN_STEP_ID, *hpu_tensor);
            // 1.
            // Notice:
            // LOG(INFO) level can output the above information or say VLOG_IS_ON(1)
          }

          GPUUtil::DeviceToDeviceCopy(
            lpu_device_context, /* ==> */ hpu_device_context,
            lpu_device, /* ==> */ hpu_device,
            lpu_alloc_attr, hpu_alloc_attr,
            lpu_tensor, /* ==> */ hpu_tensor,
            0, 
            [](const Status& s){
              if (!s.ok()){
                VLOG(0) << "HPU->LPU MemCpy Fail!";
              }
            }
          ); // End of GPUUtil::DeviceToDeviceCopy
        } // End of updating HPU stateful variables
      
      } // if branch Legacy Branch End
    } // End of for loop of each item
  } // End of for loop of each container


  // Deallocate LPU memory after LPU ==> HPU
  for (auto& lpu_container: lpu_containers) {
    lpu_resource_mgr->Cleanup(lpu_container);
  }

  uint64 lpu2hpu_eclipsed_time = Env::Default()->NowMicros() - start_time_usecs;
  VLOG(0) << "LPU to HPU transfer stateful vars time eclipsed(micro sec): " << lpu2hpu_eclipsed_time;
}
//~wxf

Status DirectSession::MaybeInitializeExecutionState(
    const GraphDef& graph, bool* out_already_initialized) {
  // If already initialized, do nothing.
  // wxf 
  if (direct_session_priority_ == DirectSessionPriority::DIRECTSESSION_PRIORITY_LOW) {
    if (flib_def_ && execution_state_ && low_priority_execution_state_) {
      *out_already_initialized = true;
      return Status::OK();
    }
  } else if (direct_session_priority_ == DirectSessionPriority::DIRECTSESSION_PRIORITY_HIGH) {
    if (flib_def_ && execution_state_) {
      *out_already_initialized = true;
      return Status::OK();
    }
  }

  // Set up the per-session execution state.
  // NOTE(mrry): The function library created here will be used for
  // all subsequent extensions of the graph.
  flib_def_.reset(
      new FunctionLibraryDefinition(OpRegistry::Global(), graph.library()));

  GraphExecutionStateOptions options;
  options.device_set = &device_set_;
  options.session_options = &options_;
  options.session_handle = session_handle_;
  // TODO(mrry,suharshs): We explicitly copy `graph` so that
  // `MakeForBaseGraph()` can take ownership of its
  // contents. Previously this happened implicitly in calls to the
  // `GraphExecutionState`. Other sessions call
  // `MakeForBaseGraph` in such a way that we can destructively read
  // the passed-in `GraphDef`. In principle we could do the same here,
  // with a wider refactoring; we might revise the direct session so
  // that it copies the graph fewer times.
  GraphDef temp(graph);
  TF_RETURN_IF_ERROR(
      GraphExecutionState::MakeForBaseGraph(&temp, options, &execution_state_));
  
  // wxf 
  // Create low_priority_executor_state_ graph
  if (direct_session_priority_ == DirectSessionPriority::DIRECTSESSION_PRIORITY_LOW){
    GraphExecutionStateOptions low_priority_options;
    low_priority_options.device_set = &low_priority_device_set_;
    low_priority_options.session_options = &options_;
    low_priority_options.session_handle = session_handle_;
    GraphDef temp2(graph);
    TF_RETURN_IF_ERROR(
        GraphExecutionState::MakeForLowPriorityBaseGraph(
          &temp2, 
          low_priority_options, 
          &low_priority_execution_state_));
  }

  graph_created_ = true;
  *out_already_initialized = false;
  return Status::OK();
}

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

Status DirectSession::Extend(const GraphDef& graph) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  mutex_lock l(graph_state_lock_);
  return ExtendLocked(graph);
}

Status DirectSession::ExtendLocked(const GraphDef& graph) {
  bool already_initialized;
  // If this is the first call, we can initialize the execution state
  // with `graph` and do not need to call `Extend()`.
  TF_RETURN_IF_ERROR(
      MaybeInitializeExecutionState(graph, &already_initialized));
  if (already_initialized) {
    TF_RETURN_IF_ERROR(flib_def_->AddLibrary(graph.library()));
    std::unique_ptr<GraphExecutionState> state;
    TF_RETURN_IF_ERROR(execution_state_->Extend(graph, &state));
    execution_state_.swap(state);
    
    // wxf
    // Extend low_priority_execution_state_
    if (direct_session_priority_ == DirectSessionPriority::DIRECTSESSION_PRIORITY_LOW){
      std::unique_ptr<GraphExecutionState> low_priority_state;
      TF_RETURN_IF_ERROR(low_priority_execution_state_->Extend(graph, &low_priority_state));
      low_priority_execution_state_.swap(low_priority_state);
    }
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

Status DirectSession::RunInternal(int64 step_id, const RunOptions& run_options,
                                  CallFrameInterface* call_frame,
                                  ExecutorsAndKeys* executors_and_keys,
                                  RunMetadata* run_metadata) {
  const uint64 start_time_usecs = Env::Default()->NowMicros();
  string session_id_meta = strings::StrCat("SessionRun #id=", step_id, "#");
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
  run_state.rendez = new IntraProcessRendezvous(device_mgr_.get());
#ifndef __ANDROID__
  // Set up for collectives if ExecutorsAndKeys declares a key.
  if (executors_and_keys->collective_graph_key !=
      BuildGraphOptions::kNoCollectiveGraphKey) {
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
  const size_t num_executors = executors_and_keys->items.size();
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors, run_state.rendez, [&run_state](const Status& ret) {
        {
          mutex_lock l(run_state.mu_);
          run_state.status.Update(ret);
        }
        run_state.executors_done.Notify();
      });

  Executor::Args args;
  // wxf
  // Priority of this ExecutorState instance inherited from DirectSession which
  // creates this ExecutorState instance. 
  // It should be initialized by const Executor::Args& args.
  args.executor_state_priority = direct_session_priority_;

  // Tell the ExecutorState which device type it is running on now.
  // It is used to construct the ExecutorState instances.
  args.device_type_executing_on = last_execute_device_;
  //~wxf

  args.step_id = step_id;
  args.call_frame = call_frame;
  args.rendezvous = run_state.rendez;
  args.collective_executor =
      (run_state.collective_executor ? run_state.collective_executor->get()
                                     : nullptr);
  CancellationManager step_cancellation_manager;
  args.cancellation_manager = &step_cancellation_manager;
  args.session_state = &session_state_;
  args.session_handle = session_handle_;
  args.tensor_store = &run_state.tensor_store;
  args.step_container = &run_state.step_container;
  args.sync_on_finish = sync_on_finish_;

  const bool do_trace = (run_options.trace_level() > RunOptions::NO_TRACE);

  bool update_cost_model = false;
  if (options_.config.graph_options().build_cost_model() > 0) {
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
  if (do_trace || update_cost_model ||
      run_options.report_tensor_allocations_upon_oom()) {
    run_state.collector.reset(
        new StepStatsCollector(run_metadata->mutable_step_stats()));
    args.stats_collector = run_state.collector.get();
  }

  std::unique_ptr<DeviceTracer> tracer;
  if (run_options.trace_level() >= RunOptions::HARDWARE_TRACE) {
    tracer = CreateDeviceTracer();
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
  const bool already_cancelled = !cancellation_manager_->RegisterCallback(
      cancellation_token, [&step_cancellation_manager]() {
        step_cancellation_manager.StartCancel();
      });
  if (already_cancelled) {
    // NOTE(mrry): If we don't explicitly notify
    // `run_state.executors_done`, the RunState destructor would
    // block on this notification.
    run_state.executors_done.Notify();
    delete barrier;
    return errors::Cancelled("Run call was cancelled");
  }

  thread::ThreadPool* pool =
      run_options.inter_op_thread_pool() >= 0
          ? thread_pools_[run_options.inter_op_thread_pool()].first
          : nullptr;

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
    VLOG(1) << "Using RunHandler to scheduler inter-op closures.";
    handler = GetOrCreateRunHandlerPool(options_)->Get();
  }
  auto* handler_ptr = handler.get();

  Executor::Args::Runner default_runner = nullptr;

  if (pool == nullptr) {
    default_runner = [](Executor::Args::Closure c) { c(); };
  } else if (handler_ptr != nullptr) {
    default_runner = [handler_ptr](Executor::Args::Closure c) {
      handler_ptr->ScheduleInterOpClosure(std::move(c));
    };
  } else {
    default_runner = [this, pool](Executor::Args::Closure c) {
      // wxf
      // Partition threads in the threadpool into two groups, high priority 
      // threads and low priority threads.
      // Schedule low priority in the range of [0, 2]
      // If high priority and low priority tasks both exist,
      if (direct_sessions_manager_->
  	        high_priority_direct_session_count_.load(std::memory_order_relaxed) &&
          direct_sessions_manager_->
            low_priority_direct_session_count_.load(std::memory_order_relaxed)) {
        // For High Priority tasks
        if (this->GetDirectSessionPriority() == DirectSessionPriority::DIRECTSESSION_PRIORITY_HIGH) {
          pool->Schedule(std::move(c));
        } else if (this->GetDirectSessionPriority() == DirectSessionPriority::DIRECTSESSION_PRIORITY_LOW) {
          // For low priority tasks
          // low_priority_thread_pool_->SleepAll();
          // Range: [start , limit]
          // TODO: The number can be set by ENV variable from python TF in the future.
          low_priority_thread_pool_->ScheduleWithHint(std::move(c), 0, 2);

          // 2019-10-30 test to not use global threadpool when we use HPU and LPU(GPU)
          //pool->Schedule(std::move(c));
        }
      } else {
        // Either only high or low, use all threads in the threadpool
        pool->Schedule(std::move(c));
      }

      //SchedClosure(pool, std::move(c));
    };
  }

  for (const auto& item : executors_and_keys->items) {
    // TODO(azaks): support partial run.
    // TODO(azaks): if the device picks its own threadpool, we need to assign
    //     less threads to the main compute pool by default.
    thread::ThreadPool* device_thread_pool =
        item.device->tensorflow_device_thread_pool();
    // TODO(crk): Investigate usage of RunHandlerPool when using device specific
    // thread pool(s).
    if (!device_thread_pool) {
      args.runner = default_runner;
    } else {
      args.runner = [this, device_thread_pool](Executor::Args::Closure c) {
        SchedClosure(device_thread_pool, std::move(c));
      };
    }
    item.executor->RunAsync(args, barrier->Get());
  }

  WaitForNotification(&run_state, &step_cancellation_manager,
                      run_options.timeout_in_ms() > 0
                          ? run_options.timeout_in_ms()
                          : operation_timeout_in_ms_);

  if (!cancellation_manager_->DeregisterCallback(cancellation_token)) {
    // The step has been cancelled: make sure we don't attempt to receive the
    // outputs as this would make it block forever.
    mutex_lock l(run_state.mu_);
    run_state.status.Update(errors::Cancelled("Run call was cancelled"));
  }

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
    TF_RETURN_IF_ERROR(run_state.tensor_store.SaveTensors(
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

Status DirectSession::Run(const RunOptions& run_options,
                          const NamedTensorList& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs,
                          RunMetadata* run_metadata) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  TF_RETURN_IF_ERROR(CheckGraphCreated("Run()"));
  direct_session_runs->GetCell()->IncrementBy(1);

  // Extract the inputs names for this run of the session.
  std::vector<string> input_tensor_names;
  input_tensor_names.reserve(inputs.size());
  for (const auto& it : inputs) {
    input_tensor_names.push_back(it.first);
  }

  // Check if we already have an executor for these arguments.
  ExecutorsAndKeys* executors_and_keys;

  const int64 step_id = step_id_counter_.fetch_add(1);

  RunStateArgs run_state_args(run_options.debug_options());
  run_state_args.collective_graph_key =
      run_options.experimental().collective_graph_key();

  // wxf: if high priority task exists, then the low priority task use the low
  // priority executor.
  if (direct_session_priority_ == DirectSessionPriority::DIRECTSESSION_PRIORITY_LOW &&
      direct_sessions_manager_->high_priority_direct_session_count_.load(std::memory_order_relaxed)) {
//  if (step_id > 20 && step_id < 23) { // wxf: only for debugging one thread task for lower priority for easy developing
    // itself is low priority and high priority exists, then get or create low priority executors
    // Get or Create low priority CPU executors
    TF_RETURN_IF_ERROR(GetOrCreateLowPriorityExecutors(input_tensor_names, output_names,
                                                       target_nodes, &executors_and_keys,
                                                       &run_state_args));

    // Start to transfer stateful data from GPU to CPU via device resource mgr.
    if (last_execute_device_ == "" || last_execute_device_ == "HPU"){
      // N.B.
      // to test transfer time between CPU to 2080 GPU, use API `TransferGPU2CPUStatefulVars`, `TransferCPU2GPUStatefulVars`.
      // to test transfer time between 1080 GPU to 2080 GPU, use API `TransferHPU2LPUStatefulVars`, `TransferLPU2HPUStatefulVars`.
      TransferGPU2CPUStatefulVars();
      //TransferHPU2LPUStatefulVars();
    }
    last_execute_device_ = "LPU";
    // End of transferring, start to execute the graph.
 
  } else {
    // High performance device branch:

    TF_RETURN_IF_ERROR(GetOrCreateExecutors(input_tensor_names, output_names,
                                            target_nodes, &executors_and_keys,
                                            &run_state_args));
  
    // Start to transfer stateful data from CPU to GPU via device resource mgr.
    if (last_execute_device_ != "" && last_execute_device_ == "LPU"){
      // N.B.
      // to test transfer time between CPU to 2080 GPU, use API `TransferGPU2CPUStatefulVars`, `TransferCPU2GPUStatefulVars`.
      // to test transfer time between 1080 GPU to 2080 GPU, use API `TransferHPU2LPUStatefulVars`, `TransferLPU2HPUStatefulVars`.
      TransferCPU2GPUStatefulVars();
      //TransferLPU2HPUStatefulVars();
    }
    last_execute_device_ = "HPU"; 
    // End of transferring, start to execute the graph.
  }

  {
    mutex_lock l(collective_graph_key_lock_);
    collective_graph_key_ = executors_and_keys->collective_graph_key;
  }

  // Configure a call frame for the step, which we use to feed and
  // fetch values to and from the executors.
  FunctionCallFrame call_frame(executors_and_keys->input_types,
                               executors_and_keys->output_types);
  gtl::InlinedVector<Tensor, 4> feed_args(inputs.size());
  for (const auto& it : inputs) {
    if (it.second.dtype() == DT_RESOURCE) {
      Tensor tensor_from_handle;
      TF_RETURN_IF_ERROR(
          ResourceHandleToInputTensor(it.second, &tensor_from_handle));
      feed_args[executors_and_keys->input_name_to_index[it.first]] =
          tensor_from_handle;
    } else {
      feed_args[executors_and_keys->input_name_to_index[it.first]] = it.second;
    }
  }
  const Status s = call_frame.SetArgs(feed_args);
  if (errors::IsInternal(s)) {
    return errors::InvalidArgument(s.error_message());
  } else if (!s.ok()) {
    return s;
  }
  
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(step_id, run_state_args.handle);
  }
  
  TF_RETURN_IF_ERROR(RunInternal(step_id, run_options, &call_frame,
                                 executors_and_keys, run_metadata));
  // Receive outputs.
  if (outputs) {
    // wxf:
    // If we detect that any output tensor is dead, re-execute it on LPU.
    if (call_frame.NeedReexecute()) {
      VLOG(0) << "NEED Re-execute";
      Run(run_options, inputs, output_names, target_nodes, outputs, run_metadata);
    }
    //~wxf

    std::vector<Tensor> sorted_outputs;
    const Status s = call_frame.ConsumeRetvals(
        &sorted_outputs, /* allow_dead_tensors = */ true); // modified by wxf
        //&sorted_outputs, /* allow_dead_tensors = */ false); // original 

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
  }

  return Status::OK();
}

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
//  // wxf
//  // Priority of this ExecutorState instance inherited from DirectSession which
//  // creates this ExecutorState instance. 
//  // It should be initialized by const Executor::Args& args.
//  args.executor_state_priority = direct_session_priority_;

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
      num_executors, run_state->rendez, [run_state](const Status& ret) {
        if (!ret.ok()) {
          mutex_lock l(run_state->mu_);
          run_state->status.Update(ret);
        }
        run_state->executors_done.Notify();
      });

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

Status DirectSession::CreateExecutors(
    const CallableOptions& callable_options,
    std::unique_ptr<ExecutorsAndKeys>* out_executors_and_keys,
    std::unique_ptr<FunctionInfo>* out_func_info,
    RunStateArgs* run_state_args) {
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

  std::unique_ptr<FunctionInfo> func_info(new FunctionInfo);
  std::unique_ptr<ExecutorsAndKeys> ek(new ExecutorsAndKeys);

  ek->callable_options = callable_options;

  std::unordered_map<string, std::unique_ptr<Graph>> graphs;
  TF_RETURN_IF_ERROR(CreateGraphs(
      options, &graphs, &func_info->flib_def, run_state_args, &ek->input_types,
      &ek->output_types, &ek->collective_graph_key));

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
  ek->items.reserve(graphs.size());
  const auto& optimizer_opts =
      options_.config.graph_options().optimizer_options();

  int graph_def_version;
  {
    mutex_lock l(graph_state_lock_);
    graph_def_version =
        execution_state_->original_graph_def().versions().producer();
  }
  func_info->proc_flr.reset(new ProcessFunctionLibraryRuntime(
      device_mgr_.get(), options_.env, graph_def_version,
      func_info->flib_def.get(), optimizer_opts, thread_pools_[0].first));

  GraphOptimizer optimizer(optimizer_opts);
  for (auto iter = graphs.begin(); iter != graphs.end(); ++iter) {
    const string& partition_name = iter->first;
    std::unique_ptr<Graph>& partition_graph = iter->second;

    Device* device;

    TF_RETURN_IF_ERROR(device_mgr_->LookupDevice(partition_name, &device));

    ek->items.resize(ek->items.size() + 1);

    auto* item = &(ek->items.back());

    auto lib = func_info->proc_flr->GetFLR(partition_name);
    if (lib == nullptr) {
      return errors::Internal("Could not find device: ", partition_name);
    }
    item->flib = lib;

    LocalExecutorParams params;
    params.device = device;

    params.function_library = lib;

    auto opseg = device->op_segment();

    params.create_kernel = [this, lib, opseg](const NodeDef& ndef,
                                              OpKernel** kernel) {
      // NOTE(mrry): We must not share function kernels (implemented
      // using `CallOp`) between subgraphs, because `CallOp::handle_`
      // is tied to a particular subgraph. Even if the function itself
      // is stateful, the `CallOp` that invokes it is not.
      if (!OpSegment::ShouldOwnKernel(lib, ndef.op())) {
        return lib->CreateKernel(ndef, kernel);
      }
      auto create_fn = [lib, &ndef](OpKernel** kernel) {
        return lib->CreateKernel(ndef, kernel);
      };
      // Kernels created for subgraph nodes need to be cached.  On
      // cache miss, create_fn() is invoked to create a kernel based
      // on the function library here + global op registry.
      return opseg->FindOrCreate(session_handle_, ndef.name(), kernel,
                                 create_fn);
    };
    params.delete_kernel = [lib](OpKernel* kernel) {
      if (kernel && !OpSegment::ShouldOwnKernel(lib, kernel->type_string()))
        delete kernel;
    };

    optimizer.Optimize(lib, options_.env, device, &partition_graph,
                       /*shape_map=*/nullptr);

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
    item->graph = partition_graph.get();
    item->executor = nullptr;
    item->device = device;
    auto executor_type = options_.config.experimental().executor_type();
    TF_RETURN_IF_ERROR(NewExecutor(
        executor_type, params, std::move(partition_graph), &item->executor));
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

// wxf
Status DirectSession::CreateLowPriorityExecutors(
    const CallableOptions& callable_options,
    std::unique_ptr<ExecutorsAndKeys>* out_executors_and_keys,
    std::unique_ptr<FunctionInfo>* out_func_info,
    RunStateArgs* run_state_args) {
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

  std::unique_ptr<FunctionInfo> func_info(new FunctionInfo);
  std::unique_ptr<ExecutorsAndKeys> ek(new ExecutorsAndKeys);

  ek->callable_options = callable_options;

  std::unordered_map<string, std::unique_ptr<Graph>> graphs;
  TF_RETURN_IF_ERROR(CreateLowPriorityGraphs(
      options, &graphs, &func_info->flib_def, run_state_args, &ek->input_types,
      &ek->output_types, &ek->collective_graph_key));

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
  ek->items.reserve(graphs.size());
  const auto& optimizer_opts =
      options_.config.graph_options().optimizer_options();

  int graph_def_version;
  {
    mutex_lock l(graph_state_lock_);
    graph_def_version =
        execution_state_->original_graph_def().versions().producer();
  }
  func_info->proc_flr.reset(new ProcessFunctionLibraryRuntime(
      device_mgr_.get(), options_.env, graph_def_version,
      func_info->flib_def.get(), optimizer_opts, thread_pools_[0].first));

  GraphOptimizer optimizer(optimizer_opts);
  for (auto iter = graphs.begin(); iter != graphs.end(); ++iter) {
    const string& partition_name = iter->first;
    std::unique_ptr<Graph>& partition_graph = iter->second;

    Device* device;

    TF_RETURN_IF_ERROR(device_mgr_->LookupDevice(partition_name, &device));

    ek->items.resize(ek->items.size() + 1);

    auto* item = &(ek->items.back());

    auto lib = func_info->proc_flr->GetFLR(partition_name);
    if (lib == nullptr) {
      return errors::Internal("Could not find device: ", partition_name);
    }
    item->flib = lib;

    LocalExecutorParams params;
    params.device = device;

    params.function_library = lib;

    auto opseg = device->op_segment();

    params.create_kernel = [this, lib, opseg](const NodeDef& ndef,
                                              OpKernel** kernel) {
      // NOTE(mrry): We must not share function kernels (implemented
      // using `CallOp`) between subgraphs, because `CallOp::handle_`
      // is tied to a particular subgraph. Even if the function itself
      // is stateful, the `CallOp` that invokes it is not.
      if (!OpSegment::ShouldOwnKernel(lib, ndef.op())) {
        return lib->CreateKernel(ndef, kernel);
      }
      auto create_fn = [lib, &ndef](OpKernel** kernel) {
        return lib->CreateKernel(ndef, kernel);
      };
      // Kernels created for subgraph nodes need to be cached.  On
      // cache miss, create_fn() is invoked to create a kernel based
      // on the function library here + global op registry.
      return opseg->FindOrCreate(session_handle_, ndef.name(), kernel,
                                 create_fn);
    };
    params.delete_kernel = [lib](OpKernel* kernel) {
      if (kernel && !OpSegment::ShouldOwnKernel(lib, kernel->type_string()))
        delete kernel;
    };

    optimizer.Optimize(lib, options_.env, device, &partition_graph,
                       /*shape_map=*/nullptr);

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
    item->graph = partition_graph.get();
    item->executor = nullptr;
    item->device = device;
    auto executor_type = options_.config.experimental().executor_type();
    TF_RETURN_IF_ERROR(NewExecutor(
        executor_type, params, std::move(partition_graph), &item->executor));
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

Status DirectSession::GetOrCreateExecutors(
    gtl::ArraySlice<string> inputs, gtl::ArraySlice<string> outputs,
    gtl::ArraySlice<string> target_nodes, ExecutorsAndKeys** executors_and_keys,
    RunStateArgs* run_state_args) {
  int64 handle_name_counter_value = -1;
  if (LogMemory::IsEnabled() || run_state_args->is_partial_run) {
    handle_name_counter_value = handle_name_counter_.fetch_add(1);
  }

  string debug_tensor_watches_summary;
  if (!run_state_args->debug_options.debug_tensor_watch_opts().empty()) {
    debug_tensor_watches_summary = SummarizeDebugTensorWatches(
        run_state_args->debug_options.debug_tensor_watch_opts());
  }

  // Fast lookup path, no sorting.
  const string key = strings::StrCat(
      str_util::Join(inputs, ","), "->", str_util::Join(outputs, ","), "/",
      str_util::Join(target_nodes, ","), "/", run_state_args->is_partial_run,
      "/", debug_tensor_watches_summary);
  // Set the handle, if it's needed to log memory or for partial run.
  if (handle_name_counter_value >= 0) {
    run_state_args->handle =
        strings::StrCat(key, ";", handle_name_counter_value);
  }

  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto it = executors_.find(key);
    if (it != executors_.end()) {
      *executors_and_keys = it->second.get();
      return Status::OK();
    }
  }

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

  const string sorted_key = strings::StrCat(
      str_util::Join(inputs_sorted, ","), "->",
      str_util::Join(outputs_sorted, ","), "/", str_util::Join(tn_sorted, ","),
      "/", run_state_args->is_partial_run, "/", debug_tensor_watches_summary);
  // Set the handle, if its needed to log memory or for partial run.
  if (handle_name_counter_value >= 0) {
    run_state_args->handle =
        strings::StrCat(sorted_key, ";", handle_name_counter_value);
  }

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

  // Nothing found, so create the executors and store in the cache.
  // The executor_lock_ is intentionally released while executors are
  // being created.
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
  std::unique_ptr<ExecutorsAndKeys> ek;
  std::unique_ptr<FunctionInfo> func_info;
  TF_RETURN_IF_ERROR(
      CreateExecutors(callable_options, &ek, &func_info, run_state_args));

  // Reacquire the lock, try to insert into the map.
  mutex_lock l(executor_lock_);
  functions_.push_back(std::move(func_info));

  // Another thread may have created the entry before us, in which case we will
  // reuse the already created one.
  auto insert_result = executors_.emplace(
      sorted_key, std::shared_ptr<ExecutorsAndKeys>(std::move(ek)));
  // Insert the value under the original key, so the fast path lookup will work
  // if the user uses the same order of inputs, outputs, and targets again.
  executors_.emplace(key, insert_result.first->second);
  *executors_and_keys = insert_result.first->second.get();

  return Status::OK();
}

// wxf
Status DirectSession::GetOrCreateLowPriorityExecutors(
    gtl::ArraySlice<string> inputs, gtl::ArraySlice<string> outputs,
    gtl::ArraySlice<string> target_nodes, ExecutorsAndKeys** executors_and_keys,
    RunStateArgs* run_state_args) {
  int64 handle_name_counter_value = -1;
  if (LogMemory::IsEnabled() || run_state_args->is_partial_run) {
    handle_name_counter_value = handle_name_counter_.fetch_add(1);
  }

  string debug_tensor_watches_summary;
  if (!run_state_args->debug_options.debug_tensor_watch_opts().empty()) {
    debug_tensor_watches_summary = SummarizeDebugTensorWatches(
        run_state_args->debug_options.debug_tensor_watch_opts());
  }

  // Fast lookup path, no sorting.
  const string key = strings::StrCat(
      "low_priority_device/",
      str_util::Join(inputs, ","), "->", str_util::Join(outputs, ","), "/",
      str_util::Join(target_nodes, ","), "/", run_state_args->is_partial_run,
      "/", debug_tensor_watches_summary);
  // Set the handle, if it's needed to log memory or for partial run.
  if (handle_name_counter_value >= 0) {
    run_state_args->handle =
        strings::StrCat(key, ";", handle_name_counter_value);
  }

  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto it = executors_.find(key);
    if (it != executors_.end()) {
      *executors_and_keys = it->second.get();
      return Status::OK();
    }
  }

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

  const string sorted_key = strings::StrCat(
      "low_priority_device/",
      str_util::Join(inputs_sorted, ","), "->",
      str_util::Join(outputs_sorted, ","), "/", str_util::Join(tn_sorted, ","),
      "/", run_state_args->is_partial_run, "/", debug_tensor_watches_summary);
  // Set the handle, if its needed to log memory or for partial run.
  if (handle_name_counter_value >= 0) {
    run_state_args->handle =
        strings::StrCat(sorted_key, ";", handle_name_counter_value);
  }

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

  // Nothing found, so create the executors and store in the cache.
  // The executor_lock_ is intentionally released while executors are
  // being created.
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
  std::unique_ptr<ExecutorsAndKeys> ek;
  std::unique_ptr<FunctionInfo> func_info;
  TF_RETURN_IF_ERROR(
      CreateLowPriorityExecutors(callable_options, &ek, &func_info, run_state_args));

  // Reacquire the lock, try to insert into the map.
  mutex_lock l(executor_lock_);
  functions_.push_back(std::move(func_info));

  // Another thread may have created the entry before us, in which case we will
  // reuse the already created one.
  auto insert_result = executors_.emplace(
      sorted_key, std::shared_ptr<ExecutorsAndKeys>(std::move(ek)));
  // Insert the value under the original key, so the fast path lookup will work
  // if the user uses the same order of inputs, outputs, and targets again.
  executors_.emplace(key, insert_result.first->second);
  *executors_and_keys = insert_result.first->second.get();

  return Status::OK();
}

Status DirectSession::CreateGraphs(
    const BuildGraphOptions& subgraph_options,
    std::unordered_map<string, std::unique_ptr<Graph>>* outputs,
    std::unique_ptr<FunctionLibraryDefinition>* flib_def,
    RunStateArgs* run_state_args, DataTypeVector* input_types,
    DataTypeVector* output_types, int64* collective_graph_key) {
  mutex_lock l(graph_state_lock_);
  std::unique_ptr<ClientGraph> client_graph;

  std::unique_ptr<GraphExecutionState> temp_exec_state_holder;
  GraphExecutionState* execution_state = nullptr;
  if (options_.config.graph_options().place_pruned_graph()) {
    // Because we are placing pruned graphs, we need to create a
    // new GraphExecutionState for every new unseen graph,
    // and then place it.
    GraphExecutionStateOptions prune_options;
    prune_options.device_set = &device_set_;
    prune_options.session_options = &options_;
    prune_options.stateful_placements = stateful_placements_;
    prune_options.session_handle = session_handle_;
    TF_RETURN_IF_ERROR(GraphExecutionState::MakeForPrunedGraph(
        execution_state_->original_graph_def().library(), prune_options,
        execution_state_->original_graph_def(), subgraph_options,
        &temp_exec_state_holder, &client_graph));
    execution_state = temp_exec_state_holder.get();
  } else {
    execution_state = execution_state_.get();
    TF_RETURN_IF_ERROR(
        execution_state->BuildGraph(subgraph_options, &client_graph));
  }
  *collective_graph_key = client_graph->collective_graph_key;

  if (subgraph_options.callable_options.feed_size() !=
      client_graph->feed_types.size()) {
    return errors::Internal(
        "Graph pruning failed: requested number of feed endpoints = ",
        subgraph_options.callable_options.feed_size(),
        " versus number of pruned feed endpoints = ",
        client_graph->feed_types.size());
  }
  if (subgraph_options.callable_options.fetch_size() !=
      client_graph->fetch_types.size()) {
    return errors::Internal(
        "Graph pruning failed: requested number of fetch endpoints = ",
        subgraph_options.callable_options.fetch_size(),
        " versus number of pruned fetch endpoints = ",
        client_graph->fetch_types.size());
  }

  auto current_stateful_placements = execution_state->GetStatefulPlacements();
  // Update our current state based on the execution_state's
  // placements.  If there are any mismatches for a node,
  // we should fail, as this should never happen.
  for (auto placement_pair : current_stateful_placements) {
    const string& node_name = placement_pair.first;
    const string& placement = placement_pair.second;
    auto iter = stateful_placements_.find(node_name);
    if (iter == stateful_placements_.end()) {
      stateful_placements_.insert(std::make_pair(node_name, placement));
    } else if (iter->second != placement) {
      return errors::Internal(
          "Stateful placement mismatch. "
          "Current assignment of ",
          node_name, " to ", iter->second, " does not match ", placement);
    }
  }

  stateful_placements_ = execution_state->GetStatefulPlacements();

  // Remember the graph in run state if this is a partial run.
  if (run_state_args->is_partial_run) {
    run_state_args->graph.reset(new Graph(flib_def_.get()));
    CopyGraph(*execution_state->full_graph(), run_state_args->graph.get());
  }

  // Partition the graph across devices.
  PartitionOptions popts;

  // wxf
  // set _Arg op node on GPUs
  char* set_reuse_flag = getenv("TF_SET_REUSE_INPUTS_FLAG");
  if (set_reuse_flag != NULL) {
    // true, i.e. SET
    TF_SET_REUSE_INPUTS_FLAG = strcmp(set_reuse_flag, "1") == 0; 
    if (TF_SET_REUSE_INPUTS_FLAG) { 
      //VLOG(0) << ">>> SET_REUSE_INPUTS_FLAG";

      // Master input names
      // Make sure you set master inputs X and y in the python side.
      char *master_input_X, *master_input_y;
      master_input_X = getenv("TF_REUSE_INPUT_OP_NAME_MASTER_X");
      master_input_y = getenv("TF_REUSE_INPUT_OP_NAME_MASTER_y");
      if (master_input_X != NULL) {
        master_input_X_name = string(master_input_X); // "XX01" 
        VLOG(0) << ">>> getenv master X: " << master_input_X_name;
      }

      if (master_input_y != NULL) {
        master_input_y_name = string(master_input_y); // "yy01"
        VLOG(0) << ">>> getenv master Y: " << master_input_y_name;
      }

      // Subsidiary input names
      char *input_ops_name_X, *input_ops_name_y;
      input_ops_name_X = getenv("TF_REUSE_INPUT_OPS_NAME_SUB_X");
      input_ops_name_y = getenv("TF_REUSE_INPUT_OPS_NAME_SUB_y");
      if (input_ops_name_X != NULL) {
        //VLOG(0) << input_ops_name;
        subsidiary_input_op_names_X = str_util::Split(input_ops_name_X, ',');
        // {'X02', ...}
        
        for (string& X_name: subsidiary_input_op_names_X) {
          VLOG(0) << ">>> getenv subsidiary input X: " << X_name;
        }
      }
      if (input_ops_name_y != NULL) {
        //VLOG(0) << input_ops_name;
        subsidiary_input_op_names_y = str_util::Split(input_ops_name_y, ',');
        // {'y02', ...}
        
        for (string& y_name: subsidiary_input_op_names_y) {
          VLOG(0) << ">>> getenv subsidiary input y: " << y_name;
        }
      }
      num_token_turn = 1 + subsidiary_input_op_names_X.size();
    }
  }
  //~wxf

  popts.node_to_loc = [&](const Node* node) {
    // wxf
    // Subsidiary inputs X are placed on GPU
    auto it = std::find_if(subsidiary_input_op_names_X.begin(), 
                subsidiary_input_op_names_X.end(), 
                [&](string & subsidiary_input) {
                  return str_util::StrContains(node->name(), subsidiary_input);
                });
    if (it != subsidiary_input_op_names_X.end()) {
      VLOG(0) << ">>> subsidiary reuse input: " << node->name();
      return string("/job:localhost/replica:0/task:0/device:GPU:0");
    }

    // Subsidiary inputs y are placed on GPU
    it = std::find_if(subsidiary_input_op_names_y.begin(), 
                subsidiary_input_op_names_y.end(), 
                [&](string & subsidiary_input) {
                  return str_util::StrContains(node->name(), subsidiary_input);
                });
    if (it != subsidiary_input_op_names_y.end()) {
      VLOG(0) << ">>> subsidiary reuse input: " << node->name();
      return string("/job:localhost/replica:0/task:0/device:GPU:0");
    }

//    if (TF_SET_REUSE_INPUTS_FLAG) {
//      for (string subsidiary_input_name: subsidiary_input_op_names) {
//        if (str_util::StrContains(node->name(), subsidiary_input_name)) {
//          VLOG(0) << ">>> subsidiary reuse input: " << node->name();
//          return string("/job:localhost/replica:0/task:0/device:GPU:0");
//        }
//      }
//    }

//    // graph 02-0N inputs are placed on GPU
//    if (node->name() == "_arg_X02_0_0" || node->name() == "_arg_y02_0_1" || 
//        node->name() == "_arg_X03_0_0" || node->name() == "_arg_y03_0_1" || 
//        node->name() == "_arg_X04_0_0" || node->name() == "_arg_y04_0_1") {
//      //VLOG(0) << "popts.node_to_loc for _Arg nodes: " << node->name();
//      return string("/job:localhost/replica:0/task:0/device:GPU:0");
//    }
//    //~wxf

    return node->assigned_device_name();
  };
  popts.new_name = [this](const string& prefix) {
    return strings::StrCat(prefix, "/_", edge_name_counter_.fetch_add(1));
  };
  popts.get_incarnation = [](const string& name) {
    // The direct session does not have changing incarnation numbers.
    // Just return '1'.
    return 1;
  };
  popts.flib_def = &client_graph->graph.flib_def();
  popts.control_flow_added = false;

  std::unordered_map<string, GraphDef> partitions;
  TF_RETURN_IF_ERROR(Partition(popts, &client_graph->graph, &partitions));

  std::vector<string> device_names;
  for (auto device : devices_) {
    // Extract the LocalName from the device.
    device_names.push_back(DeviceNameUtils::LocalName(device->name()));
  }

  // Check for valid partitions.
  for (const auto& partition : partitions) {
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
    std::unique_ptr<Graph> device_graph(
        new Graph(client_graph->flib_def.get()));
    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(device_opts, partition.second,
                                              device_graph.get()));
    outputs->emplace(partition.first, std::move(device_graph));
  }

  // wxf
  VLOG(1) << "OptimizationPassRegistry::POST_PARTITIONING \n";

  GraphOptimizationPassOptions optimization_options;
  optimization_options.session_options = &options_;
  optimization_options.flib_def = client_graph->flib_def.get();
  optimization_options.partition_graphs = outputs;
  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_PARTITIONING, optimization_options));

  Status s;
  for (auto& partition : *outputs) {
    const string& partition_name = partition.first;
    std::unique_ptr<Graph>* graph = &partition.second;

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
}

// wxf
Status DirectSession::CreateLowPriorityGraphs(
    const BuildGraphOptions& subgraph_options,
    std::unordered_map<string, std::unique_ptr<Graph>>* outputs,
    std::unique_ptr<FunctionLibraryDefinition>* flib_def,
    RunStateArgs* run_state_args, DataTypeVector* input_types,
    DataTypeVector* output_types, int64* collective_graph_key) {
  mutex_lock l(graph_state_lock_);
  std::unique_ptr<ClientGraph> client_graph;

  std::unique_ptr<GraphExecutionState> temp_exec_state_holder;
  GraphExecutionState* execution_state = nullptr;
  if (options_.config.graph_options().place_pruned_graph()) {
    // Because we are placing pruned graphs, we need to create a
    // new GraphExecutionState for every new unseen graph,
    // and then place it.
    GraphExecutionStateOptions prune_options;
    prune_options.device_set = &device_set_;
    prune_options.session_options = &options_;
    prune_options.stateful_placements = stateful_placements_;
    prune_options.session_handle = session_handle_;
    TF_RETURN_IF_ERROR(GraphExecutionState::MakeForPrunedGraph(
        execution_state_->original_graph_def().library(), prune_options,
        execution_state_->original_graph_def(), subgraph_options,
        &temp_exec_state_holder, &client_graph));
    execution_state = temp_exec_state_holder.get();
  } else {
    // wxf  
    // DirectSession::MaybeInitializeExecutionState init low_priority_execution_state_
    execution_state = low_priority_execution_state_.get();
    TF_RETURN_IF_ERROR(
        execution_state->BuildGraph(subgraph_options, &client_graph));
  }
  *collective_graph_key = client_graph->collective_graph_key;

  if (subgraph_options.callable_options.feed_size() !=
      client_graph->feed_types.size()) {
    return errors::Internal(
        "Graph pruning failed: requested number of feed endpoints = ",
        subgraph_options.callable_options.feed_size(),
        " versus number of pruned feed endpoints = ",
        client_graph->feed_types.size());
  }
  if (subgraph_options.callable_options.fetch_size() !=
      client_graph->fetch_types.size()) {
    return errors::Internal(
        "Graph pruning failed: requested number of fetch endpoints = ",
        subgraph_options.callable_options.fetch_size(),
        " versus number of pruned fetch endpoints = ",
        client_graph->fetch_types.size());
  }

  // wxf
  // low priority Graph stateful placement information should not update
  // DirectSession::stateful_placements_ 
  // So, I comment this part.
//  auto current_stateful_placements = execution_state->GetStatefulPlacements();
//  // Update our current state based on the execution_state's
//  // placements.  If there are any mismatches for a node,
//  // we should fail, as this should never happen.
//  for (auto placement_pair : current_stateful_placements) {
//    const string& node_name = placement_pair.first;
//    const string& placement = placement_pair.second;
//    auto iter = stateful_placements_.find(node_name);
//    if (iter == stateful_placements_.end()) {
//      stateful_placements_.insert(std::make_pair(node_name, placement));
//    } else if (iter->second != placement) {
//      return errors::Internal(
//          "Stateful placement mismatch. "
//          "Current assignment of ",
//          node_name, " to ", iter->second, " does not match ", placement);
//    }
//  }
//
//  stateful_placements_ = execution_state->GetStatefulPlacements();

  // Remember the graph in run state if this is a partial run.
  if (run_state_args->is_partial_run) {
    run_state_args->graph.reset(new Graph(flib_def_.get()));
    CopyGraph(*execution_state->full_graph(), run_state_args->graph.get());
  }

  // Partition the graph across devices.
  PartitionOptions popts;
  popts.node_to_loc = [](const Node* node) {
    return node->assigned_device_name();
  };
  popts.new_name = [this](const string& prefix) {
    return strings::StrCat(prefix, "/_", edge_name_counter_.fetch_add(1));
  };
  popts.get_incarnation = [](const string& name) {
    // The direct session does not have changing incarnation numbers.
    // Just return '1'.
    return 1;
  };
  popts.flib_def = &client_graph->graph.flib_def();
  popts.control_flow_added = false;

  std::unordered_map<string, GraphDef> partitions;
  TF_RETURN_IF_ERROR(Partition(popts, &client_graph->graph, &partitions));

  std::vector<string> device_names;
  for (auto device : devices_) {
    // Extract the LocalName from the device.
    device_names.push_back(DeviceNameUtils::LocalName(device->name()));
  }

  // Check for valid partitions.
  for (const auto& partition : partitions) {
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
    std::unique_ptr<Graph> device_graph(
        new Graph(client_graph->flib_def.get()));
    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(device_opts, partition.second,
                                              device_graph.get()));
    outputs->emplace(partition.first, std::move(device_graph));
  }

  GraphOptimizationPassOptions optimization_options;
  optimization_options.session_options = &options_;
  optimization_options.flib_def = client_graph->flib_def.get();
  optimization_options.partition_graphs = outputs;
  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_PARTITIONING, optimization_options));

  Status s;
  for (auto& partition : *outputs) {
    const string& partition_name = partition.first;
    std::unique_ptr<Graph>* graph = &partition.second;

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
}

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
    const std::vector<string>& pending_output_names, int64 step_id,
    const std::vector<Device*>* devices)
    : step_container(step_id, [devices, step_id](const string& name) {
        for (auto d : *devices) {
          if (!d->resource_manager()->Cleanup(name).ok()) {
            // Do nothing...
          }
          ScopedAllocatorMgr* sam = d->GetScopedAllocatorMgr();
          if (sam) sam->Cleanup(step_id);
        }
      }) {
  // Initially all the feeds and fetches are pending.
  for (auto& name : pending_input_names) {
    pending_inputs[name] = false;
  }
  for (auto& name : pending_output_names) {
    pending_outputs[name] = false;
  }
}

DirectSession::RunState::RunState(int64 step_id,
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

void DirectSession::WaitForNotification(RunState* run_state,
                                        CancellationManager* cm,
                                        int64 timeout_in_ms) {
  const Status status =
      WaitForNotification(&run_state->executors_done, timeout_in_ms);
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

::tensorflow::Status DirectSession::WaitForNotification(
    Notification* notification, int64 timeout_in_ms) {
  if (timeout_in_ms > 0) {
    const int64 timeout_in_us = timeout_in_ms * 1000;
    const bool notified =
        WaitForNotificationWithTimeout(notification, timeout_in_us);
    if (!notified) {
      return Status(error::DEADLINE_EXCEEDED,
                    "Timed out waiting for notification");
    }
  } else {
    notification->WaitForNotification();
  }
  return Status::OK();
}

Status DirectSession::MakeCallable(const CallableOptions& callable_options,
                                   CallableHandle* out_handle) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  TF_RETURN_IF_ERROR(CheckGraphCreated("MakeCallable()"));

  std::unique_ptr<ExecutorsAndKeys> ek;
  std::unique_ptr<FunctionInfo> func_info;
  RunStateArgs run_state_args(callable_options.run_options().debug_options());
  TF_RETURN_IF_ERROR(
      CreateExecutors(callable_options, &ek, &func_info, &run_state_args));
  {
    mutex_lock l(callables_lock_);
    *out_handle = next_callable_handle_++;
    callables_[*out_handle] = {std::move(ek), std::move(func_info)};
  }

  // -----------------------------------------------------------------------

  // wxf
  // For low priority task, we create a backup LPU-only executors;
  // For high priority task, we don't create LPU-only executors.
  // LPU: Low Performance Processing Unit
  // HPU: High Performance Processing Unit
  if(direct_session_priority_ == DirectSessionPriority::DIRECTSESSION_PRIORITY_LOW){
    CallableHandle low_priority_handle = 0;
    // 1.
    // low_priority_handle explanation:
    // the low_priority_handle is hid from the user.
  
    std::unique_ptr<ExecutorsAndKeys> low_priority_ek;
    std::unique_ptr<FunctionInfo> low_priority_func_info;
    RunStateArgs low_priority_run_state_args(callable_options.run_options().debug_options());
    TF_RETURN_IF_ERROR(
        CreateLowPriorityExecutors(
          callable_options,
          &low_priority_ek,
          &low_priority_func_info,
          &low_priority_run_state_args));
    {
      mutex_lock l(callables_lock_);
      low_priority_handle = next_low_priority_callable_handle_++;
      // take down the mapping for DirectSession::RunCallable
      usr_handle_to_low_priority_handle_[*out_handle] = low_priority_handle;
      low_priority_callables_[low_priority_handle] = {std::move(low_priority_ek),
                                                      std::move(low_priority_func_info)};
    }
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

  Status GetArg(int index, Tensor* val) const override {
    if (index > feed_tensors_->size()) {
      return errors::Internal("Args index out of bounds: ", index);
    } else if (executors_and_keys_->input_types[index] == DT_RESOURCE) {
      TF_RETURN_IF_ERROR(
          session_->ResourceHandleToInputTensor((*feed_tensors_)[index], val));
    } else {
      *val = (*feed_tensors_)[index];
    }
    return Status::OK();
  }

  Status SetRetval(int index, const Tensor& val) override {
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

::tensorflow::Status DirectSession::RunCallable(
    CallableHandle handle, const std::vector<Tensor>& feed_tensors,
    std::vector<Tensor>* fetch_tensors, RunMetadata* run_metadata) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  TF_RETURN_IF_ERROR(CheckGraphCreated("RunCallable()"));
  direct_session_runs->GetCell()->IncrementBy(1);

  // Check if we already have an executor for these arguments.
  std::shared_ptr<ExecutorsAndKeys> executors_and_keys;
  const int64 step_id = step_id_counter_.fetch_add(1);

  // wxf 
  // if high priority task exists, then the low priority task uses the low
  // priority executor.
  if (direct_session_priority_ == DirectSessionPriority::DIRECTSESSION_PRIORITY_LOW &&
      direct_sessions_manager_->high_priority_direct_session_count_.load(std::memory_order_relaxed)) {
//  if (step_id > 20 && step_id < 23) { // wxf: only for debugging one thread task for lower priority for easy developing
    {
      tf_shared_lock l(callables_lock_);
      CallableHandle low_priority_handle = usr_handle_to_low_priority_handle_[handle];
      if (low_priority_handle >= next_low_priority_callable_handle_) {
        return errors::InvalidArgument("Low Priority case: No such callable handle: ", low_priority_handle);
      }
      executors_and_keys = low_priority_callables_[low_priority_handle].executors_and_keys;
    }

    if (!executors_and_keys) {
      return errors::InvalidArgument(
          "Low Priority case: Attempted to run callable after handle was released: ", handle);
      // wxf
      // we still use handle here since low_priority_handle var is out of scope.
    }

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

    // Start to transfer stateful data from HPU to LPU via device resource mgr.
    if (last_execute_device_ == "" || last_execute_device_ == "HPU"){
      // N.B.
      // to test transfer time between CPU to 2080 GPU, use API `TransferGPU2CPUStatefulVars`, `TransferCPU2GPUStatefulVars`.
      // to test transfer time between 1080 GPU to 2080 GPU, use API `TransferHPU2LPUStatefulVars`, `TransferLPU2HPUStatefulVars`.
      TransferGPU2CPUStatefulVars();
      //TransferHPU2LPUStatefulVars();
    }
    last_execute_device_ = "LPU";
    // End of transferring, start to execute the graph.

    RunCallableCallFrame call_frame(this, executors_and_keys.get(), &feed_tensors,
                                    fetch_tensors);

    if (LogMemory::IsEnabled()) {
      LogMemory::RecordStep(step_id, run_state_args.handle);
    }

    TF_RETURN_IF_ERROR(
        RunInternal(step_id, executors_and_keys->callable_options.run_options(),
                    &call_frame, executors_and_keys.get(), run_metadata));
      
  } else {
    // normal case: original code logic
    // High performance branch:
    {
      tf_shared_lock l(callables_lock_);
      if (handle >= next_callable_handle_) {
        return errors::InvalidArgument("No such callable handle: ", handle);
      }
      executors_and_keys = callables_[handle].executors_and_keys;
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
  
    // Start to transfer stateful data from LPU to HPU via device resource mgr.
    if (last_execute_device_ != "" && last_execute_device_ == "LPU"){
      // N.B.
      // to test transfer time between CPU to 2080 GPU, use API `TransferGPU2CPUStatefulVars`, `TransferCPU2GPUStatefulVars`.
      // to test transfer time between 1080 GPU to 2080 GPU, use API `TransferHPU2LPUStatefulVars`, `TransferLPU2HPUStatefulVars`.
      TransferCPU2GPUStatefulVars();
      //TransferLPU2HPUStatefulVars();
    }
    last_execute_device_ = "HPU"; 
    // End of transferring, start to execute the graph.

    // A specialized CallFrame implementation that takes advantage of the
    // optimized RunCallable interface.
  
    RunCallableCallFrame call_frame(this, executors_and_keys.get(), &feed_tensors,
                                    fetch_tensors);
  
    if (LogMemory::IsEnabled()) {
      LogMemory::RecordStep(step_id, run_state_args.handle);
    }
  
    TF_RETURN_IF_ERROR(
        RunInternal(step_id, executors_and_keys->callable_options.run_options(),
                    &call_frame, executors_and_keys.get(), run_metadata));
  }

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

void DirectSessionsManager::AddDirectSessionAndPriority(
    DirectSession** direct_session){
  // N.B. priority is hardcoded as:
  // 0: user doesn't set
  // 1: low
  // 2: high

  add_mu_.lock();

  // Use the tid to retrieve the priority level number stored in the
  // mapping of {tid : priority_level_num}
  std::thread::id tid = std::this_thread::get_id();

  auto iter = tid_execution_priority_map_.find(tid); 
  if (iter == tid_execution_priority_map_.end()){
    // Priority 0, if the client doesn't define priority by 
    // python API tf.set_execution_priority
    direct_session_priority_map_.insert({*direct_session, 0});
	  // -- debug
	  //std::cout << "Add DirectSession::Pirority (0)" << std::endl;
	  // ~~ debug
  }else{
    // Get the priority level number based on tid
    int direct_session_priority = iter->second;

    direct_session_priority_map_.insert(
        {*direct_session, direct_session_priority});

    // 1 is defined as low priority; 2 is high.
    if (direct_session_priority == 
          DirectSessionPriority::DIRECTSESSION_PRIORITY_LOW){
      low_priority_direct_session_count_.fetch_add(1, 
          std::memory_order_relaxed);

	    // -- debug
      //std::cout << "Add DirectSession::Low Pirority (1), #Low: " <<
		  //  low_priority_direct_session_count_.load(
      //      std::memory_order_relaxed) << std::endl;
	    // ~~ debug
    }

    if (direct_session_priority ==
          DirectSessionPriority::DIRECTSESSION_PRIORITY_HIGH){
      high_priority_direct_session_count_.fetch_add(1,
          std::memory_order_relaxed);

      // -- debug
      //std::cout << "Add DirectSession::High Pirority (2), #High: " << 
      //  high_priority_direct_session_count_.load(
      //      std::memory_order_relaxed) << std::endl;
      // ~~ debug
    }
  }
  add_mu_.unlock();
}

void DirectSessionsManager::DeleteDirectSession(DirectSession* direct_session){
  delete_mu_.lock();
  // TODO: Error handling when not found, although it's not possible
  int direct_session_priority = direct_session_priority_map_[direct_session];

  // 1 is defined as low priority; 
  if(direct_session_priority == 
      DirectSessionPriority::DIRECTSESSION_PRIORITY_LOW){
    low_priority_direct_session_count_.fetch_sub(1,
        std::memory_order_relaxed);

	  // -- debug
    //std::cout << "Delete DirectSession::Low Pirority (1), #Low: " <<
		//  low_priority_direct_session_count_.load(
    //      std::memory_order_relaxed) << std::endl;
	  // ~~ debug
  }

  // 2 is high.
  if(direct_session_priority == 
      DirectSessionPriority::DIRECTSESSION_PRIORITY_HIGH){
    high_priority_direct_session_count_.fetch_sub(1, 
        std::memory_order_relaxed);
    // -- debug
    //std::cout << "Delete DirectSession::High Pirority (2), #High: " << 
    //  high_priority_direct_session_count_.load(
    //      std::memory_order_relaxed) << std::endl;
    // ~~ debug
  }

//  // --debug
//  if(direct_session_priority == 0){
//    std::cout << "Delete DirectSession::Default Pirority (0)" << std::endl;
//  }
//  // ~~debug

  direct_session_priority_map_.erase(direct_session);
  delete_mu_.unlock();
}

int DirectSessionsManager::InquirePriorityByDirectSession(
    const DirectSession* direct_session){
  return direct_session_priority_map_[const_cast<DirectSession*>(direct_session)];
}

}  // namespace tensorflow
