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

// Master implements the service MasterSerivce.
//
// A Master maintains the state of live graph computation
// sessions, each session orchestrates both local and remote devices
// to carry out the graph computation.
//
// A Master knows ahead of time local devices available as
// client devices.
//
// A Master discovers remote devices on-demand and keeps track of
// statistics of those remote devices.
//
// Each session analyzes the graph, places nodes across available
// devices, and ultimately drives the graph computation by initiating
// RunGraph on the workers.

#include "tensorflow/core/distributed_runtime/master.h"

#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/remote_device.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

namespace {
const char* const kGrpcProtocol = "grpc://";
}  // namespace

Master::Master(MasterEnv* env, double session_gc_seconds)
    : env_(env),
      last_1000_steps_(1000),
      step_count_(0),
      session_gc_seconds_(session_gc_seconds),
      recent_request_ids_(10000) {
  // 1.
  // ps callstack to here is :
  //
  // Thread #1 [python] 64502 [core: 32] (Suspended : Breakpoint)
  // 	tensorflow::Master::Master() at master.cc:72 0x7f5f8a9d553d
  // 	tensorflow::GrpcServer::CreateMaster() at grpc_server_lib.cc:423 0x7f5f8a99888b
  // 	tensorflow::GrpcServer::Init() at grpc_server_lib.cc:196 0x7f5f8a9965ac
  //✅tensorflow::GrpcServer::Create() at grpc_server_lib.cc:434 0x7f5f8a9989be
  // 	tensorflow::(anonymous namespace)::GrpcServerFactory::NewServer at grpc_server_lib.cc:469 0x7f5f8a998dfc
  // 	tensorflow::NewServer() at server_lib.cc:76 0x7f5f8ee754d5
  // 	TF_NewServer() at c_api.cc:2,922 0x7f5f8ed34fa8
  // 	_wrap_TF_NewServer() at pywrap_tensorflow_internal.cc:19,069 0x7f5f8a648146
  // 	_PyCFunction_FastCallDict() at methodobject.c:234 0x5623a6693271
  // 	call_function() at ceval.c:4,851 0x5623a671ab70
  // 	<...more frames...>

  // 2.
  //

  // Right now, a master service must be co-located with a device.
  // Otherwise, fetches do not work.
  CHECK(!env->local_devices.empty());

  if (session_gc_seconds_ > 0.0) {
    gc_thread_ = env_->env->StartThread(ThreadOptions(), "TF_master_GC",
                                        [this]() { GC(); });
  } else {
    gc_thread_ = nullptr;
  }
}

Master::~Master() {
  if (gc_thread_) {
    mutex_lock l(mu_);
    shutdown_ = true;
    shutdown_cv_.notify_all();
    delete gc_thread_;
  }
}

// Cleanup unused session.
void Master::GC() {
  Env* env = Env::Default();
  while (true) {
    mutex_lock l(mu_);
    const int kTimeoutMilliseconds = 10 * 1000;  // 10 seconds.
    WaitForMilliseconds(&l, &shutdown_cv_, kTimeoutMilliseconds);
    if (shutdown_) {
      break;
    }
    std::vector<string> handles;
    const int64 num_micros = static_cast<int64>(session_gc_seconds_ * 1000000);
    for (const auto& entry : sessions_) {
      int64 lat = entry.second->last_access_time_usec();
      if (static_cast<int64>(env->NowMicros()) - lat > num_micros) {
        handles.push_back(entry.first);
        auto* sess = entry.second;
        SchedClosure([this, sess]() {
          LOG(WARNING) << "GC session " << sess->handle() << " after "
                       << session_gc_seconds_ << " seconds.  "
                       << "Note that if you are starting multiple replicas "
                       << "on a staggered delay, session_gc_seconds may need "
                       << "to be raised.";
          sess->GarbageCollect();
        });
      }
    }
    for (const auto& handle : handles) sessions_.erase(handle);
  }
}

/** \brief Return the corresponding master session handler to the calling client
 *
 *  \param handle: const string&;
 *
 */
MasterSession* Master::FindMasterSession(const string& handle) {
  // 1.
  // handle
  // "2668d59701ccc3f7"

  // 2.
  // Maps session handles to sessions.
  // std::unordered_map<string, MasterSession*> sessions_ GUARDED_BY(mu_);

  // 3.
  // (gdb) p sessions_
  // $13 = std::unordered_map with 1 element = {["2668d59701ccc3f7"] = 0x7f566c00da60}



  MasterSession* session = nullptr;
  {
    mutex_lock l(mu_);
    session = gtl::FindPtrOrNull(sessions_, handle);
    if (session != nullptr) {
      session->Ref();
    }
  }
  return session;
}

class DeviceFinder {
 public:
  /** \brief
   *
   *  \param[in] device_filters: const protobuf::RepeatedPtrField<string>& ;
   *         device_filters is defined in core/protobuf/config.proto.
   *         message ConfigProto::string device_filters
   *         protobuf::RepeatedPtrField is similar to STL's vector, but include
   *         a number of optimizations found to be useful specifically in the
   *         case of Protocol Buffers.
   *         When any filters are present sessions will ignore all devices which
   *         do not match the filters. Each filter can be partially specified,
   *         e.g. "/job:ps", "/job:worker/replica:3", etc.
   *
   *  \param[in] env: MasterEnv* ;
   *         The master environment class, which holds a bag of pointers to
   *         per-master state. MasterEnv does not own its member pointers.
   *         It stores pointers to 1. OS environment 2. WorkerCacheInterface
   *         3. OpRegistryInterface 4. a list of local_devices 5. a lambda of
   *         factory lambda function for creating master sessions 6. a lambda of
   *         worker_cache_factory 7. CollectiveExecutorMgrInterface.
   *
   *  \param[in] worker_cache: WorkerCacheInterface* ;
   *         WorkerCacheInterface provides interface for 1. list workers' names;
   *         2. list workers' name of a job name; 3. create worker with a name;
   *         4. destory a worker; 5. get device locality information of a device;
   *         6. logging.
   *
   *  \param[out] out_remote: std::vector<std::unique_ptr<Device>>* ;
   *         Device is the hardware device to do computations.
   *
   *  \return Status
   */
  static Status GetRemoteDevices(
      const protobuf::RepeatedPtrField<string>& device_filters,
      MasterEnv* env,
      WorkerCacheInterface* worker_cache,
      std::vector<std::unique_ptr<Device>>* out_remote) {
    // 1.
    // device_filters: const protobuf::RepeatedPtrField<string>&
    // 是什么啊???
    //

    DeviceFinder finder(device_filters, env, worker_cache);

    finder.Start();

    /// wait until get all devices.
    // ===================================================
    TF_RETURN_IF_ERROR(finder.Wait());
    // ===================================================

    finder.GetRemoteDevices(env->local_devices, out_remote);
    // 1.
    // out_remote 是返回值

    return Status::OK();
  }

  /** \brief Enumerates all known workers' target and get qualified candidates
   *         according to the filter requirement.
   *
   *  \param[in] device_filters: const protobuf::RepeatedPtrField<string>& ;
   *         device_filters is defined in core/protobuf/config.proto.
   *         message ConfigProto::string device_filters
   *         protobuf::RepeatedPtrField is similar to STL's vector, but include
   *         a number of optimizations found to be useful specifically in the
   *         case of Protocol Buffers.
   *         When any filters are present sessions will ignore all devices which
   *         do not match the filters. Each filter can be partially specified,
   *         e.g. "/job:ps", "/job:worker/replica:3", etc.
   *
   *  \param[in] env: MasterEnv* ;
   *         The master environment class, which holds a bag of pointers to
   *         per-master state. MasterEnv does not own its member pointers.
   *         It stores pointers to 1. OS environment 2. WorkerCacheInterface
   *         3. OpRegistryInterface 4. a list of local_devices 5. a lambda of
   *         factory lambda function for creating master sessions 6. a lambda of
   *         worker_cache_factory 7. CollectiveExecutorMgrInterface.
   *
   *  \param[in] worker_cache: WorkerCacheInterface* ;
   *         WorkerCacheInterface provides interface for 1. list workers' names;
   *         2. list workers' name of a job name; 3. create worker with a name;
   *         4. destory a worker; 5. get device locality information of a device;
   *         6. logging.
   *
   *  \param[out] workers: std::vector<string>* ;
   *         name of workers.
   */
  static void GetRemoteWorkers(
      const protobuf::RepeatedPtrField<string>& device_filters,
      MasterEnv* env,
      WorkerCacheInterface* worker_cache,
      std::vector<string>* workers) {

    DeviceFinder finder(device_filters, env, worker_cache);
    // 1.
    // DeviceFinder constructor. Enumerates all known workers' target and
    // get qualified candidates according to the filter requirement.

    *workers = finder.targets_;
  }

 private:

  /** \brief DeviceFinder constructor. Enumerates all known workers' target and
   *         get qualified candidates according to the filter requirement.
   *
   *  \param device_filters: const protobuf::RepeatedPtrField<string>& ;
   *         device_filters is defined in core/protobuf/config.proto.
   *         message ConfigProto::string device_filters
   *         protobuf::RepeatedPtrField is similar to STL's vector, but include
   *         a number of optimizations found to be useful specifically in the
   *         case of Protocol Buffers.
   *         When any filters are present sessions will ignore all devices which
   *         do not match the filters. Each filter can be partially specified,
   *         e.g. "/job:ps", "/job:worker/replica:3", etc.
   *
   *  \param env: MasterEnv* ;
   *         The master environment class, which holds a bag of pointers to
   *         per-master state. MasterEnv does not own its member pointers.
   *         It stores pointers to 1. OS environment 2. WorkerCacheInterface
   *         3. OpRegistryInterface 4. a list of local_devices 5. a lambda of
   *         factory lambda function for creating master sessions 6. a lambda of
   *         worker_cache_factory 7. CollectiveExecutorMgrInterface.
   *
   *  \param worker_cache: WorkerCacheInterface* ;
   *         WorkerCacheInterface provides interface for 1. list workers' names;
   *         2. list workers' name of a job name; 3. create worker with a name;
   *         4. destory a worker; 5. get device locality information of a device;
   *         6. logging.
   */
  explicit DeviceFinder(
      const protobuf::RepeatedPtrField<string>& device_filters, MasterEnv* env,
      WorkerCacheInterface* worker_cache)
      : env_(env), worker_cache_(worker_cache) {

    CHECK(worker_cache) << "Worker cache was null!";

    auto process_filter = [this](const string& filter) {
      DeviceNameUtils::ParsedName parsed;
      /// Parse the fullname string filter into a structed data type of
      /// ParsedName which can tell job, replica, task, id and bool of them
      /// immediately.
      if (DeviceNameUtils::ParseFullName(filter, &parsed)) {
        /// std::vector<DeviceNameUtils::ParsedName> filters_ in master.cc
        filters_.push_back(parsed);
      } else {
        LOG(FATAL) << "Skipping invalid filter: " << filter;
      }
    };

    for (const string& filter : device_filters) {
      process_filter(filter);
    }

    // Enumerates all known workers' target. A target name is a
    // prefix of a device name. E.g., /job:mnist/replica:0/task:10.
    // I don't specify the filter, so it goes to the first if branch.
    if (filters_.empty()) {
      // If no filters were specified, we list all known workers in
      // `worker_cache`.
      std::vector<string> workers;

      // \brief Get all workers from all ip:port, and store them to a string in
      //        the of ("/job:", job, "/replica:0/task:", task).
      //
      // \param workers: std::vector<string>*
      //        The return value of ListWorkers, and store them to a string in
      //        the of ("/job:", job, "/replica:0/task:", task).
      //
      // \note call GrpcWorkerCache::ListWorkers,
      //       worker_cache is GrpcWorkerCache.
      //       GrpcWorkerCache derives WorkerCachePartial, which derives
      //       WorkerCacheInterface.
      //
      worker_cache->ListWorkers(&workers);   // 为什么会报错呢?????????
      // 1.
      // output of workers are:
      // p workers
      // $5 = std::vector of length 3, capacity 3 = {"/job:ps/replica:0/task:0", "/job:worker/replica:0/task:0", "/job:worker/replica:0/task:1"}
      //
      // p workers
      // $11 = std::vector of length 3, capacity 3 = {"/job:ps/replica:0/task:0", "/job:worker/replica:0/task:0", "/job:worker/replica:0/task:1"}

      std::swap(workers, targets_);
    } else {
      // 未进入!

      // When applying filters, we must include the local worker, even if it
      // does not match any of the filters.
      CHECK_GT(env_->local_devices.size(), 0) << "No local devices provided.";
      const string& local_device_name = env_->local_devices[0]->name();
      DeviceNameUtils::ParsedName local_parsed_name;
      CHECK(DeviceNameUtils::ParseFullName(local_device_name,
                                           &local_parsed_name));
      bool all_filters_have_job = true;
      std::unordered_set<string> filter_job_names({local_parsed_name.job});
      for (const DeviceNameUtils::ParsedName& filter : filters_) {
        all_filters_have_job = all_filters_have_job && filter.has_job;
        if (filter.has_job) {
          filter_job_names.insert(filter.job);
        }
      }

      std::vector<string> workers;
      if (all_filters_have_job) {
        // If all of the device filters have a job specified, then we only need
        // to list the workers in the jobs named in the filter, because a worker
        // in any other job would not match any filter.
        for (const string& job_name : filter_job_names) {
          VLOG(2) << "Selectively listing workers in job: " << job_name;
          std::vector<string> workers_in_job;
          worker_cache->ListWorkersInJob(job_name, &workers_in_job);
          workers.insert(workers.end(), workers_in_job.begin(),
                         workers_in_job.end());
        }
      } else {
        // If any of the device filters does not have a job specified, then we
        // must list the workers from all jobs.
        VLOG(2) << "Listing workers in all jobs because some device "
                << "filter has no job specified. Filters were:";
        if (device_filters.empty()) {
          VLOG(2) << "- <NO FILTERS>";
        } else {
          for (const string& filter : device_filters) {
            VLOG(2) << "- " << filter;
          }
        }
        worker_cache->ListWorkers(&workers);
      }
      for (const string& name : workers) {
        if (MatchFilters(name) ||
            DeviceNameUtils::IsSameAddressSpace(name, local_device_name)) {
          targets_.push_back(name);
        }
      }
    }
    seen_targets_.assign(targets_.size(), false);
    // 1.
    // seen_targets_ size is 3 in my example.
    // targets_: std::vector of length 3, capacity 3 = {"/job:ps/replica:0/task:0", "/job:worker/replica:0/task:0", "/job:worker/replica:0/task:1"}
  }

  ~DeviceFinder() {
    for (Device* dev : found_) delete dev;
  }

  /** \brief Start finding all devices from all ip:port from all targets.
   */
  void Start() {
    {
      mutex_lock l(mu_);
      /// targets_ is all job:task_index from all ip:port .
      num_pending_ = targets_.size();
      if (num_pending_ == 0) {
        pending_zero_.notify_all();
      }
    }
    // Talk to all workers to get the list of available devices.
    using std::placeholders::_1;
    using std::placeholders::_2;
    for (size_t i = 0; i < targets_.size(); ++i) {
      // 1.
      // targets_ 是:
      // std::vector of length 3, capacity 3 = {"/job:ps/replica:0/task:0", "/job:worker/replica:0/task:0", "/job:worker/replica:0/task:1"}

      // TODO(mrry): Propagate a timeout here, since `this->WhenFound()` may
      // never be called.
      /// Create a work mapping to a target, i.e., job:task_index.
      NewRemoteDevices(
        env_->env,
        worker_cache_,
        targets_[i],
        std::bind(&ME::WhenFound, this, i, _1, _2));
      // 1.
      // NewRemoteDevices
      // tensorflow/core/distributed_runtime/remote_device.cc

      // 2.
      // targets_ are
      //  std::vector of length 3, capacity 3 = {"/job:ps/replica:0/task:0", "/job:worker/replica:0/task:0", "/job:worker/replica:0/task:1"}
      // 逐个放到 NewRemoteDevices 里面.

      // 3.
      // WhenFound
      // 在本文件下面.

      // 4.
      // ME:
      // typedef DeviceFinder ME

    }
  }

  // Every `kLoggingPeriodMs`, while the DeviceFinder is still waiting
  // to hear from workers, log a list of the workers who have not
  // responded.
  const int32 kLoggingPeriodMs = 10 * 1000;

  // CreateSession will wait for response from the unresponsive workers.
  // ===================================================
  Status Wait() {
  // ===================================================
    mutex_lock l(mu_);
    // TODO(mrry): Propagate a timeout here, since `num_pending_` may
    // never become zero.
    while (num_pending_ != 0) {
      /// condition_variable pending_zero_; is defined in master.cc
      pending_zero_.wait_for(l, std::chrono::milliseconds(kLoggingPeriodMs));

      if (num_pending_ != 0) {
        for (size_t i = 0; i < targets_.size(); ++i) {
          if (!seen_targets_[i]) {
            LOG(INFO)
                << "CreateSession still waiting for response from worker: "
                << targets_[i];
          }
        }
      }
    }
    return status_;
  }

  /** \brief Get all devices from all workers remotely.
   *
   *  \param local: const std::vector<Device*>&
   *
   *  \param remote: std::vector<std::unique_ptr<Device>>*
   *
   *  \remark No return value.
   */
  // The caller takes the ownership of returned remote devices.
  void GetRemoteDevices(const std::vector<Device*>& local,
                        std::vector<std::unique_ptr<Device>>* remote) {
    // 1.
    // remote : output

    std::unordered_set<string> names(local.size());
    for (Device* dev : local) names.insert(dev->name());
    mutex_lock l(mu_);
    for (Device* dev : found_) {
      // 1.
      // std::vector<Device*> found_ GUARDED_BY(mu_);
      // std::vector of length 10, capacity 12 = {0x7f57f40066d0,
      // 0x7f57f4006a10, 0x7f5800006aa0, 0x7f5800006de0, 0x7f5800007190,
      // 0x7f58000075b0, 0x7f566c00f6e0, 0x7f566c00d570, 0x7f566c00f270,
      // 0x7f566c00d740}

      const string& name = dev->name();
      // 1.
      // name
      // e.g.,
      // "/job:ps/replica:0/task:0/device:XLA_CPU:0"

      if (names.insert(name).second && MatchFilters(name)) {
        remote->push_back(std::unique_ptr<Device>(dev));

      } else {
        delete dev;
      }
    }
    found_.clear();
  }

  typedef DeviceFinder ME;
  const MasterEnv* env_;
  WorkerCacheInterface* worker_cache_;
  // 1.
  // core/distributed_runtime/worker_cache.h:32:
  // class WorkerCacheInterface

  std::vector<DeviceNameUtils::ParsedName> filters_;

  mutex mu_;
  /// num of unresponsive workers when create session.
  int num_pending_ GUARDED_BY(mu_);
  condition_variable pending_zero_;
  std::vector<Device*> found_ GUARDED_BY(mu_);
  // 1.
  // 找到的 devices


  // List of targets to be contacted by this DeviceFinder. The
  // respective `bool` in `seen_targets_` indicates whether we have
  // heard from this target or not.
  std::vector<string> targets_;
  std::vector<bool> seen_targets_ GUARDED_BY(mu_);
  Status status_;

  /** \brief Specify the range of [devices->begin(), devices->end()), inserted
   *         at the position of found_.end().
   *
   *  \param target_index: int ;
   *
   *  \param s: const Status& ;
   *
   *  \param devices: std::vector<Device*>* ;
   *
   *  \remark No return value.
   */
  void WhenFound(int target_index, const Status& s,
                 std::vector<Device*>* devices) {
    // 1.
    // size of devices is 2

    mutex_lock l(mu_);
    /// std::vector<bool> seen_targets_  defined in master.cc
    seen_targets_[target_index] = true;

    if (!s.ok()) {
      LOG(ERROR) << "CreateSession failed because worker "
                 << targets_[target_index] << " returned error: " << s;
      status_.Update(s);
    } else {
      found_.insert(found_.end(), devices->begin(), devices->end());
      // 1.
      // std::vector<Device*> found_ GUARDED_BY(mu_); 定义在上面
      // 找到的 devices

      // 2.
      // Specify the range of [devices->begin(), devices->end()), inserted at
      // the position of found_.end().

      devices->clear();
    }

    --num_pending_;
    if (num_pending_ == 0) {
      pending_zero_.notify_all();
      // 1.
      // pending_zero_: condition_variable

    }
  }

  // Returns true iff the set of devices allowed by 'x' intersects
  // with the set of devices allowed by 'y'.
  bool Intersects(const DeviceNameUtils::ParsedName& x,
                  const DeviceNameUtils::ParsedName& y) {
    return (!x.has_job || !y.has_job || x.job == y.job) &&
           (!x.has_replica || !y.has_replica || x.replica == y.replica) &&
           (!x.has_task || !y.has_task || x.task == y.task) &&
           (!x.has_type || !y.has_type || x.type == y.type) &&
           (!x.has_id || !y.has_id || x.id == y.id);
  }

  // Returns true iff 'name' matches one of the filters_.
  bool MatchFilters(const string& name) {
    if (filters_.empty()) return true;
    DeviceNameUtils::ParsedName x;
    if (DeviceNameUtils::ParseFullName(name, &x)) {
      for (const auto& filter : filters_) {
        if (Intersects(x, filter)) return true;
      }
    }
    return false;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceFinder);
};

/** \brief Master grpc server side handler function.
 *         Create a session in another thread, requested by a client to a master
 *         at Server side. Then, Master sends a request to worker to create a
 *         session.
 *
 *  \param[in] req: CreateSessionRequest* ;
 *         message CreateSessionRequest includes
 *         1. GraphDef: NodeDef, versionDef, FunctionDefLibrary;
 *         2. ConfigProto, including Session configuration parameters.
 *         3. The target string used from the client's perspective.
 *            e.g., "grpc://localhost:2223"
 *
 *  \param[out] resp: CreateSessionResponse* ;
 *         message CreateSessionResponse includes
 *         1. session_handle string, and
 *         2. graph version for the subsequent call to ExtendSession.
 *
 *  \param[in] done: MyClosure;
 *         A lambda function to be called at the very end of CreateSession to
 *         send response back to the client
 *         ```cpp
 *         [call, rewritten_req](const Status& status) {
 *           call->SendResponse(ToGrpcStatus(status));
 *           delete rewritten_req;
 *         }
 *         ```
 *         from grpc_master_service.cc
 *
 *  \note SchedClosure: std::thread;
 *        Start a new thread to run the SchedClosure lambda function, and
 *        then detach this thread.
 *
 *  \details
 *   1. class Master must be used to manage all master sessions requested by
 *      all clients.
 *   - std::unordered_map< string, MasterSession * > sessions_
 *
 *   2. It is called by GrpcServer::master_impl_
 *      - master_impl_: std::unique_ptr< Master >
 *      - Master is constructed in GrpcServer::CreateMaster
 *      - master_session_factory is initialized when grpc server is initialized.
 */
void Master::CreateSession(const CreateSessionRequest* req,
                           CreateSessionResponse* resp,
                           MyClosure done) {
  // SchedClosure
  //
  // SchedClosure is at core/common_runtime/process_util.cc
  // It is a function, its input is a lambda.
  SchedClosure([this, req, resp, done]() {
    // The following code is executed by another thread.
    Status status;

    WorkerCacheFactoryOptions worker_cache_factory_options;
    // 1.
    // worker_cache_factory_options: WorkerCacheFactoryOptions
    // To fill in fields in the struct of WorkerCacheFactoryOptions
    // Worker cache factory options are pointers to cluster, job, task,
    // protocol, host:port information to be used in
    // GrpcServer::WorkerCacheFactory to Construct a new GrpcWorkerCache
    // which can access all workers responsible for graph computing.

    string grpc_protocol("grpc");
    worker_cache_factory_options.protocol = &grpc_protocol;

    auto call_done = gtl::MakeCleanup([&status, &done] { done(status); });
    // 1.
    // gtl::MakeCleanup can make an RAII cleanup object that will call
    // the lambda function when it go out of this scope.

    status = ValidateExternalGraphDefSyntax(req->graph_def());

    if (!status.ok()) return;

    // The following 4 variables are set differently, depending on whether this
    // session uses a client-provided clusterspec or not.

    WorkerCacheInterface* worker_cache = nullptr;
    // 1.
    // worker_cache: WorkerCacheInterface*
    // WorkerCacheInterface provides interface for 1. list workers' names;
    // 2. list workers' name of a job name; 3. create worker with a name;
    // 4. destory a worker; 5. get device locality information of a device;
    // 6. logging.

    std::unique_ptr<WorkerCacheInterface> worker_cache_ptr;
    // worker_cache_ptr will be null except if this session is using a
    // client-supplied ClusterDef (ClusterSpec propagation).

    std::unique_ptr<DeviceSet> device_set;
    // DeviceSet:
    // DeviceSet is a container class for managing the various types of
    // devices used by a model, registering all device name and Device object.
    // utilities includes 1. add device, 2. lookup device by name or type.

    // TODO(saeta): Convert to std::make_unique when available.
    std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devices(
        new std::vector<std::unique_ptr<Device>>());

    if (req->config().has_cluster_def()) {
      // 未进入!!!

      worker_cache_factory_options.cluster_def = &req->config().cluster_def();

      // Set the server_def's job_name and task_index fields.
      string normalized_string;
      string grpc_protocol(kGrpcProtocol);
      if (req->target().compare(0, grpc_protocol.length(), grpc_protocol) ==
          0) {
        normalized_string =
            req->target().substr(grpc_protocol.length(), string::npos);
      } else {
        normalized_string = req->target();
      }
      for (auto&& job : req->config().cluster_def().job()) {
        /// \note job.tasks(): map<int32, string> tasks;
        /// Mapping from task ID to "hostname:port" string.
        for (auto&& task : job.tasks()) {
          /// task.second is "hostname:port".
          if (task.second == normalized_string) {
            if (worker_cache_factory_options.job_name != nullptr) {
              status = errors::InvalidArgument(
                  "Found multiple matching tasks that correspond to "
                  "to the master. Master target: '",
                  req->target(), "'. ClusterDef: ",
                  req->config().cluster_def().ShortDebugString());
              LOG(ERROR) << status;
              return;
            }
            if (env_->local_devices[0]->parsed_name().job == job.name() &&
                env_->local_devices[0]->parsed_name().task == task.first) {
              // TODO(b/37868888): Remove this limitation when resolved
              status = errors::InvalidArgument(
                  "The ClusterSpec names the job and task index to be the same "
                  "names that were provided when the server booted. This is "
                  "currently not allowed. Job: ",
                  job.name(), ", task index: ", task.first);
              return;
            }
            worker_cache_factory_options.job_name = &job.name();
            worker_cache_factory_options.task_index = task.first;
          }
        }
      }

      // Create the worker cache from the computed server_def.
      status = env_->worker_cache_factory(worker_cache_factory_options,
                                          &worker_cache);
      // 1.
      // Indirectly call the GrpcServer::WorkerCacheFactory in
      // distributed_runtime/rpc/grpc_server_lib.cc
      // to construct a new GrpcWorkerCache which can access all workers
      // responsible for graph computing.

      if (!status.ok()) return;

      worker_cache_ptr = std::unique_ptr<WorkerCacheInterface>(worker_cache);

      //=======================================================================
      // Ping all the workers and build the list of devices that the
      // session will use.
      //=======================================================================
      // Q. What if a second remote session is created? If there is no left
      // devices for the second session?
      // A. I guess the 2nd session creation will also get the same
      // devices.
      //=======================================================================
      status =
          DeviceFinder::GetRemoteDevices(
            req->config().device_filters(),
            env_,
            worker_cache,
            remote_devices.get());
      //=======================================================================

      if (!status.ok()) return;

      device_set.reset(new DeviceSet);

      for (auto&& d : *remote_devices) {
        device_set->AddDevice(d.get());

        DeviceNameUtils::ParsedName name = d->parsed_name();

        if (name.job == *worker_cache_factory_options.job_name &&
            name.task == worker_cache_factory_options.task_index &&
            name.type == "CPU" && name.id == 0) {
          device_set->set_client_device(d.get());
        }
      }

    } else {
      // 进入!

      worker_cache = env_->worker_cache;
      // 1.
      // worker_cache: WorkerCacheInterface*, nullptr

      // 2.

      // =============================================================
      // Ping all the workers and build the list of devices that the
      // session will use.
      status =
          DeviceFinder::GetRemoteDevices(
            req->config().device_filters(),
            env_,
            worker_cache,
            remote_devices.get());
      // =============================================================
      // 1.
      // callstack
      //
      // 0  tensorflow::NewGrpcRemoteWorker (channel=..., completion_queue=0x55ab5e705db0, callback_threadpool=0x55ab5e707870, logger=0x55ab5e705cd0) at
      //  tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc:313
      //
      // 1  0x00007f21dadce250 in tensorflow::(anonymous namespace)::GrpcWorkerCache::CreateWorker (this=0x55ab5e705c60, target=...) at
      //  tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc:80
      //
      // 2  0x00007f21dae28566 in tensorflow::NewRemoteDevices(tensorflow::Env*, tensorflow::WorkerCacheInterface*, std::string const&, std::function<void (tensorflow::Status const&, std::vector<tensorflow::Device*, std::allocator<tensorflow::Device*> >*)>) (env=0x55ab5b737820, worker_cache=0x55ab5e705c60, worker_name=..., done=...) at
      //  tensorflow/core/distributed_runtime/remote_device.cc:59
      //
      // 3  0x00007f21d69d736f in tensorflow::DeviceFinder::Start (this=0x7f1d11ffa6e0) at
      //  tensorflow/core/distributed_runtime/master.cc:249
      //
      // 4  0x00007f21d69d5d9f in tensorflow::DeviceFinder::GetRemoteDevices (device_filters=..., env=0x55ab5cfc9b68, worker_cache=0x55ab5e705c60, out_remote=0x7f1d080014f0) at
      //  tensorflow/core/distributed_runtime/master.cc:140 还在这个文件里面.
      //
      // 5  0x00007f21d69d895c in tensorflow::Master::<lambda()>::operator()(void) const (__closure=0x55ab5e9b56c0) at
      //  tensorflow/core/distributed_runtime/master.cc:444

      // 2.
      // GetRemoteDevices
      //


      if (!status.ok()) return;
      device_set.reset(new DeviceSet);
      for (auto&& d : *remote_devices) {
        device_set->AddDevice(d.get());
      }

      int num_local_devices = 0;

      for (Device* d : env_->local_devices) {
        device_set->AddDevice(d);
        if (num_local_devices == 0) {
          // Uses the first local device as the client device.
          device_set->set_client_device(d);
        }
        num_local_devices++;
      }
    }

    CHECK(device_set->client_device()) << "No client device found. Missing "
                                       << "CPU:0 device?";

    SessionOptions options;
    options.config = req->config();

    std::vector<string> filtered_worker_list;
    DeviceFinder::GetRemoteWorkers(req->config().device_filters(), env_,
                                   worker_cache, &filtered_worker_list);

    MasterSession* session = env_->master_session_factory(
        options,
        env_,
        std::move(remote_devices),
        std::move(worker_cache_ptr),
        std::move(device_set),
        std::move(filtered_worker_list));
    // 1.
    // master_session_factory
    // lambda function defined in rpc/grpc_server_lib.cc
    //
    // master_env_.master_session_factory =
    //     [config, stats_factory](
    //         SessionOptions options,
    //         const MasterEnv* env,
    //         std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs,
    //         std::unique_ptr<WorkerCacheInterface> worker_cache,
    //         std::unique_ptr<DeviceSet> device_set,
    //         std::vector<string> filtered_worker_list) {
    //       // 函数体:
    //       options.config.MergeFrom(config);
    //       return new MasterSession(options, env, std::move(remote_devs),
    //                                std::move(worker_cache), std::move(device_set),
    //                                std::move(filtered_worker_list),
    //                                stats_factory);
    //     };

    GraphDef* gdef =
        const_cast<CreateSessionRequest*>(req)->mutable_graph_def();
    // 1.
    // 这里面就有图啊
    // 就是需要被构造的图啊

    /// \note session: MasterSession*
    ///       MasterSession::Create in master_session.cc
    ///
    /// \fn Create
    ///  MasterSession::Create is responsible for creating Worker Session
    ///  asynchronously.
    status = session->Create(gdef, worker_cache_factory_options);
    //

    if (!status.ok()) {
      session->Close().IgnoreError();
      session->Unref();
      return;
    }
    resp->set_session_handle(session->handle());
    // Insert into the session map, which takes ownership of the session.
    {
      mutex_lock l(mu_);
      /// \todo What if a second remote session is created? If there is no left
      ///       devices for the second session?
      CHECK(sessions_.insert({session->handle(), session}).second);
    }
  });
}

void Master::ExtendSession(const ExtendSessionRequest* req,
                           ExtendSessionResponse* resp, MyClosure done) {
  auto session = FindMasterSession(req->session_handle());
  if (session == nullptr) {
    done(errors::Aborted("Session ", req->session_handle(), " is not found."));
    return;
  }

  SchedClosure([session, req, resp, done]() {
    Status status = ValidateExternalGraphDefSyntax(req->graph_def());
    if (status.ok()) {
      status = session->Extend(req, resp);
    }
    session->Unref();
    done(status);
  });
}

void Master::PartialRunSetup(const PartialRunSetupRequest* req,
                             PartialRunSetupResponse* resp, MyClosure done) {
  Status s = recent_request_ids_.TrackUnique(req->request_id(),
                                             "PartialRunSetup (Master)", *req);
  if (!s.ok()) {
    done(s);
    return;
  }
  auto session = FindMasterSession(req->session_handle());
  if (session == nullptr) {
    done(errors::Aborted("Session ", req->session_handle(), " is not found."));
    return;
  }

  SchedClosure([session, req, resp, done]() {
    Status s = session->PartialRunSetup(req, resp);
    session->Unref();
    done(s);
  });
}

/** \brief Server side handler: Client send a request to master to run one step via a new thread.
 *
 *  \param opts: CallOptions* ;
 *
 *  \param req: const RunStepRequestWrapper* ;
 *         Wrapper classes for the `MasterService.RunStep` request message, for
 *         performance.
 *
 *  \param resp: MutableRunStepResponseWrapper* ;
 *
 *  \param done: MyClosure;
 *         A callback function ran by a new thread.
 *
 *  \remark No return value.
 *
 */
void Master::RunStep(CallOptions* opts,
                     const RunStepRequestWrapper* req,
                     MutableRunStepResponseWrapper* resp,
                     MyClosure done) {
  // 1.
  // 所以, master_impl_->RunStep 的 RunStep 是怎么转交给 worker 的?
  // ...

  Status s = recent_request_ids_.TrackUnique(req->request_id(),
                                             "RunStep (Master)", req);

  if (!s.ok()) {
    done(s);
    return;
  }

  auto start_time = env_->env->NowMicros();

  auto session = FindMasterSession(req->session_handle());
  // 1.
  // session: MasterSession
  // core/distributed_runtime/master_session.h:42:
  // class MasterSession : public core::RefCounted


  if (session == nullptr) {
    done(errors::Aborted("Session ", req->session_handle(), " is not found."));
    return;
  }

  /// Start a new thread to run the closure.
  SchedClosure([this, start_time, session, opts, req, resp, done]() {
    // 1.
    // 1.1
    // SchedClosure 功能:
    // Start a new thread to run the closure.

    // 1.2.
    // SchedClosure 函数定义:
    //
    // 接口:
    // core/platform/env.h:286:
    // virtual void SchedClosure(std::function<void()> closure) = 0;
    //
    // 实现:
    // core/platform/posix/env.cc:118:
    // void SchedClosure(std::function<void()> closure) override {
    //
    // 其他:
    // core/common_runtime/process_util.cc:109:
    // void SchedClosure(std::function<void()> closure)

    // =============================================================
    Status status = session->Run(opts, *req, resp);
    // =============================================================
    // 1.
    // req, resp 的概念如下完美解释.

    // 2.
    // RPC is a request–response protocol.
    // Wiki: https://en.wikipedia.org/wiki/Remote_procedure_call
    //
    // An RPC is initiated by the client, which sends a request message to a
    // known remote server to execute a specified procedure with supplied
    // parameters.
    //
    // Sequence of events
    // - The client calls the client stub. The call is a local procedure call, with parameters pushed on to the stack in the normal way.
    // - The client stub packs the parameters into a message and makes a system call to send the message. Packing the parameters is called marshalling.
    // - The client's local operating system sends the message from the client machine to the server machine.
    // - The local operating system on the server machine passes the incoming packets to the server stub.
    // - The server stub unpacks the parameters from the message. Unpacking the parameters is called unmarshalling.
    // - Finally, the server stub calls the server procedure. The reply traces the same steps in the reverse direction.

    // 2.
    // session :  real type = tensorflow::MasterSession *
    // tensorflow/core/distributed_runtime/master_session.cc

    // 3.
    // session->Run(opts, *req, resp)
    // 定义在
    // tensorflow/core/distributed_runtime/master_session.cc
    //
    // Status MasterSession::Run(CallOptions* opts, const RunStepRequestWrapper& req,
    //                          MutableRunStepResponseWrapper* resp) {
    //

    // 4.
    // 思想:
    // RegisterPartitions 切割图, 然后去执行.
    // 关键函数: RunPartitionsHelper in
    // tensorflow/core/distributed_runtime/master_session.cc
    // - part.worker->RunGraphAsync

    // 5.
    // 1 step of the above log: (a lot!!!)
    // https://gist.github.com/shizukanaskytree/d3b382ff22187520d6b481f166b92541

    session->Unref();
    uint64 done_time = env_->env->NowMicros();
    done(status);
    // 1.
    // done 函数是什么
    // tensorflow/core/distributed_runtime/local_master.cc
    // LocalMaster::RunStep 内
    //
    // Notification n;
    // [&n, &ret](const Status& s) {
    //   ret.Update(s);
    //   n.Notify();
    // }

    mutex_lock l(mu_);
    last_1000_steps_.AddValue((done_time - start_time) / 1e9);
    ++step_count_;
  });
}

void Master::CloseSession(const CloseSessionRequest* req,
                          CloseSessionResponse* resp, MyClosure done) {
  MasterSession* session = nullptr;
  {
    mu_.lock();
    auto iter = sessions_.find(req->session_handle());
    if (iter == sessions_.end()) {
      mu_.unlock();
      done(errors::Aborted(
          "Session ", req->session_handle(),
          " is not found. Possibly, this master has restarted."));
      return;
    }
    // NOTE(mrry): One reference to the session is transferred from
    // `sessions_[req->session_handle()]` to `session`.
    session = iter->second;
    sessions_.erase(iter);
    mu_.unlock();
  }

  // Session Close() blocks on thread shutdown. Therefore, we need to
  // delete it in non-critical thread.
  SchedClosure([session, done]() {
    Status s = session->Close();
    session->Unref();
    done(s);
  });
}

void Master::ListDevices(const ListDevicesRequest* req,
                         ListDevicesResponse* resp, MyClosure done) {
  SchedClosure([this, req, resp, done]() {
    if (!req->session_handle().empty()) {
      auto session = FindMasterSession(req->session_handle());
      if (session == nullptr) {
        done(errors::InvalidArgument(
            "Session ", req->session_handle(),
            " is not found. Possibly, this master has restarted."));
        return;
      }
      core::ScopedUnref ref(session);
      Status s = session->ListDevices(resp);
      done(s);
      return;
    }
    std::vector<std::unique_ptr<Device>> remote_devices;
    Status s = DeviceFinder::GetRemoteDevices({}, env_, env_->worker_cache,
                                              &remote_devices);
    if (s.ok()) {
      for (Device* dev : env_->local_devices) {
        *(resp->add_local_device()) = dev->attributes();
      }
      for (auto&& dev : remote_devices) {
        *(resp->add_remote_device()) = dev->attributes();
      }
    }
    done(s);
  });
}

void Master::CleanupWorkers(const ResetRequest& reset) {
  std::vector<string> worker_names;
  DeviceFinder::GetRemoteWorkers(reset.device_filters(), env_,
                                 env_->worker_cache, &worker_names);
  if (!worker_names.empty()) {
    const int num_workers = worker_names.size();
    std::vector<Notification> n(num_workers);
    CleanupAllRequest req;
    (*req.mutable_container()) = reset.container();
    std::vector<CleanupAllResponse> resp(num_workers);
    int c = 0;
    for (int i = 0; i < num_workers; ++i) {
      const string& worker_name = worker_names[i];
      auto worker = env_->worker_cache->CreateWorker(worker_name);
      if (worker) {
        worker->CleanupAllAsync(
            &req, &resp[i], [this, &n, worker_name, worker, c](Status s) {
              TF_CHECK_OK(s);
              env_->worker_cache->ReleaseWorker(worker_name, worker);
              n[c].Notify();
            });
      } else {
        n[c].Notify();
      }
      ++c;
    }
    for (size_t i = 0; i < n.size(); ++i) {
      n[i].WaitForNotification();
    }
  }
}

void Master::Reset(const ResetRequest* req, ResetResponse* resp,
                   MyClosure done) {
  // Vector to hold the session pointers present in the sessions_
  // (string->Session*) map.
  std::vector<MasterSession*> sessions_to_close;
  {
    mutex_lock l(mu_);
    // NOTE(mrry): Transfer one reference to each session from the
    // `sessions_` map to the `sessions_to_close` vector.
    for (const auto& entry : sessions_) {
      sessions_to_close.push_back(entry.second);
    }
    sessions_.clear();
  }

  CleanupWorkers(*req);

  SchedClosure([sessions_to_close, done]() {
    Status s;
    for (MasterSession* session : sessions_to_close) {
      s.Update(session->Close());
      session->Unref();
    }
    done(s);
  });
}

void Master::MakeCallable(const MakeCallableRequest* req,
                          MakeCallableResponse* resp, MyClosure done) {
  Status s = recent_request_ids_.TrackUnique(req->request_id(),
                                             "MakeCallable (Master)", *req);
  if (!s.ok()) {
    done(s);
    return;
  }
  auto session = FindMasterSession(req->session_handle());
  if (session == nullptr) {
    done(errors::Aborted("Session ", req->session_handle(), " is not found."));
    return;
  }

  SchedClosure(std::bind(
      [session, req, resp](MyClosure done) {
        Status s = session->MakeCallable(*req, resp);
        session->Unref();
        done(s);
      },
      std::move(done)));
}

void Master::RunCallable(CallOptions* opts, const RunCallableRequest* req,
                         RunCallableResponse* resp, MyClosure done) {
  Status s = recent_request_ids_.TrackUnique(req->request_id(),
                                             "RunCallable (Master)", *req);
  if (!s.ok()) {
    done(s);
    return;
  }
  auto session = FindMasterSession(req->session_handle());
  if (session == nullptr) {
    done(errors::Aborted("Session ", req->session_handle(), " is not found."));
    return;
  }

  SchedClosure(std::bind(
      [session, opts, req, resp](MyClosure done) {
        Status s = session->RunCallable(opts, *req, resp);
        session->Unref();
        done(s);
      },
      std::move(done)));
}

void Master::ReleaseCallable(const ReleaseCallableRequest* req,
                             ReleaseCallableResponse* resp, MyClosure done) {
  auto session = FindMasterSession(req->session_handle());
  if (session == nullptr) {
    done(errors::Aborted("Session ", req->session_handle(), " is not found."));
    return;
  }

  SchedClosure(std::bind(
      [session, req, resp](MyClosure done) {
        Status s = session->ReleaseCallable(*req, resp);
        session->Unref();
        done(s);
      },
      std::move(done)));
}

}  // end namespace tensorflow
