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

#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"

#include <unordered_map>

#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_cache_partial.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

namespace {
/** \class GrpcWorkerCache
 *
 *  \details GrpcWorkerCache is a derived class of WorkerCacheInterface, which
 *           provides interface for 1. list workers' names; 2. list workers' name
 *           of a job name; 3. create worker with a name; 4. destory a worker;
 *           5. get device locality information of a device; 6. logging.
 */
class GrpcWorkerCache : public WorkerCachePartial {
 public:
  // TODO(ncteisen): consider adding a config var or flag for this
  static constexpr const size_t kGrpcWorkerCacheThreadCount = 8;

  explicit GrpcWorkerCache(std::shared_ptr<GrpcChannelCache> channel_cache,
                           WorkerInterface* local_worker,
                           const string& local_target)
      : local_target_(local_target),
        local_worker_(local_worker),
        channel_cache_(channel_cache),
        threads_(kGrpcWorkerCacheThreadCount),
        next_round_robin_assignment_(0) {
    // NOTE: We don't yet have any reason to assign NUMA affinity to this
    // ThreadPool.  If there's only a single NIC it shouldn't make any
    // difference since presumably it is handling memory from all nodes.
    ThreadOptions options;
    options.numa_node = port::kNUMANoAffinity;
    const int kNumCallbackThreads = 10;
    callback_threadpool_.reset(new thread::ThreadPool(
        Env::Default(), options, "grpc_wcache_callback", kNumCallbackThreads,
        false /*low_latency_hint*/, nullptr /*allocator*/));
  }

  // Explicit destructor to control destruction order.
  ~GrpcWorkerCache() override {
    threads_.clear();  // Blocks until threads exit.
  }

  /** \brief Get all workers from all ip:port, and store them to a string in the
   *         of ("/job:", job, "/replica:0/task:", task).
   *
   *  \param workers: std::vector<string>*
   *         The return value of ListWorkers, and store them to a string in the
   *         of ("/job:", job, "/replica:0/task:", task).
   *
   *  \remark No return values.
   */
  void ListWorkers(std::vector<string>* workers) const override {
    /// std::shared_ptr<GrpcChannelCache> channel_cache_;
    /// But, channel_cache_ will call SparseGrpcChannelCache::ListWorkers at
    /// grpc_channel.cc.
    /// SparseGrpcChannelCache --> CachingGrpcChannelCache --> GrpcChannelCache
    channel_cache_->ListWorkers(workers);
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) const override {
    channel_cache_->ListWorkersInJob(job_name, workers);
  }

  /** \brief Create a remote worker and initialize their grpc services.
   *
   *  \param target: const string&
   *         worker name.
   */
  WorkerInterface* CreateWorker(const string& target) override {
    // 1.
    // target example:
    // $21 = "/job:worker/replica:0/task:1"

    if (target == local_target_) {
      return local_worker_;
    } else {
      SharedGrpcChannelPtr channel = channel_cache_->FindWorkerChannel(target);
      if (!channel) return nullptr;
      return NewGrpcRemoteWorker(
          channel,
          threads_[AssignWorkerToThread(target)].completion_queue(),
          callback_threadpool_.get(),
          &logger_);
      // 1.
      // NewGrpcRemoteWorker 在哪?
      // tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.cc:32:
      // WorkerInterface* NewGrpcRemoteWorker(SharedGrpcChannelPtr channel,

      // NewGrpcRemoteWorker 做了什么?

    }
  }

  /** \brief
   *
   *  \param target: const string&
   *
   *  \param worker: WorkerInterface*
   *
   */
  void ReleaseWorker(const string& target, WorkerInterface* worker) override {
    if (target == local_target_) {
      CHECK_EQ(worker, local_worker_)
          << "Releasing a worker that was not returned by this WorkerCache";
    } else {
      WorkerCacheInterface::ReleaseWorker(target, worker);
    }
  }

  void SetLogging(bool v) override { logger_.SetLogging(v); }

  void ClearLogs() override { logger_.ClearLogs(); }

  bool RetrieveLogs(int64 step_id, StepStats* ss) override {
    return logger_.RetrieveLogs(step_id, ss);
  }

 private:
  // Thread wrapping class that drives work over a single gRPC
  // CompletionQueue.
  class GrpcWorkerCacheThread {
   public:
    /** \brief GrpcWorkerCacheThread constructor used in class GrpcWorkerCache
     *
     *  \details reset value of thread_ to Thread*, return from StartThread.
     *           The lambda function is starting to run.
     */
    GrpcWorkerCacheThread() {
      thread_.reset(Env::Default()->StartThread(
          ThreadOptions(), "grpc_worker_cache", [this]() {
            void* tag;
            bool ok;
            while (completion_queue_.Next(&tag, &ok)) {
              GrpcClientCQTag* callback_tag =
                  static_cast<GrpcClientCQTag*>(tag);
              callback_tag->OnCompleted(ok);
            }
          }));
    }

    ~GrpcWorkerCacheThread() {
      completion_queue_.Shutdown();
      thread_.reset();
    }

    ::grpc::CompletionQueue* completion_queue() { return &completion_queue_; }

   private:
    ::grpc::CompletionQueue completion_queue_;
    std::unique_ptr<Thread> thread_;
  };  // GrpcWorkerCacheThread

  size_t AssignWorkerToThread(const string& target) {
    // Round-robin target assignment, but keeps the same target on the same
    // polling thread always, as this is important for gRPC performance
    mutex_lock lock(assignment_mu_);
    auto it = target_assignments_.find(target);
    if (it == target_assignments_.end()) {
      it = target_assignments_
               .insert(std::make_pair(
                   target, (next_round_robin_assignment_++) % threads_.size()))
               .first;
    }
    return it->second;
  }

  const string local_target_;
  WorkerInterface* const local_worker_;  // Not owned.
  std::shared_ptr<GrpcChannelCache> channel_cache_;
  WorkerCacheLogger logger_;
  std::vector<GrpcWorkerCacheThread> threads_;

  std::unique_ptr<thread::ThreadPool> callback_threadpool_;

  mutex assignment_mu_;
  std::unordered_map<std::string, size_t> target_assignments_
      GUARDED_BY(assignment_mu_);
  size_t next_round_robin_assignment_ GUARDED_BY(assignment_mu_);
};

}  // namespace

WorkerCacheInterface* NewGrpcWorkerCache(std::shared_ptr<GrpcChannelCache> cc) {
  return new GrpcWorkerCache(cc, nullptr, "");
}

/** \brief NewGrpcWorkerCacheWithLocalWorker helps construct a new
 *         GrpcWorkerCache, which can access all workers doing graph computing.
 *
 *  \param cc: std::shared_ptr<GrpcChannelCache>
 *         GrpcChannelCache is an interface handling 1. listing all worker names,
 *         2. listing names of all workers of a specific job name, 3. finding
 *         ::grpc::channel of a worker by its name in the form of
 *         /job:<job identifier>/task:<task id>, and 4. extract host:port string
 *         via worker name "/job:X/task:Z".
 *
 *  \param local_worker: WorkerInterface*
 *         WorkerInterface is an interface for talking with the TensorFlow
 *         Worker grpc service functions, e.g., CreateWorkerSessionAsync,
 *         RunGraphAsync.
 *
 *  \param local_target: string&. Name of the worker.
 *
 *  \return WorkerCacheInterface*
 *          WorkerCacheInterface manages Worker (interface is defined in class
 *          WorkerInterface), like 1. listing all worker names; 2. listing all
 *          worker names in a specified job name; 3. create a worker by passing
 *          a name to it; 4. destory a worker created by CreateWorker;
 *          5. device locality information and logging setting.
 */
WorkerCacheInterface* NewGrpcWorkerCacheWithLocalWorker(
    std::shared_ptr<GrpcChannelCache> cc, WorkerInterface* local_worker,
    const string& local_target) {
  return new GrpcWorkerCache(cc, local_worker, local_target);
}

}  // namespace tensorflow
