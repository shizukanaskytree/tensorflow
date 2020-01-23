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

#include "tensorflow/core/distributed_runtime/session_mgr.h"

#include <utility>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_cache_wrapper.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

/** \brief SessionMgr manages a list of Sessions on a worker.
 *         1. create a session to a list of sessions ; 2. delete a session;
 *         3. lookup a session; 4. logging.
 *
 *  \param worker_env: WorkerEnv*
 *         WorkerEnv includes 1. environment, 2. session manager, 3. device,
 *         4. rendezvous, 5. collective, 6. compute thread pool.
 *
 *  \param default_worker_name: const string& ;
 *
 *  \param default_worker_cache: std::unique_ptr<WorkerCacheInterface>
 *         WorkerCacheInterface provides interface for 1. list workers' names;
 *         2. list workers' name of a job name; 3. create worker with a name;
 *         4. destory a worker; 5. get device locality information of a device;
 *         6. logging.
 *
 *  \param worker_cache_factory: WorkerCacheFactory
 *         A GrpcServer member fucntion, constructing a new GrpcWorkerCache
 *         which can access all workers responsible for graph computing.
 */
SessionMgr::SessionMgr(
    WorkerEnv* worker_env, const string& default_worker_name,
    std::unique_ptr<WorkerCacheInterface> default_worker_cache,
    WorkerCacheFactory worker_cache_factory)
    : worker_env_(worker_env),
      default_worker_cache_(std::move(default_worker_cache)),
      /// CreateWithBorrowedDeviceMgr create a WorkerSession.
      legacy_session_(WorkerSession::CreateWithBorrowedDeviceMgr(
          "", default_worker_name,
          std::unique_ptr<WorkerCacheInterface>(
              /// WorkerCacheWrapper manages 1. create of a worker; 2. delete a
              /// a worker; 3. list all workers; 4. logging.
              new WorkerCacheWrapper(default_worker_cache_.get())),
          worker_env->device_mgr,
          std::unique_ptr<GraphMgr>(
              /// GraphMgr keeps track of a set of graphs that are registered
              /// with a TensorFlow worker.
              new GraphMgr(worker_env, worker_env->device_mgr)))),
      worker_cache_factory_(std::move(worker_cache_factory)) {}

/* static */
string SessionMgr::WorkerNameFromServerDef(const ServerDef& server_def) {
  return strings::StrCat("/job:", server_def.job_name(), "/replica:0/task:",
                         server_def.task_index());
}

/** \brief Worker server side grpc implementation of creating session.
 *
 *  \param[in] session: const string&
 *         worker session name, session handler is a string.
 *
 *  \param[in] server_def: const ServerDef&
 *
 *  \param[in] isolate_session_state: bool
 *
 *  \return Status.
 *
 *  \details It finally sets
 *   std::map< string, std::shared_ptr< WorkerSession > > sessions_ of this
 *   SessionMgr instance.
 *
 *   class SessionMgr serves class WorkerEnv, indirectly serves class Worker.
 */
Status SessionMgr::CreateSession(const string& session,
                                 const ServerDef& server_def,
                                 bool isolate_session_state) {
  mutex_lock l(mu_);
  if (session.empty()) {
    return errors::InvalidArgument("Session must be non-empty.");
  }

  WorkerCacheInterface* worker_cache = nullptr;
  string worker_name;
  if (server_def.cluster().job().empty()) {
    /// Go into this if branch
    worker_cache = new WorkerCacheWrapper(default_worker_cache_.get());
    worker_name = legacy_session_->worker_name;
  } else {
    TF_RETURN_IF_ERROR(worker_cache_factory_(server_def, &worker_cache));
    worker_name = WorkerNameFromServerDef(server_def);
  }

  if (worker_cache != nullptr && default_worker_cache_ != nullptr) {
    worker_cache->SetLogging(this->is_logging_active_);
  }

  CHECK(!worker_env_->local_devices.empty())
      << "The WorkerEnv must have at least one device in `local_devices`.";

  std::shared_ptr<WorkerSession> worker_session;

  if (isolate_session_state) {
    // Create a private copy of the DeviceMgr for the WorkerSession.
    std::vector<std::unique_ptr<Device>> renamed_devices;
    for (Device* d : worker_env_->local_devices) {
      renamed_devices.push_back(RenamedDevice::NewRenamedDevice(
          worker_name, d, false, isolate_session_state));
    }

    auto device_mgr = MakeUnique<DeviceMgr>(std::move(renamed_devices));
    /// GraphMgr keeps track of a set of graphs that are registered with a
    /// TensorFlow worker. Each registered graph is identified by a handle
    /// that is generated by GraphMgr and returned to the caller.
    auto graph_mgr = MakeUnique<GraphMgr>(worker_env_, device_mgr.get());
    worker_session.reset(
        new WorkerSession(session, worker_name,
                          std::unique_ptr<WorkerCacheInterface>(worker_cache),
                          std::move(device_mgr), std::move(graph_mgr)));
  } else {
    /// Go into this if branch.
    // Borrown the WorkerEnv's DeviceMgr for the WorkerSession, so
    // that resources using it can use its devices after the
    // WorkerSession has been deleted.
    auto graph_mgr = MakeUnique<GraphMgr>(worker_env_, worker_env_->device_mgr);
    /// CreateWithBorrowedDeviceMgr
    /// Indirectly call WorkerSession construct with the same parameters.
    worker_session = WorkerSession::CreateWithBorrowedDeviceMgr(
        session, worker_name,
        std::unique_ptr<WorkerCacheInterface>(worker_cache),
        worker_env_->device_mgr, std::move(graph_mgr));
  }

  sessions_.insert(std::make_pair(session, std::move(worker_session)));
  return Status::OK();
}

Status SessionMgr::DeleteSession(const string& session) {
  mutex_lock l(mu_);
  auto it = sessions_.find(session);
  if (it != sessions_.end()) {
    sessions_.erase(it);
  }
  return Status::OK();
}

Status SessionMgr::WorkerSessionForSessionLocked(
    const string& session_handle, std::shared_ptr<WorkerSession>* out_session) {
  if (session_handle.empty()) {
    *out_session = legacy_session_;
  } else {
    auto it = sessions_.find(session_handle);
    if (it == sessions_.end()) {
      return errors::Aborted("Session handle is not found: ", session_handle,
                             ". Possibly this worker (\"",
                             legacy_session_->worker_name,
                             "\") just restarted.");
    } else {
      *out_session = it->second;
    }
  }
  return Status::OK();
}

Status SessionMgr::WorkerSessionForSession(
    const string& session_handle, std::shared_ptr<WorkerSession>* out_session) {
  mutex_lock l(mu_);
  return WorkerSessionForSessionLocked(session_handle, out_session);
}

std::shared_ptr<WorkerSession> SessionMgr::LegacySession() {
  return legacy_session_;
}

void SessionMgr::SetLogging(bool active) {
  mutex_lock l(mu_);
  this->is_logging_active_ = active;
  // Legacy Session
  if (legacy_session_) {
    auto* worker_cache = legacy_session_->worker_cache.get();
    if (worker_cache) {
      worker_cache->SetLogging(active);
    }
  }

  for (const auto& session_kv : sessions_) {
    auto session = session_kv.second.get();
    if (session) {
      auto* worker_cache = session->worker_cache.get();
      if (worker_cache) {
        worker_cache->SetLogging(active);
      }
    }
  }
}

void SessionMgr::RetrieveLogs(tensorflow::int64 step_id,
                              LoggingResponse* response) {
  mutex_lock l(mu_);
  // Legacy Session
  if (legacy_session_) {
    auto* worker_cache = legacy_session_->worker_cache.get();
    if (worker_cache) {
      auto step_stats = StepStats();
      if (worker_cache->RetrieveLogs(step_id, &step_stats)) {
        auto* labeled_step_stats = response->add_step();
        labeled_step_stats->set_step_id(step_id);
        labeled_step_stats->mutable_step_stats()->Swap(&step_stats);
      }
    }
  }
  for (const auto& session_kv : sessions_) {
    auto session = session_kv.second.get();
    if (session) {
      auto* worker_cache = session->worker_cache.get();
      if (worker_cache) {
        auto step_stats = StepStats();
        if (worker_cache->RetrieveLogs(step_id, &step_stats)) {
          auto* labeled_step_stats = response->add_step();
          labeled_step_stats->set_step_id(step_id);
          labeled_step_stats->mutable_step_stats()->Swap(&step_stats);
        }
      }
    }
  }
}

void SessionMgr::ClearLogs() {
  mutex_lock l(mu_);
  // Legacy Session
  if (legacy_session_) {
    auto* worker_cache = legacy_session_->worker_cache.get();
    if (worker_cache) {
      worker_cache->ClearLogs();
    }
  }

  for (const auto& session_kv : sessions_) {
    auto session = session_kv.second.get();
    if (session) {
      auto* worker_cache = session->worker_cache.get();
      if (worker_cache) {
        worker_cache->ClearLogs();
      }
    }
  }
}
}  // namespace tensorflow
