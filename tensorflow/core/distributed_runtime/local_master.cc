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

#include "tensorflow/core/distributed_runtime/local_master.h"

#include <unordered_map>

#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

namespace {
Status WaitForNotification(CallOptions* call_options,
                           const int64 default_timeout_in_ms, Notification* n) {
  int64 timeout_in_ms = call_options->GetTimeout();
  if (timeout_in_ms == 0) {
    timeout_in_ms = default_timeout_in_ms;
  }
  if (timeout_in_ms > 0) {
    int64 timeout_in_us = timeout_in_ms * 1000;
    bool notified = WaitForNotificationWithTimeout(n, timeout_in_us);
    if (!notified) {
      call_options->StartCancel();
      // The call has borrowed pointers to the request and response
      // messages, so we must still wait for the call to complete.
      n->WaitForNotification();
      return errors::DeadlineExceeded("Operation timed out.");
    }
  } else {
    n->WaitForNotification();
  }
  return Status::OK();
}
}  // namespace

LocalMaster::LocalMaster(Master* master_impl, const int64 default_timeout_in_ms)
    : master_impl_(master_impl),
      default_timeout_in_ms_(default_timeout_in_ms) {}

Status LocalMaster::CreateSession(CallOptions* call_options,
                                  const CreateSessionRequest* request,
                                  CreateSessionResponse* response) {
  // 1.
  // 执行逻辑:
  // 首先,
  // main thread is blocking here when another thread is
  // `master_impl_->CreateSession`
  // 其次,
  // master_impl_->CreateSession 是第二个线程分出去执行的
  // 最终,
  // 只有 master_impl_->CreateSession 这个线程结束了以后 n.Notify() 后, main thread
  // 才会继续执行.

  // 1.1
  // 截图看:
  // https://docs.google.com/document/d/1OGmW02cLX686fhgaB5Vh4JgHLhdmXyQ3CtU3MAJOi4A/edit

  Notification n;
  Status ret;
  master_impl_->CreateSession(request, response, [&n, &ret](const Status& s) {
    ret.Update(s);
    n.Notify();
  });
  // 1.
  // master_impl_:
  // Master*


  TF_RETURN_IF_ERROR(
      WaitForNotification(call_options, default_timeout_in_ms_, &n));
  // 1.
  // main thread is blocking here when another thread is
  // `master_impl_->CreateSession`

  return ret;
}

Status LocalMaster::ExtendSession(CallOptions* call_options,
                                  const ExtendSessionRequest* request,
                                  ExtendSessionResponse* response) {
  Notification n;
  Status ret;
  master_impl_->ExtendSession(request, response, [&n, &ret](const Status& s) {
    ret.Update(s);
    n.Notify();
  });
  TF_RETURN_IF_ERROR(
      WaitForNotification(call_options, default_timeout_in_ms_, &n));
  return ret;
}

Status LocalMaster::PartialRunSetup(CallOptions* call_options,
                                    const PartialRunSetupRequest* request,
                                    PartialRunSetupResponse* response) {
  Notification n;
  Status ret;
  master_impl_->PartialRunSetup(request, response, [&n, &ret](const Status& s) {
    ret.Update(s);
    n.Notify();
  });
  TF_RETURN_IF_ERROR(
      WaitForNotification(call_options, default_timeout_in_ms_, &n));
  return ret;
}

Status LocalMaster::RunStep(CallOptions* call_options,
                            RunStepRequestWrapper* request,
                            MutableRunStepResponseWrapper* response) {
// 1.
// 输入:
// CallOptions* call_options
// RunStepRequestWrapper* req,
// 输出:
// MutableRunStepResponseWrapper* resp

// 2.
// RunStepRequestWrapper 在
// tensorflow/core/distributed_runtime/message_wrappers.h

  Notification n;
  // 1.
  // Notification 数据结构在哪里
  // tensorflow/core/platform/default/notification.h

  // 2.
  // Notification 怎么用啊?
  //

  Status ret;

  master_impl_->RunStep(call_options,
                        request,
                        response,
                        [&n, &ret](const Status& s) {
                          ret.Update(s);
                          n.Notify();
                        });
  // 1.
  // RunStep 在
  // tensorflow/core/distributed_runtime/master.cc
  // Master::RunStep

  // 2.
  // master_impl_: Master*

  // 3.
  // 所以, master_impl_->RunStep 的 RunStep 是怎么转交给 worker 的?
  // ...


  TF_RETURN_IF_ERROR(
      WaitForNotification(call_options, default_timeout_in_ms_, &n));
  return ret;
}

MutableRunStepRequestWrapper* LocalMaster::CreateRunStepRequest() {
  return new InMemoryRunStepRequest;
}

MutableRunStepResponseWrapper* LocalMaster::CreateRunStepResponse() {
  return new InMemoryRunStepResponse;
}

Status LocalMaster::CloseSession(CallOptions* call_options,
                                 const CloseSessionRequest* request,
                                 CloseSessionResponse* response) {
  Notification n;
  Status ret;
  master_impl_->CloseSession(request, response, [&n, &ret](const Status& s) {
    ret.Update(s);
    n.Notify();
  });
  TF_RETURN_IF_ERROR(
      WaitForNotification(call_options, default_timeout_in_ms_, &n));
  return ret;
}

Status LocalMaster::ListDevices(CallOptions* call_options,
                                const ListDevicesRequest* request,
                                ListDevicesResponse* response) {
  Notification n;
  Status ret;
  master_impl_->ListDevices(request, response, [&n, &ret](const Status& s) {
    ret.Update(s);
    n.Notify();
  });
  TF_RETURN_IF_ERROR(
      WaitForNotification(call_options, default_timeout_in_ms_, &n));
  return ret;
}

Status LocalMaster::Reset(CallOptions* call_options,
                          const ResetRequest* request,
                          ResetResponse* response) {
  Notification n;
  Status ret;
  master_impl_->Reset(request, response, [&n, &ret](const Status& s) {
    ret.Update(s);
    n.Notify();
  });
  TF_RETURN_IF_ERROR(
      WaitForNotification(call_options, default_timeout_in_ms_, &n));
  return ret;
}

Status LocalMaster::MakeCallable(CallOptions* call_options,
                                 const MakeCallableRequest* request,
                                 MakeCallableResponse* response) {
  Notification n;
  Status ret;
  master_impl_->MakeCallable(request, response, [&n, &ret](const Status& s) {
    ret.Update(s);
    n.Notify();
  });
  TF_RETURN_IF_ERROR(
      WaitForNotification(call_options, default_timeout_in_ms_, &n));
  return ret;
}
Status LocalMaster::RunCallable(CallOptions* call_options,
                                const RunCallableRequest* request,
                                RunCallableResponse* response) {
  Notification n;
  Status ret;
  master_impl_->RunCallable(call_options, request, response,
                            [&n, &ret](const Status& s) {
                              ret.Update(s);
                              n.Notify();
                            });
  TF_RETURN_IF_ERROR(
      WaitForNotification(call_options, default_timeout_in_ms_, &n));
  return ret;
}
Status LocalMaster::ReleaseCallable(CallOptions* call_options,
                                    const ReleaseCallableRequest* request,
                                    ReleaseCallableResponse* response) {
  Notification n;
  Status ret;
  master_impl_->ReleaseCallable(request, response, [&n, &ret](const Status& s) {
    ret.Update(s);
    n.Notify();
  });
  TF_RETURN_IF_ERROR(
      WaitForNotification(call_options, default_timeout_in_ms_, &n));
  return ret;
}

namespace {
mutex* get_local_master_registry_lock() {
  static mutex local_master_registry_lock(LINKER_INITIALIZED);
  return &local_master_registry_lock;
}

struct MasterInfo {
  Master* master;
  const int64 default_timeout_in_ms;

  MasterInfo(Master* master, const int64 default_timeout_in_ms)
      : master(master), default_timeout_in_ms(default_timeout_in_ms) {}
};

typedef std::unordered_map<string, MasterInfo> LocalMasterRegistry;
/** \brief Create an unorder map to store the target string (ip:port string)
 *         with Master grpc service into an unorder map.
 *         i.e., ip:port string <-> {class Master, timeout}
 *         class Master is responsible for grpc service functions:
 *         1. Create a session; 2. extend a session; 3. close a session;
 *         4. run a step of the graph; 5. list devices;
 */
LocalMasterRegistry* local_master_registry() {
  static LocalMasterRegistry* local_master_registry_ = new LocalMasterRegistry;
  return local_master_registry_;
}
}  // namespace

/** \brief Store the target string (ip:port string) with Master grpc service
 *         into an unorder map, i.e., ip:port string <-> {class Master, timeout}
 *
 *  \param target: const string& ;
 *         For example, "localhost:2223"
 *
 *  \param master: Master*
 *         class Master is responsible for grpc service functions:
 *         1. Create a session; 2. extend a session; 3. close a session;
 *         4. run a step of the graph; 5. list devices;
 *
 *  \param default_timeout_in_ms: int64
 *
 */
/* static */
void LocalMaster::Register(const string& target, Master* master,
                           int64 default_timeout_in_ms) {
  mutex_lock l(*get_local_master_registry_lock());
  local_master_registry()->insert(
      {target, MasterInfo(master, default_timeout_in_ms)});
}

/** \brief Get the LocalMaster class which provide Session interface functions.
 *         class Master is responsible for grpc service functions:
 *         1. Create a session; 2. extend a session; 3. close a session;
 *         4. run a step of the graph; 5. list devices;
 *
 *  \param[in] target: const string& ;
 *         The name of the target, i.e., "ip:port", e.g., "localhost:2223".
 *
 *  \return std::unique_ptr<LocalMaster> ;
 *         As a contrast to Remote Master, Remote Master is in a process, client
 *         is in another process. LocalMaster enables direct intraprocess
 *         communication between the client and master implementation.
 *
 *   \details Local master registry has local master inside, so ret is nullptr.
 */
/* static */
std::unique_ptr<LocalMaster> LocalMaster::Lookup(const string& target) {
  std::unique_ptr<LocalMaster> ret;
  mutex_lock l(*get_local_master_registry_lock());
  auto iter = local_master_registry()->find(target);
  if (iter != local_master_registry()->end()) {
    ret.reset(new LocalMaster(iter->second.master,
                              iter->second.default_timeout_in_ms));
  }
  return ret;
}

}  // namespace tensorflow
