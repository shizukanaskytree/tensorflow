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

#include "tensorflow/core/framework/cancellation.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

const CancellationToken CancellationManager::kInvalidToken = -1;

CancellationManager::CancellationManager()
    : is_cancelling_(false),
      is_cancelled_(false),
      next_cancellation_token_(0) {}

void CancellationManager::Reset() {
  mutex_lock l(mu_);
  is_cancelling_ = false;
  is_cancelled_.store(false);
}

// 执行注册的 CancelCallback lambda function
void CancellationManager::StartCancel() {

  gtl::FlatMap<CancellationToken, CancelCallback> callbacks_to_run;

  {
    mutex_lock l(mu_);
    if (is_cancelled_.load(std::memory_order_relaxed) || is_cancelling_) {
      return;
    }
    is_cancelling_ = true;
    std::swap(callbacks_, callbacks_to_run);
  }

  // We call these callbacks without holding mu_, so that concurrent
  // calls to DeregisterCallback, which can happen asynchronously, do
  // not block. The callbacks remain valid because any concurrent call
  // to DeregisterCallback will block until the
  // cancelled_notification_ is notified.
  for (auto key_and_value : callbacks_to_run) {
    key_and_value.second();
  }

  {
    mutex_lock l(mu_);
    is_cancelling_ = false;
    is_cancelled_.store(true, std::memory_order_release);
  }

  cancelled_notification_.Notify();
  // 1.
  // cancelled_notification_ 变量说明
  // cancelled_notification_: class Notification
  // tensorflow/core/platform/default/notification.h

  // 2.
  // class Notification 数据结构
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
}

CancellationToken CancellationManager::get_cancellation_token() {
  mutex_lock l(mu_);
  return next_cancellation_token_++;
}

bool CancellationManager::RegisterCallback(CancellationToken token,
                                           CancelCallback callback) {
  mutex_lock l(mu_);
  CHECK_LT(token, next_cancellation_token_) << "Invalid cancellation token";

  bool should_register = !is_cancelled_ && !is_cancelling_;
  // 1.
  // QQQ. is_cancelled_ 和 is_cancelling_ 是在哪里被构造和初始化的?
  // AAA.
  // tensorflow/core/common_runtime/direct_session.cc 内的
  // step_cancellation_manager: CancellationManager 被构造时构造了
  // - CancellationManager step_cancellation_manager;
  // 全部为 false

  // 2.
  // CancellationManager 构造函数说明:
  // CancellationManager::CancellationManager()
  // : is_cancelling_(false),
  //   is_cancelled_(false),
  //   next_cancellation_token_(0) {}
  // - is_cancelling_ : false
  // - is_cancelled_: false
  // - next_cancellation_token_: 0

  // 2.
  // CancellationManager::is_cancelled_: std::atomic_bool

  // 3.
  // CancellationManager::is_cancelling_: bool

  // 4.
  // 打印
  // 4.1
  // p is_cancelled_
  // $5 = {_M_base = {static _S_alignment = 1, _M_i = false}}

  // 4.2
  // p is_cancelling_
  // $6 = false

  if (should_register) {
    // 1.
    // should_register 变量说明
    // 只有进入这个分支了才会去注册这个 callback lambda function

    std::swap(callbacks_[token], callback);
    // 1.
    // callback lambda 函数说明:
    // [&step_cancellation_manager]() {
    //   step_cancellation_manager.StartCancel();
    // }

    // 2.
    // void CancellationManager::StartCancel() 函数说明:
    // tensorflow/core/framework/cancellation.cc
    //
    // 概述:
    // 执行注册的 CancelCallback lambda function

    // 3.
    // callbacks_ 变量说明:
    // gtl::FlatMap<CancellationToken, CancelCallback> callbacks_
    // within class CancellationManager
    //
    // 数据结构图
    //
    // +---+       +--------------------------+
    // | 0 |+----> | callback lambda function |
    // +---+       +--------------------------+
    //
    // +---+       +--------------------------+
    // | 1 |+----> | callback lambda function |
    // +---+       +--------------------------+
    //   .                     .
    //   .                     .
    //   .                     .
    // +---+       +--------------------------+
    // | n |+----> | callback lambda function |
    // +---+       +--------------------------+
    //
  }
  return should_register;
}


bool CancellationManager::DeregisterCallback(CancellationToken token) {
  mu_.lock();

  if (is_cancelled_) {
    mu_.unlock();
    return false;
  } else if (is_cancelling_) {
    mu_.unlock();
    // Wait for all of the cancellation callbacks to be called. This
    // wait ensures that the caller of DeregisterCallback does not
    // return immediately and free objects that may be used in the
    // execution of any currently pending callbacks in StartCancel.
    cancelled_notification_.WaitForNotification();
    return false;
  } else {
    callbacks_.erase(token);
    mu_.unlock();
    return true;
  }
}


bool CancellationManager::TryDeregisterCallback(CancellationToken token) {
  mutex_lock lock(mu_);
  if (is_cancelled_ || is_cancelling_) {
    return false;
  } else {
    callbacks_.erase(token);
    return true;
  }
}

CancellationManager::~CancellationManager() {
  if (!callbacks_.empty()) {
    StartCancel();
  }
}

}  // end namespace tensorflow
