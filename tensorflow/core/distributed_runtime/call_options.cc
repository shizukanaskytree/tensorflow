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

#include "tensorflow/core/distributed_runtime/call_options.h"

#include "tensorflow/core/platform/mutex.h"

#include "tensorflow/core/util/write_log.h"
#include <boost/stacktrace.hpp>
#define BOOST_STACKTRACE_USE_ADDR2LINE

namespace tensorflow {

CallOptions::CallOptions() {
  //write_log(getpid(), __func__, __LINE__, __FILE__, "/home/wxf/tf2/tensorflow/cc_debug_var.log");
}

void CallOptions::StartCancel() {
  //write_log(getpid(), __func__, __LINE__, __FILE__, "/home/wxf/tf2/tensorflow/cc_debug_var.log");
  mutex_lock l(mu_);
  if (cancel_func_ != nullptr) {
    // NOTE: We must call the cancel_func_ with mu_ held. This ensure
    // that ClearCancelCallback() does not race with StartCancel().
    cancel_func_();
    // NOTE: We can clear cancel_func_ if needed.
  }
}

void CallOptions::SetCancelCallback(CancelFunction cancel_func) {
  //write_log(getpid(), __func__, __LINE__, __FILE__, "/home/wxf/tf2/tensorflow/cc_debug_var.log");
  mutex_lock l(mu_);
  cancel_func_ = std::move(cancel_func);
}

void CallOptions::ClearCancelCallback() {
  //write_log(getpid(), __func__, __LINE__, __FILE__, "/home/wxf/tf2/tensorflow/cc_debug_var.log");
  mutex_lock l(mu_);
  cancel_func_ = nullptr;
}

int64_t CallOptions::GetTimeout() {
  //write_log(getpid(), __func__, __LINE__, __FILE__, "/home/wxf/tf2/tensorflow/cc_debug_var.log");
  mutex_lock l(mu_);
  return timeout_in_ms_;
}

void CallOptions::SetTimeout(int64_t ms) {
  //write_log(getpid(), __func__, __LINE__, __FILE__, "/home/wxf/tf2/tensorflow/cc_debug_var.log");
  mutex_lock l(mu_);
  timeout_in_ms_ = ms;
}

}  // end namespace tensorflow
