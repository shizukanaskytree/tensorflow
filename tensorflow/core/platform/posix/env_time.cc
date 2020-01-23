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

#include <sys/time.h>
#include <time.h>

#include "tensorflow/core/platform/env_time.h"

namespace tensorflow {

namespace {

class PosixEnvTime : public EnvTime {
 public:
  PosixEnvTime() {}

  // 返回自 1970 年开始至今的 总 elapsed time (单位:NanoSeconds)
  uint64 NowNanos() override {
    struct timespec ts;
    // 1.
    // struct timespec 数据结构:
    // https://embeddedartistry.com/blog/2019/1/31/converting-between-timespec-amp-stdchrono
    // - tv_sec: time_t
    // - tv_nsec: long

    clock_gettime(CLOCK_REALTIME, &ts);
    return (static_cast<uint64>(ts.tv_sec) * kSecondsToNanos +
            static_cast<uint64>(ts.tv_nsec));
  }
  // 1.
  // kSecondsToNanos 变量说明：
  // static constexpr uint64 kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;
  // tensorflow/core/platform/env_time.h:

  // 2.
  // timespec::tv_sec 变量说明:
  // The tv_sec field represents either a general number of seconds, or seconds elapsed since 1970.

  // 3.
  // timespec::tv_nsec 变量说明:
  // tv_nsec represents the count of nanoseconds.

};

}  // namespace

#if defined(PLATFORM_POSIX) || defined(__ANDROID__)
EnvTime* EnvTime::Default() {
  static EnvTime* default_env_time = new PosixEnvTime;
  return default_env_time;
}
#endif

}  // namespace tensorflow
