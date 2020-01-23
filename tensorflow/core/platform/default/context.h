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

#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_CONTEXT_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_CONTEXT_H_

namespace tensorflow {

class Context {
 public:
  Context() {}
  Context(const ContextKind kind) {}
  // 1.
  // enum class ContextKind 数据结构
  // core/platform/context.h
  // - kDefault
  //   Initial state with default (empty) values.
  // - kThread
  //   Initial state inherited from the creating or scheduling thread.

  bool operator==(const Context& other) const { return true; }
};
// 1.
// class Context 数据结构
// tensorflow/core/platform/default/context.h
// 1.1 概述
// Context is a container for request-specific information that should be passed
// to threads that perform related work. The default constructor should capture
// all relevant context.

// 2.
// enum class ContextKind 数据结构
// core/platform/context.h
// - kDefault
//   Initial state with default (empty) values.
// - kThread
//   Initial state inherited from the creating or scheduling thread.


class WithContext {
 public:
  explicit WithContext(const Context& x) {}
  // 1.
  // class Context 数据结构
  // 这个文件最上面写着呢。

  ~WithContext() {}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_DEFAULT_CONTEXT_H_
