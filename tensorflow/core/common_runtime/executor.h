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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/session_state.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class StepStatsCollector;

// Executor runs a graph computation.
// Example:
//   Graph* graph = ...;
//      ... construct graph ...
//   Executor* executor;
//   TF_CHECK_OK(NewSimpleExecutor(my_device, graph, &executor));
//   Rendezvous* rendezvous = NewNaiveRendezvous();
//   TF_CHECK_OK(rendezvous->Send("input", some_input_tensor));
//   TF_CHECK_OK(executor->Run({ExecutorOpts, rendezvous, nullptr}));
//   TF_CHECK_OK(rendezvous->Recv("output", &output_tensor));
//   ... ...
//
// Multiple threads can call Executor::Run concurrently.
class Executor {
 public:
  virtual ~Executor() {}

  // RunAsync() executes the graph computation. "done" is run when the
  // graph computation completes. If any error happens during the
  // computation, "done" is run and the error is passed to "done".
  //
  ///
  /// struct Args 有很多，我需要都把他们搞到清楚
  ///   GraphMgr::StartParallelExecutors, graph_mgr.cc 里面赋值了 Args
  ///   我想到，那些赋值的变量可以向上追溯，一路追啊追的，就能看到他们的本质了。
  ///
  // RunAsync() is given a few arguments in Args. The caller must
  // ensure objects passed in Args (rendezvous, stats_collector, etc.)
  // are alive at least until done is invoked. All pointers to the
  // argument objects can be nullptr.
  //
  // "step_id" is a process-wide unique identifier for the step being
  // run. Executors on different devices may receive the same step_id
  // in the case that a step runs Ops on more than one device. The
  // step_id is used for tracking resource usage of a given step.
  //
  // RunAsync() uses the given "rendezvous", if not null, as the
  // mechanism to communicate inputs and outputs of the underlying
  // graph computation.
  //
  // RunAsync() calls "stats_collector", if not null, to keep track of
  // stats. This allows us to collect statistics and traces on demand.
  //
  // RunAsync() is provided a "call_frame", if the executor is used
  // for executing a function, is used to pass arguments and return
  // values between the caller and the callee.
  //
  // RunAsync() uses "cancellation_manager", if not nullptr, to
  // register callbacks that should be called if the graph computation
  // is canceled. Note that the callbacks merely unblock any
  // long-running computation, and a canceled step will terminate by
  // returning/calling the DoneCallback as usual.
  //
  // RunAsync() dispatches closures to "runner". Typically, "runner"
  // is backed up by a bounded threadpool.
  struct Args {
    int64 step_id = 0;
    Rendezvous* rendezvous = nullptr;
    StepStatsCollectorInterface* stats_collector = nullptr;
    CallFrameInterface* call_frame = nullptr;
    CancellationManager* cancellation_manager = nullptr;
    SessionState* session_state = nullptr;

    // Unique session identifier. Can be empty.
    string session_handle;
    TensorStore* tensor_store = nullptr;
    ScopedStepContainer* step_container = nullptr;
    CollectiveExecutor* collective_executor = nullptr;

    // If true, calls Sync() on the device.
    bool sync_on_finish = false;

    typedef std::function<void()> Closure;
    typedef std::function<void(Closure)> Runner;
    Runner runner = nullptr;
  };

  typedef std::function<void(const Status&)> DoneCallback;

  /// 由于这里是接口定义，必然是被 ExecutorImpl::RunAsync override
  /// 必然是去调用 ExecutorImpl::RunAsync 的定义
  virtual void RunAsync(const Args& args, DoneCallback done) = 0;

  // Synchronous wrapper for RunAsync().
  Status Run(const Args& args) {
    Status ret;
    Notification n;
    /// 必然是去调用 ExecutorImpl::RunAsync 的定义
    RunAsync(args, [&ret, &n](const Status& s) {
      ret = s;
      n.Notify();
    });
    n.WaitForNotification();
    return ret;
  }

};

// Creates an Executor that computes the given "graph".
// If successful, returns the constructed executor in "*executor". Otherwise,
// returns an error status.
// "params" provides a set of context for the executor. We expect that
// different context would provide different implementations.
struct LocalExecutorParams {
  Device* device;

  // The library runtime support.
  FunctionLibraryRuntime* function_library = nullptr;

  // create_kernel returns an instance of op kernel based on NodeDef.
  // delete_kernel is called for every kernel used by the executor
  // when the executor is deleted.
  std::function<Status(const NodeDef&, OpKernel**)> create_kernel;
  std::function<void(OpKernel*)> delete_kernel;
};

::tensorflow::Status NewLocalExecutor(const LocalExecutorParams& params,
                                      std::unique_ptr<const Graph> graph,
                                      Executor** executor);

// A class to help run multiple executors in parallel and wait until
// all of them are complete.
//
// ExecutorBarrier deletes itself after the function returned by Get()
// is called.
class ExecutorBarrier {
 public:
  typedef std::function<void(const Status&)> StatusCallback;

  // Create an ExecutorBarrier for 'num' different executors.
  //
  // 'r' is the shared Rendezvous object that is used to communicate
  // state.  If any of the executors experiences an error, the
  // rendezvous object will be aborted exactly once.
  //
  // 'done' is called after the last executor completes, and
  // ExecutorBarrier is deleted.
  ExecutorBarrier(size_t num, Rendezvous* r, StatusCallback done)
      : rendez_(r), done_cb_(done), pending_(num) {}

  ~ExecutorBarrier() {}

  // Returns a closure that Executors must call when they are done
  // computing, passing the status of their execution as an argument.
  StatusCallback Get() {
    return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1);
  }

 private:
  Rendezvous* rendez_ = nullptr;
  StatusCallback done_cb_ = nullptr;

  mutable mutex mu_;
  int pending_ GUARDED_BY(mu_) = 0;
  StatusGroup status_group_ GUARDED_BY(mu_);

  void WhenDone(const Status& s) {
    Rendezvous* error_rendez = nullptr;
    StatusCallback done = nullptr;
    Status status;

    {
      mutex_lock l(mu_);

      // If we are the first error encountered, trigger an abort of the
      // Rendezvous object by this thread only.
      if (status_group_.ok() && !s.ok()) {
        error_rendez = rendez_;
        error_rendez->Ref();
      }

      status_group_.Update(s);

      // =======================================================================
      // If this is the last call to WhenDone, call the final callback
      // below.
      if (--pending_ == 0) {
        CHECK(done_cb_ != nullptr);
        std::swap(done, done_cb_);
        status = status_group_.as_status();
      }
      // =======================================================================
      // 1.
      // pending_ 变量说明
      // 如果有 2 个 executors, 则 pending_ 为 2, 所以我苦苦未能找到的答案，为什么不能 call 这个
      // done_cb_ 也就是 tensorflow/core/common_runtime/direct_session.cc 里面的原因就在这里。
      // [&run_state](const Status& ret) {
      //   {
      //     mutex_lock l(run_state.mu_);
      //     run_state.status.Update(ret);
      //   }
      //   run_state.executors_done.Notify(); // 核心 notify 去唤醒 sleep 的 main 函数
      // }
      //

    }

    if (error_rendez != nullptr) {
      error_rendez->StartAbort(
          errors::Aborted("Stopping remaining executors."));
      error_rendez->Unref();
    }

    if (done != nullptr) {
      delete this;
      done(status);
    }
  }
  // 1.
  // ExecutorBarrier::WhenDone 函数说明:
  // tensorflow/core/common_runtime/executor.h
  // 概述:
  // 更新 Status, 然后如果 !s.ok() 就 StartAbort 否则 就 执行 callback done 函数



  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorBarrier);
};
// 1.
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



// A few helpers to facilitate create/delete kernels.

// Creates a kernel based on "ndef" on device "device". The kernel can
// access the functions in the "flib". The caller takes ownership of
// returned "*kernel".
Status CreateNonCachedKernel(Device* device, FunctionLibraryRuntime* flib,
                             const NodeDef& ndef, int graph_def_version,
                             OpKernel** kernel);

// Deletes "kernel" returned by CreateKernel.
void DeleteNonCachedKernel(OpKernel* kernel);

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_H_
