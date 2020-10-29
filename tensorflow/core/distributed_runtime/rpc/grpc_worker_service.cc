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

// file note:
// 如下文件多数是在执行了如下 python 后执行的
//
// server = tf.train.Server(cluster,
//       job_name="worker",
//       task_index=FLAGS.task_index,
//       config=config)
//

// logging 看逻辑
// https://gist.github.com/shizukanaskytree/e8ce8e25e59e88f9de4ee6846c90522f

#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "grpcpp/alarm.h"
#include "grpcpp/server_builder.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/common_runtime/buf_rendezvous.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_call.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_response_cache.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"
#include "tensorflow/core/distributed_runtime/worker.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/transport_options.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

namespace {

// This macro creates a new request for the given RPC method name
// (e.g., `ENQUEUE_REQUEST(GetStatus, false);`), and enqueues it on
// `this->cq_`.
//
// This macro is invoked one or more times for each RPC method to
// ensure that there are sufficient completion queue entries to
// handle incoming requests without blocking.
//
// The implementation of the request handler for each RPC method
// must ensure that it calls ENQUEUE_REQUEST() for that RPC method,
// to keep accepting new requests.
#define ENQUEUE_REQUEST(method, supports_cancel)                             \
  do {                                                                       \
    mutex_lock l(shutdown_mu_);                                              \
    if (!is_shutdown_) {                                                     \
      Call<GrpcWorkerServiceThread, grpc::WorkerService::AsyncService,       \
           method##Request, method##Response>::                              \
          EnqueueRequestForMethod(                                           \
              worker_service_, cq_.get(),                                    \
              static_cast<int>(GrpcWorkerMethod::k##method),                 \
              &GrpcWorkerServiceThread::method##Handler, (supports_cancel)); \
    }                                                                        \
  } while (0)

// 1.
// 调用的是什么? 本质上是:
// Call 内的 member function :: EnqueueRequestForMethod
//

// 2.
// tensorflow/core/distributed_runtime/rpc/grpc_call.h:218:
// static void EnqueueRequestForMethod(...)

// 3.
// Call<types...>
//
// template <class Service, // 1.
//           class GrpcService, // 2.
//           class RequestMessage, // 3.
//           class ResponseMessage> //  4.
//
// 3.1
// GrpcWorkerServiceThread
//
// 3.2
// grpc::WorkerService::AsyncService
//
// 3.3
// method##Request, e.g.,
//
// 3.4
// method##Response

// 4.
// GrpcWorkerMethod::k##method
// tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h
// Names of worker methods.
//
// enum class GrpcWorkerMethod {
//   kGetStatus,
//   kCreateWorkerSession,
//   kDeleteWorkerSession,
//   kRegisterGraph,
//   kDeregisterGraph,
//   kRunGraph,
//   kCleanupGraph,
//   kCleanupAll,
//   kRecvTensor,
//   kRecvBuf,
//   kLogging,
//   kTracing,
//   kCompleteGroup,
//   kCompleteInstance,
//   kGetStepSequence,
// };


#define SETUP_FOR_REQUEST(method, default_depth, supports_cancel)              \
  for (int i = 0;                                                              \
       i < gtl::FindWithDefault(queue_depth_,                                  \
                                static_cast<int>(GrpcWorkerMethod::k##method), \
                                default_depth);                                \
       ++i) {                                                                  \
    ENQUEUE_REQUEST(method, supports_cancel);                                  \
  }

// 感受:
// 好恶心的写法, 干嘛不写个 function ?!

// 1.
// e.g.,
// SETUP_FOR_REQUEST(RunGraph, 100, true);
// for 循环重复 100 次执行 ENQUEUE_REQUEST(method, supports_cancel);
//

// 2.
// gtl::FindWithDefault
// tensorflow/core/lib/gtl/map_util.h
//

// 3.
//
// std::unordered_map<int, int> queue_depth_;

// 4.
// 这里的 k 表示 konst, const!
// static_cast<int>(GrpcWorkerMethod::k##method

// ===================================================

// GrpcWorkerService spawns one or more GrpcWorkerServiceThreads to service
// requests.  Each thread operates on an independent completion queue.
class GrpcWorkerServiceThread {
 public:
  explicit GrpcWorkerServiceThread(
      GrpcWorker* worker, // 这个是干嘛的? ==> worker_, thread serving a request.
      ::grpc::ServerBuilder* builder,
      std::unordered_map<int, int> queue_depth,
      GrpcResponseCache* cache,
      grpc::WorkerService::AsyncService* worker_service)
      : worker_(worker),
        queue_depth_(queue_depth),
        cache_(cache),
        worker_service_(worker_service),
        is_shutdown_(false) {
  // 1.
  // grpc::WorkerService::AsyncService* worker_service 是什么?
  // tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h

    cq_ = builder->AddCompletionQueue();
    // 1.
    // Add a completion queue for handling asynchronous services.
    //

    // 根据 server 的提示:
    // https://grpc.io/docs/languages/cpp/async/
    // Async server
    // The server implementation requests an RPC call with a tag
    // and then waits for the completion queue to return the tag.

  }

  void Start() {
    thread_.reset(
        worker_->env()->env->StartThread(ThreadOptions(), "grpc_worker_service",
                                         [this]() { HandleRPCsLoop(); }));
    // 1.
    // HandleRPCsLoop 在下面.
  }

  void Join() { thread_.reset(); }  // Blocks until thread exits

  void Shutdown() {
    {
      mutex_lock lock(shutdown_mu_);
      is_shutdown_ = true;
    }
    cq_->Shutdown();
  }

 private:

  // Add one or more completion queue entries for each worker method, then
  // begin servicing requests from the completion queue.
  void HandleRPCsLoop() {
    // TODO(ncteisen): This may require performance engineering. We can
    // change the number of threads, the number of handlers per thread,
    // or even decide to specialize certain threads to certain methods.
    SETUP_FOR_REQUEST(GetStatus, 1, false);
    SETUP_FOR_REQUEST(CreateWorkerSession, 1, false);
    SETUP_FOR_REQUEST(DeleteWorkerSession, 1, false);
    SETUP_FOR_REQUEST(CleanupAll, 1, false);
    SETUP_FOR_REQUEST(RegisterGraph, 1, false);
    SETUP_FOR_REQUEST(DeregisterGraph, 1, false);
    SETUP_FOR_REQUEST(Logging, 1, false);
    SETUP_FOR_REQUEST(Tracing, 1, false);
    SETUP_FOR_REQUEST(CompleteGroup, 10, true);
    SETUP_FOR_REQUEST(CompleteInstance, 10, true);
    SETUP_FOR_REQUEST(GetStepSequence, 10, true);
    SETUP_FOR_REQUEST(RecvBuf, 500, true);

    SETUP_FOR_REQUEST(RunGraph, 100, true);
    // 1.
    // 相关:
    // tensorflow/core/protobuf/worker_service.proto
    // rpc RunGraph(RunGraphRequest) returns (RunGraphResponse);

    // 2.
    // void RunGraphHandler(WorkerCall<RunGraphRequest, RunGraphResponse>* call)
    // 这个函数是执行者.

    // 3.
    // SETUP_FOR_REQUEST 定义:
    // tensorflow/core/distributed_runtime/rpc/grpc_worker_service.cc:87:
    // #define SETUP_FOR_REQUEST(method, default_depth, supports_cancel)
    //

    // e.g.,
    // RunGraph
    // RunGraphRequest
    // RunGraphResponse
    // ...


    SETUP_FOR_REQUEST(CleanupGraph, 100, false);

    // TODO(ncteisen): Determine a better policy for enqueuing the
    // appropriate number of each request type.
    for (int i = 0;
         i < gtl::FindWithDefault(
                    queue_depth_,
                    static_cast<int>(GrpcWorkerMethod::kRecvTensor),
                    1000);
         ++i) {
      EnqueueRecvTensorRequestRaw();
    }

    void* tag;
    bool ok;

    while (cq_->Next(&tag, &ok)) {
      // 1.
      // 观感像 server
      // https://grpc.io/docs/languages/cpp/async/

      UntypedCall<GrpcWorkerServiceThread>::Tag* callback_tag =
          static_cast<UntypedCall<GrpcWorkerServiceThread>::Tag*>(tag);
      // 1.
      // tensorflow/core/distributed_runtime/rpc/grpc_call.h:76:
      // class UntypedCall : public core::RefCounted
      //

      CHECK(callback_tag);
      callback_tag->OnCompleted(this, ok);
    }
  } // end of void HandleRPCsLoop()

 private:

  void Schedule(std::function<void()> f) {
    // 1.
    // 下面 HANDLE_CALL 用了

    // 2.
    // 这个文件里面的 Schedule 都用了.

    worker_->env()->compute_pool->Schedule(std::move(f));
  }

  // The following section contains one request handler method per
  // RPC. The `FooHandler` method is called (indirectly) by
  // `HandleRPCsLoop()` when the next Foo RPC is received. Each
  // `FooHandler` call schedules a closure on `worker_->env()->compute_pool`,
  // and is responsible for requesting the next Foo call by calling
  // `ENQUEUE_REQUEST(Foo)`.
  template <class RequestMessage, class ResponseMessage>
  using WorkerCall =
      Call<GrpcWorkerServiceThread,
           grpc::WorkerService::AsyncService,
           RequestMessage,
           ResponseMessage>;
  // 1.
  // class RequestMessage, class ResponseMessage 定义在:
  // tensorflow/core/protobuf/worker.proto

  // 2.
  // tensorflow/core/distributed_runtime/rpc/grpc_worker_service.cc:98:
  // class GrpcWorkerServiceThread

  // Handle all non-cancellable simple methods with a standard wrapper.
#define HANDLE_CALL(method)                                                   \
  void method##Handler(WorkerCall<method##Request, method##Response>* call) { \
    Schedule([this, call]() {                                                 \
      // 1.
      // 上面 Schedule 定义的.

      Status s = worker_->method(&call->request, &call->response);            \
      if (!s.ok()) {                                                          \
        VLOG(1) << "Bad response from " << #method << ": " << s;              \
      }                                                                       \
      call->SendResponse(ToGrpcStatus(s));                                    \
    });                                                                       \
    ENQUEUE_REQUEST(method, false);                                           \
  }

  HANDLE_CALL(GetStatus);
  HANDLE_CALL(CreateWorkerSession);
  HANDLE_CALL(DeleteWorkerSession);
  HANDLE_CALL(CleanupAll);
  HANDLE_CALL(RegisterGraph);
  HANDLE_CALL(DeregisterGraph);
  HANDLE_CALL(CleanupGraph);
  HANDLE_CALL(Logging);
  HANDLE_CALL(Tracing);

#undef HANDLE_CALL

  void GetStepSequenceHandler(
      WorkerCall<GetStepSequenceRequest,
                 GetStepSequenceResponse>* call) {
    Schedule([this, call]() {
      worker_->GetStepSequenceAsync(
          &call->request,
          &call->response,
          [call](const Status& s) {
            VLOG(1) << "Bad response from GetStepSequence:" << s;
            call->SendResponse(ToGrpcStatus(s));
          });
    });
    ENQUEUE_REQUEST(GetStepSequence, true);
  }

  void RunGraphHandler(
    WorkerCall<RunGraphRequest, RunGraphResponse>* call) {

    Schedule([this, call]() {

      CallOptions* call_opts = new CallOptions;

      ProtoRunGraphRequest* wrapped_request =
          new ProtoRunGraphRequest(&call->request);

      NonOwnedProtoRunGraphResponse* wrapped_response =
          new NonOwnedProtoRunGraphResponse(&call->response);

      call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });

      // =====================================================================

      // callback function, until somepoint in the future. so currently,
      // it is registered.
      auto done_cb = [call,
                      call_opts,
                      wrapped_request,
                      wrapped_response](const Status& s) {
        VLOG(1) << "RunGraph::Done";

        if (!s.ok()) {
          VLOG(1) << "Bad response from RunGraph:" << s;
        }

        // ---

        call->ClearCancelCallback();
        delete call_opts;
        delete wrapped_request;
        delete wrapped_response;
        // =================================================================
        call->SendResponse(ToGrpcStatus(s));
        // =================================================================
        // 1.
        // SendResponse
        // tensorflow/core/distributed_runtime/rpc/grpc_call.h
        //
      };

      auto compute_fn = [this,
                         call_opts,
                         wrapped_request,
                         wrapped_response](StatusCallback done) {
        // ===================================================================
        worker_->RunGraphAsync(call_opts,
                               wrapped_request,
                               wrapped_response,
                               done);
        // ===================================================================
      };

      if (cache_) {
        string request_key = call->request.ShortDebugString();
        cache_->LookupOrCompute(request_key, RPCResponse(&call->response),
                                compute_fn, done_cb);
      } else {
        compute_fn(done_cb);
      }
    });
    ENQUEUE_REQUEST(RunGraph, true);
  }

  void RecvTensorHandlerRaw(
      WorkerCall<RecvTensorRequest, ::grpc::ByteBuffer>* call) {
    Schedule([this, call]() {
      CallOptions* call_opts = new CallOptions;
      call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });

      auto done_cb = [call, call_opts](const Status& s) {
        call->ClearCancelCallback();
        delete call_opts;
        if (!s.ok()) {
          VLOG(1) << "Bad response from RecvTensor:" << s;
        }
        call->SendResponse(ToGrpcStatus(s));
        // 1.
        //
      };

      auto compute_fn = [this, &call_opts, &call](StatusCallback done) {
        worker_->GrpcRecvTensorAsync(call_opts, &call->request, &call->response,
                                     done);
        // 1.
        // done 就是上面的 done_cb
        //
      };

      if (cache_) {
        string request_key = call->request.ShortDebugString();
        cache_->LookupOrCompute(request_key, RPCResponse(&call->response),
                                compute_fn, done_cb);
      } else {
        compute_fn(done_cb);
        // 1.
        // callstack
        // compute_fn
        //
      }
    });
    EnqueueRecvTensorRequestRaw();
  }

  void RecvBufHandler(WorkerCall<RecvBufRequest, RecvBufResponse>* call) {
    Schedule([this, call]() {
      CallOptions* call_opts = new CallOptions;
      call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
      worker_->RecvBufAsync(call_opts, &call->request, &call->response,
                            [call, call_opts](const Status& s) {
                              call->ClearCancelCallback();
                              delete call_opts;
                              if (!s.ok()) {
                                VLOG(1) << "Bad response from RecvBuf:" << s;
                              }
                              call->SendResponse(ToGrpcStatus(s));
                            });
    });
    ENQUEUE_REQUEST(RecvBuf, true);
  }

  void CompleteGroupHandler(
      WorkerCall<CompleteGroupRequest, CompleteGroupResponse>* call) {
    Schedule([this, call]() {
      CallOptions* call_opts = new CallOptions;
      call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
      worker_->CompleteGroupAsync(
          call_opts, &call->request, &call->response,
          [call, call_opts](const Status& s) {
            call->ClearCancelCallback();
            delete call_opts;
            if (!s.ok()) {
              VLOG(1) << "Bad response from CompleteGroup:" << s;
            }
            call->SendResponse(ToGrpcStatus(s));
          });
    });
    ENQUEUE_REQUEST(CompleteGroup, true);
  }

  void CompleteInstanceHandler(
      WorkerCall<CompleteInstanceRequest, CompleteInstanceResponse>* call) {
    Schedule([this, call]() {
      CallOptions* call_opts = new CallOptions;
      call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
      worker_->CompleteInstanceAsync(
          call_opts, &call->request, &call->response,
          [call, call_opts](const Status& s) {
            call->ClearCancelCallback();
            delete call_opts;
            if (!s.ok()) {
              VLOG(1) << "Bad response from CompleteInstance:" << s;
            }
            call->SendResponse(ToGrpcStatus(s));
          });
    });
    ENQUEUE_REQUEST(CompleteInstance, false);
  }
#undef ENQUEUE_REQUEST

  void EnqueueRecvTensorRequestRaw() {
    mutex_lock l(shutdown_mu_);
    if (!is_shutdown_) {
      Call<GrpcWorkerServiceThread,
           grpc::WorkerService::AsyncService,
           RecvTensorRequest,
           ::grpc::ByteBuffer>::
          EnqueueRequestForMethod(
              worker_service_,
              cq_.get(),
              static_cast<int>(GrpcWorkerMethod::kRecvTensor),
              &GrpcWorkerServiceThread::RecvTensorHandlerRaw,
              true /* supports cancel*/);
    }
  }

  GrpcWorker* const worker_ = nullptr;  // Not owned.

  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  // 1.
  // CompletionQueue 可以看这个:
  // tutorial 参考:
  // https://grpc.io/docs/languages/cpp/async/
  // 我发现说这个是 一个 async client

  // 2.
  // 为什么要用 tag?
  // uniquely identifies a request.

  // 3.
  //

  std::unique_ptr<Thread> thread_;
  std::unordered_map<int, int> queue_depth_;
  GrpcResponseCache* cache_;


  grpc::WorkerService::AsyncService* const worker_service_;
  // 1.
  // 相关资料:
  // https://www.cnblogs.com/deep-learning-stacks/p/10355770.html

  mutex shutdown_mu_;
  bool is_shutdown_ GUARDED_BY(shutdown_mu_);
  TF_DISALLOW_COPY_AND_ASSIGN(GrpcWorkerServiceThread);
};



class GrpcWorkerService : public AsyncServiceInterface {
  // 1.
  // service
  // gRPC lets you define four kinds of service method: Unary RPCs where the
  // client sends a single request to the server and gets a single response
  // back, just like a normal function call. rpc SayHello(HelloRequest)
  // returns (HelloResponse);

  // 2.
  // 图示: what is service.
  // https://keep.google.com/u/1/#NOTE/1ptjqroVL0xS38J_UGMdDqBXHju0TeTqDPT4VUVHmegSOOCn66AfWr8BWgH8JIw

 public:

  GrpcWorkerService(
    GrpcWorker* worker,
    ::grpc::ServerBuilder* builder,
    GrpcWorkerServiceOptions options)
      : is_shutdown_(false) {
  // 1.
  // GrpcWorkerService spawns one or more GrpcWorkerServiceThreads to service
  // requests.  Each thread operates on an independent completion queue.

  // 2.
  // 什么是 service 呢?
  // Desriptor of an RPC service and its various RPC methods.
  // https://grpc.github.io/grpc/cpp/classgrpc_1_1_service.html

  // 3.
  // worker: GrpcWorker*
  // tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h:38:
  // class GrpcWorker : public Worker

  // 4.
  // tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h:64:
  // struct GrpcWorkerServiceOptions

    builder->RegisterService(&worker_service_);
    // 1.
    // builder : ::grpc::ServerBuilder*
    //
    // API 文档:
    // https://grpc.github.io/grpc/cpp/classgrpc_1_1_server_builder.html
    //
    // 大概说明:
    // A builder class for the creation and startup of grpc::Server instances.
    // ...

    // 2.
    // builder->RegisterService(&worker_service_); 说明
    // worker_service_: grpc::WorkerService::AsyncService
    // "Register a service."

    // 3.
    // 所以, 什么是 service 呢?
    // Desriptor of an RPC service and its various RPC methods.
    // https://grpc.github.io/grpc/cpp/classgrpc_1_1_service.html

    if (options.response_cache_bytes > 0) {
      // 未进入
      // tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h
      // struct GrpcWorkerServiceOptions :: int64 response_cache_bytes = 0;

      cache_.reset(
          new GrpcResponseCache(options.response_cache_bytes,
                                options.response_cache_expires_seconds));
          // 1.
          // GrpcResponseCache
          // tensorflow/core/distributed_runtime/rpc/grpc_response_cache.h:64:
          // class GrpcResponseCache

          // 2.
          // cache !
    }

    for (int i = 0; i < options.num_serving_threads; i++) {
      // 1.
      // struct GrpcWorkerServiceOptions :: int num_serving_threads = 8;
      // for loop: 8 times

      threads_.emplace_back(
          new GrpcWorkerServiceThread(
            worker, // worker: GrpcWorker*
            builder,
            options.queue_depth,
            cache_.get(),
            &worker_service_));
      // 1.
      // GrpcWorkerServiceThread
      // tensorflow/core/distributed_runtime/rpc/grpc_worker_service.cc
      //
      // 描述:
      // GrpcWorkerService spawns one or more GrpcWorkerServiceThreads to service
      // requests.  Each thread operates on an independent completion queue.

      // 2.
      // worker_service_ : grpc::WorkerService::AsyncService* const
      // tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h <== v
      // tensorflow/core/distributed_runtime/rpc/grpc_worker_service.cc <== x

      // 3.
      // threads_ 是什么:
      // std::vector<std::unique_ptr<GrpcWorkerServiceThread>> threads_;

    }
  }

  void Shutdown() override {
    bool did_shutdown = false;
    {
      mutex_lock l(service_shutdown_mu_);
      if (!is_shutdown_) {
        LOG(INFO) << "Shutting down GrpcWorkerService.";
        is_shutdown_ = true;
        did_shutdown = true;
      }
    }

    if (did_shutdown) {
      for (auto& worker_thread : threads_) {
        worker_thread->Shutdown();
      }
    }
  }

  // This method blocks forever handling requests from the completion queue.
  void HandleRPCsLoop() override {
    // 1.
    //
    // tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc
    //
    // HandleRPCsLoop 调用来自 GrpcServer::Start()
    //
    // Status GrpcServer::Start() {
    //   mutex_lock l(mu_);
    //   switch (state_) {
    //     case NEW: {
    //       master_thread_.reset(
    //           env_->StartThread(ThreadOptions(), "TF_master_service",
    //                             [this] { master_service_->HandleRPCsLoop(); }));
    //
    // =======================================================================
    //       worker_thread_.reset(
    //           env_->StartThread(ThreadOptions(), "TF_worker_service",
    //                             [this] { worker_service_->HandleRPCsLoop(); }));
    // =======================================================================
    //
    //       eager_thread_.reset(
    //           env_->StartThread(ThreadOptions(), "TF_eager_service",
    //                             [this] { eager_service_->HandleRPCsLoop(); }));
    //       state_ = STARTED;
    //       LOG(INFO) << "Started server with target: " << target();
    //       return Status::OK();
    //     }
    //     case STARTED:
    //       LOG(INFO) << "Server already started (target: " << target() << ")";
    //       return Status::OK();
    //     case STOPPED:
    //       return errors::FailedPrecondition("Server has stopped.");
    //     default:
    //       LOG(FATAL);
    //   }
    // }

    for (auto& worker_thread : threads_) {
      // 1.
      // threads_ 是什么:
      // std::vector<std::unique_ptr<GrpcWorkerServiceThread>> threads_;

      worker_thread->Start();
      // 1.
      // go to GrpcWorkerServiceThread::Start()
      //
    }


    for (auto& worker_thread : threads_) {
      worker_thread->Join();
    }
  }

 private:

  grpc::WorkerService::AsyncService worker_service_;
  // 1.
  // grpc::WorkerService::AsyncService 是什么? 怎么用的?
  //
  // Asynchronous-API tutorial
  // https://grpc.io/docs/languages/cpp/async/
  //
  // This tutorial shows you how to write a simple server and client in C++
  // using gRPC’s asynchronous/non-blocking APIs.
  //
  // gRPC uses the CompletionQueue API for asynchronous operations.
  //
  // 1.1 bind a CompletionQueue to an RPC call
  // 1.2 do something like a read or write, present with a unique void* tag
  // 1.3 call CompletionQueue::Next to wait for operations to complete. If a tag appears, it indicates that the corresponding operation is complete.
  //
  // === Async client ===
  // To use an asynchronous client to call a remote method,
  // you first create a channel and stub.
  //
  // === server ===
  //

  // 2.
  // tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h
  // grpc::WorkerService::AsyncService
  //
  // 接口定义 (service, rpc methods)
  // tensorflow/core/protobuf/worker_service.proto
  // ...

  // 3.
  // 这段文字写的太好了.
  // tensorflow/core/protobuf/worker_service.proto
  // service WorkerService
  //
  ////////////////////////////////////////////////////////////////////////////////
  //
  // WorkerService defines a TensorFlow service that executes dataflow
  // graphs on a set of local devices, on behalf of a MasterService.
  //
  // A worker service keeps track of multiple "registered graphs". Each
  // registered graph is a subgraph of a client's graph, corresponding to
  // only the nodes that should execute on this worker (and any
  // additional nodes necessary for inter-process communication using
  // the `RecvTensor` method).
  //
  ////////////////////////////////////////////////////////////////////////////////

  // 4.
  // 相关:
  // tensorflow/core/protobuf/worker.proto

  std::vector<std::unique_ptr<GrpcWorkerServiceThread>> threads_;
  // 1.
  // threads_ 会被 GrpcWorkerService 构造函数填充 8 个, 默认情况下.

  std::unique_ptr<GrpcResponseCache> cache_;
  // 1.
  // GrpcResponseCache 是什么
  // tensorflow/core/distributed_runtime/rpc/grpc_response_cache.h:64:
  // class GrpcResponseCache
  // 猜测: worker 的返回值的 cache 吧.

  mutex service_shutdown_mu_;
  bool is_shutdown_ GUARDED_BY(service_shutdown_mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcWorkerService);
};

}  // namespace


GrpcWorker::GrpcWorker(WorkerEnv* worker_env, const ConfigProto& config)
    : Worker(worker_env),
      recent_request_ids_(100000),
      recv_buf_max_chunk_(
          config.experimental().recv_buf_max_chunk() > 0
              ? config.experimental().recv_buf_max_chunk()
              : (config.experimental().recv_buf_max_chunk() < 0 ? 0 : 4096)) {}
// 1.
// recv_buf_max_chunk 是什么?
//
// Guidance to formatting of large RecvBuf fields for transfer.
// Any positive value sets the max chunk size.  0 defaults to 4096.
// Any negative value indicates no max, i.e. one chunk only.
//
// int32 recv_buf_max_chunk;

// 2.


// GrpcRecvTensorAsync: unlike the other Worker methods, which use protocol
// buffers for a response object, to avoid extra protocol buffer serialization
// overhead we generate our response directly into a ::grpc::ByteBuffer object
void GrpcWorker::GrpcRecvTensorAsync(CallOptions* opts,
                                     const RecvTensorRequest* request,
                                     ::grpc::ByteBuffer* response,
                                     StatusCallback done) {
  Status s = recent_request_ids_.TrackUnique(
      request->request_id(), "RecvTensor (GrpcWorker)", *request);
  if (!s.ok()) {
    done(s);
    return;
  }

  const int64 step_id = request->step_id();
  const string& key = request->rendezvous_key();
  TRACEPRINTF("RecvTensor: %lld %s", step_id, key.c_str());
  Rendezvous::ParsedKey parsed;
  s = Rendezvous::ParseKey(key, &parsed);
  Device* src_dev = nullptr;
  if (s.ok()) {
    s = PrepareRecvTensor(parsed, &src_dev);
  }
  if (!s.ok()) {
    done(s);
    return;
  }

  // Request the tensor associated with the rendezvous key.
  // Note that we log the cancellation here but do not abort the current step.
  // gRPC can generate cancellations in response to transient network failures,
  // and aborting the step eliminates the opportunity for client side retries.
  // Repeated client failures will eventually cause the step to be aborted by
  // the client.
  opts->SetCancelCallback(
      [step_id]() { LOG(WARNING) << "RecvTensor cancelled for " << step_id; });

  env_->rendezvous_mgr->RecvLocalAsync(
    // 1.
    // RecvLocalAsync
    // tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc
    // BaseRendezvousMgr::RecvLocalAsync

      step_id,
      parsed,
      [opts, response, done, src_dev, request](
          const Status& status,
          const Rendezvous::Args& send_args,
          const Rendezvous::Args& recv_args,
          const Tensor& val,
          const bool is_dead) {

        opts->ClearCancelCallback();

        if (status.ok()) {
          // DMA can only be used for Tensors that do not fall into
          // the following three odd edge cases: 1) a zero-size
          // buffer, 2) a dead tensor which has an uninit value, and
          // 3) the tensor has the on_host allocation attribute,
          // i.e. it's in CPU RAM *independent of its assigned
          // device type*.
          const bool on_host = send_args.alloc_attrs.on_host();

          {
            // Non-DMA cases.
            if (src_dev->tensorflow_gpu_device_info() && (!on_host)) {
              DeviceContext* send_dev_context = send_args.device_context;
              AllocatorAttributes alloc_attrs;
              alloc_attrs.set_gpu_compatible(true);
              alloc_attrs.set_on_host(true);
              Allocator* alloc = src_dev->GetAllocator(alloc_attrs);
              Tensor* copy = new Tensor(alloc, val.dtype(), val.shape());

              CHECK(send_dev_context)
                  << "send dev name: " << src_dev->name()
                  << " gpu_info: " << src_dev->tensorflow_gpu_device_info();

              // "val" is on an accelerator device. Uses the device_context to
              // fill the copy on host.
              StatusCallback copy_ready = [response, done, copy,
                                           is_dead](const Status& s) {
                // The value is now ready to be returned on the wire.
                grpc::EncodeTensorToByteBuffer(is_dead, *copy, response);
                done(s);
                delete copy;
              };

              send_dev_context->CopyDeviceTensorToCPU(
                  &val,
                  request->rendezvous_key(),
                  src_dev,
                  copy,
                  copy_ready);

            } else {
              grpc::EncodeTensorToByteBuffer(
                is_dead,
                val,
                response);
              done(Status::OK());
            }
          }
        } else {
          //  !s.ok()
          done(status);
        }
      });
}

namespace {
// If RecvBufRespExtra.tensor_content is a single large string, then gRPC
// can stall on the recv side when the string buffer needs to be enlarged,
// since the size is not sent in advance.  Changing this field to a sequence
// of small strings costs some extra time on the send side, since we do
// some otherwise unnecessary copies, but it improves runtime overall by
// improving flow control.  Best performance is likely achieved with a
// max_chunk_bytes equal to the memory page size.
//
// TODO(tucker): When proto3 supports [ctype=CORD] then change
// RecvBufRespExtra.tensor_content to a cord instead of a repeated string,
// and remove this function.
void SetTensorInRecvBufResp(int64 max_chunk_bytes, const Tensor* tensor,
                            int64 num_bytes, RecvBufResponse* response) {
  RecvBufRespExtra extra;
  const char* head = reinterpret_cast<const char*>(DMAHelper::base(tensor));
  while (num_bytes > 0) {
    int64 bytes =
        max_chunk_bytes > 0 ? std::min(num_bytes, max_chunk_bytes) : num_bytes;
    extra.add_tensor_content(std::string(head, bytes));
    head += bytes;
    num_bytes -= bytes;
  }
  response->mutable_transport_options()->PackFrom(extra);
}
}  // namespace

void GrpcWorker::RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                              RecvBufResponse* response, StatusCallback done) {
  // This is a generic, low performance implementation appropriate for grpc.
  Status s = recent_request_ids_.TrackUnique(request->request_id(),
                                             "RecvBuf (GrpcWorker)", *request);
  if (!s.ok()) {
    done(s);
    return;
  }
  CollectiveExecutor::Handle ce_handle(
      env_->collective_executor_mgr->FindOrCreate(request->step_id()), true);
  CollectiveRemoteAccess* rma = ce_handle.get()->remote_access();
  rma->buf_rendezvous()->ConsumeBuf(
      request->buf_rendezvous_key(),
      [this, request, response, done](const Status& status,
                                      BufRendezvous::Hook* hook) {
        Status s = status;
        if (s.ok()) {
          if (!DMAHelper::CanUseDMA(hook->prod_value)) {
            s = errors::Internal("Tensor value for key ",
                                 request->buf_rendezvous_key(),
                                 " is not of a type supported by RecvBuf");
          }
        }
        if (s.ok()) {
          // The RPC source tensor needs to be in CPU RAM.  If not already
          // there make a copy using memory appropriate to the purpose.
          const size_t num_bytes = hook->prod_value->TotalBytes();
          const bool on_host =
              hook->prod_dev->attributes().device_type() == "CPU" ||
              hook->prod_attr.on_host();
          if ((!on_host) && (num_bytes > 0)) {
            Device* cpu_dev = nullptr;
            s = env_->device_mgr->LookupDevice("CPU:0", &cpu_dev);
            if (s.ok()) {
              AllocatorAttributes cpu_attr;
              cpu_attr.set_gpu_compatible(true);
              cpu_attr.set_nic_compatible(true);
              Tensor* cpu_tensor = new Tensor(cpu_dev->GetAllocator(cpu_attr),
                                              hook->prod_value->dtype(),
                                              hook->prod_value->shape());
              hook->prod_ctx->CopyDeviceTensorToCPU(
                  hook->prod_value, "empty_name", hook->prod_dev, cpu_tensor,
                  [this, num_bytes, response, done, hook,
                   cpu_tensor](const Status& s) {
                    if (s.ok()) {
                      SetTensorInRecvBufResp(recv_buf_max_chunk_, cpu_tensor,
                                             num_bytes, response);
                    }
                    response->set_send_start_micros(env_->env->NowMicros());
                    done(s);
                    BufRendezvous::DoneWithHook(hook);
                    delete cpu_tensor;
                  });
              return;
            }
          } else {
            // Tensor is on CPU.
            SetTensorInRecvBufResp(recv_buf_max_chunk_, hook->prod_value,
                                   num_bytes, response);
          }
        }
        response->set_send_start_micros(env_->env->NowMicros());
        done(s);
        BufRendezvous::DoneWithHook(hook);
      });
}

void GrpcWorker::LoggingAsync(const LoggingRequest* request,
                              LoggingResponse* response, StatusCallback done) {
  auto env = this->env();
  if (env) {
    auto session_mgr = env->session_mgr;
    if (session_mgr) {
      if (request->enable_rpc_logging()) {
        session_mgr->SetLogging(true);
      }
      // NOTE(mrry): Handle old masters that disable RPC logging by setting
      // `request->enable_rpc_logging` to `false`.
      if (request->disable_rpc_logging() ||
          (!request->enable_rpc_logging() &&
           request->fetch_step_id_size() == 0)) {
        session_mgr->SetLogging(false);
      }
      for (const auto& step_id : request->fetch_step_id()) {
        session_mgr->RetrieveLogs(step_id, response);
      }
      if (request->clear()) {
        session_mgr->ClearLogs();
      }
    }
  }
  done(Status::OK());
}

WorkerEnv* GrpcWorker::env() { return env_; }

// =======================================================================
// 一个是 worker
// 一个是 service
// =======================================================================
std::unique_ptr<GrpcWorker> NewGrpcWorker(WorkerEnv* env,
                                          const ConfigProto& config) {
  return std::unique_ptr<GrpcWorker>(new GrpcWorker(env, config));
}

std::unique_ptr<AsyncServiceInterface> NewGrpcWorkerService(
    GrpcWorker* worker,
    ::grpc::ServerBuilder* builder,
    GrpcWorkerServiceOptions options) {

  return std::unique_ptr<AsyncServiceInterface>(
      new GrpcWorkerService(worker, builder, options));
}

}  // namespace tensorflow
