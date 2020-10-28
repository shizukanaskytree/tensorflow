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

#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"

#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#include <stdlib.h>

#include "grpc/support/alloc.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_builder.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/local_master.h"
#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/master_session.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_cache_wrapper.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

// Define an option subclass in order to disable SO_REUSEPORT for the
// server socket.
class NoReusePortOption : public ::grpc::ServerBuilderOption {
 public:
  void UpdateArguments(::grpc::ChannelArguments* args) override {
    args->SetInt(GRPC_ARG_ALLOW_REUSEPORT, 0);
  }

  void UpdatePlugins(std::vector<std::unique_ptr<::grpc::ServerBuilderPlugin>>*
                         plugins) override {}
};

// static utility function
RendezvousMgrInterface* NewRpcRendezvousMgr(const WorkerEnv* env) {
  return new RpcRendezvousMgr(env);
}

}  // namespace

GrpcServer::GrpcServer(const ServerDef& server_def, Env* env)
    : server_def_(server_def), env_(env), state_(NEW) {}

GrpcServer::~GrpcServer() {
  TF_CHECK_OK(Stop());
  TF_CHECK_OK(Join());

  delete master_service_;
  delete worker_service_;
  delete eager_service_;

  // TODO(mrry): Refactor the *Env classes so that it is less fiddly
  // to destroy them.

  // Shut down all outstanding rendezvous.
  delete worker_env_.rendezvous_mgr;

  // We must delete graph_mgr before device_mgr, due to shared
  // ownership of OpKernels in the executors. (The graph_mgr will
  // free all stateless OpKernels, and pass over borrowed stateful
  // OpKernels, which are also held in their respective devices'
  // OpSegments.)
  if (worker_env_.session_mgr != nullptr) {
    delete worker_env_.session_mgr;  // Deletes graph_mgr's.
  } else {
    // Note: session_mgr's legacy_session_ deletes device_mgr now.
    delete worker_env_.device_mgr;
  }

  // Do not delete (as these are not owned by the server):
  // - master_env_.env
  // - worker_env_.env
  // - worker_env_.compute_pool
}

void GrpcServer::MaybeMutateBuilder(::grpc::ServerBuilder* builder) {}

Status GrpcServer::Init(const GrpcServerOptions& opts) {
  // 1.
  // cmt:
  // 需要重写一个 ReInit function 内部的很多东西都要重写呢. 那就全部重写吧.

  mutex_lock l(mu_);
  CHECK_EQ(state_, NEW);
  master_env_.env = env_;
  worker_env_.env = env_;

  // Check parameters before DeviceFactory::AddDevices,
  // otherwise if 'task_index=-1' the program will abort.

  // Look up the port that has been requested for this task in `server_def_`.
  int requested_port = -1;
  
  // debug
  VLOG(0) << "server_def_:" << server_def_.DebugString(); 

  for (const auto& job : server_def_.cluster().job()) {
    if (job.name() == server_def_.job_name()) {
      auto iter = job.tasks().find(server_def_.task_index());
      if (iter == job.tasks().end()) {
        return errors::InvalidArgument("Task ", server_def_.task_index(),
                                       " was not defined in job \"",
                                       server_def_.job_name(), "\"");
      }
      auto colon_index = iter->second.find_last_of(':');
      if (!strings::safe_strto32(iter->second.substr(colon_index + 1),
                                 &requested_port)) {
        return errors::InvalidArgument(
            "Could not parse port for local server from \"", iter->second,
            "\".");
      }
      break;
    }
  }
  if (requested_port == -1) {
    return errors::Internal("Job \"", server_def_.job_name(),
                            "\" was not defined in cluster");
  }

  SessionOptions sess_opts;
  ConfigProto config = server_def_.default_session_config();
  sess_opts.config = config;

  // Configure shared devices between master and worker.
  string name_prefix =
      strings::StrCat("/job:", server_def_.job_name(), "/replica:0",
                      "/task:", server_def_.task_index());
  std::vector<std::unique_ptr<Device>> devices;
  // 1.
  // devices 在添加前应该有变. 腾出位置给新的 process, 新的 session.

  TF_RETURN_IF_ERROR(
      DeviceFactory::AddDevices(sess_opts, name_prefix, &devices));
  // 1.
  // AddDevices, output is devices
  // 重写, 重写!!! 还不如重写呢

  worker_env_.device_mgr = new DeviceMgr(std::move(devices));
  master_env_.local_devices = worker_env_.device_mgr->ListDevices();
  worker_env_.local_devices = worker_env_.device_mgr->ListDevices();
  worker_env_.rendezvous_mgr = opts.rendezvous_mgr_func == nullptr
                                   ? new RpcRendezvousMgr(&worker_env_)
                                   : opts.rendezvous_mgr_func(&worker_env_);
  string unused;
  string default_worker_name;
  if (!DeviceNameUtils::SplitDeviceName(master_env_.local_devices[0]->name(),
                                        &default_worker_name, &unused)) {
    return errors::Internal("Could not parse worker name.");
  }

  // N.B. The order of initialization here is intricate, because we
  // wish to allow `requested_port == 0` (for choosing any port,
  // mostly for testing). Therefore, the construction of the channel
  // and worker caches depends on `bound_port_`, which is not set
  // until we call `builder.BuildAndStart()`. We must create the
  // service objects before calling `builder.BuildAndStart()`, but
  // `master_env_` and `worker_env_` are only partially
  // configured. However, this is not dangerous, because we do not
  // start serving requests until `this->Start()` is called, which
  // happens after this method returns.
  //
  // TODO(mrry): Provide a general mechanism for dynamically setting
  // the identities of tasks in the worker pool after the service is
  // running.
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port),
                           GetServerCredentials(server_def_), &bound_port_);
  builder.SetMaxMessageSize(std::numeric_limits<int32>::max());

  builder.SetOption(
      std::unique_ptr<::grpc::ServerBuilderOption>(new NoReusePortOption));
  // Allow subclasses to specify more args to pass to the gRPC server.
  MaybeMutateBuilder(&builder);
  master_impl_ = CreateMaster(&master_env_);
  master_service_ = NewGrpcMasterService(master_impl_.get(), config, &builder);
  worker_impl_ = opts.worker_func ? opts.worker_func(&worker_env_, config)
                                  : NewGrpcWorker(&worker_env_, config);
  worker_service_ = NewGrpcWorkerService(worker_impl_.get(), &builder,
                                         opts.worker_service_options)
                        .release();
  eager_service_ = new eager::GrpcEagerServiceImpl(&worker_env_, &builder);

  // extra service:
  if (opts.service_func != nullptr) {
    opts.service_func(&worker_env_, &builder);
  }
  server_ = builder.BuildAndStart();

  if (!server_) {
    return errors::Unknown("Could not start gRPC server");
  }

  WorkerCacheInterface* worker_cache;
  WorkerCacheFactoryOptions worker_cache_factory_options(server_def_);
  TF_RETURN_IF_ERROR(
      WorkerCacheFactory(worker_cache_factory_options, &worker_cache));
  CHECK_NE(nullptr, worker_cache);

  if (opts.collective_mgr_func) {
    worker_env_.collective_executor_mgr =
        opts.collective_mgr_func(config, &worker_env_, worker_cache);
    if (!worker_env_.collective_executor_mgr) {
      return errors::Internal(
          "collective_mgr_func did not return CollectiveExecutorMgr");
    }
  } else {
    std::unique_ptr<DeviceResolverDistributed> dev_resolver(
        new DeviceResolverDistributed(worker_env_.device_mgr, worker_cache,
                                      default_worker_name));
    std::unique_ptr<CollectiveParamResolverDistributed> param_resolver(
        new CollectiveParamResolverDistributed(config, worker_env_.device_mgr,
                                               dev_resolver.get(), worker_cache,
                                               default_worker_name));
    worker_env_.collective_executor_mgr = new RpcCollectiveExecutorMgr(
        config, worker_env_.device_mgr, std::move(dev_resolver),
        std::move(param_resolver), worker_cache, default_worker_name);
  }

  // Set up worker environment.
  worker_env_.session_mgr = new SessionMgr(
      &worker_env_, SessionMgr::WorkerNameFromServerDef(server_def_),
      std::unique_ptr<WorkerCacheInterface>(worker_cache),
      [this](const ServerDef& server_def, WorkerCacheInterface** worker_cache) {
        WorkerCacheFactoryOptions options(server_def);
        return WorkerCacheFactory(options, worker_cache);
      });
  worker_env_.compute_pool = ComputePool(sess_opts);
  // 1.
  // 这个不能删, 不属于 worker_env_

  // Finish setting up master environment.
  master_env_.ops = OpRegistry::Global();
  // 1.
  // OpRegistry::Global() 这个不能删

  master_env_.worker_cache = worker_cache;
  master_env_.collective_executor_mgr = worker_env_.collective_executor_mgr;
  StatsPublisherFactory stats_factory = opts.stats_factory;
  master_env_.master_session_factory =
      [config, stats_factory](
          SessionOptions options, const MasterEnv* env,
          std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs,
          std::unique_ptr<WorkerCacheInterface> worker_cache,
          std::unique_ptr<DeviceSet> device_set,
          std::vector<string> filtered_worker_list) {
        options.config.MergeFrom(config);
        return new MasterSession(options, env, std::move(remote_devs),
                                 std::move(worker_cache), std::move(device_set),
                                 std::move(filtered_worker_list),
                                 stats_factory);
      };
  master_env_.worker_cache_factory =
      [this](const WorkerCacheFactoryOptions& options,
             WorkerCacheInterface** worker_cache) {
        return WorkerCacheFactory(options, worker_cache);
      };

  // Provide direct access to the master from in-process clients.
  LocalMaster::Register(target(), master_impl_.get(),
                        config.operation_timeout_in_ms());

  return Status::OK();
}

Status GrpcServer::Restart(int selected_dev) {
  // Arg:
  //   selected_dev 描述: platform_gpu_id
  // 
  // -1: CPU
  // 
  //      0 1 2 3
  // 0:   N Y Y Y
  // 1:   Y N Y Y
  // 2:   Y Y N Y
  // 3:   Y Y Y N
  // 
  // 添加专属的 devices, 因为 devices 生变, 所以 sess_opts 内指定一下新的 
  // device 是什么情况.
  // visible devices 4 个, 但是不会全用了, 而且对于这个 server 只是挑了其中一个
  // 而且最差情况是只能用 CPU 而不能用 GPU 了.
  // 还是在 Restart() 里设定参数吧. 用 int 吗? -1, 0, 1, 2, 3, 4, 5, 6, 7
  // 
  // -1: CPU, GPU_0, GPU_1, GPU_2 ...
  // 
  // -1 表示没有 GPU, 就初始化为 CPU server. 可以.
  //
  // hardcode 了试试先, -1: CPU: PASS
  // 0: GPU_0: DOING...
  //
  // int selected_dev = 0; 
  // 
  // 之前的 devices_ 都已经存了, 我想应该要析构掉之前的, 包括 stream_executor 等等
  // 
  // 不行
  // global devices pool --> pick device you want.
  // get all available uninitialized occupied devices, 
  // then pick 1 for the following initialization process.
  // 
  // cmt:
  // 模仿 GrpcServer::Init
  // GrpcServer::Init again! 而不是删除之前的!
  // 
  // 感觉需要所有的 server 都重启, 否则怎么办呢? 每个 server 里面都存着 remote devices 的信息
  // 这个需要被更新啊.
  // 那就都重启喽

  mutex_lock l(mu_);
  
  // 也要参考一下 GrpcServer::~GrpcServer() 
  // -------------------------------------------------------------
  // reset everything in grpc server: GrpcServer.
  // shutdown the server before a new one.
  server_->Shutdown();

  // shutdown services 
  master_service_->Shutdown();
  //master_service_->ShutdownServer();
  worker_service_->Shutdown();
  //worker_service_->ShutdownServer();
  eager_service_->Shutdown();
  //eager_service_->ShutdownServer();

  // -------------------------------------------------------------
  // MasterEnv
  // -------------------------------------------------------------
  // 二次删除, 在 delete worker_env_.session_mgr; 所以不写了.
  // delete master_env_.worker_cache; 
  //delete master_env_.ops;

  master_impl_.reset();
  delete master_service_;
  channel_cache_.reset();

  // -------------------------------------------------------------
  // WorkerEnv
  // -------------------------------------------------------------
  // from GrpcServer::~GrpcServer()
  // We must delete graph_mgr before device_mgr, due to shared
  // ownership of OpKernels in the executors. (The graph_mgr will
  // free all stateless OpKernels, and pass over borrowed stateful
  // OpKernels, which are also held in their respective devices'
  // OpSegments.)
  // -------------------------------------------------------------
  if (worker_env_.session_mgr != nullptr) {
    VLOG(0) << "1. delete worker_env_.session_mgr";
    delete worker_env_.session_mgr;  // Deletes graph_mgr's.
  } else {
    VLOG(0) << "2. delete worker_env_.device_mgr";
    // Note: session_mgr's legacy_session_ deletes device_mgr now.
    delete worker_env_.device_mgr;
  }
  // Shut down all outstanding rendezvous.
  delete worker_env_.rendezvous_mgr;
  // deallocate the old one.
  delete worker_env_.collective_executor_mgr;

  worker_impl_.reset();

  delete worker_service_;
  // -------------------------------------------------------------
  delete eager_service_;
  // -------------------------------------------------------------
  server_.reset();

  master_env_ = {}; // reset
  worker_env_ = {}; // reset
  // -------------------------------------------------------------

  state_ = STARTED;

  master_env_.env = env_;
  worker_env_.env = env_;

  VLOG(0) << "Restart server;";

  GrpcServerOptions opts;
  opts.rendezvous_mgr_func = NewRpcRendezvousMgr;

  // 实际上, 我在这里需要尝试的是 Init 内的重新构建
  SessionOptions sess_opts;
  ConfigProto config = server_def_.default_session_config();
  sess_opts.config = config;
  
  // Configure shared devices between master and worker.
  string name_prefix =
      strings::StrCat("/job:", server_def_.job_name(), "/replica:0",
                      "/task:", server_def_.task_index());

  // --------------------------------------------------------------------
  std::vector<std::unique_ptr<Device>> devices;

  // Returns a string listing all devices.
  //VLOG(0) << worker_env_.device_mgr->DebugString();

  // Returns a string of all the device mapping.
  //VLOG(0) << worker_env_.device_mgr->DeviceMappingString();
  // device_mgr 里面没有 clear devices 的操作.

  // remove all existing devices if it is gpu or xla_gpu 
  // 删除以前要把 variables 备份到 cpu 
  // 迁移一下.

  // 那么 device 自己有没有自焚的呢? 
  // std::vector<Device*> 
  //c// std::vector<Device*> devs = worker_env_.device_mgr->ListDevices();
  //c// for (auto& dev : devs) {
  //c//   VLOG(0) << dev->DebugString();
  //c//   delete dev;
  //c// }

  // note: 删除了 devices 后好像不能再进行如下操作了.
  // 先把以前的 device mgr 清理掉, 因为我要重新构造新的.
  //delete worker_env_.device_mgr;

  TF_RETURN_IF_ERROR(
      DeviceFactory::AddSelectedDevices(sess_opts, name_prefix, 
                                        selected_dev, &devices));

  worker_env_.device_mgr = new DeviceMgr(std::move(devices));

  // print again.
  // Returns a string listing all devices.
  VLOG(0) << worker_env_.device_mgr->DebugString();
  // Returns a string of all the device mapping.
  VLOG(0) << worker_env_.device_mgr->DeviceMappingString();
  
  //master_env_.local_devices.clear();
  master_env_.local_devices = worker_env_.device_mgr->ListDevices();
  
  //worker_env_.local_devices.clear();
  worker_env_.local_devices = worker_env_.device_mgr->ListDevices();

  //worker_env_.rendezvous_mgr->CleanupAll();
  //delete worker_env_.rendezvous_mgr;
  worker_env_.rendezvous_mgr = opts.rendezvous_mgr_func == nullptr
                                   ? new RpcRendezvousMgr(&worker_env_)
                                   : opts.rendezvous_mgr_func(&worker_env_);

  // -------------------------------------------------------------
  // Look up the port that has been requested for this task in `server_def_`.
  int requested_port = -1;
  for (const auto& job : server_def_.cluster().job()) {
    if (job.name() == server_def_.job_name()) {
      auto iter = job.tasks().find(server_def_.task_index());
      if (iter == job.tasks().end()) {
        return errors::InvalidArgument("Task ", server_def_.task_index(),
                                       " was not defined in job \"",
                                       server_def_.job_name(), "\"");
      }
      auto colon_index = iter->second.find_last_of(':');
      if (!strings::safe_strto32(iter->second.substr(colon_index + 1),
                                 &requested_port)) {
        return errors::InvalidArgument(
            "Could not parse port for local server from \"", iter->second,
            "\".");
      }
      break;
    }
  }
  if (requested_port == -1) {
    return errors::Internal("Job \"", server_def_.job_name(),
                            "\" was not defined in cluster");
  }
  // -------------------------------------------------------------

  string unused;
  string default_worker_name;
  if (!DeviceNameUtils::SplitDeviceName(master_env_.local_devices[0]->name(),
                                        &default_worker_name, &unused)) {
    return errors::Internal("Could not parse worker name.");
  }

  // -------------------------------------------------------------

  // grpc server
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port),
                           GetServerCredentials(server_def_), &bound_port_);
  
  VLOG(0) << "Debug, bound_port_:" << bound_port_;

  builder.SetMaxMessageSize(std::numeric_limits<int32>::max());

  builder.SetOption(
      std::unique_ptr<::grpc::ServerBuilderOption>(new NoReusePortOption));

  // Allow subclasses to specify more args to pass to the gRPC server.
  MaybeMutateBuilder(&builder);

  // -------------------------------------------------------------
  master_impl_ = CreateMaster(&master_env_);
  master_service_ = NewGrpcMasterService(master_impl_.get(), config, &builder);
  // -------------------------------------------------------------
  worker_impl_ = opts.worker_func ? opts.worker_func(&worker_env_, config)
                                  : NewGrpcWorker(&worker_env_, config);
  worker_service_ = NewGrpcWorkerService(worker_impl_.get(), &builder,
                                         opts.worker_service_options)
                        .release();
  // -------------------------------------------------------------
  eager_service_ = new eager::GrpcEagerServiceImpl(&worker_env_, &builder);
  // -------------------------------------------------------------

  // extra service:
  if (opts.service_func != nullptr) {
    VLOG(0) << "Enter! opts.service_func != nullptr";
    opts.service_func(&worker_env_, &builder);
  }

  server_ = builder.BuildAndStart(); // OK!
  if (!server_) {
    return errors::Unknown("Could not start gRPC server");
  }

  // 在 WorkerCacheFactory 里面会 new channel_cache_.
  WorkerCacheInterface* worker_cache;
  WorkerCacheFactoryOptions worker_cache_factory_options(server_def_);
  TF_RETURN_IF_ERROR(
      WorkerCacheFactory(worker_cache_factory_options, &worker_cache));
  CHECK_NE(nullptr, worker_cache);

  // -------------------------------------------------------------

  if (opts.collective_mgr_func) {
    worker_env_.collective_executor_mgr =
        opts.collective_mgr_func(config, &worker_env_, worker_cache);
    if (!worker_env_.collective_executor_mgr) {
      return errors::Internal(
          "collective_mgr_func did not return CollectiveExecutorMgr");
    }
  } else {
    std::unique_ptr<DeviceResolverDistributed> dev_resolver(
        new DeviceResolverDistributed(worker_env_.device_mgr, worker_cache,
                                      default_worker_name));
    std::unique_ptr<CollectiveParamResolverDistributed> param_resolver(
        new CollectiveParamResolverDistributed(config, worker_env_.device_mgr,
                                               dev_resolver.get(), worker_cache,
                                               default_worker_name));
    worker_env_.collective_executor_mgr = new RpcCollectiveExecutorMgr(
        config, worker_env_.device_mgr, std::move(dev_resolver),
        std::move(param_resolver), worker_cache, default_worker_name);
  }

  // Set up worker environment.
  worker_env_.session_mgr = new SessionMgr(
      &worker_env_, SessionMgr::WorkerNameFromServerDef(server_def_),
      std::unique_ptr<WorkerCacheInterface>(worker_cache),
      [this](const ServerDef& server_def, WorkerCacheInterface** worker_cache) {
        WorkerCacheFactoryOptions options(server_def);
        return WorkerCacheFactory(options, worker_cache);
      });
  
  // Do not delete (as these are not owned by the server):
  // - master_env_.env
  // - worker_env_.env
  // - worker_env_.compute_pool
  worker_env_.compute_pool = ComputePool(sess_opts);

  // Finish setting up master environment.
  master_env_.ops = OpRegistry::Global();
  master_env_.worker_cache = worker_cache;
  master_env_.collective_executor_mgr = worker_env_.collective_executor_mgr;
  StatsPublisherFactory stats_factory = opts.stats_factory;
  master_env_.master_session_factory =
      [config, stats_factory](
          SessionOptions options, const MasterEnv* env,
          std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs,
          std::unique_ptr<WorkerCacheInterface> worker_cache,
          std::unique_ptr<DeviceSet> device_set,
          std::vector<string> filtered_worker_list) {
        options.config.MergeFrom(config);
        return new MasterSession(options, env, std::move(remote_devs),
                                 std::move(worker_cache), std::move(device_set),
                                 std::move(filtered_worker_list),
                                 stats_factory);
      };
  master_env_.worker_cache_factory =
      [this](const WorkerCacheFactoryOptions& options,
             WorkerCacheInterface** worker_cache) {
        return WorkerCacheFactory(options, worker_cache);
      };

  // Provide direct access to the master from in-process clients.
  LocalMaster::Register(target(), master_impl_.get(),
                        config.operation_timeout_in_ms());

  // then restart threads of master, worker, and eager.
  master_thread_.reset(
      env_->StartThread(ThreadOptions(), "TF_master_service",
                        [this] { master_service_->HandleRPCsLoop(); }));
  worker_thread_.reset(
      env_->StartThread(ThreadOptions(), "TF_worker_service",
                        [this] { worker_service_->HandleRPCsLoop(); }));

  eager_thread_.reset(
      env_->StartThread(ThreadOptions(), "TF_eager_service",
                        [this] { eager_service_->HandleRPCsLoop(); }));

  return Status::OK();
}

Status GrpcServer::Shutdown(int selected_dev) {
  // 会挂, 为什么呢???
  // 这所谓的 shutdown 就是个坑!!!

  mutex_lock l(mu_);
 
  // 也要参考一下 GrpcServer::~GrpcServer() 
  // -------------------------------------------------------------
  // reset everything in grpc server: GrpcServer.

  // shutdown the server before a new one.
  server_->Shutdown();
  // shutdown services 
  master_service_->Shutdown();
  worker_service_->Shutdown();
  eager_service_->Shutdown();
  //bug?// master_service_->ShutdownServer();
  //bug?// worker_service_->ShutdownServer();
  //bug?// eager_service_->ShutdownServer();

  // 如下是删除什么都会错错错. 不妨重新 init 吧, 这里的 service 都 shutdown 了
  GrpcServerOptions opts;
  opts.rendezvous_mgr_func = NewRpcRendezvousMgr;

  VLOG(0) << "state: NEW:0, STARTED:1, STOPPED:2 " << state_;
  master_env_.env = env_;
  worker_env_.env = env_;

  SessionOptions sess_opts;
  ConfigProto config = server_def_.default_session_config();
  sess_opts.config = config;

  string name_prefix =
      strings::StrCat("/job:", server_def_.job_name(), "/replica:0",
                      "/task:", server_def_.task_index());
  std::vector<std::unique_ptr<Device>> devices;

  // 总得把以前的删除吧
  //t// if (worker_env_.session_mgr != nullptr) {
  //t//   VLOG(0) << "1. delete worker_env_.session_mgr";
  //t//   delete worker_env_.session_mgr;  // Deletes graph_mgr's.
  //t// } else {
  //t//   VLOG(0) << "2. delete worker_env_.device_mgr";
  //t//   // Note: session_mgr's legacy_session_ deletes device_mgr now.
  //t//   delete worker_env_.device_mgr;
  //t// }
  delete worker_env_.session_mgr;  // Deletes graph_mgr's.
  delete worker_env_.device_mgr;

  // hardcoded now
  VLOG(0) << "selected_dev: " << selected_dev;
  TF_RETURN_IF_ERROR(
      DeviceFactory::AddSelectedDevices(sess_opts, name_prefix, 
                                        selected_dev, &devices));
  
  worker_env_.device_mgr = new DeviceMgr(std::move(devices));

  worker_env_.device_mgr->DebugString();
  worker_env_.device_mgr->DeviceMappingString();

  master_env_.local_devices = worker_env_.device_mgr->ListDevices();
  worker_env_.local_devices = worker_env_.device_mgr->ListDevices();
  worker_env_.rendezvous_mgr = opts.rendezvous_mgr_func == nullptr
                                   ? new RpcRendezvousMgr(&worker_env_)
                                   : opts.rendezvous_mgr_func(&worker_env_);
  
  string unused;
  string default_worker_name;
  if (!DeviceNameUtils::SplitDeviceName(master_env_.local_devices[0]->name(),
                                        &default_worker_name, &unused)) {
    return errors::Internal("Could not parse worker name.");
  }
  VLOG(0) << "default_worker_name: " << default_worker_name;

  // Look up the port that has been requested for this task in `server_def_`.
  int requested_port = -1;
  // debug
  VLOG(0) << "server_def_:" << server_def_.DebugString(); 
  for (const auto& job : server_def_.cluster().job()) {
    if (job.name() == server_def_.job_name()) {
      auto iter = job.tasks().find(server_def_.task_index());
      if (iter == job.tasks().end()) {
        return errors::InvalidArgument("Task ", server_def_.task_index(),
                                       " was not defined in job \"",
                                       server_def_.job_name(), "\"");
      }
      auto colon_index = iter->second.find_last_of(':');
      if (!strings::safe_strto32(iter->second.substr(colon_index + 1),
                                 &requested_port)) {
        return errors::InvalidArgument(
            "Could not parse port for local server from \"", iter->second,
            "\".");
      }
      break;
    }
  }
  if (requested_port == -1) {
    return errors::Internal("Job \"", server_def_.job_name(),
                            "\" was not defined in cluster");
  }

  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port),
                           GetServerCredentials(server_def_), &bound_port_);
  builder.SetMaxMessageSize(std::numeric_limits<int32>::max());
  builder.SetOption(
      std::unique_ptr<::grpc::ServerBuilderOption>(new NoReusePortOption));
  // Allow subclasses to specify more args to pass to the gRPC server.
  MaybeMutateBuilder(&builder);

  master_impl_ = CreateMaster(&master_env_);
  master_service_ = NewGrpcMasterService(master_impl_.get(), config, &builder);
  worker_impl_ = opts.worker_func ? opts.worker_func(&worker_env_, config)
                                  : NewGrpcWorker(&worker_env_, config);
  worker_service_ = NewGrpcWorkerService(worker_impl_.get(), &builder,
                                         opts.worker_service_options)
                        .release();
  eager_service_ = new eager::GrpcEagerServiceImpl(&worker_env_, &builder);

  // extra service:
  if (opts.service_func != nullptr) {
    opts.service_func(&worker_env_, &builder);
  }
  server_ = builder.BuildAndStart();

  if (!server_) {
    return errors::Unknown("Could not start gRPC server");
  }

  WorkerCacheInterface* worker_cache;
  WorkerCacheFactoryOptions worker_cache_factory_options(server_def_);
  TF_RETURN_IF_ERROR(
      WorkerCacheFactory(worker_cache_factory_options, &worker_cache));
  CHECK_NE(nullptr, worker_cache);

  if (opts.collective_mgr_func) {
    worker_env_.collective_executor_mgr =
        opts.collective_mgr_func(config, &worker_env_, worker_cache);
    if (!worker_env_.collective_executor_mgr) {
      return errors::Internal(
          "collective_mgr_func did not return CollectiveExecutorMgr");
    }
  } else {
    std::unique_ptr<DeviceResolverDistributed> dev_resolver(
        new DeviceResolverDistributed(worker_env_.device_mgr, worker_cache,
                                      default_worker_name));
    std::unique_ptr<CollectiveParamResolverDistributed> param_resolver(
        new CollectiveParamResolverDistributed(config, worker_env_.device_mgr,
                                               dev_resolver.get(), worker_cache,
                                               default_worker_name));
    worker_env_.collective_executor_mgr = new RpcCollectiveExecutorMgr(
        config, worker_env_.device_mgr, std::move(dev_resolver),
        std::move(param_resolver), worker_cache, default_worker_name);
  }

    // Set up worker environment.
  worker_env_.session_mgr = new SessionMgr(
      &worker_env_, SessionMgr::WorkerNameFromServerDef(server_def_),
      std::unique_ptr<WorkerCacheInterface>(worker_cache),
      [this](const ServerDef& server_def, WorkerCacheInterface** worker_cache) {
        WorkerCacheFactoryOptions options(server_def);
        return WorkerCacheFactory(options, worker_cache);
      });
  worker_env_.compute_pool = ComputePool(sess_opts);

  // Finish setting up master environment.
  master_env_.ops = OpRegistry::Global();

  master_env_.worker_cache = worker_cache;
  master_env_.collective_executor_mgr = worker_env_.collective_executor_mgr;
  StatsPublisherFactory stats_factory = opts.stats_factory;
  master_env_.master_session_factory =
      [config, stats_factory](
          SessionOptions options, const MasterEnv* env,
          std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs,
          std::unique_ptr<WorkerCacheInterface> worker_cache,
          std::unique_ptr<DeviceSet> device_set,
          std::vector<string> filtered_worker_list) {
        options.config.MergeFrom(config);
        return new MasterSession(options, env, std::move(remote_devs),
                                 std::move(worker_cache), std::move(device_set),
                                 std::move(filtered_worker_list),
                                 stats_factory);
      };
  master_env_.worker_cache_factory =
      [this](const WorkerCacheFactoryOptions& options,
             WorkerCacheInterface** worker_cache) {
        return WorkerCacheFactory(options, worker_cache);
      };

  // 能不能 de-register 呢???
  LocalMaster::Erase(target());

  // Provide direct access to the master from in-process clients.
  LocalMaster::Register(target(), master_impl_.get(),
                        config.operation_timeout_in_ms());

    // then restart threads of master, worker, and eager.
  master_thread_.reset(
      env_->StartThread(ThreadOptions(), "TF_master_service",
                        [this] { master_service_->HandleRPCsLoop(); }));
  worker_thread_.reset(
      env_->StartThread(ThreadOptions(), "TF_worker_service",
                        [this] { worker_service_->HandleRPCsLoop(); }));

  eager_thread_.reset(
      env_->StartThread(ThreadOptions(), "TF_eager_service",
                        [this] { eager_service_->HandleRPCsLoop(); }));


  // -------------------------------------------------------------
  // MasterEnv
  // -------------------------------------------------------------
  // 二次删除, 在 delete worker_env_.session_mgr; 所以不写了.
  //dumped// delete master_env_.worker_cache; 

  // -------------------------------------------------------------
  // 是不是应该按照先构建的后消除的逻辑???
  //t// delete master_env_.collective_executor_mgr;
  //t// delete master_env_.worker_cache;

  // 这样顺带就把 worker_env_.collective_executor_mgr 给删了..
  // master_env_.collective_executor_mgr = worker_env_.collective_executor_mgr;

  //t// channel_cache_.reset();

  //t// server_.reset(); // ok...

  //t// delete master_service_;
  //t// delete worker_service_;
  //t// delete eager_service_;

  //t// worker_impl_.reset();
  //t// master_impl_.reset();

  //t// delete worker_env_.rendezvous_mgr;
  
  //dumped// delete worker_env_.device_mgr; 
  // 不能再删了,因为 worker_env_.session_mgr 里面的原因.

  // We must delete graph_mgr before device_mgr, due to shared
  // ownership of OpKernels in the executors. (The graph_mgr will
  // free all stateless OpKernels, and pass over borrowed stateful
  // OpKernels, which are also held in their respective devices'
  // OpSegments.)
  // -------------------------------------------------------------
  //t// if (worker_env_.session_mgr != nullptr) {
  //t//   VLOG(0) << "1. delete worker_env_.session_mgr";
  //t//   delete worker_env_.session_mgr;  // Deletes graph_mgr's.
  //t// } else {
  //t//   VLOG(0) << "2. delete worker_env_.device_mgr";
  //t//   // Note: session_mgr's legacy_session_ deletes device_mgr now.
  //t//   delete worker_env_.device_mgr;
  //t// }

  //t// delete worker_env_.device_mgr;

  VLOG(0) << "Shutdown the server and release resource in the server!"; 
  return Status::OK();
}

Status GrpcServer::ParseChannelSpec(const WorkerCacheFactoryOptions& options,
                                    GrpcChannelSpec* channel_spec) {
  for (const auto& job : options.cluster_def->job()) {
    std::map<int, string> host_ports;
    for (const auto& task : job.tasks()) {
      string& host_port = host_ports[task.first];
      if (!host_port.empty()) {
        return errors::InvalidArgument("JobDef for job \"", job.name(),
                                       "\" specified two addresses for task \"",
                                       task.first, "\": ", host_port, " and ",
                                       task.second);
      }
      if (job.name() == *options.job_name && task.first == options.task_index) {
        host_port = strings::StrCat("localhost:", bound_port_);
      } else {
        host_port = task.second;
      }
    }
    TF_RETURN_IF_ERROR(channel_spec->AddHostPortsJob(job.name(), host_ports));
  }
  return Status::OK();
}

Status GrpcServer::WorkerCacheFactory(const WorkerCacheFactoryOptions& options,
                                      WorkerCacheInterface** worker_cache) {
  if (options.job_name == nullptr || options.job_name->empty()) {
    Status s = errors::InvalidArgument(
        "The master (current machine) is not included in the provided "
        "cluster_def. ",
        options.cluster_def->DebugString());
    LOG(WARNING) << s;
    return s;
  }

  GrpcChannelSpec channel_spec;
  TF_RETURN_IF_ERROR(ParseChannelSpec(options, &channel_spec));

  channel_cache_.reset(
      NewGrpcChannelCache(channel_spec, GetChannelCreationFunction()));

  string name_prefix = strings::StrCat("/job:", *options.job_name, "/replica:0",
                                       "/task:", options.task_index);

  const string host_port = channel_cache_->TranslateTask(name_prefix);
  int requested_port;

  auto colon_index = host_port.find_last_of(':');
  if (!strings::safe_strto32(host_port.substr(colon_index + 1),
                             &requested_port)) {
    return errors::Internal("Could not parse port for local server from \"",
                            host_port, "\".");
  }

  if (requested_port != bound_port_) {
    return errors::InvalidArgument("Requested port ", requested_port,
                                   " differs from expected port ", bound_port_);
  }

  *worker_cache = NewGrpcWorkerCacheWithLocalWorker(
      channel_cache_, worker_impl_.get(), name_prefix);
  return Status::OK();
}

Status GrpcServer::Start() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW: {
      master_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_master_service",
                            [this] { master_service_->HandleRPCsLoop(); }));
      worker_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_worker_service",
                            [this] { worker_service_->HandleRPCsLoop(); }));
      eager_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_eager_service",
                            [this] { eager_service_->HandleRPCsLoop(); }));
      state_ = STARTED;
      LOG(INFO) << "Started server with target: " << target();
      return Status::OK();
    }
    case STARTED:
      LOG(INFO) << "Server already started (target: " << target() << ")";
      return Status::OK();
    case STOPPED:
      return errors::FailedPrecondition("Server has stopped.");
    default:
      LOG(FATAL);
  }
}

Status GrpcServer::Stop() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      state_ = STOPPED;
      return Status::OK();
    case STARTED:
      return errors::Unimplemented(
          "Clean shutdown is not currently implemented");
    case STOPPED:
      LOG(INFO) << "Server already stopped (target: " << target() << ")";
      return Status::OK();
    default:
      LOG(FATAL);
  }
}

Status GrpcServer::Join() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      // Prevent the server from being started subsequently.
      state_ = STOPPED;
      return Status::OK();
    case STARTED:
    case STOPPED:
      master_thread_.reset();
      worker_thread_.reset();
      eager_thread_.reset();
      return Status::OK();
    default:
      LOG(FATAL);
  }
}

const string GrpcServer::target() const {
  return strings::StrCat("grpc://localhost:", bound_port_);
}

std::shared_ptr<::grpc::ServerCredentials> GrpcServer::GetServerCredentials(
    const ServerDef& server_def) const {
  return ::grpc::InsecureServerCredentials();
}

ChannelCreationFunction GrpcServer::GetChannelCreationFunction() const {
  // We can do this because SparseGrpcChannelCache is robust to nullptr being
  // returned by the channel creation function
  return ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
}

std::unique_ptr<Master> GrpcServer::CreateMaster(MasterEnv* master_env) {
  return std::unique_ptr<Master>(new Master(master_env, 0.0));
}

/* static */
Status GrpcServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<GrpcServer> ret(
      new GrpcServer(server_def, env == nullptr ? Env::Default() : env));
  // 为什么不在这里重来呢? 是不是? 哈哈哈
  // 删掉以前的, 重建一个吧.
  // 如何 stop the server 呢?

  ServiceInitFunction service_func = nullptr;
  GrpcServerOptions options;
  options.rendezvous_mgr_func = NewRpcRendezvousMgr;
  Status s = ret->Init(options);
  if (!s.ok()) {
    LOG(ERROR) << s;
    return s;
  }
  *out_server = std::move(ret);
  return Status::OK();
}

/* static */
Status GrpcServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<GrpcServer>* out_server) {
  std::unique_ptr<GrpcServer> ret(
      new GrpcServer(server_def, env == nullptr ? Env::Default() : env));
  GrpcServerOptions options;
  options.rendezvous_mgr_func = NewRpcRendezvousMgr;
  Status s = ret->Init(options);
  if (!s.ok()) {
    LOG(ERROR) << s;
    return s;
  }
  *out_server = std::move(ret);
  return Status::OK();
}

namespace {

class GrpcServerFactory : public ServerFactory {
 public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return server_def.protocol() == "grpc";
  }

  Status NewServer(const ServerDef& server_def,
                   std::unique_ptr<ServerInterface>* out_server) override {
    return GrpcServer::Create(server_def, Env::Default(), out_server);
  }
};

// Registers a `ServerFactory` for `GrpcServer` instances.
class GrpcServerRegistrar {
 public:
  GrpcServerRegistrar() {
    gpr_allocation_functions alloc_fns;
    memset(&alloc_fns, 0, sizeof(alloc_fns));
    alloc_fns.malloc_fn = port::Malloc;
    alloc_fns.realloc_fn = port::Realloc;
    alloc_fns.free_fn = port::Free;
    gpr_set_allocation_functions(alloc_fns);
    ServerFactory::Register("GRPC_SERVER", new GrpcServerFactory());
  }
};
static GrpcServerRegistrar registrar;

}  // namespace
}  // namespace tensorflow
