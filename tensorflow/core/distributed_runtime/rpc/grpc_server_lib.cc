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

/** \file grpc_server_lib.cc
 *
 *  \brief Server side grpc handler function to handle various rpc request from
 *         client side.
 *
 *  \details
 *
 *
 */

// 这个 code file 的逻辑和框架定义在了这个 interface 内
// tensorflow/core/distributed_runtime/server_lib.h
//
// 其次, 拎清楚:
// tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h

// =======================================================================
// 设计思想
// https://stackoverflow.com/questions/40979688/tensorflow-using-parameter-servers-in-distributed-training
// https://stackoverflow.com/questions/48318936/how-tensorflow-worker-driver-training-process-and-cause-variables-update-on-ps-j
// =======================================================================

// 例子 ps , how to place parameters.
// https://stackoverflow.com/questions/37712509/how-to-run-tensorflow-distributed-mnist-example/37823338#37823338

#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"

#include <cstring>
#include <limits>
#include <memory>
#include <vector>

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

// =======================================================================
#include "tensorflow/core/distributed_runtime/master.h"
// =======================================================================

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
  // 1.
  // class RpcRendezvousMgr : public BaseRendezvousMgr
  // tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h:45:

  // 2.
  // 

}

}  // namespace


/** \brief Grpc Server Constructor
 *
 *  \param server_def
 *         Server definition from message ServerDef, including 1. ClusterDef,
 *         2.job name, 3. task index, 4. configuration for sessions on this
 *         server, 5. protocol used by this server.
 *
 *  \param env
 *         tensorflow/core/platform/env.h defines class Env
 *         Here, it is be a class PosixEnv : public Env in
 *         tensorflow/core/platform/posix/env.cc
 *
 */
GrpcServer::GrpcServer(const ServerDef& server_def, Env* env)
    : server_def_(server_def), env_(env), state_(NEW) {}
// 1.
// why env_ ?
// StartThread()

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

  // cmt:
  // 随便搞搞了, 有的问题谁知道呢, 比如上面的
  //~cmt// - worker_env_.compute_pool
}

void GrpcServer::MaybeMutateBuilder(::grpc::ServerBuilder* builder) {}


/** \brief Initialize Grpc Server
 *
 *  \details 1. Initialize `Env* env` to `struct MasterEnv master_env_` and
 *           `struct WorkerEnv worker_env_`.
 *           2.
 *
 *  \param opts A collection of function pointer to service init function,
 *              rendezvous manager creation function, collective manager
 *              creation function worker creation function, statistics publisher
 *              factory, grpc worker service Options.
 *
 *  \return Status
 */
Status GrpcServer::Init(const GrpcServerOptions& opts) {
  // 1.
  // Tutorial about "Starting the server"
  // https://grpc.io/docs/languages/cpp/basics/#starting-the-server

  // 2.
  // design note
  // 就这个设计而言, 如果 device 数量变化, 那么就要重新进行 GrpcServer::Init 了

  // 3.
  // struct GrpcServerOptions
  // tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h:63:

  // 4.
  // Why? Purpose?
  //

  mutex_lock l(mu_);
  CHECK_EQ(state_, NEW);
  master_env_.env = env_;
  worker_env_.env = env_;
  // 1.
  // master and worker
  // https://keep.google.com/u/1/#NOTE/1W1lFHD5dvpy5oFxFNLB1mTtcm2Ojw2PDEqt_pgZvklWP3Aetn1FKnnR04xm1

  // 2.
  // gRPC implementation
  // https://keep.google.com/u/1/#NOTE/13QMApUGKMwckCOFeWjs3I7DXC_qrtg1NyhboiogDjlih0SxuR8bVA3melV93

  // ===

  // Check parameters before DeviceFactory::AddDevices,
  // otherwise if 'task_index=-1' the program will abort.



  // 2.
  // master_env_ 和 worker_env_ 最后怎么样了呢? 用在哪里了呢?
  //

  int requested_port = -1;
  for (const auto& job : server_def_.cluster().job()) {
    // 1.
    // Look up the port that has been requested for this task in `server_def_`.
    //
    // server_def_
    //
    // server_def_:
    // cluster {
    //   job {
    //     name: "ps"
    //     tasks {
    //       key: 0
    //       value: "localhost:2222"
    //     }
    //   }
    //   job {
    //     name: "worker"
    //     tasks {
    //       key: 0
    //       value: "localhost:2223"
    //     }
    //     tasks {
    //       key: 1
    //       value: "localhost:2224"
    //     }
    //   }
    // }
    // job_name: "ps"
    // default_session_config {
    // }
    // protocol: "grpc"
    //

    // 2.
    // ServerDef 是什么?
    // message ServerDef
    // tensorflow/core/protobuf/tensorflow_server.proto:28
    //
    // ServerDef
    // - cluster
    // - job_name
    //
    // 具体的话:
    //
    // server = tf.train.Server(cluster,
    //       job_name="ps",
    //       task_index=FLAGS.task_index,
    //       config=config


    // 2.
    // cluster = tf.train.ClusterSpec({
    //   'ps':['localhost:2222'],
    //   'worker':['localhost:2223', 'localhost:2224']
    // }

    // 每个 Job 是什么?
    // job 1.
    //   job {
    //     name: "ps"
    //     tasks {
    //       key: 0
    //       value: "localhost:2222"
    //     }
    //   }

    // job 2.
    //   job {
    //     name: "worker"
    //     tasks {
    //       key: 0
    //       value: "localhost:2223"
    //     }
    //     tasks {
    //       key: 1
    //       value: "localhost:2224"
    //     }
    //   }

    if (job.name() == server_def_.job_name()) {
      // See definition of `JobDef` in tensorflow/core/protobuf/cluster.proto;
      // Template of job string is /job:JobDef.name/task:JobDef.tasks.index;
      // e.g., /job:worker/task:0;
      // job.tasks() is a map from task index (int) to "hostname:port" string.
      // So, job.tasks().find(server_def_.task_index()) returns iter points to
      //    std::pair of {task index , "hostname:port"}

      // 1.
      // 打印:
      // "worker": server_def_.job_name()

      auto iter = job.tasks().find(server_def_.task_index());
      // 1.
      // p server_def_.task_index()
      // $4 = 0

      // 2.
      // p job.tasks().size()
      // $6 = 2
      // 应该是这两个:
      //   job {
      //     name: "worker"
      //     tasks {
      //       key: 0
      //       value: "localhost:2223"
      //     }
      //     tasks {
      //       key: 1
      //       value: "localhost:2224"
      //     }
      //   }

      // 3.
      // job.tasks() type
      // (google::protobuf::Map<int, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > &)

      if (iter == job.tasks().end()) {
        return errors::InvalidArgument("Task ", server_def_.task_index(),
                                       " was not defined in job \"",
                                       server_def_.job_name(), "\"");
      }

      auto colon_index = iter->second.find_last_of(':');
      // 1.
      // p iter->second
      // $7 = "localhost:2223"
      //
      // iter points to the "hostname:port" string
      // `JobDef.tasks.string` can be `"ip:2222"`

      if (!strings::safe_strto32(iter->second.substr(colon_index + 1),
                                 &requested_port)) {
        // 1.
        // \fn strings::safe_strto32
        //
        // \brief string number to int32 number
        //        For example, "2223" --> 2223
        //        The result is stored into requested_port.

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
  // 1.
  // struct SessionOptions
  // core/public/session_options.h:28
  //
  // Initially, sess_opts is empty without any parameters being set.
  // struct SessionOptions is configuration information for a Session.
  // It includes
  // 1. the environment to use;
  // 2. target string to connect to at runtime;
  // 2.1 The target string is used from the client's perspective;
  //     It is defined in  message CreateSessionRequest in master rpc service.
  //     tensorflow/core/protobuf/master.proto;
  // 2.2 target string can be local; ip:port; host:port;
  // 3. ConfigProto which contains Session configuration parameters.

  // 2.
  // +--------------------+
  // |SessionOptions      |
  // |+-----------------+ |
  // ||ConfigProto      | |
  // ||...              | |
  // |+-----------------+ |
  // +--------------------+

  /// ConfigProto contains Session configuration parameters.
  ConfigProto config = server_def_.default_session_config();
  // 1.
  // message ConfigProto
  // tensorflow/core/protobuf/config.proto
  // Briefly speaking:
  // all 开关 s, options

  sess_opts.config = config;

  // Configure shared devices between master and worker.
  string name_prefix =
      strings::StrCat("/job:", server_def_.job_name(), "/replica:0",
                      "/task:", server_def_.task_index());
  // 1.
  // eg.
  // name_prefix
  // -----------------------------------------------------------------------
  // "/job:worker/replica:0/task:1"
  // -----------------------------------------------------------------------
  //
  // when server_def_ is
  // python ssgd.py --job_name "worker" --task_index 1 --cuda_devices "1"

  // =======================================================================
  std::vector<std::unique_ptr<Device>> devices;
  // =======================================================================
  // 1.
  // 对于 PS 我都是用 CPU
  // python ssgd.py --job_name "ps" --task_index 0 --cuda_devices ""

  // =======================================================================
  TF_RETURN_IF_ERROR(
      DeviceFactory::AddDevices(sess_opts, name_prefix, &devices));
  // =======================================================================
  // 1.
  // device! device! device!
  // Construct CPU first and then GPU devices.
  // ...

  // worker
  // master

  /// Construct a device manager to store and read device basic information
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
    // 1.
    // what is default_worker_name?
    //

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
  // 1.
  // build a server:
  // tutorial: https://grpc.io/docs/languages/cpp/basics/#starting-the-server

  builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port),
                           GetServerCredentials(server_def_), &bound_port_);

  builder.SetMaxMessageSize(std::numeric_limits<int32>::max());

  builder.SetOption(
      std::unique_ptr<::grpc::ServerBuilderOption>(new NoReusePortOption));

  // Allow subclasses to specify more args to pass to the gRPC server.
  MaybeMutateBuilder(&builder);

  // 1.
  // master and worker 的概念和逻辑关系.
  // https://etd.ohiolink.edu/!etd.send_file?accession=osu1531827968620294&disposition=inline
  // Figure 2.1 Overview of TensorFlow

  master_impl_ = CreateMaster(&master_env_);
  // Construct a Master and return a unique pointer to it.

  // 1.
  // master 在哪?
  // core/distributed_runtime/rpc/grpc_server_lib.h:109

  // 2.
  // virtual std::unique_ptr<Master> CreateMaster(MasterEnv* master_env);
  // tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc

  // 3.
  // class Master
  // core/distributed_runtime/master.cc:36
  // class Master have the same interface as class Session. It is
  // responsible for grpc service functions:
  // 1. Create a session; 2. extend a session; 3. close a session;
  // 4. run a step of the graph; 5. list devices;

  // 3.
  // Master 的初始化要 device 的信息吗?
  // e.g.,
  // DeviceFinder
  // NewRemoteDevices
  // remote_devices

  // Initialize grpc service in Master
  master_service_ = NewGrpcMasterService(master_impl_.get(), config, &builder);

  // WorkerSession related: NewGrpcWorker
  worker_impl_ = opts.worker_func ? opts.worker_func(&worker_env_, config)
                                  : NewGrpcWorker(&worker_env_, config);
  // 1.
  // NewGrpcWorker 是在  grpc_worker_service.cc 内

  /// Initialize grpc service in Worker
  worker_service_ = NewGrpcWorkerService(worker_impl_.get(), &builder,
                                         opts.worker_service_options)
                        .release();
  // 1.
  // NewGrpcWorkerService 是在  grpc_worker_service.cc 内

  // Initialize grpc service in Eager
  eager_service_ = new eager::GrpcEagerServiceImpl(&worker_env_, &builder);

  // extra service:
  if (opts.service_func != nullptr) {
    opts.service_func(&worker_env_, &builder);
  }
  server_ = builder.BuildAndStart();

  if (!server_) {
    return errors::Unknown("Could not start gRPC server");
  }

  // Worker cache is to store cluster, job, task, protocol, host:port
  // information.
  WorkerCacheInterface* worker_cache;

  // Worker cache factory options are pointers to cluster, job, task, protocol,
  // host:port information.
  WorkerCacheFactoryOptions worker_cache_factory_options(server_def_);

  TF_RETURN_IF_ERROR(
      WorkerCacheFactory(worker_cache_factory_options, &worker_cache));
  // WorkerCacheFactory constructs a new GrpcWorkerCache which can access all
  // workers responsible for graph computing.

  CHECK_NE(nullptr, worker_cache);

  if (opts.collective_mgr_func) {
    worker_env_.collective_executor_mgr =
        opts.collective_mgr_func(config, &worker_env_, worker_cache);
    if (!worker_env_.collective_executor_mgr) {
      return errors::Internal(
          "collective_mgr_func did not return CollectiveExecutorMgr");
    }
  } else {
    // 进入!

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
  /// IMPT: SessionMgr manage a list of Sessions on a worker.
  /// 1. create a session to a list of sessions ; 2. delete a session;
  /// 3. lookup a session; 4. logging.
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

  /// It is used when master creates session.
  master_env_.master_session_factory =
      [config, stats_factory](
          SessionOptions options,
          const MasterEnv* env,
          std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs,
          std::unique_ptr<WorkerCacheInterface> worker_cache,
          std::unique_ptr<DeviceSet> device_set,
          std::vector<string> filtered_worker_list) {
        options.config.MergeFrom(config);

        // =============================================================
        // MasterSession is responsible for a graph computation.
        return new MasterSession(
          options,
          env,
          std::move(remote_devs),
          std::move(worker_cache),
          std::move(device_set), // 🤔 👍
          std::move(filtered_worker_list),
          stats_factory);
        // =============================================================
        // MasterSession 在
        // tensorflow/core/distributed_runtime/master_session.cc
      };

  master_env_.worker_cache_factory =
      [this](const WorkerCacheFactoryOptions& options,
             WorkerCacheInterface** worker_cache) {
        return WorkerCacheFactory(options, worker_cache);
      };

  // Master is local to the server.
  // LocalMaster:
  // A master service that has been created in the same process as the client.
  // But I don't think I will use this design. My design requires that client
  // and master are in different processes.
  // Provide direct access to the master from in-process clients.
  LocalMaster::Register(target(), master_impl_.get(),
                        config.operation_timeout_in_ms());

  return Status::OK();
}

/** \brief Parse job id, host:port information from worker cache factory options,
 *         and store it into grpc channel specification(channel_spec).
 *
 *  \param options
 *         Pointers to ClusterDef, job_name, and protocol and task_index.
 *
 *  \param channel_spec
 *         grpc channel specification. Speak in more detail, it stores
 *         information about a list of job id to host:port.
 *
 *  \return Status
 */
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

/** \brief Construct a new GrpcWorkerCache which can access all workers
 *         responsible for graph computing.
 *
 *  \param options: WorkerCacheFactoryOptions
 *         Worker cache factory options are pointers to cluster, job, task,
 *         protocol, host:port information.
 *
 *  \param worker_cache: WorkerCacheInterface**
 *         An interface related to list workers, release workers, list jobs in
 *         workers
 *
 *   \return Status
 */
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

  // GrpcChannelSpec:
  // grpc channel specification. Speak in more detail, it stores information
  // about a list of job id to host:port.
  GrpcChannelSpec channel_spec;

  /// Parse job id, host:port information from worker cache factory options (options),
  /// and return grpc channel specification(channel_spec).
  TF_RETURN_IF_ERROR(ParseChannelSpec(options, &channel_spec));

  /// Assign channel_cache_ with a new pointer to grpc channel cache, which is
  /// by NewGrpcChannelCache.  Speak in more details,
  /// - /job:<job identifier>/task:<task id> is called target or worker name
  /// - host:port
  channel_cache_.reset(
      /// NewGrpcChannelCache creates grpc channel information of
      /// job id, host:port and a lambda function. If there are multiple grpc
      /// channel caches, create multiple channel cache.
      NewGrpcChannelCache(channel_spec, GetChannelCreationFunction()));

  string name_prefix = strings::StrCat("/job:", *options.job_name, "/replica:0",
                                       "/task:", options.task_index);

  /// TranslateTask find the host:port string by task index from the device
  /// fullname via task_index of the tf device fullname name_prefix.
  const string host_port = channel_cache_->TranslateTask(name_prefix);
  int requested_port;

  auto colon_index = host_port.find_last_of(':');
  if (!strings::safe_strto32(host_port.substr(colon_index + 1),
                             &requested_port)) {
    return errors::Internal("Could not parse port for local server from \"",
                            host_port, "\".");
  }

  /// The requested port should be the same as the port to which this server is
  /// bound (bound_port_).
  /// \todo Why I think it is an unnecessary move to check the so called
  ///       requested_port and bound_port_?
  if (requested_port != bound_port_) {
    return errors::InvalidArgument("Requested port ", requested_port,
                                   " differs from expected port ", bound_port_);
  }

  *worker_cache = NewGrpcWorkerCacheWithLocalWorker(
      channel_cache_, worker_impl_.get(), name_prefix);
  // 1.
  // NewGrpcWorkerCacheWithLocalWorker:
  // NewGrpcWorkerCacheWithLocalWorker helps construct a new GrpcWorkerCache
  // , which can access all workers doing graph computing.

  return Status::OK();
}

/** \brief Start master thread, worker thread, and eager thread to serve grpc
 *         service functions.
 *
 *  \remark No params.
 */
Status GrpcServer::Start() {
  // 1.
  // state:
  // GrpcServer::Start() : state_ = STARTED

  mutex_lock l(mu_);
  switch (state_) {
    case NEW: {
      // 1.
      // 自问自答:
      // GrpcServer::GrpcServer(): state_(NEW) {}

      master_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_master_service",
                            [this] { master_service_->HandleRPCsLoop(); }));
      worker_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_worker_service",
                            [this] { worker_service_->HandleRPCsLoop(); }));
      eager_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_eager_service",
                            [this] { eager_service_->HandleRPCsLoop(); }));
      // ====================================================================
      state_ = STARTED;
      // ====================================================================
      // 1.
      // 状态的转化真是神奇啊
      // 本质上用的是 if-else 分支来管理在当前状态下下一步要执行的函数是什么,
      // 虽然简单, 但是好神奇, 好有用啊!!!
      // s

      LOG(INFO) << "Started server with target: " << target();
      // 1.
      // target(): grpc://localhost:2223
      // Started server with target: grpc://localhost:2223

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
  // 1.
  // who calls it?
  //
  // server.__del__()
  //  TF_ServerStop()
  //    GrpcServer::Stop()

  mutex_lock l(mu_);

  switch (state_) {
    case NEW:
      state_ = STOPPED;
      return Status::OK();

    case STARTED:
      // 1.
      // 如果说先 Start() --> Stop() 那么就要挂(出错)!
      // 说来说去就是不能 stop 喽...

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
      // 通常呢, 这里
      // 也就是没有操作喽, 啥都没干啊, tmd.

    case STOPPED:
      master_thread_.reset();
      worker_thread_.reset();
      eager_thread_.reset();
      return Status::OK();

    default:
      LOG(FATAL);
  }
}

/** \brief Hardcode the bound(束缚(bind的过去式，过去分词)) port of a grpc server
 *
 *  \details For example, logging shows multiple host:port information, but the
 *           current one will show
 *           Initialize GrpcChannelCache for job worker -> {0 -> localhost:2223}
 *           Other host:port shows the exact ip:port information. Not the
 *           hardcoded "localhost".
 *
 *  \remark No params.
 */
const string GrpcServer::target() const {
  return strings::StrCat("grpc://localhost:", bound_port_);
}

std::shared_ptr<::grpc::ServerCredentials> GrpcServer::GetServerCredentials(
    const ServerDef& server_def) const {
  // 1.
  // message ServerDef
  // Defines the configuration of a **single** TensorFlow server.
  // tensorflow/core/protobuf/tensorflow_server.proto:28:
  return ::grpc::InsecureServerCredentials();
}

/** \brief Get the converted function ChannelCreationFunction from
 *         NewHostPortGrpcChannel.
 *
 *  \return ChannelCreationFunction
 */
ChannelCreationFunction GrpcServer::GetChannelCreationFunction() const {
  // We can do this because SparseGrpcChannelCache is robust to nullptr being
  // returned by the channel creation function
  return ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
}

std::unique_ptr<Master> GrpcServer::CreateMaster(MasterEnv* master_env) {
  return std::unique_ptr<Master>(new Master(master_env, 0.0));
  // 1.
  // class Master
  // tensorflow/core/distributed_runtime/master.h:36:

}

/* static */
Status GrpcServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<ServerInterface>* out_server) {
  // 1.
  // server_def: ServerDef
  // Server definition from message ServerDef, including
  // 1. ClusterDef,
  // 2. job name,
  // 3. task index,
  // 4. configuration for sessions on this server,
  // 5. protocol used by this server.

  std::unique_ptr<GrpcServer> ret(
      new GrpcServer(server_def, env == nullptr ? Env::Default() : env));

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
/** \class GrpcServerFactory
 *
 *  \brief For the purpose of constructing a new GRPC server at runtime
 *         according to the protocol from server definition.
 */
class GrpcServerFactory : public ServerFactory {
 public:
  /** \brief To determine whether to new a grpc server by GrpcServerFactory
   *         or a gdr server by GdrServerFactory according to the protocol
   *
   *  \param server_def
   *         Server definition from message ServerDef, including 1. ClusterDef,
   *         2.job name, 3. task index, 4. configuration for sessions on this
   *         server, 5. protocol used by this server.
   *
   *  \return true if the protocol provided by server definition is "grpc",
   *          flase if the protocol provided by server definition is not "grpc".
   */
  bool AcceptsOptions(const ServerDef& server_def) override {
    return server_def.protocol() == "grpc";
  }

  /** \brief Create a grpc server.
   *
   *  \param[in] server_def
   *         Server definition from message ServerDef, including 1. ClusterDef,
   *         2.job name, 3. task index, 4. configuration for sessions on this
   *         server, 5. protocol used by this server.
   *   ```cpp
   *   cluster {
   *       job {
   *         name: "worker"
   *         tasks {
   *           key: 0
   *           value: "localhost:2223"
   *         }
   *       }
   *   }
   *   job_name: "worker"
   *   default_session_config {
   *       gpu_options {
   *         allocator_type: "BFC"
   *         allow_growth: true
   *       }
   *       allow_soft_placement: true
   *   }
   *   protocol: "grpc"
   *   ```
   *
   *  \param[out] out_server: std::unique_ptr<ServerInterface>*
   *
   *  \return Status
   *
   */
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
// 1.
// 妙哉

// 2.
// static 为什么这么用?
// Since declaring a variable static this way means internal linkage,
// every translation unit #includeing your header file gets its own,
// individual variable (which is not visible outside your translation unit).

}  // namespace
}  // namespace tensorflow
