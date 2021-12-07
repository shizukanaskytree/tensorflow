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

#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"

#include <cstdlib>
#include <limits>
#include <map>
#include <unordered_map>

#include "grpcpp/create_channel.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

namespace {

string MakeAddress(const string& job, int task) {
  return strings::StrCat("/job:", job, "/replica:0/task:", task);
}
// 1.
// 对我来说是重要的.
//
// 2.
// There's the worker, which does actual work. 
// If your graph has a with tf.device(job:worker/task:0):, block,
// then computation in that block should be executed on task:0
// https://stackoverflow.com/questions/41067398/task-assignment-in-tensorflow-distributed-process
//
// 3.
// tf.device(job:worker/task:0):
// 这个是其使用场景
// address 的两个重要标记: job name, task id.





// 这个好!
// Allows the host to be a raw IP (either v4 or v6).
Status ValidateHostPortPair(const string& host_port) {

  string bns_prefix = "/bns/";

  if (host_port.substr(0, bns_prefix.length()) == bns_prefix) {

    return Status::OK();

  }

  uint32 port;

  auto colon_index = host_port.find_last_of(':');
  
  if (!strings::safe_strtou32(host_port.substr(colon_index + 1), &port) ||
      host_port.substr(0, colon_index).find('/') != string::npos) {

    return errors::InvalidArgument("Could not interpret \"", host_port,
                                   "\" as a host-port pair.");
  }

  return Status::OK();
}
// 




::grpc::ChannelArguments* CreateDefaultChannelArguments() {
  ::grpc::ChannelArguments* args = new ::grpc::ChannelArguments();
  const char* env = std::getenv("TF_GRPC_DEFAULT_OPTIONS");
  if (env != nullptr) {
    for (auto& grpc_option : absl::StrSplit(env, ',')) {
      std::vector<string> name_value = absl::StrSplit(grpc_option, '=');
      if (name_value.size() != 2) {
        LOG(ERROR) << "Invalid GRPC options format: " << grpc_option;
        continue;
      }
      VLOG(3) << "Setting GRPC default for '" << name_value[0] << "' to '"
              << name_value[1] << "'";
      if (name_value[1].size() >= 2 && name_value[1][0] == '"') {
        string ue_value = name_value[1].substr(1, name_value[1].size() - 2);
        string value;
        string error;
        if (!absl::CUnescape(ue_value, &value, &error)) {
          LOG(ERROR) << "Failed to parse escaped string for " << grpc_option
                     << ": " << error;
        } else {
          args->SetString(name_value[0], value);
        }
      } else {
        int64_t value;
        if (strings::safe_strto64(name_value[1], &value)) {
          args->SetInt(name_value[0], value);
        } else {
          LOG(ERROR) << "Invalid integer value: " << grpc_option;
        }
      }
    }
  }
  return args;
}

const ::grpc::ChannelArguments* GetDefaultChannelArguments() {
  static const ::grpc::ChannelArguments* args = CreateDefaultChannelArguments();
  return args;
}

}  // namespace

::grpc::ChannelArguments GetChannelArguments(const RPCOptions* rpc_options) {
  // TODO(mrry): Implement secure channels.
  ::grpc::ChannelArguments args = *GetDefaultChannelArguments();
  args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH, std::numeric_limits<int32>::max());
  // NOTE(mrry): Some versions of gRPC use a 20-second minimum backoff
  // on connection failure, which makes our tests time out.
  args.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 1000);
  if (rpc_options != nullptr) {
    if (rpc_options->compression_algorithm() == "deflate") {
      args.SetCompressionAlgorithm(GRPC_COMPRESS_DEFLATE);
      args.SetInt(GRPC_COMPRESSION_CHANNEL_DEFAULT_LEVEL,
                  rpc_options->compression_level());
      VLOG(5) << "Setting GRPC compression : algo='"
              << rpc_options->compression_algorithm()
              << "' level=" << rpc_options->compression_level();
    } else if (rpc_options->compression_algorithm() == "gzip") {
      args.SetCompressionAlgorithm(GRPC_COMPRESS_GZIP);
      args.SetInt(GRPC_COMPRESSION_CHANNEL_DEFAULT_LEVEL,
                  rpc_options->compression_level());
      VLOG(5) << "Setting GRPC compression : algo='"
              << rpc_options->compression_algorithm()
              << "' level=" << rpc_options->compression_level();
    } else if (!rpc_options->compression_algorithm().empty()) {
      LOG(ERROR) << "Invalid compression algorithm: "
                 << rpc_options->compression_algorithm();
    }
    if (rpc_options->disable_session_connection_sharing()) {
      VLOG(5) << "Disabling TCP connection sharing";
      args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, true);
    }
  }
  return args;
}


// 我为什么要看这个 NewHostPortGrpcChannel?
// - 因为我想着说如果有新的 server 加入, 那么需要动态构造 channel.

// wxf: where is this function used?
// - 
// - 
Status NewHostPortGrpcChannel(const string& target,
                              const RPCOptions* rpc_options,
                              SharedGrpcChannelPtr* channel_pointer) {
  // - Summary:
  //   - create a channel to the target address
  //
  // - input
  //   - target
  //   - rpc_options
  // 
  // - output
  //   - channel_pointer

  // Minimally ensure that the target is valid
  TF_RETURN_IF_ERROR(ValidateHostPortPair(target));
  // host:port 可以用 raw ip:port 的形式来写.
  // 

  
  ::grpc::ChannelArguments args = GetChannelArguments(rpc_options);
  // 在 test 里面, rpc_options 设置成 nullptr 没有问题.
  // 


  *channel_pointer = ::grpc::CreateCustomChannel(
      "dns:///" + target, ::grpc::InsecureChannelCredentials(), args);
  // https://grpc.github.io/grpc/cpp/namespacegrpc.html
  // create a channel targetting to the address of "dns:///" + target

  return Status::OK();
}



// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc

// ===

// (hm) wxf@seir19:~/tf2/tensorflow$ grep -nwr "NewHostPortGrpcChannel"

// 忽略
// tensorflow/core/distributed_runtime/cluster_function_library_runtime_test.cc:38:        ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/master_test.cc:53:    TF_CHECK_OK(NewHostPortGrpcChannel(


// 忽略
// tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:140:Status NewHostPortGrpcChannel(const string& target,


// 忽略
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.cc:173:Status NewHostPortGrpcChannel(const string& target,


// 忽略
// tensorflow/core/distributed_runtime/rpc/grpc_channel.h:96:Status NewHostPortGrpcChannel(const string& target,

// used 1:
// tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:536:  return ConvertToChannelCreationFunction(NewHostPortGrpcChannel);


// 忽略
// tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client_test.cc:34:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.h:102:Status NewHostPortGrpcChannel(const string& target,
// tensorflow/core/distributed_runtime/rpc/grpc_server_lib_doc.cc:549:  return ConvertToChannelCreationFunction(NewHostPortGrpcChannel);

// 忽略
// tensorflow/core/distributed_runtime/rpc/grpc_worker_cache_test.cc:33:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_worker_cache_test.cc:68:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);

// used 2:
// tensorflow/core/distributed_runtime/rpc/grpc_session.cc:58:        NewHostPortGrpcChannel(options.target.substr(kSchemePrefixLength),
// tensorflow/core/distributed_runtime/rpc/grpc_session.cc:413:      NewHostPortGrpcChannel(options.target.substr(kSchemePrefixLength),

// 忽略
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:63:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:126:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:207:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:296:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:358:  EXPECT_TRUE(NewHostPortGrpcChannel("127.0.0.1:2222", /*rpc_options=*/nullptr,
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:361:  EXPECT_TRUE(NewHostPortGrpcChannel("example.com:2222",
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:364:  EXPECT_TRUE(NewHostPortGrpcChannel("fqdn.example.com.:2222",
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:367:  EXPECT_TRUE(NewHostPortGrpcChannel("[2002:a9c:258e::]:2222",
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:371:      NewHostPortGrpcChannel("[::]:2222", /*rpc_options=*/nullptr, &mock_ptr)
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:374:  EXPECT_FALSE(NewHostPortGrpcChannel("example.com/abc:2222",
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:377:  EXPECT_FALSE(NewHostPortGrpcChannel("127.0.0.1:2222/",
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:380:  EXPECT_FALSE(NewHostPortGrpcChannel(
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:384:      NewHostPortGrpcChannel("[::]/:2222", /*rpc_options=*/nullptr, &mock_ptr)
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:387:      NewHostPortGrpcChannel("[::]:2222/", /*rpc_options=*/nullptr, &mock_ptr)
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:390:      NewHostPortGrpcChannel("[::]:", /*rpc_options=*/nullptr, &mock_ptr).ok());
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:393:      NewHostPortGrpcChannel("/bns/example", /*rpc_options=*/nullptr, &mock_ptr)
// tensorflow/core/distributed_runtime/remote_device_test.cc:52:        ConvertToChannelCreationFunction(NewHostPortGrpcChannel);


ChannelCreationFunction ConvertToChannelCreationFunction(
    const std::function<Status(string, const RPCOptions*,
                               SharedGrpcChannelPtr*)>& new_channel_func_ptr) {
  return [new_channel_func_ptr](const string& target) -> SharedGrpcChannelPtr {
    SharedGrpcChannelPtr channel_ptr;
    if (new_channel_func_ptr(target, /*rpc_options=*/nullptr, &channel_ptr)
            .ok()) {
      return channel_ptr;
    } else {
      return nullptr;
    }
  };
}
// - where is ConvertToChannelCreationFunction used?
//   - tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:536:  return ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// 


// ====================================================
// (hm) wxf@seir19:~/tf2/tensorflow$ grep -wnr "ConvertToChannelCreationFunction" 

// dismiss
// tensorflow/core/distributed_runtime/cluster_function_library_runtime_test.cc:38:        ConvertToChannelCreationFunction(NewHostPortGrpcChannel);

// dismiss
// tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:152:ChannelCreationFunction ConvertToChannelCreationFunction(


// dismiss
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.cc:219:// tensorflow/core/distributed_runtime/cluster_function_library_runtime_test.cc:38:        ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.cc:235:// tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:536:  return ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.cc:239:// tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client_test.cc:34:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.cc:241:// tensorflow/core/distributed_runtime/rpc/grpc_server_lib_doc.cc:549:  return ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.cc:244:// tensorflow/core/distributed_runtime/rpc/grpc_worker_cache_test.cc:33:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.cc:245:// tensorflow/core/distributed_runtime/rpc/grpc_worker_cache_test.cc:68:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.cc:252:// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:63:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.cc:253:// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:126:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.cc:254:// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:207:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.cc:255:// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:296:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.cc:268:// tensorflow/core/distributed_runtime/remote_device_test.cc:52:        ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.cc:271:ChannelCreationFunction ConvertToChannelCreationFunction(
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.cc:284:// - where is ConvertToChannelCreationFunction used?

// tensorflow/core/distributed_runtime/rpc/grpc_channel.h:92:ChannelCreationFunction ConvertToChannelCreationFunction(

// used 1: 
// tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:536:  return ConvertToChannelCreationFunction(NewHostPortGrpcChannel);

// dismiss
// tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client_test.cc:34:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_doc.h:98:ChannelCreationFunction ConvertToChannelCreationFunction(
// tensorflow/core/distributed_runtime/rpc/grpc_server_lib_doc.cc:549:  return ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_worker_cache_test.cc:33:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_worker_cache_test.cc:68:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:63:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:126:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:207:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc:296:      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// tensorflow/core/distributed_runtime/remote_device_test.cc:52:        ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
// ====================================================



Status GrpcChannelSpec::AddHostPortsJob(const string& job_id,
                                        const std::vector<string>& host_ports) {
  std::map<int, string> host_ports_map;
  for (size_t i = 0; i < host_ports.size(); ++i) {
    host_ports_map[i] = host_ports[i];
  }
  return AddHostPortsJob(job_id, host_ports_map);
}

Status GrpcChannelSpec::AddHostPortsJob(
    const string& job_id, const std::map<int, string>& host_ports) {
  if (!job_ids_.insert(job_id).second) {
    return errors::InvalidArgument(
        "Duplicate job ID in cluster specification: ", job_id);
  }
  for (const auto& id_host_port : host_ports) {
    TF_RETURN_IF_ERROR(ValidateHostPortPair(id_host_port.second));
  }
  host_ports_jobs_.emplace_back(job_id, host_ports);
  return Status::OK();
}

namespace {

// GrpcChannelCache that caches results to FindWorkerChannel() calls.
using CachingGrpcChannelCache = GenericCachingChannelCache<GrpcChannelCache>;

// A ChannelCache that is the union of multiple ChannelCaches.
// Takes ownership of the caches passed to the constructor.
class MultiGrpcChannelCache : public CachingGrpcChannelCache {
 public:
  explicit MultiGrpcChannelCache(const std::vector<GrpcChannelCache*>& caches,
                                 int num_channels_per_target)
      : CachingGrpcChannelCache(num_channels_per_target), caches_(caches) {}

  ~MultiGrpcChannelCache() override {
    for (GrpcChannelCache* cache : caches_) {
      delete cache;
    }
  }

  void ListWorkers(std::vector<string>* workers) override {
    for (GrpcChannelCache* cache : caches_) {
      cache->ListWorkers(workers);
    }
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) override {
    for (GrpcChannelCache* cache : caches_) {
      cache->ListWorkersInJob(job_name, workers);
    }
  }

  string TranslateTask(const string& target) override {
    mutex_lock l(mu_);  // could use reader lock
    GrpcChannelCache* cache = gtl::FindPtrOrNull(target_caches_, target);
    if (cache == nullptr) {
      for (GrpcChannelCache* c : caches_) {
        string r = c->TranslateTask(target);
        if (!r.empty()) {
          target_caches_.insert({target, c});
          cache = c;
          break;
        }
      }
    }
    CHECK(cache) << "Could not find GrpcChannelCache holding channel for "
                 << target;
    return cache->TranslateTask(target);
  }

 protected:
  SharedGrpcChannelPtr FindChannelOnce(const string& target) override {
    for (GrpcChannelCache* cache : caches_) {
      SharedGrpcChannelPtr ch(cache->FindWorkerChannel(target));
      if (ch) {
        mutex_lock l(mu_);
        target_caches_.insert({target, cache});
        return ch;
      }
    }
    return nullptr;
  }

 private:
  // List of channels used by this MultiGrpcChannelCache.
  const std::vector<GrpcChannelCache*> caches_;

  mutex mu_;
  // Cache of channels keyed by the target they are handling.
  // The same GrpcChannelCache can appear multiple times in the cache.
  std::unordered_map<string, GrpcChannelCache*> target_caches_
      TF_GUARDED_BY(mu_);
};

class SparseGrpcChannelCache : public CachingGrpcChannelCache {
 public:
  SparseGrpcChannelCache(const string& job_id,
                         const std::map<int, string>& host_ports,
                         ChannelCreationFunction channel_func,
                         int num_channels_per_target)
      : CachingGrpcChannelCache(num_channels_per_target),
        job_id_(job_id),
        host_ports_(host_ports),
        channel_func_(std::move(channel_func)) {
    LOG(INFO) << "Initialize GrpcChannelCache for job " << ToString();
  }
  ~SparseGrpcChannelCache() override {}

  void ListWorkers(std::vector<string>* workers) override {
    workers->reserve(workers->size() + host_ports_.size());
    for (const auto& id_host_port : host_ports_) {
      workers->emplace_back(MakeAddress(job_id_, id_host_port.first));
    }
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) override {
    if (job_name == job_id_) {
      ListWorkers(workers);
    }
  }

  string TranslateTask(const string& target) override {
    DeviceNameUtils::ParsedName parsed;
    if (!DeviceNameUtils::ParseFullName(target, &parsed)) {
      LOG(WARNING) << "Invalid target: " << target;
      return "";
    }

    if (!parsed.has_job || parsed.job != job_id_) {
      return "";
    }
    if (!parsed.has_replica || parsed.replica != 0) {
      LOG(WARNING) << "Replica ID must be 0 in target: " << target;
      return "";
    }
    int32_t task = parsed.has_task ? parsed.task : -1;
    auto iter = host_ports_.find(task);
    if (iter == host_ports_.end()) {
      LOG(WARNING) << "Task " << task << " was not defined in sparse job "
                   << job_id_ << ": " << target;
      return "";
    }
    return iter->second;
  }

 protected:
  SharedGrpcChannelPtr FindChannelOnce(const string& target) override {
    const string host_port = TranslateTask(target);
    if (host_port.empty()) {
      return nullptr;
    }
    auto chan_ptr = channel_func_(host_port);
    VLOG(5) << "Channel created for: job: " << job_id_
            << " host_port: " << host_port << " target : " << target
            << " Ptr: " << chan_ptr.get();
    return chan_ptr;
  }

 private:
  string ToString() {
    std::vector<string> task_strings;
    task_strings.reserve(host_ports_.size());
    for (const auto& id_host_port : host_ports_) {
      task_strings.emplace_back(
          strings::StrCat(id_host_port.first, " -> ", id_host_port.second));
    }
    return strings::StrCat(job_id_, " -> {", absl::StrJoin(task_strings, ", "),
                           "}");
  }

  const string job_id_;
  const std::map<int, string> host_ports_;
  const ChannelCreationFunction channel_func_;
  TF_DISALLOW_COPY_AND_ASSIGN(SparseGrpcChannelCache);
};

}  // namespace

GrpcChannelCache* NewGrpcChannelCache(const GrpcChannelSpec& spec,
                                      ChannelCreationFunction channel_func,
                                      const RPCOptions& options) {
  const int num_jobs = spec.host_ports_jobs().size();
  if (!num_jobs) {
    LOG(ERROR) << "Empty channel spec.";
    return nullptr;
  }
  std::vector<GrpcChannelCache*> caches;
  caches.reserve(num_jobs);
  for (auto& job : spec.host_ports_jobs()) {
    VLOG(2) << "Creating Grpc Channel Cache for: " << job.job_id;
    caches.push_back(
        new SparseGrpcChannelCache(job.job_id, job.host_ports, channel_func,
                                   options.num_channels_per_target()));
  }
  return caches.size() == 1 ? caches[0]
                            : new MultiGrpcChannelCache(
                                  caches, options.num_channels_per_target());
}

}  // end namespace tensorflow
