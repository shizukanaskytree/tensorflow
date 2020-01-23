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

/** \file grpc_channel.cc
 *
 *  \brief channel means host_port.
 */
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"

#include <limits>
#include <map>
#include <unordered_map>

#include "grpcpp/create_channel.h"

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

/** \brief Make a string of a job and task.
 *         The return value is in the of ("/job:", job, "/replica:0/task:", task).
 */
string MakeAddress(const string& job, int task) {
  return strings::StrCat("/job:", job, "/replica:0/task:", task);
}

// Allows the host to be a raw IP (either v4 or v6).
Status ValidateHostPortPair(const string& host_port) {
  uint32 port;
  auto colon_index = host_port.find_last_of(':');
  if (!strings::safe_strtou32(host_port.substr(colon_index + 1), &port) ||
      host_port.substr(0, colon_index).find("/") != string::npos) {
    return errors::InvalidArgument("Could not interpret \"", host_port,
                                   "\" as a host-port pair.");
  }
  return Status::OK();
}

}  // namespace

/** \brief Extract configuration parameters from tf RPCOptions to construct
 *         grpc ChannelArguments, which are options for channel creation.
 *
 *  \param rpc_options: const RPCOptions* ;
 *         message RPCOptions have 1. in-process master setting; 2. compression
 *         algorithm setting; 3. compression level setting.
 *
 *  \return ::grpc::ChannelArguments
 *          Options for channel creation.
 */
::grpc::ChannelArguments GetChannelArguments(const RPCOptions* rpc_options) {
  // TODO(mrry): Implement secure channels.
  ::grpc::ChannelArguments args;
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
  }
  return args;
}

/** \brief Create a new custom grpc Channel pointing to target.
 *
 *  \param[in] target: const string& ;
 *
 *  \param[in] rpc_options: const RPCOptions* ;
 *
 *  \param[out] channel_pointer: SharedGrpcChannelPtr* ;
 *         std::shared_ptr<::grpc::Channel> ; It is the return value, a channel
 *         pointer to the target.
 *
 *  \return Status ;
 *
 *  \details To call service methods, we first need to create a stub. First, we
 *           need to create a gRPC channel for our stub, specifying the server
 *           address and port we want to connect to.
 *           In order to set additional options for the channel, use the
 *           grpc::CreateCustomChannel() api with any special channel arguments
 *           - grpc::ChannelArguments.
 *           from: https://grpc.io/docs/tutorials/basic/c/
 */
Status NewHostPortGrpcChannel(const string& target,
                              const RPCOptions* rpc_options,
                              SharedGrpcChannelPtr* channel_pointer) {
  // Minimally ensure that the target is valid
  TF_RETURN_IF_ERROR(ValidateHostPortPair(target));

  ::grpc::ChannelArguments args = GetChannelArguments(rpc_options);

  /// \fn ::grpc::CreateCustomChannel
  ///
  /// \brief Create a new custom Channel pointing to target.
  ///
  /// \param "dns:///" + target: const grpc::string &target;
  ///        The target string that a new custom Channel created pointing to.
  ///
  /// \param ::grpc::InsecureChannelCredentials():
  ///        const std::shared_ptr<::grpc::ChannelCredentials > & ;
  ///        A channel credentials object encapsulates all the state needed by a client to authenticate with a server for a given channel.
  ///
  /// \param args: const ::grpc::ChannelArguments &;
  ///
  /// \return SharedGrpcChannelPtr* channel_pointer:
  ///         std::shared_ptr<::grpc::Channel> ;
  *channel_pointer = ::grpc::CreateCustomChannel(
      "dns:///" + target, ::grpc::InsecureChannelCredentials(), args);
      
  return Status::OK();
}

/** \brief Convert NewHostPortGrpcChannel function to ChannelCreationFunction
 *
 *  \details ChannelCreationFunction signature, input, function body, return
 *           NewHostPortGrpcChannel is shown above.
 *           - The signature of ChannelCreationFunction is
 *             std::function<SharedGrpcChannelPtr(string target)>
 *           - The input is from NewHostPortGrpcChannel's input: target string.
 *           - The function body is a specialized invocation of
 *             NewHostPortGrpcChannel(target, nullptr, &channel_ptr)
 *           - The return value of ChannelCreationFunction is a pointer to
 *             SharedGrpcChannelPtr (channel_ptr).
 *
 *  \param new_channel_func_ptr Take NewHostPortGrpcChannel(above) as an example,
 *
 *  \return ChannelCreationFunction is std::function<SharedGrpcChannelPtr(string)>
 *          SharedGrpcChannelPtr is a pointer to grpc channel
 */
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
class CachingGrpcChannelCache : public GrpcChannelCache {
 public:
  CachingGrpcChannelCache() {}

  ~CachingGrpcChannelCache() override {}

  SharedGrpcChannelPtr FindWorkerChannel(const string& target) override {
    SharedGrpcChannelPtr ch = nullptr;
    {
      mutex_lock l(mu_);  // could use reader lock
      ch = gtl::FindPtrOrNull(channels_, target);
      if (ch) {
        return ch;
      }
    }
    ch = FindChannelOnce(target);
    if (ch) {
      mutex_lock l(mu_);
      channels_.insert({target, ch});
    }
    return ch;
  }

 protected:
  // Find the ClientChannel for "target".  Only called when no channel was
  // found in the channels_ cache for "target".  A non nullptr result will be
  // cached in channels_.
  virtual SharedGrpcChannelPtr FindChannelOnce(const string& target) = 0;

 private:
  // TODO(zhifengc): Eviction when the map becomes too big.
  mutex mu_;
  std::unordered_map<string, SharedGrpcChannelPtr> channels_ GUARDED_BY(mu_);
};

// A ChannelCache that is the union of multiple ChannelCaches.
// Takes ownership of the caches passed to the constructor.
class MultiGrpcChannelCache : public CachingGrpcChannelCache {
 public:
  explicit MultiGrpcChannelCache(const std::vector<GrpcChannelCache*>& caches)
      : CachingGrpcChannelCache(), caches_(caches) {}

  ~MultiGrpcChannelCache() override {
    for (GrpcChannelCache* cache : caches_) {
      delete cache;
    }
  }

  /** \brief
   *
   *  \param workers: [out] std::vector<string>* ;
   *
   */
  void ListWorkers(std::vector<string>* workers) override {
    /// caches_: List of channels used by this MultiGrpcChannelCache
    /// caches_: std::vector<GrpcChannelCache*>,
    /// GrpcChannelCache is an interface class.
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
  std::unordered_map<string, GrpcChannelCache*> target_caches_ GUARDED_BY(mu_);
};

class SparseGrpcChannelCache : public CachingGrpcChannelCache {
 public:
  SparseGrpcChannelCache(const string& job_id,
                         const std::map<int, string>& host_ports,
                         ChannelCreationFunction channel_func)
      : job_id_(job_id),
        host_ports_(host_ports),
        channel_func_(std::move(channel_func)) {
    LOG(INFO) << "Initialize GrpcChannelCache for job " << ToString();
  }
  ~SparseGrpcChannelCache() override {}

  /** \brief Get all workers from all ip:port, and store them to a string in the
   *         of ("/job:", job, "/replica:0/task:", task_index).
   *
   *  \param workers: std::vector<string>*
   *         The return value of ListWorkers, and store them to a string in the
   *         of ("/job:", job, "/replica:0/task:", task_index).
   *
   *  \remark No return values.
   */
  void ListWorkers(std::vector<string>* workers) override {
    /// \todo Why do we count workers->size() ?
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

  /** \brief Find the host:port string by task index from the device fullname
   *         target.
   *
   *  \param target: string
   *         The fullname of a device name in the form of
   *        /job:<name>/replica:<replica>/task:<task>/device:<type>:<device_num>.
   *
   *  \return A host:port string by task index from the device fullname of target.
   *
   */
  string TranslateTask(const string& target) override {
    /// class DeviceNameUtils represents a device name:
    /// /job:<name>/replica:<replica>/task:<task>/device:<type>:<device_num> .
    /// While DeviceNameUtils::ParsedName is a struct storing information of
    /// each item i.e. job, replica, task id, device, type, and whether it has
    /// each item or not.
    DeviceNameUtils::ParsedName parsed;

    /// DeviceNameUtils::ParseFullName parses "fullname" target string to the
    /// struct of DeviceNameUtils::ParsedName, and return the result via parsed.
    /// The ParseFullName returns true iff succeeds.
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
    int32 task = parsed.has_task ? parsed.task : -1;
    /// Find host:port string from a map of task index to host:port string via
    /// task id
    auto iter = host_ports_.find(task);
    if (iter == host_ports_.end()) {
      LOG(WARNING) << "Task " << task << " was not defined in sparse job "
                   << job_id_ << ": " << target;
      return "";
    }
    /// host_ports_ : std::map<int, string>, if found via task id, return
    /// host:port string
    return iter->second;
  }

 protected:
  SharedGrpcChannelPtr FindChannelOnce(const string& target) override {
    const string host_port = TranslateTask(target);
    if (host_port.empty()) {
      return nullptr;
    }
    return channel_func_(host_port);
  }

 private:
  string ToString() {
    std::vector<string> task_strings;
    task_strings.reserve(host_ports_.size());
    for (const auto& id_host_port : host_ports_) {
      task_strings.emplace_back(
          strings::StrCat(id_host_port.first, " -> ", id_host_port.second));
    }
    return strings::StrCat(job_id_, " -> {", str_util::Join(task_strings, ", "),
                           "}");
  }

  const string job_id_;
  const std::map<int, string> host_ports_;
  const ChannelCreationFunction channel_func_;
  TF_DISALLOW_COPY_AND_ASSIGN(SparseGrpcChannelCache);
};

}  // namespace

/** \brief Create grpc channel information of job id, host:port and a lambda
 *         function. If there are multiple grpc channel caches, create multiple
 *         channel cache.
 *
 *  \param spec
 *         grpc channel specification. Speak in more detail, it stores
 *         information about a list of job id to host:port.
 *
 *  \param channel_func
 *         A lambda function ChannelCreationFunction(string) -> ::grpc::Channel
 *
 *  \return A pointer to grpc channel cache. Speak in more details,
 *          - /job:<job identifier>/task:<task id> is called target or worker
 *            name
 *          - host:port
 */
GrpcChannelCache* NewGrpcChannelCache(const GrpcChannelSpec& spec,
                                      ChannelCreationFunction channel_func) {
  const int num_jobs = spec.host_ports_jobs().size();
  if (!num_jobs) {
    LOG(ERROR) << "Empty channel spec.";
    return nullptr;
  }
  std::vector<GrpcChannelCache*> caches;
  caches.reserve(num_jobs);
  for (auto& job : spec.host_ports_jobs()) {
    caches.push_back(
        new SparseGrpcChannelCache(job.job_id, job.host_ports, channel_func));
  }
  return caches.size() == 1 ? caches[0] : new MultiGrpcChannelCache(caches);
}

}  // end namespace tensorflow
