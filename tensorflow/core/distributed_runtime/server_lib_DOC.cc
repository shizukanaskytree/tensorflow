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

#include "tensorflow/core/distributed_runtime/server_lib.h"

#include <unordered_map>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

namespace {
mutex* get_server_factory_lock() {
  static mutex server_factory_lock(LINKER_INITIALIZED);
  return &server_factory_lock;
}

typedef std::unordered_map<string, ServerFactory*> ServerFactories;
ServerFactories* server_factories() {
  static ServerFactories* factories = new ServerFactories;
  return factories;
}

 0# 0x00007FEDAE837398 in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::ServerFactory::Register(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tensorflow::ServerFactory*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# 0x00007FEDA67D9392 in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 3# 0x00007FEDCA4F96FA in /lib64/ld-linux-x86-64.so.2

}  // namespace

/* static */
void ServerFactory::Register(const string& server_type,
                             ServerFactory* factory) {
  mutex_lock l(*get_server_factory_lock());
  if (!server_factories()->insert({server_type, factory}).second) {
    LOG(ERROR) << "Two server factories are being registered under "
               << server_type;
  }
}

0# tensorflow::ServerFactory::Register(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tensorflow::ServerFactory*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
1# 0x00007FEDA67D9392 in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
2# 0x00007FEDCA4F96FA in /lib64/ld-linux-x86-64.so.2


/* static */
Status ServerFactory::GetFactory(const ServerDef& server_def,
                                 ServerFactory** out_factory) {
  mutex_lock l(*get_server_factory_lock());
  for (const auto& server_factory : *server_factories()) {
    if (server_factory.second->AcceptsOptions(server_def)) {
      *out_factory = server_factory.second;
      return Status::OK();
    }
  }

  std::vector<string> server_names;
  for (const auto& server_factory : *server_factories()) {
    server_names.push_back(server_factory.first);
  }

  return errors::NotFound(
      "No server factory registered for the given ServerDef: ",
      server_def.DebugString(), "\nThe available server factories are: [ ",
      absl::StrJoin(server_names, ", "), " ]");
}

// Creates a server based on the given `server_def`, and stores it in
// `*out_server`. Returns OK on success, otherwise returns an error.
Status NewServer(const ServerDef& server_def,
                 std::unique_ptr<ServerInterface>* out_server) {
  ServerFactory* factory;
  TF_RETURN_IF_ERROR(ServerFactory::GetFactory(server_def, &factory));
  return factory->NewServer(server_def, ServerFactory::Options(), out_server);
}

 0# tensorflow::NewServer(tensorflow::ServerDef const&, std::unique_ptr<tensorflow::ServerInterface, std::default_delete<tensorflow::ServerInterface> >*) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 1# tensorflow::EagerContextDistributedManager::EnableCollectiveOps(tensorflow::ServerDef const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 2# tensorflow::EagerContext::EnableCollectiveOps(tensorflow::ServerDef const&) in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 3# TFE_EnableCollectiveOps in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
 4# 0x00007FED9BA13208 in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tfe.so
 5# 0x00007FED9BA20932 in /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/tensorflow/python/_pywrap_tfe.so
 6# PyCFunction_Call in python
 7# _PyObject_MakeTpCall in python


// Creates a server based on the given `server_def`, and stores it in
// `*out_server`. Returns OK on success, otherwise returns an error.
Status NewServerWithOptions(const ServerDef& server_def,
                            const ServerFactory::Options& options,
                            std::unique_ptr<ServerInterface>* out_server) {
  ServerFactory* factory;
  TF_RETURN_IF_ERROR(ServerFactory::GetFactory(server_def, &factory));
  return factory->NewServer(server_def, options, out_server);
}

}  // namespace tensorflow
