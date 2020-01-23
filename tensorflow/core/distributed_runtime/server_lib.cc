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
/** \file server_lib.cc
 *
 *  \brief 1. Register different kind of server's factory. 2. Then new server
 *         the registered map. Other functions are utility functions.
 *
 *  \details 1. Register is the first to call. 2. It use server_factory function
 *           to new a register map to take all kinds of server and their factory.
 *           3. NewServer extracts registered server and their factory to
 *           construct server. 4. In the course of NewServer, GetFactory is used.
 *           5. get_server_factory_lock is a mutex utility function.
 */


#include "tensorflow/core/distributed_runtime/server_lib.h"

#include <unordered_map>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

namespace {
/** \brief New a mutex to protect only one thread to construct the server
 *         factory
 *
 *  \remark No params
 *
 *  \return mutex* a pointer to mutex to protect multiple threads access the
 *          server factory
 *
 *  \todo mutex with LINKER_INITIALIZED ???
 */
mutex* get_server_factory_lock() {
  static mutex server_factory_lock(LINKER_INITIALIZED);
  return &server_factory_lock;
}

typedef std::unordered_map<string, ServerFactory*> ServerFactories;


/** \brief New a map from server type(e.g., "GRPC_SERVER") to a pointer to
 *         ServerFactory (e.g., GrpcServerFactory)
 *
 *  \remark No params.
 *
 *  \return Return a pointer to static ServerFactories, specifing the name of
 *          server type and its ServerFactory pointer.
 *          ServerFactories is an alias of an unordered map, mapping from a
 *          string of server type name to server factory.
 *
 *  \note We only have one pointer factories to ServerFactories whose lifespan
 *        is the program. No matter how many times we call `server_factories()`,
 *        Only be initialized once and last till the end of the program.
 *
 *  \todo It is better to use `std::shared_ptr` to avoid memory leakage.
 *        \code{.cpp}
 *        static std::shared_ptr<ServerFactories> factories (new ServerFactories)
 *        \endcode
 */
ServerFactories* server_factories() {
  static ServerFactories* factories = new ServerFactories;
  return factories;
}
}  // namespace


/** \brief Add, for example, a map of "GRPC_SERVER": a pointer to
 *         GrpcServerFactory into a map whose lifetime span the program.
 *
 *  \details So, it holds the information of the kind/name of server type and
 *           the corresponding server factory class which can new such kind of
 *           server. Also, you cannot insert two times of the same kind of
 *           server.
 *
 *  \param server_type A string to describe the server type.
 *
 *  \param factory A type of server factory who can new server. For example,
 *                 `class GrpcServerFactory`, which has procedure to new server.
 *
 *  \details In tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc,
 *           around line 483, the only invocation of
 *           \code{.cpp}
 *           ServerFactory::Register("GRPC_SERVER", new GrpcServerFactory());
 *           \endcode
 *
 *  \return void
 *
 */
/* static */
void ServerFactory::Register(const string& server_type,
                             ServerFactory* factory) {
  mutex_lock l(*get_server_factory_lock());
  if (!server_factories()->insert({server_type, factory}).second) {
    LOG(ERROR) << "Two server factories are being registered under "
               << server_type;
  }
}


/** \brief Get the right factory to new a server according to protocol from
 *         the server definition, and return the factory from the register map
 *
 *  \param server_def
 *         Server definition from message ServerDef, including 1. ClusterDef,
 *         2.job name, 3. task index, 4. configuration for sessions on this
 *         server, 5. protocol used by this server.
 *         Here, according to the protocol("grpc"/"grpc+gdr") from the server
 *         definition to determine what kind of server factory to use.
 *         If the protocol is :
 *         - 1. "grpc"(default), GrpcServerFactory is used.
 *         - 2. "grpc+gdr", GdrServerFactory is used.
 *
 *  \param out_factory Return value of this function. It is the derived class of
 *                     ServerFactory, GrpcServerFactory or GdrServerFactory.
 *
 *  \return Status
 *          return Status::OK() if success, otherwise print all available
 *          ServerFactory, i.e., GrpcServerFactory's name or GdrServerFactory's
 *          name.
 */
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
      str_util::Join(server_names, ", "), " ]");
}


/** \brief New a server according to the definition of the server
 *
 *  \param[in] server_def: ServerDef
 *         Server definition from message ServerDef, including 1. ClusterDef,
 *         2.job name, 3. task index, 4. configuration for sessions on this
 *         server, 5. protocol used by this server.
 *
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
 *  \param[out] out_server
 *         a unique pointer to the constructed server, providing interface of
 *         1. Start 2. Stop 3. Join
 *
 *  \return Status
 *          return Code::OK if success, otherwise return error code from Code
 *          Status are defined in tensorflow/core/lib/core/error_codes.proto
 *          See header file of tensorflow/core/lib/core/status.h from
 *          server_lib.h !!!
 */
// Creates a server based on the given `server_def`, and stores it in
// `*out_server`. Returns OK on success, otherwise returns an error.
Status NewServer(const ServerDef& server_def,
                 std::unique_ptr<ServerInterface>* out_server) {
  ServerFactory* factory;
  TF_RETURN_IF_ERROR(ServerFactory::GetFactory(server_def, &factory));
  return factory->NewServer(server_def, out_server);
}

}  // namespace tensorflow
