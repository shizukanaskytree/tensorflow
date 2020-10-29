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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_ASYNC_SERVICE_INTERFACE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_ASYNC_SERVICE_INTERFACE_H_

namespace tensorflow {

// Represents an abstract asynchronous service that handles incoming
// RPCs with a polling loop.
// 1.
// 轮询（Polling）是一种 CPU 决策如何提供周边设备服务的方式，又称“程控输入输出”
//（Programmed I/O）。轮询法的概念是：由 CPU 定时发出询问，依序询问每一个周边设备是否需要
// 其服务，有即给予服务，服务结束后再问下一个周边，接着不断周而复始。
// 轮询法实现容易，但效率偏低。

class AsyncServiceInterface {
// - GrpcEagerServiceImpl, grpc_eager_service_impl.h
// - GrpcWorkerService, grpc_worker_service.cc
// - GrpcMasterService, grpc_master_service.cc
// - GrpcVerbsService, grpc_verbs_service.h

 public:
  virtual ~AsyncServiceInterface() {}

  // A blocking method that should be called to handle incoming RPCs.
  // This method will block until the service shuts down.
  virtual void HandleRPCsLoop() = 0;

  // Starts shutting down this service.
  //
  // NOTE(mrry): To shut down this service completely, the caller must
  // also shut down any servers that might share ownership of this
  // service's resources (e.g. completion queues).
  virtual void Shutdown() = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_ASYNC_SERVICE_INTERFACE_H_
