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


关于设计:
如果需要增加 cluster 的配置, 是否可以通过:

ServerDef:
serverdef.cluster.

// tensorflow/core/protobuf/cluster.proto
// Defines a TensorFlow cluster as a set of jobs.
message ClusterDef {
  // The jobs that comprise the cluster.
  repeated JobDef job = 1;
}


// ===================================================
// ***************************************************
// ****************** Next topic *********************
// ***************************************************
// ===================================================

// API to use in cc file
// =====================

// create a instance/obj:
// =====================
// ServerDef* options;
// ClusterDef* const cluster = options->mutable_cluster();
//                                      ^^^^^^^
// Returns a pointer to the mutable object that stores the field's value.
// doc: https://developers.google.com/protocol-buffers/docs/reference/cpp-generated
//
// 
// ===
//
// int *const is a constant pointer to integer
// This means that the variable being declared is a constant pointer pointing to an integer.
// this implies that the pointer shouldn’t point to some other address.
// 
// ===
//
// from `tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server_doc.cc`
// https://stackoverflow.com/questions/53648009/google-protobuf-mutable-foo-or-set-allocated-foo

// 
// JobDef* const job_def = cluster->add_job();
//                                  ^^^

// set the value:
// ==============
// options->set_job_name(job_name);
//          ^^^
// 




syntax = "proto3";

package tensorflow;

import "tensorflow/core/protobuf/cluster.proto";
import "tensorflow/core/protobuf/config.proto";
import "tensorflow/core/protobuf/device_filters.proto";

option cc_enable_arenas = true;
option java_outer_classname = "ServerProtos";
option java_multiple_files = true;
option java_package = "org.tensorflow.distruntime";
option go_package = "github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto";

// Defines the configuration of a single TensorFlow server.
message ServerDef {

  // 为什么一定要 ClusterDef 呢?
  // tensorflow/core/protobuf/cluster_doc.proto
  // 如果不写这个 server 就定义不起来吗? 
  // 哪个步骤定义不起来?
  // ================

  // The cluster of which this server is a member.
  ClusterDef cluster = 1;

  // The name of the job of which this server is a member.
  //
  // NOTE(mrry): The `cluster` field must contain a `JobDef` with a `name` field
  // that matches this name.
  string job_name = 2;

  // The task index of this server in its job.
  //
  // NOTE: The `cluster` field must contain a `JobDef` with a matching `name`
  // and a mapping in its `tasks` field for this index.
  int32 task_index = 3;

  // The default configuration for sessions that run on this server.
  ConfigProto default_session_config = 4;
  // 如果你不想写, 你可以写 NULL.
  // NULL 怎么写?

  // The protocol to be used by this server.
  //
  // Acceptable values include: "grpc", "grpc+verbs".
  string protocol = 5;

  // The server port. If not set, then we identify the port from the job_name.
  int32 port = 6;

  // Device filters for remote tasks in the cluster.
  // NOTE: This is an experimental feature and only effective in TensorFlow 2.x.
  ClusterDeviceFilters cluster_device_filters = 7;
}

