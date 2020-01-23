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

// A Device is a something that can perform computations as part of a
// model.  Devices can be local (runs computation on this machine), or
// remote (contacts a device local to another machine using an RPC to
// do the work).  Devices are registered in a DeviceSet, which is also
// responsible for the Device <-> id mapping.
//
// Device names
// * Every Device should have a unique name with the format:
//     /job:___/replica:___/task:___/(gpu|cpu):___
//   An example name would be "/job:train/replica:0/task:3/device:GPU:2".
// * Task numbers are within the specified replica, so there are as
//   many "task zeros" as replicas.

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_H_

#include <memory>
#include <string>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_attributes.pb_text.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

class DeviceMgr;

class Device : public DeviceBase {
 public:
  // Callback type that takes a Status and returns void.
  typedef std::function<void(const Status&)> DoneCallback;

  Device(Env* env, const DeviceAttributes& device_attributes);
  ~Device() override;

  // Full name of this device (see top comment).
  const string& name() const override { return device_attributes_.name(); }

  // Parsed name of this device
  const DeviceNameUtils::ParsedName& parsed_name() const {
    return parsed_name_;
  }

  // Describes what kind of device this is.  This is intended to be
  // human-readable and not computer-parsed, except that two devices
  // with the same device_type() are expected to perform similarly
  // (both from a computation and communication perspective).
  const string& device_type() const { return device_attributes_.device_type(); }

  // Returns an aggregation of device attributes.
  const DeviceAttributes& attributes() const override {
    return device_attributes_;
    // 1.
    // message DeviceAttributes 数据结构
    // tensorflow/core/framework/device_attributes.proto
    // 1.1
    // message DeviceAttributes 数据结构
    // tensorflow/core/framework/device_attributes.proto
    // - name: string
    // - device_type: string
    // - memory_limit: int64
    // - locality: DeviceLocality
    // - incarnation: fixed64
    // - physical_device_desc: string

    // 1.2
    // message DeviceLocality 数据结构说明:
    // - bus_id: int32
    // - numa_node: int32
    // - links: LocalLinks

    // 1.3
    // message LocalLinks 数据结构说明:
    // - link: repeated InterconnectLink , i.e., vector of InterconnectLink instance

    // 1.4
    // message InterconnectLink 数据结构说明:
    // - device_id: int32
    // - type: string
    // - strength: int32


    // 2 打印
    /*
    device_attributes_.DebugString() =

    name: "/job:worker/replica:0/task:0/device:GPU:0"
    device_type: "GPU"
    memory_limit: 10399953716
    locality {
      bus_id: 1
      links {
        link {
          device_id: 1
          type: "StreamExecutor"
          strength: 1
        }
      }
    }
    incarnation: 1405998625628097843
    physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1"
    */
  }

  // Performs the actual compute function.
  // Subclasses may override this function if they wish to perform
  // some initialization before each compute.
  virtual void Compute(
    OpKernel* op_kernel,          // input
    OpKernelContext* context) {   // output

      // 纯虚函数，所以是看具体哪个 op ，执行对应的 Compute() 函数
      op_kernel->Compute(context);

    // OpKernel::Compute 函数说明:
    // virtual void Compute(OpKernelContext* context) = 0;
    // tensorflow/core/framework/op_kernel.h
  }

  ////////////////////////////////////////////////////////////////////////

  // Asynchronous kernel's compute.
  virtual void ComputeAsync(
                 AsyncOpKernel* op_kernel,  // input
                 OpKernelContext* context,  // input
                 AsyncOpKernel::DoneCallback done) {  // input
    // AsyncOpKernel::DoneCallback 数据结构
    // core/framework/op_kernel.h
    // Asynchronous compute.
    //
    // Implementations of ComputeAsync() must run "done" to signal the
    // completion of the computation. "context" is guaranteed to be
    // alive until the "done" callback starts.
    // typedef std::function<void()> DoneCallback;

    op_kernel->ComputeAsync(
                 context,
                 std::move(done));
                 // 1.
                 // done lambda 的函数体的定义在
                 // tensorflow/core/common_runtime/executor.cc
                 // Asynchronous computes.
                 // ExecutorState::Process 内
                 //   auto done = [this, state]() {

                 // 2.
                 // RecvOp::ComputeAsync 函数说明:
                 // core/kernels/sendrecv_ops.cc
  }

  ////////////////////////////////////////////////////////////////////////

  // Takes ownership of the references in tensors. If necessary, a
  // device may override this method to keep a reference to the
  // accessed tensors until the async computation has completed.
  virtual void ConsumeListOfAccessedTensors(
      DeviceContext* context, const TensorReferenceVector& tensors) {
    for (const auto& ref : tensors) {
      ref.Unref();
    }
  }

  // If true, and tracing is enabled, the `tracing::ScopedAnnotation()` tracing
  // mechanism will be used instead of `tracing::ScopedActivity()`. Some devices
  // may override this method to use annotations, which enable child activities
  // (such as GPU kernel launches) to be related to the OpKernel invocation.
  virtual bool TraceUsingAnnotations() const { return false; }

  // Blocks until all operations queued on the device at the time of
  // the call have completed.  Returns any error pending on the device
  // at completion.
  virtual Status Sync() = 0;

  // Calls the given callback when all operations queued on the device at the
  // time of the call have completed. The callback is passed any error pending
  // on the device at completion.
  // TODO(b/112409994): Consolidate these two APIs, removing the synchronous
  // version.
  virtual void Sync(const DoneCallback& done);

  // On session completion, the executor may call Device::Sync() depending on
  // flag settings. Override this to return false for devices that don't allow
  // such calls. Instead, these devices must use other mechanisms (such as
  // num_deferred_ops) to ensure the device has finished processing necessary
  // work at session completion. In addition, for these devices, RefreshStatus
  // must be called at session completion to retrieve execution result status.
  //
  // Devices that override this function must also implement RefreshStatus.
  virtual bool AllowsSyncOnCompletion() const { return true; }

  // This is used in conjunction with AllowsSyncOnCompletion to allow the
  // executor to get execution result status at session completion.
  //
  // For supported devices, this call returns the underlying device stream's
  // current status in a non-blocking way, without using blocking calls such as
  // Stream::BlockHostUntilDone or Device::Sync. When applicable, the device
  // status is also updated with the retrieved stream status.
  virtual Status RefreshStatus() {
    return errors::Unimplemented(
        "RefreshStatus is not supported on this device.");
  }

  // Optionally modify the device's GraphDef before execution.
  //
  // This method should be considered experimental and is supplied to enable
  // prototyping of TensorFlow device implementations that need to modify
  // the GraphDef before execution.
  //
  // 'graph' supplies the partition of the graph assigned to this
  // device.
  virtual Status MaybeRewriteGraph(std::unique_ptr<Graph>* /*graph*/) {
    return Status::OK();
  }

  // Fill in the context map for the graph. Default behavior is to do
  // nothing.
  //
  // The caller takes ownership over the DeviceContext objects given
  // by the device.
  virtual Status FillContextMap(const Graph* graph,
                                DeviceContextMap* device_context_map) {
    return Status::OK();
  }

  // Returns the op segment of this device.  The caller can reuse op
  // kernels registered for the same session running on this device.
  OpSegment* op_segment() { return &op_seg_; }

  // Returns the resource manager associated w/ this device.
  // -----------------------------------------------------------------------
  virtual ResourceMgr* resource_manager() { return rmgr_; }
  // -----------------------------------------------------------------------
  // 1.
  // 打印:
  /*
  p devices_
  $10 = std::vector of length 10, capacity 16 = {0x5619adf32fa0, 0x5619afbfcb90, 0x5619afe53ad0, 0x5619afe5c5f0, 0x5619afe648c0, 0x5619afe6d8b0, 0x5619afe84500, 0x5619afe95df0, 0x5619afea3620, 0x5619afeb0f20}
  p devices_[0].name()
  Cannot resolve method tensorflow::Device::name to any overloaded instance
  p devices_[0]->name()
  $11 = "/job:localhost/replica:0/task:0/device:CPU:0"
  p devices_[1]->name()
  $12 = "/job:localhost/replica:0/task:0/device:XLA_CPU:0"
  p devices_[2]->name()
  $13 = "/job:localhost/replica:0/task:0/device:XLA_GPU:0"
  p devices_[3]->name()
  $14 = "/job:localhost/replica:0/task:0/device:XLA_GPU:1"
  p devices_[4]->name()
  $15 = "/job:localhost/replica:0/task:0/device:XLA_GPU:2"
  p devices_[5]->name()
  $16 = "/job:localhost/replica:0/task:0/device:XLA_GPU:3"
  p devices_[6]->name()
  $17 = "/job:localhost/replica:0/task:0/device:GPU:0"
  */

  // 2.
  // 打印:
  // p devices_[6]->resource_manager()->DebugString()
  // "localhost | N10tensorflow9LegacyVarE | v1 | float/[25,35]\nlocalhost | N10tensorflow9LegacyVarE | v2 | float/[30,20]\nlocalhost | N10tensorflow9LegacyVarE | v3 | float/[10,40]\nlocalhost | N10tensorflow9LegacyVarE | v4 | float/[25,15]"
  //
  // 排版后
  // localhost | N10tensorflow9LegacyVarE | v1 | float/[25,35]
  // localhost | N10tensorflow9LegacyVarE | v2 | float/[30,20]
  // localhost | N10tensorflow9LegacyVarE | v3 | float/[10,40]
  // localhost | N10tensorflow9LegacyVarE | v4 | float/[25,15]

  // 1.
  // p devices_[0]->resource_manager()->DebugString()
  // $19 = ""

  // 3.
  // vgg16 keras application 的打印
  // https://gist.github.com/shizukanaskytree/c373da3bac4b18551a91c7bff7015333
  // localhost            | N10tensorflow3VarE                       | block1_conv1/bias                        | float/[64]
  // localhost            | N10tensorflow3VarE                       | block1_conv1/kernel                      | float/[3,3,3,64]
  // localhost            | N10tensorflow3VarE                       | block1_conv2/bias                        | float/[64]
  // localhost            | N10tensorflow3VarE                       | block1_conv2/kernel                      | float/[3,3,64,64]

  // 4.
  // N10tensorflow3VarE 解析:
  // class Var : public ResourceBase
  // tensorflow/core/framework/resource_var.h


  // Returns the device manager that owns this device, or nullptr if this Device
  // is not owned by a device manager.
  DeviceMgr* device_mgr() const { return device_mgr_; }

  // Summarizes the status of this Device, for debugging.
  string DebugString() const { return ProtoDebugString(device_attributes_); }

  // Assembles the parameter components into a complete DeviceAttributes value.
  static DeviceAttributes BuildDeviceAttributes(
      const string& name, DeviceType device, Bytes memory_limit,
      const DeviceLocality& locality, const string& physical_device_desc);

  static DeviceAttributes BuildDeviceAttributes(
      const string& name, DeviceType device, Bytes memory_limit,
      const DeviceLocality& locality) {
    // Pass in an empty string as physical device name.
    return BuildDeviceAttributes(name, device, memory_limit, locality, "");
  }

  // Clears the resource manager associated with this device.
  void ClearResourceMgr() { rmgr_->Clear(); }

 protected:
  void DeleteResourceMgr() {
    delete rmgr_;
    rmgr_ = nullptr;
  }

 private:
  friend class DeviceMgr;

  // Pointer to the device manager that owns this device. Not owned.
  DeviceMgr* device_mgr_ = nullptr;

  const DeviceAttributes device_attributes_;
  // message DeviceAttributes 数据结构
  // tensorflow/core/framework/device_attributes.proto
  // - name: string
  // - device_type: string
  // - memory_limit: int64
  // - locality: DeviceLocality
  // - incarnation: fixed64
  // - physical_device_desc: string

  DeviceNameUtils::ParsedName parsed_name_;

  // op_seg_ maps session handle and op name to OpKernel objects.
  OpSegment op_seg_;
  // class OpSegment 数据结构
  // tensorflow/core/framework/op_segment.h
  //
  // 概述:
  //
  // OpSegment keeps track of OpKernels registered for sessions running
  // on a device.
  //
  // The implementation maintains a two-level map. The 1st level maps
  // session handle to the map of registered OpKernels. The 2nd level
  // map maps node names to instantiated OpKernel objects.
  //
  // Each 2-nd level map is reference-counted and the caller can call
  // AddHold to obtain a reference on all kernels of a session and
  // ensure these kernels are alive until a corresponding RemoveHold is
  // called on the same session.
  //
  // - SessionMap: typedef std::unordered_map<string, Item*> SessionMap
  // - KernelMap: typedef std::unordered_map<string, OpKernel*> KernelMap
  // - sessions_: SessionMap
  // - Item
  //    - num_holds: int
  //    - name_kernel: KernelMap
  //      session handle -> item.
  /*
  +--------------------------------------------------------------+
  |                      +-------------+                         |
  | session handle+-------->  Item     |                         |
  |                      |-------------|                         |
  |                      |             |   +-------------------+ |
  |                      | name_kernel+--->|op name+-->OpKernel| |
  |                      |             |   +-------------------+ |
  |                      |             |   +-------------------+ |
  |                      +-------------+   |op name+-->OpKernel| |
  |                                        +-------------------+ |
  |                                        +-------------------+ |
  |                                        |op name+-->OpKernel| |
  |                                        +-------------------+ |
  |                                                              |
  |                                                              |
  |                                                              |
  |                      +-------------+                         |
  | session handle+-------->  Item     |                         |
  |                      |-------------|                         |
  |                      |             |   +-------------------+ |
  |                      | name_kernel+--->|op name+-->OpKernel| |
  |                      |             |   +-------------------+ |
  |                      |             |   +-------------------+ |
  |                      +-------------+   |op name+-->OpKernel  |
  |                                        +-------------------  |
  |                                        +-------------------  |
  |                                        |op name+-->OpKernel  |
  |                                        +-------------------  |
  */
  // Resources associated w/ this device. E.g., shared variables, etc.
  ResourceMgr* rmgr_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(Device);
};
// 1.
// class Device 数据结构
//
// class Device : public DeviceBase
// tensorflow/core/common_runtime/device.h
//
// 重要接口
// - Compute
// - ComputeAsync
// - FillContextMap
// - resource_manager
// 成员变量:
// - device_mgr_: DeviceMgr*
// - device_attributes_: const DeviceAttributes
// - parsed_name_: DeviceNameUtils::ParsedName
// - op_seg_: OpSegment
// - rmgr_: ResourceMgr*
// friend class:
// friend class DeviceMgr;

// 2.
// class DeviceBase 数据结构
// tensorflow/core/framework/device_base.h
// - struct CpuWorkerThreads
//    - num_threads: int , default : 0
//    - workers: thread::ThreadPool*, default : nullptr
// - struct GpuDeviceInfo
//    - stream: stream_executor::Stream*
//    - default_context: DeviceContext*
//    - event_mgr: EventMgr*
//    - gpu_id: int, default: -1
// - env_: Env* const
// - cpu_worker_threads_: CpuWorkerThreads* , default : nullptr
// - gpu_device_info_: GpuDeviceInfo* , default : nullptr
// - device_thread_pool_: thread::ThreadPool* , default : nullptr
// - eigen_cpu_devices_: std::vector<Eigen::ThreadPoolDevice*>
// - eigen_sycl_device_: Eigen::SyclDevice*, default : nullptr

// 3.
// class DeviceContext: public core::RefCounted
// tensorflow/core/framework/device_base.h
// 概述:
// 接口定义
//
// 接口:
// - stream()
// - MaintainLifetimeOnStream
// - CopyCPUTensorToDevice
// - CopyTensorInSameDevice
// - CopyDeviceTensorToCPU
// - ThenExecute

// 4.
// class DeviceMgr 数据结构说明:
// tensorflow/core/common_runtime/device_mgr.h


}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_H_
