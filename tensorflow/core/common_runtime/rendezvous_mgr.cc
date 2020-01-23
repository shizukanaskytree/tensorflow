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

#include "tensorflow/core/common_runtime/rendezvous_mgr.h"

#include <unordered_set>

#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

IntraProcessRendezvous::IntraProcessRendezvous(const DeviceMgr* device_mgr)
    : device_mgr_(device_mgr), local_(NewLocalRendezvous()) {}

IntraProcessRendezvous::~IntraProcessRendezvous() { local_->Unref(); }


Status IntraProcessRendezvous::Send(const ParsedKey& parsed,
                                    const Rendezvous::Args& args,
                                    const Tensor& val,
                                    const bool is_dead) {

  VLOG(1) << "IntraProcessRendezvous Send " << this << " " << parsed.FullKey();
  /*
  logging

  2019-09-26 23:11:20.888187: I
  tensorflow/core/common_runtime/rendezvous_mgr.cc:42]
  IntraProcessRendezvous Send 0x564c977b0150
  /job:localhost/replica:0/task:0/device:CPU:0;0000000000000001;
  /job:localhost/replica:0/task:0/device:GPU:0;edge_8_y/RandomStandardNormal;0:0
  */

  {
    mutex_lock l(mu_);
    // 异常处理
    if (!status_.ok()) return status_;
  }

  // 正常处理
  // Buffers "val" and "device_context" in local_.
  return local_->Send(parsed, args, val, is_dead);

  // 0.
  // LocalRendezvousImpl::Send 函数说明
  // tensorflow/core/framework/rendezvous.cc
  //

  // 1.
  // local_ 变量含义
  // local_: IntraProcessRendezvous::Rendezvous*

  // 2.
  // class IntraProcessRendezvous : public Rendezvous
  // IntraProcessRendezvous is a Rendezvous which expects all producers
  // and consumers to be devices immediately accessible within the
  // process.  That is, it will never be necessary to perform an RPC to
  // communicate with either.
  //
  // Buffering of Tensor values is delegated to a "local" Rendezvous
  // obtained from NewLocalRendezvous().  This class just adds
  // functionality to coordinate multiple process-local devices.

  // 3.
  // parsed 变量说明
  // parsed: const ParsedKey&
  /*
  p parsed
  $54 = (const tensorflow::Rendezvous::ParsedKey &) @0x564c977c26e0: {
    src_device = {
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x564c977fe198 "/job:localhost/replica:0/task:0/device:CPU:0;", '0' <repeats 15 times>, "1;/job:localhost/replica:0/task:0/device:GPU:0;edge_8_y/RandomStandardNormal;0:0",
      length_ = 44
    },
    src = {
      has_job = true,
      job = "localhost",
      has_replica = true,
      replica = 0,
      has_task = true,
      task = 0,
      has_type = true,
      type = "CPU",
      has_id = true,
      id = 0
    },
    src_incarnation = 1,
    dst_device = {
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x564c977fe1d6 "/job:localhost/replica:0/task:0/device:GPU:0;edge_8_y/RandomStandardNormal;0:0",
      length_ = 44
    },
    dst = {
      has_job = true,
      job = "localhost",
      has_replica = true,
      replica = 0,
      has_task = true,
      task = 0,
      has_type = true,
      type = "GPU",
      has_id = true,
      id = 0
    },
    edge_name = {
      static npos = 18446744073709551615,
      static kMaxSize = 9223372036854775807,
      ptr_ = 0x564c977fe203 "edge_8_y/RandomStandardNormal;0:0",
      length_ = 29
    },
    buf_ = "/job:localhost/replica:0/task:0/device:CPU:0;", '0' <repeats 15 times>, "1;/job:localhost/replica:0/task:0/device:GPU:0;edge_8_y/RandomStandardNormal;0:0"
  }
  */


  // 4.
  /*
  args 变量说明

  p args
  $55 = (const tensorflow::Rendezvous::Args &) @0x7f1b3dffdbf0: {
    device_context = 0x0,
    alloc_attrs = {
      value = 4,
      scope_id = 0
    }
  }

  00000000
  ^^^^^^^^
  ||||||||
  |||||||+----+on host          1
  ||||||+-----+nic compatible   2
  |||||+------+gpu compatible   4   所以 4 的含义是这个
  ||||+-------+
  |||+--------+
  ||+---------+
  |+----------+
  +-----------+
  */

  // 5.
  /*
  p val
  $56 = (const tensorflow::Tensor &) @0x564c9ff5dea8: {
    shape_ = {
      <tensorflow::TensorShapeBase<tensorflow::TensorShape>> = {
        <tensorflow::TensorShapeRep> = {
          static kMaxRep16 = 65534,
          static kMaxRep32 = 4294967294,
          static kUnknownRep16 = 65535,
          static kUnknownRep32 = 4294967295,
          static kUnknownRank = 255 '\377',
          u_ = {
            buf =             "\036\000\024\000\033\177\000\000\217/-f\036\001\002",
            unused_aligner = 0x7f1b0014001e
          },
          num_elements_ = 600
        },
        members of tensorflow::TensorShapeBase<tensorflow::TensorShape>:
        static kIsPartial = false
      }, <No data fields>},
    buf_ = 0x7f1b04001170
  }
  */

  // 6.
  // p is_dead, $57 = false

}

Status IntraProcessRendezvous::ParseKey(const string& key, bool is_src,
                                        Rendezvous::ParsedKey* parsed) {
  {
    mutex_lock l(mu_);
    if (!status_.ok()) return status_;
  }
  TF_RETURN_IF_ERROR(Rendezvous::ParseKey(key, parsed));
  return Status::OK();
}

// context:
// 被调用:
// if (status.ok() && in.IsInitialized()) {
//   SameWorkerRecvDone(
//     parsed,
//     send_args,
//     recv_args,
//     in,
//     out,
//     std::move(final_callback));

void IntraProcessRendezvous::SameWorkerRecvDone(
    const Rendezvous::ParsedKey& parsed,
    const Rendezvous::Args& send_args,
    const Rendezvous::Args& recv_args,
    const Tensor& in,
    Tensor* out,
    StatusCallback done) {
  // 1.
  // StatusCallback done 说明:
  // done 的函数体是 final_callback, tensorflow/core/common_runtime/rendezvous_mgr.cc (本文件)
  // 定义在 IntraProcessRendezvous::RecvAsync 内的 local_->RecvAsync 内。

  // 2.
  // parsed 变量说明:
  // tensorflow/core/framework/rendezvous.h:46:
  // class Rendezvous : public core::RefCounted
  //
  // Parses the key constructed by CreateKey and parse src/dst device
  // names into structures respectively.
  // struct ParsedKey {
  //   StringPiece src_device;
  //   DeviceNameUtils::ParsedName src;
  //   uint64 src_incarnation = 0;
  //   StringPiece dst_device;
  //   DeviceNameUtils::ParsedName dst;
  //   StringPiece edge_name;
  //   ParsedKey() {}
  //   ParsedKey(const ParsedKey& b) { *this = b; }
  //   ParsedKey& operator=(const ParsedKey& b);
  //   StringPiece FullKey() const { return buf_; }
  //  private:
  //   friend class Rendezvous;
  //   friend class SendOp;
  //   friend class RecvOp;
  //   string buf_;
  // };

  // 3.
  // Args 数据结构
  // class Rendezvous :: struct Args {
  //   DeviceContext* device_context = nullptr;
  //   AllocatorAttributes alloc_attrs;
  // };

  // 4.
  // DeviceContext 数据结构
  // tensorflow/core/framework/device_base.h:68:
  // class DeviceContext : public core::RefCounted


  // Do a quick copy (sharing the underlying buffer) if both tensors
  // are on host memory.
  const bool src_host =
      (send_args.alloc_attrs.on_host() || parsed.src.type == "CPU");
  const bool dst_host =
      (recv_args.alloc_attrs.on_host() || parsed.dst.type == "CPU");
  if (src_host && dst_host) {
    *out = in;

    done(Status::OK());
    // 1.
    // StatusCallback done 说明:
    // done 的函数体是 final_callback, tensorflow/core/common_runtime/rendezvous_mgr.cc (本文件)
    // 定义在 IntraProcessRendezvous::RecvAsync 内的 local_->RecvAsync 内。

    return;
  }

  // This copy must involve a non-CPU device. Hence, "in" must support DMA
  // (e.g., string tensors do not work on GPU).  Variant copy DMA
  // checks happen inside CopyTensor::ViaDMA.
  if (!DataTypeCanUseMemcpy(in.dtype()) && in.dtype() != DT_VARIANT) {
    done(errors::InvalidArgument("Non-DMA-safe ", DataTypeString(in.dtype()),
                                 " tensor may not be copied from/to a GPU."));
    return;
  }

  // -----------------------------------------------------------------------
  Device* src_device;
  Status s = device_mgr_->LookupDevice(parsed.src_device, &src_device);
  if (!s.ok()) {
    done(s);
    return;
  }
  Device* dst_device;
  s = device_mgr_->LookupDevice(parsed.dst_device, &dst_device);
  if (!s.ok()) {
    done(s);
    return;
  }
  // -----------------------------------------------------------------------

  AllocatorAttributes attr = recv_args.alloc_attrs;
  // 1.
  // AllocatorAttributes 数据结构
  // tensorflow/core/framework/allocator.h
  //
  // - uint32 value = 0;
  //   value 是比特的方
  //
  //  00000000
  //  ^^^^^^^^
  //  ||||||||
  //  |||||||+----+ on host ?
  //  ||||||+-----+ nic compatible ?
  //  |||||+------+ gpu compatible ?
  //  ||||+-------+
  //  |||+--------+
  //  ||+---------+
  //  |+----------+
  //  +-----------+

  // 1.1
  // 介绍:
  // A tensorflow Op may need access to different kinds of memory that
  // are not simply a function of the device to which the Op has been
  // assigned.  For example, an Op executing on a GPU may still need
  // to allocate CPU RAM for some purpose.  Internal to the tensorflow
  // runtime we may choose to allocate CPU ram from special regions
  // that have been prepared for higher performance in some use
  // contexts, e.g. doing DMA with particular devices.  For these
  // reasons, the Device interface does not expose just one memory
  // Allocator, but instead provides an accessor that takes a
  // specification of the desired memory attributes in order to select
  // an Allocator.
  //
  // Example use:
  //  // Allocator for ordinary device memory:
  //  Allocator* a = allocator(AllocatorAttributes());
  // ...
  //  // Allocator for CPU RAM, regardless of where Op is executing:
  //  AllocatorAttributes attr;
  //  attr.set_on_host(true);
  //  Allocator* a = allocator(attr);

  // 2.
  // QQQ. recv_args.alloc_attrs 是怎么被初始化的?
  // recv_args: Rendezvous::Args&

  // 2.1
  // AAA.
  // RecvOp::ComputeAsyn 内
  // args.alloc_attrs = ctx->output_alloc_attr(0); // OpKernelContext* ctx
  // tensorflow/core/kernels/sendrecv_ops.cc

  // 2.2
  // OpKernelContext* ctx from ExecutorState::Process(), executor.cc
  // by params
  // only inited by
  // params.input_alloc_attrs = &input_alloc_attrs;
  // params.output_attr_array = item.output_attrs();


  attr.set_gpu_compatible(send_args.alloc_attrs.gpu_compatible() ||
                          recv_args.alloc_attrs.gpu_compatible());


  Allocator* out_allocator = dst_device->GetAllocator(attr);

  if (in.dtype() != DT_VARIANT) {
    // Variants are handled by CopyTensor::ViaDMA.
    Tensor copy(out_allocator, in.dtype(), in.shape());
    *out = copy;
    // 1.
    // QQQ. out 会指向 copy 吗?

    // 2.
    // Tensor 构造函数
    // \brief Creates a tensor with the input `type` and `shape`, using
    // the allocator `a` to allocate the underlying buffer. If
    // LogMemory::IsEnabled() the allocation is logged as coming from
    // an unknown kernel and step. Calling the Tensor constructor
    // directly from within an Op is deprecated: use the
    // OpKernelConstruction/OpKernelContext allocate_* methods to
    // allocate a new tensor, which record the kernel and step.
    //
    // `a` must outlive the lifetime of this Tensor.
    // - Tensor(Allocator* a, DataType type, const TensorShape& shape);
    // tensorflow/core/framework/tensor.cc

  }

  CopyTensor::ViaDMA(parsed.edge_name,
                     send_args.device_context,
                     recv_args.device_context,
                     src_device,
                     dst_device,
                     send_args.alloc_attrs,
                     recv_args.alloc_attrs,
                     &in,
                     out,
                     0 /*dev_to_dev_stream_index*/,
                     std::move(done));
  // 1.
  // class CopyTensor 数据结构
  // common_runtime/copy_tensor.h

  // 2.
  // CopyTensor::ViaDMA 函数说明:
  // common_runtime/copy_tensor.cc
  // Copies "input" to "output" between devices accessible to the
  // local process via some DMA-like method.  "edge_name" is the name
  // of the tensor being copied, for debugging purposes. Depending on
  // the type of devices and memory in use, the copy may be performed
  // synchronously or asynchronously.  'done' will be invoked only
  // after the copy is actually complete.
  //
  // static void ViaDMA(StringPiece edge_name,
  //                    DeviceContext* send_dev_context,
  //                    DeviceContext* recv_dev_context,
  //                    Device* src, Device* dst,
  //                    const AllocatorAttributes src_alloc_attr,
  //                    const AllocatorAttributes dst_alloc_attr,
  //                    const Tensor* input, Tensor* output,
  //                    int dev_to_dev_stream_index, StatusCallback done);
}


////////////////////////////////////////////////////////////////////////


void IntraProcessRendezvous::RecvAsync(const ParsedKey& parsed,
                                       const Rendezvous::Args& recv_args,
                                       DoneCallback done) {

  // 1.
  // DoneCallback 数据结构说明
  // tensorflow/core/framework/rendezvous.h
  // 说明
  // Callback provided by a tensor consumer waiting on the rendezvous.
  // It will be invoked when the tensor is available, or when a non-OK
  // status arises in the production of that tensor.  It also gets
  // two Rendezvous::Args, one provided by the sender, the other by the
  // receiver, which may be needed when a non-CPU device is in use
  // by either side.
  // typedef std::function<void(
  //                         const Status&,
  //                         const Args&, // provided by the sender
  //                         const Args&, // provided by receiver
  //                         const Tensor&,
  //                         const bool)>
  //         DoneCallback;

  // 1.1
  // done lambda 变量说明:
  // AsyncOpKernel::DoneCallback done
  // AsyncOpKernel::DoneCallback done 来自 executor.cc 内 Process 里面定义的 done

  // 2.
  // struct Args 数据结构
  // tensorflow/core/framework/rendezvous.h
  // struct Args {
  //   DeviceContext* device_context = nullptr;
  //   AllocatorAttributes alloc_attrs;
  // };


  VLOG(1) << "IntraProcessRendezvous Recv " << this << " " << parsed.FullKey();

  // Recv the tensor from local_.
  local_->RecvAsync(  // LocalRendezvousImpl::RecvAsync, rendezvous.cc
    parsed,
    recv_args,
    /*DoneCallback done=*/std::bind(
        [this, parsed](DoneCallback done,  // 提前绑定的(bound)参数, , 来自 executor.cc 内 Process 里面定义的 done
                       // Begin unbound arguments.
                       const Status& status,
                       const Rendezvous::Args& send_args,
                       const Rendezvous::Args& recv_args,
                       const Tensor& in,
                       bool is_dead) {

          // If "in" is an uninitialized tensor, do copy-construction to
          // preserve the uninitialized state, along with data type and shape
          // info, which is useful for debugger purposes.
          Tensor* out = in.IsInitialized() ? new Tensor : new Tensor(in);

          // ------------------------------------------------------------------
          // lambda 函数定义
          auto final_callback = std::bind(
              [send_args, recv_args, out, is_dead](
                  DoneCallback done,
                  const Status& s)
              {
                // 1.
                // 提前绑定 done, 来自 executor.cc 内 Process 里面定义的 done
                done(
                  s,
                  send_args,
                  recv_args,
                  *out,
                  is_dead);
                // 1.
                // done lambda 函数的定义是:
                // tensorflow/core/kernels/sendrecv_ops.cc
                // make_recv_callback 函数内

                delete out;
                // 把这个域内的 Tenor 临时变量 out 销毁而已，因为外部的那个正在的 Tensor 已经接收到了 copy 出去的 Tensor, 所以这里可以销毁这个 临时变量了。
              },

              // 如下是参数
              std::move(done),
              std::placeholders::_1);
          // lambda final_callback 函数体定义结束
          // ------------------------------------------------------------------

          if (status.ok() && in.IsInitialized()) {
            SameWorkerRecvDone(
              parsed,
              send_args,
              recv_args,
              in,
              out,
              std::move(final_callback));
            // 1.
            // SameWorkerRecvDone 函数说明:
            // rendezvous_mgr.cc
            //
          } else {
            final_callback(status);
          }

        }, // lambda function done.

        std::move(done), // 提前绑定 done, 来自 executor.cc 内 Process 里面定义的 done
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4,
        std::placeholders::_5));

  // 1.
  // local_ 变量说明
  // IntraProcessRendezvous::local_: Rendezvous*
  // 实际变量类型是: local_: LocalRendezvousImpl*
  //
  // 打印
  // p local_
  // $2 = (tensorflow::LocalRendezvousImpl *) 0x5619aff3c440

  // 2.
  // done lambda 变量说明:
  // AsyncOpKernel::DoneCallback done
  // AsyncOpKernel::DoneCallback done 来自 executor.cc 内 Process 里面定义的 done

  // 3.
  // LocalRendezvousImpl::RecvAsync 函数说明
  // tensorflow/core/framework/rendezvous.cc
  // void RecvAsync(
  //   const ParsedKey& key,
  //   const Args& recv_args,
  //   DoneCallback done) override

  // 4.
  // class Rendezvous
  // tensorflow/core/framework/rendezvous.h:46:
  // class Rendezvous : public core::RefCounted
  //
  // 真实的类型
  // tensorflow/core/framework/rendezvous.cc:149:
  // class LocalRendezvousImpl : public Rendezvous

  // 5.
  // std::bind( F&& f, Args&&... args ) 说明
  // The function template bind generates a forwarding call wrapper for f.
  // Calling this wrapper is equivalent to invoking f with some of its
  // arguments bound to args.
  // 上面的等价于提前预定了一个输入参数:
  // lambda_function(
  //   std::move(done),
  //   const Status& status,
  //   const Rendezvous::Args& send_args,
  //   const Rendezvous::Args& recv_args,
  //   const Tensor& in,
  //   bool is_dead)
}


void IntraProcessRendezvous::StartAbort(const Status& s) {
  CHECK(!s.ok());
  local_->StartAbort(s);
}

}  // end namespace tensorflow
