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

#include "tensorflow/core/kernels/sendrecv_ops.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

static string GetRendezvousKeyPrefix(const string& send_device,
                                     const string& recv_device,
                                     const uint64 send_device_incarnation,
                                     const string& tensor_name) {
  return strings::StrCat(send_device, ";",
                         strings::FpToString(send_device_incarnation), ";",
                         recv_device, ";", tensor_name);
}

static void GetRendezvousKey(const string& key_prefix,
                             const FrameAndIter& frame_iter, string* key) {
  key->clear();
  strings::StrAppend(key, key_prefix, ";", frame_iter.frame_id, ":",
                     frame_iter.iter_id);
}

static FrameAndIter GetFrameAndIter(OpKernelContext* ctx,
                                    bool hostmem_sendrecv) {
  if (hostmem_sendrecv && ctx->call_frame() != nullptr) {
    // Host memory send/recv pairs are added by
    // common_runtime/memory_types.cc.  When the pair of nodes are
    // added inside a function, we need to use the function call frame
    // to formulate the unique rendezvous key.
    return FrameAndIter(reinterpret_cast<uint64>(ctx->call_frame()), 0);
  } else {
    return ctx->frame_iter();
  }
}

SendOp::SendOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  string send_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device));
  string recv_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device));
  uint64 send_device_incarnation;
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr("send_device_incarnation",
                        reinterpret_cast<int64*>(&send_device_incarnation)));
  string tensor_name;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name));
  key_prefix_ = GetRendezvousKeyPrefix(send_device, recv_device,
                                       send_device_incarnation, tensor_name);
  // The vast majority of Send nodes are outside any loop context, so
  // proactively cache the rendezvous key for the top-level.
  GetRendezvousKey(key_prefix_, {0, 0}, &parsed_key_.buf_);
  OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(parsed_key_.buf_, &parsed_key_));
  if (!ctx->GetAttr("_hostmem_sendrecv", &hostmem_sendrecv_).ok()) {
    hostmem_sendrecv_ = false;
  }
}


// -----------------------------------------------------------------------

void SendOp::Compute(OpKernelContext* ctx) {

  OP_REQUIRES(
      ctx, ctx->rendezvous() != nullptr,
      errors::Internal("Op kernel context needs to provide a rendezvous."));

  // The device context may be passed between the Send/Recv
  // boundary, so that the device context used to produce the Tensor
  // is used when performing the copy on the recv side (which may be
  // a different device).

  Rendezvous::Args args;
  // 1.
  // Rendezvous 数据结构：
  // tensorflow/core/framework/rendezvous.h
  // 原理说明：
  // A Rendezvous is an abstraction for passing tensors from producers
  // to consumers. A rendezvous is a table of channels. Each channel is
  // keyed by a rendezvous key. The key encodes a pair of <producer,
  // consumer>, where the producer and the consumer are tensorflow
  // devices.
  //
  // The producer calls the Send() method to send one tensor over one
  // named channel. The consumer calls the Recv() method to receive one
  // tensor from a named channel. A sequence of tensors can be passed
  // from the producer to the consumer.  The consumer receives them in
  // the order as the producer sends them.
  //
  // A consumer may safely request the tensor before or after it has
  // been produced.  A consumer has the choice of making a blocking call
  // or providing a callback: in either case, the consumer receives the
  // Tensor as soon as it is available.  A producer never blocks.


  // 2.
  // struct Rendezvous::Args 数据结构
  // tensorflow/core/framework/rendezvous.h
  // - device_context: DeviceContext*
  // - alloc_attrs: AllocatorAttribute


  args.device_context = ctx->op_device_context();
  /*
  gdb print:

  p args.device_context
  $46 = (tensorflow::DeviceContext *) 0x0
  */

  args.alloc_attrs = ctx->input_alloc_attr(0);

  FrameAndIter frame_iter = GetFrameAndIter(ctx, hostmem_sendrecv_);
  /*
  gdb print
  p hostmem_sendrecv_
  $47 = false  # why? Because it is only host memory send not host memory send + recv.

  p frame_iter
  $48 = {
    frame_id = 0,
    iter_id = 0
  }
  */


  if (frame_iter == FrameAndIter(0, 0)) {
    // Use the cached rendezvous key.
    VLOG(2) << "Send " << parsed_key_.buf_;
    /*
    2019-09-26 22:52:14.603571: I tensorflow/core/kernels/sendrecv_ops.cc:93]
    Send
    /job:localhost/replica:0/task:0/device:CPU:0;0000000000000001;
    /job:localhost/replica:0/task:0/device:GPU:0;edge_8_y/RandomStandardNormal;0:0
    */

    ctx->SetStatus(
      ctx->rendezvous()->Send(
        parsed_key_,
        args,
        ctx->input(0),
        ctx->is_input_dead()));

    // 0.
    // IntraProcessRendezvous::Send 函数说明
    // tensorflow/core/common_runtime/rendezvous_mgr.cc

    // 1.
    // ctx->rendezvous()
    // framework/op_kernel.h
    // rendezvous() 函数说明:
    // An op kernel communicates with outside environment through
    // Rendezvous Send() and Recv().
    // Rendezvous* rendezvous() const { return params_->rendezvous; }

    // 2.
    // params_: class OpKernelContext::Params*

    // 3.
    // params_->rendezvous 说明
    // Mechanism used by this op kernel invocation to communicate with
    // computations running on other devices.
    // rendezvous: class OpKernelContext::struct Params::Rendezvous*

    // 4.
    // ctx->is_input_dead()
    // gdb print: false

    // 5.
    // class OpKernelContext::is_input_dead() 函数说明:
    // params_->is_input_dead
    // framework/op_kernel.cc

    // 6.
    // OpKernelContext::input 函数说明
    //
    // const Tensor& OpKernelContext::input(int index)
    // framework/op_kernel.cc
    // 取出的是第一个，可能也是唯一的一个吧
    // 核心的一步:
    // const Tensor& tensor = *((*params_->inputs)[index].tensor);

    /*
    7.

    p parsed_key_
    $51 = {
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

    /*
    8.

    p args
    $52 = {
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

    return;

  } else {

    Rendezvous::ParsedKey in_loop_parsed;

    GetRendezvousKey(key_prefix_, frame_iter, &in_loop_parsed.buf_);

    VLOG(2) << "Send " << in_loop_parsed.buf_;

    OP_REQUIRES_OK(ctx,
                   Rendezvous::ParseKey(in_loop_parsed.buf_, &in_loop_parsed));

    ctx->SetStatus(ctx->rendezvous()->Send(in_loop_parsed, args, ctx->input(0),
                                           ctx->is_input_dead()));
    return;
  }

}

REGISTER_KERNEL_BUILDER(Name("_Send").Device(DEVICE_CPU), SendOp);
REGISTER_KERNEL_BUILDER(Name("_Send").Device(DEVICE_GPU), SendOp);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("_Send").Device(DEVICE_SYCL), SendOp);
REGISTER_KERNEL_BUILDER(
    Name("_HostSend").Device(DEVICE_SYCL).HostMemory("tensor"), SendOp);
#endif  // TENSORFLOW_USE_SYCL

REGISTER_KERNEL_BUILDER(Name("_HostSend").Device(DEVICE_CPU), SendOp);
REGISTER_KERNEL_BUILDER(
    Name("_HostSend").Device(DEVICE_GPU).HostMemory("tensor"), SendOp);

RecvOp::RecvOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
  string send_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device));
  string recv_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device));
  uint64 send_device_incarnation;
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr("send_device_incarnation",
                        reinterpret_cast<int64*>(&send_device_incarnation)));
  string tensor_name;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name));
  key_prefix_ = GetRendezvousKeyPrefix(send_device, recv_device,
                                       send_device_incarnation, tensor_name);
  // The vast majority of Recv nodes are outside any loop context, so
  // proactively cache the rendezvous key for the top-level.
  GetRendezvousKey(key_prefix_, {0, 0}, &parsed_key_.buf_);
  OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(parsed_key_.buf_, &parsed_key_));
  if (!ctx->GetAttr("_hostmem_sendrecv", &hostmem_sendrecv_).ok()) {
    hostmem_sendrecv_ = false;
  }
}

namespace {


Rendezvous::DoneCallback make_recv_callback(OpKernelContext* ctx,
                                            AsyncOpKernel::DoneCallback done) {
  // 1.
  // AsyncOpKernel::DoneCallback 数据结构
  // std::function<void()>
  //

  // 2.
  // done lambda 的函数体的定义在
  // tensorflow/core/common_runtime/executor.cc
  // Asynchronous computes.
  // ExecutorState::Process 内
  //   auto done = [this, state]() { ... }

  // 3.
  // Rendezvous::DoneCallback 数据结构
  //
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

  using namespace std::placeholders;

  return std::bind(
      // lambda 函数体
      [ctx](AsyncOpKernel::DoneCallback done,
            // Begin unbound arguments.
            const Status& s,
            const Rendezvous::Args& send_args,
            const Rendezvous::Args& recv_args,
            const Tensor& val,
            bool is_dead) {

        ctx->SetStatus(s);

        if (s.ok()) {
          // 'ctx' allocates the output tensor of the expected type.
          // The runtime checks whether the tensor received here is
          // the same type.
          if (!is_dead) {
            ctx->set_output(0, val);
          }

        }

        done();
        // done lambda 的函数体的定义在
        // tensorflow/core/common_runtime/executor.cc
        // Asynchronous computes.
        // ExecutorState::Process 内
        //   auto done = [this, state]() { ... }
      }, // lambda function done.

      // 参数
      std::move(done), _1, _2, _3, _4, _5);
}
}  // namespace


////////////////////////////////////////////////////////////////////////

void RecvOp::ComputeAsync(
               OpKernelContext* ctx,
               DoneCallback done) {
  // 1.
  // done lambda 的函数体的定义在:
  // tensorflow/core/common_runtime/executor.cc
  // Asynchronous computes.
  // ExecutorState::Process 内
  //   auto done = [this, state]()  { ... }

  // 2.
  // class OpKernelContext 数据结构
  // tensorflow/core/framework/op_kernel.h:567
  // 真真正正的图计算的集大成者

  OP_REQUIRES_ASYNC(
      ctx,
      ctx->rendezvous() != nullptr,
      errors::Internal("Op kernel context needs to provide a rendezvous."),
      done);


  Rendezvous::Args args;
  // Rendezvous::Args 数据结构
  // tensorflow/core/framework/rendezvous.h
  //
  //  struct Args {
  //    DeviceContext* device_context = nullptr;
  //    AllocatorAttributes alloc_attrs;
  //  };

  args.device_context = ctx->op_device_context();
  args.alloc_attrs = ctx->output_alloc_attr(0);


  FrameAndIter frame_iter = GetFrameAndIter(ctx, hostmem_sendrecv_);


  if (frame_iter == FrameAndIter(0, 0)) {

    VLOG(2) << "Recv " << parsed_key_.buf_;

    ctx->rendezvous()->RecvAsync(parsed_key_,
                                 args,
                                 make_recv_callback(
                                   ctx,
                                   std::move(done)));
    // 1.
    // make_recv_callback 函数说明
    //
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
    // DoneCallback 数据结构的使用是在:
    // tensorflow/core/kernels/sendrecv_ops.cc
    // make_recv_callback 函数的返回值

    // 2.
    // done lambda 变量说明:
    // AsyncOpKernel::DoneCallback done
    // AsyncOpKernel::DoneCallback done 来自 executor.cc 内 Process 里面定义的 done

    // 3.
    // ctx->rendezvous()->RecvAsync 函数说明:
    // IntraProcessRendezvous::RecvAsync
    // common_runtime/rendezvous_mgr.cc


  } else {

    Rendezvous::ParsedKey in_loop_parsed;

    GetRendezvousKey(key_prefix_, frame_iter, &in_loop_parsed.buf_);

    VLOG(2) << "Recv " << in_loop_parsed.buf_;

    OP_REQUIRES_OK_ASYNC(
        ctx,
        Rendezvous::ParseKey(
          in_loop_parsed.buf_,
          &in_loop_parsed),
          done);

    ctx->rendezvous()->RecvAsync(
                         in_loop_parsed,
                         args,
                         make_recv_callback(ctx, std::move(done)));

  }
}

REGISTER_KERNEL_BUILDER(Name("_Recv").Device(DEVICE_CPU), RecvOp);
REGISTER_KERNEL_BUILDER(Name("_Recv").Device(DEVICE_GPU), RecvOp);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("_Recv").Device(DEVICE_SYCL), RecvOp);
#endif  // TENSORFLOW_USE_SYCL

REGISTER_KERNEL_BUILDER(Name("_HostRecv").Device(DEVICE_CPU), RecvOp);
REGISTER_KERNEL_BUILDER(
    Name("_HostRecv").Device(DEVICE_GPU).HostMemory("tensor"), RecvOp);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(
    Name("_HostRecv").Device(DEVICE_SYCL).HostMemory("tensor"), RecvOp);
#endif  // TENSORFLOW_USE_SYCL

}  // end namespace tensorflow
