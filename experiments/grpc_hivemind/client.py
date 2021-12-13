from typing import Any, Dict, Optional, Tuple

# from proto import runtime_pb2, runtime_pb2_grpc as runtime_grpc
import proto.runtime_pb2 as runtime_pb2
import proto.runtime_pb2_grpc as runtime_grpc
from utils.networking import Endpoint
from utils.grpc import ChannelCache

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

# 主要参考 hivemind/moe/client folder.
#
# Steps to run
# 1. create the stub
# 2. call RPC function

def _get_stub(endpoint: Endpoint, *extra_options: Tuple[str, Any]):
    """Create a gRPC stub to access remote expert or use previously created stub from a process-wide cache"""
    channel_options = (
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1),
    ) + extra_options

    return ChannelCache.get_stub(
        endpoint, runtime_grpc.ConnectionHandlerStub, aio=False, options=channel_options
    )


port_num = "28282"
endpoint = "localhost:" + port_num
client_stub = _get_stub(endpoint)


# How to construct a runtime_pb2.Tensor
# =====================================
# runtime_pb2.Tensor
#   bytes buffer = 1;
#   repeated uint32 size = 2;
#   bool requires_grad = 3;
#   string dtype = 4;
#   CompressionType compression = 5;
#   int32 chunks = 6;
# 
# >>> def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
# >>>     dtype_name = tensor.numpy().dtype.name
# >>>     tensor = tensor.detach().cpu().float()
# >>>     tensor = tensor if allow_inplace else tensor.clone()
# >>>     tensor = tensor.clamp_(self.FP16_MIN, self.FP16_MAX).to(torch.float16)
# >>>     return runtime_pb2.Tensor(
# >>>         compression=self.compression_type,
# >>>         buffer=tensor.numpy().tobytes(),
# >>>         size=tensor.shape,
# >>>         dtype=dtype_name,
# >>>         requires_grad=tensor.requires_grad,
# >>>     )
uid = "uid_test_string"
serialized_tensors = [runtime_pb2.Tensor()]
outputs = client_stub.forward(runtime_pb2.ExpertRequest(uid=uid, tensors=serialized_tensors))
outputs = client_stub.backward(runtime_pb2.ExpertRequest(uid=uid, tensors=serialized_tensors))
outputs = client_stub.info(runtime_pb2.ExpertUID(uid=uid))

# Example code in the hivemind to use grpc in the client side.
# ============================================================
#
# forward rpc function
# * class RemoteExpert(nn.Module)
# * _RemoteModuleCall.apply
# * outputs = stub.forward(runtime_pb2.ExpertRequest(uid=ctx.uid, tensors=serialized_tensors))
#
# >>> class RemoteExpert(nn.Module):
# >>>     """
# >>>     A simple module that runs forward/backward of an expert hosted on a remote machine.
# >>>     Works seamlessly with pytorch autograd. (this is essentially a simple RPC function)

# >>>     Warning: RemoteExpert currently assumes that you provide it with correct input shapes.
# >>>     Sending wrong input shapes can cause RemoteExpert to freeze indefinitely due to error in runtime.

# >>>     :param uid: unique expert identifier
# >>>     :param endpoint: network endpoint of a server that services that expert, e.g. "201.123.321.99:1337" or "[::]:8080"
# >>>     """

# >>>     def __init__(self, uid, endpoint: Endpoint):
# >>>         super().__init__()
# >>>         self.uid, self.endpoint = uid, endpoint
# >>>         self._info = None

# >>>     @property
# >>>     def stub(self):
# >>>         return _get_expert_stub(self.endpoint)

# >>>     def forward(self, *args, **kwargs):
# >>>         """Call RemoteExpert for the specified inputs and return its output(s). Compatible with pytorch.autograd."""
# >>>         assert len(kwargs) == len(
# >>>             self.info["keyword_names"]
# >>>         ), f"Keyword args should be {self.info['keyword_names']}"

# >>>         kwargs = {key: kwargs[key] for key in self.info["keyword_names"]}

# >>>         # Note: we put keyword arguments in the same order as on a server to prevent f(a=1, b=2) != f(b=2, a=1) errors

# >>>         forward_inputs = (args, kwargs)

# >>>         if not nested_compare(forward_inputs, self.info["forward_schema"]):
# >>>             raise TypeError(
# >>>                 f"Inputs do not match expert input schema. Did you pass the right number of parameters?"
# >>>             )

# >>>         flat_outputs = _RemoteModuleCall.apply(
# >>>             DUMMY, self.uid, self.stub, self.info, *nested_flatten(forward_inputs)
# >>>         )
# >>>         # Note: we send DUMMY to prevent torch from excluding expert from backward if no other inputs require grad
# >>>         return nested_pack(flat_outputs, structure=self.info["outputs_schema"])

# >>>     @property
# >>>     def info(self):
# >>>         if self._info is None:
# >>>             outputs = self.stub.info(runtime_pb2.ExpertUID(uid=self.uid))
# >>>             self._info = MSGPackSerializer.loads(outputs.serialized_info)
# >>>         return self._info

# >>>     def extra_repr(self):
# >>>         return f"uid={self.uid}, endpoint={self.endpoint}"


# >>> class _RemoteModuleCall(torch.autograd.Function):
# >>>     """Internal autograd-friendly call of a remote module. For applications, use RemoteExpert instead."""

# >>>     @staticmethod
# >>>     def forward(
# >>>         ctx,
# >>>         dummy: torch.Tensor,
# >>>         uid: str,
# >>>         stub: runtime_grpc.ConnectionHandlerStub,
# >>>         info: Dict[str, Any],
# >>>         *inputs: torch.Tensor,
# >>>     ) -> Tuple[torch.Tensor, ...]:
# >>>         # Note: *inputs are flattened input tensors that follow the expert's info['input_schema']
# >>>         # detach to avoid pickling the computation graph
# >>>         inputs = tuple(tensor.cpu().detach() for tensor in inputs)
# >>>         ctx.uid, ctx.stub, ctx.info = uid, stub, info
# >>>         ctx.save_for_backward(*inputs)

# >>>         serialized_tensors = [
# >>>             serialize_torch_tensor(inp, proto.compression)
# >>>             for inp, proto in zip(inputs, nested_flatten(info["forward_schema"]))
# >>>         ]

# >>>         outputs = stub.forward(
# >>>             runtime_pb2.ExpertRequest(uid=ctx.uid, tensors=serialized_tensors)
# >>>         )

# >>>         deserialized_outputs = [
# >>>             deserialize_torch_tensor(tensor) for tensor in outputs.tensors
# >>>         ]

# >>>         return tuple(deserialized_outputs)

# >>>     @staticmethod
# >>>     @once_differentiable
# >>>     def backward(ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:
# >>>         grad_outputs_cpu = tuple(tensor.cpu() for tensor in grad_outputs)
# >>>         inputs_and_grad_outputs = tuple(
# >>>             nested_flatten((ctx.saved_tensors, grad_outputs_cpu))
# >>>         )
# >>>         backward_schema = tuple(
# >>>             nested_flatten((ctx.info["forward_schema"], ctx.info["outputs_schema"]))
# >>>         )
# >>>         serialized_tensors = [
# >>>             serialize_torch_tensor(tensor, proto.compression)
# >>>             for tensor, proto in zip(inputs_and_grad_outputs, backward_schema)
# >>>         ]

# >>>         grad_inputs = ctx.stub.backward(
# >>>             runtime_pb2.ExpertRequest(uid=ctx.uid, tensors=serialized_tensors)
# >>>         )

# >>>         deserialized_grad_inputs = [
# >>>             deserialize_torch_tensor(tensor) for tensor in grad_inputs.tensors
# >>>         ]
# >>>         return (DUMMY, None, None, None, *deserialized_grad_inputs)
