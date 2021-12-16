from grpctimedstorage.utils.grpc import ChannelCache
from grpctimedstorage.proto import runtime_pb2, runtime_pb2_grpc as runtime_grpc

import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
debugpy.breakpoint()

# def _get_expert_stub(endpoint: Endpoint, *extra_options: Tuple[str, Any]):
#     """Create a gRPC stub to access remote expert or use previously created stub from a process-wide cache"""
#     channel_options = (("grpc.max_send_message_length", -1), ("grpc.max_receive_message_length", -1)) + extra_options
#     return ChannelCache.get_stub(endpoint, runtime_grpc.ConnectionHandlerStub, aio=False, options=channel_options)

### extra_options: Tuple[str, Any]
extra_options = ()
channel_options = (("grpc.max_send_message_length", -1), ("grpc.max_receive_message_length", -1)) + extra_options

# 不需要直接调用 get_singleton, 因为 ChannelCache.get_stub 
# channel_cache = ChannelCache.get_singleton()
ChannelCache.get_stub(
    target="localhost:28282",
    stub_type=runtime_grpc.ConnectionHandlerStub,
    aio=False
)

print('end')