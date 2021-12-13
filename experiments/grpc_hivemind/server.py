import multiprocessing as mp
import os
from typing import Dict

import grpc
import torch

# from hivemind.compression import deserialize_torch_tensor, serialize_torch_tensor
# from hivemind.moe.server.expert_backend import ExpertBackend
from proto import runtime_pb2, runtime_pb2_grpc as runtime_grpc
from utils.logging import get_logger
from utils.nested import nested_flatten
from utils.serializer import MSGPackSerializer
from utils.asyncio import switch_to_uvloop
from utils.networking import Endpoint

GRPC_KEEPALIVE_OPTIONS = (
    ("grpc.keepalive_time_ms", 60 * 1000),
    ("grpc.keepalive_timeout_ms", 60 * 1000),
    ("grpc.keepalive_permit_without_calls", True),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.http2.min_time_between_pings_ms", 30 * 1000),
    ("grpc.http2.min_ping_interval_without_data_ms", 10 * 1000),
)


logger = get_logger(__name__)


class ExpertBackend:
    ...


class ConnectionHandler(mp.context.ForkProcess):
    """
    A process that accepts incoming requests to experts and submits them into the corresponding TaskPool.


    :param listen_on: network interface, e.g. "0.0.0.0:1337" or "localhost:*" (* means pick any port) or "[::]:7654"
    :param experts: a dict [UID -> ExpertBackend] with all active experts
    :note: ConnectionHandler is designed so as to allow using multiple handler processes for the same port.

    Code logic:
    1. define the grpc server
    2. add service to server
    3. add listening port to the server
    4. start the server
    5. wait the server to terminate if needed.
    """

    def __init__(self, listen_on: Endpoint):
        """[summary]

        Args:
            listen_on (Endpoint): [description]
            experts (Dict[str, ExpertBackend]): 这个例子里面现在没有写
        """
        super().__init__()
        self.listen_on = listen_on
        self.ready = mp.Event()

    def run(self):
        torch.set_num_threads(1)

        loop = switch_to_uvloop()

        async def _run():
            grpc.aio.init_grpc_aio()

            logger.debug(f"Starting, pid {os.getpid()}")

            # grpc max_send_message_length, max_receive_message_length 参数含义和使用 (server side and client side)
            # https://stackoverflow.com/a/65009436/7748163
            server = grpc.aio.server(
                options=GRPC_KEEPALIVE_OPTIONS
                + (
                    ("grpc.so_reuseport", 1),
                    ("grpc.max_send_message_length", -1),
                    ("grpc.max_receive_message_length", -1),
                )
            )

            # Boilerplate code
            runtime_grpc.add_ConnectionHandlerServicer_to_server(self, server)

            """
            server.add_insecure_port 分析
            Opens an insecure port for accepting RPCs.
            """
            found_port = server.add_insecure_port(self.listen_on)

            assert found_port != 0, f"Failed to listen to {self.listen_on}"

            await server.start()
            self.ready.set()

            await server.wait_for_termination()
            logger.debug(f"ConnectionHandler terminated: (pid={os.getpid()})")

        try:
            loop.run_until_complete(_run())
        except KeyboardInterrupt:
            logger.debug("Caught KeyboardInterrupt, shutting down")

    async def info(self, request: runtime_pb2.ExpertUID, context: grpc.ServicerContext):
        """[summary]

        Args:
            request (runtime_pb2.ExpertUID): [description]
            context (grpc.ServicerContext): [description]

        Returns:
            [type]: [description]
        """
        print("info-ing")
        print("info-ed")
        return runtime_pb2.ExpertInfo(
            serialized_info=MSGPackSerializer.dumps(
                # self.experts[request.uid].get_info()
                "test content"
            )
        )

    async def forward(
        self, request: runtime_pb2.ExpertRequest, context: grpc.ServicerContext
    ):
        print("forward-ing")

        # inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        # future = self.experts[request.uid].forward_pool.submit_task(*inputs)
        # serialized_response = [
        #     serialize_torch_tensor(tensor, proto.compression, allow_inplace=True)
        #     for tensor, proto in zip(await future, nested_flatten(self.experts[request.uid].outputs_schema))
        # ]

        # return runtime_pb2.ExpertResponse(tensors=serialized_response)
        print("forward-ed")
        return runtime_pb2.ExpertResponse()

    async def backward(
        self, request: runtime_pb2.ExpertRequest, context: grpc.ServicerContext
    ):
        print("backward-ing")
        # inputs_and_grad_outputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        # future = self.experts[request.uid].backward_pool.submit_task(*inputs_and_grad_outputs)
        # serialized_response = [
        #     serialize_torch_tensor(tensor, proto.compression, allow_inplace=True)
        #     for tensor, proto in zip(await future, nested_flatten(self.experts[request.uid].grad_inputs_schema))
        # ]
        # return runtime_pb2.ExpertResponse(tensors=serialized_response)
        print("backward-ed")
        return runtime_pb2.ExpertResponse()

endpoint = "localhost:28282"
conn = ConnectionHandler(listen_on=endpoint)
conn.run()