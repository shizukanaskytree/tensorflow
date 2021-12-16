"""
Utilities for running GRPC services: compile protobuf, patch legacy versions, etc
"""

from __future__ import annotations

import os
import threading
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import grpc

# How does python import from upper directory?

import proto.runtime_pb2 as runtime_pb2
from utils.logging import get_logger
from utils.networking import Endpoint
from utils.timed_storage import TimedStorage, ValueWithExpiration, get_dht_time

logger = get_logger(__name__)

Stub = TypeVar("Stub")

GRPC_KEEPALIVE_OPTIONS = (
    ("grpc.keepalive_time_ms", 60 * 1000),
    ("grpc.keepalive_timeout_ms", 60 * 1000),
    ("grpc.keepalive_permit_without_calls", True),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.http2.min_time_between_pings_ms", 30 * 1000),
    ("grpc.http2.min_ping_interval_without_data_ms", 10 * 1000),
)


class ChannelInfo(NamedTuple):
    target: Endpoint
    aio: bool
    options: Tuple[Tuple[str, str], ...]
    credentials: Optional[grpc.ChannelCredentials]
    compression: Optional[grpc.Compression]


class ChannelCache(
    TimedStorage[ChannelInfo, Tuple[Union[grpc.Channel, grpc.aio.Channel], Dict]]
):
    """
    A process-wide cache of gRPC channels, supports both normal and aio channels, secure/insecure channels, etc
    Based on grpcio internal channel cache by Richard Belleville and Lidi Zheng (thanks!)
    Unlike TimedStorage, ChannelCache actively evicts stale channels even if the cache is not accessed
    Unlike grpc._simple_stubs.ChannelCache, this implementation supports aio and does not forcibly close active channels

    Takeaway:
    Stale channels should be evicted.
    """

    MAXIMUM_CHANNELS = int(os.environ.get("GRPC_PYTHON_MANAGED_CHANNEL_MAXIMUM", 4096))
    EVICTION_PERIOD_SECONDS = float(
        os.environ.get("GRPC_PYTHON_MANAGED_CHANNEL_EVICTION_SECONDS", 10 * 60)
    )
    logger.debug(
        f"Eviction period = {EVICTION_PERIOD_SECONDS}s, max channels = {MAXIMUM_CHANNELS}"
    )

    _singleton: Optional[ChannelCache] = None
    _singleton_pid: int = os.getpid()
    _lock: threading.RLock = threading.RLock()
    _update_eviction_evt: threading.Event = threading.Event()

    def __init__(self, _created_as_singleton=False):
        assert (
            _created_as_singleton
        ), f"Please use {self.__class__.__name__}.get_singleton()"
        super().__init__(maxsize=self.MAXIMUM_CHANNELS)

        self._is_active = True

        self._nearest_expiration_time = float("inf")

        self._eviction_thread = threading.Thread(
            target=self._evict_stale_channels_in_background, daemon=True
        )
        self._eviction_thread.start()

    @classmethod
    def get_singleton(cls):
        """Get or create the channel cache for the current process"""
        with cls._lock:
            if cls._singleton is None or cls._singleton_pid != os.getpid():
                if cls._singleton is not None:
                    cls._singleton._stop_background_thread()
                cls._singleton, cls._singleton_pid = (
                    cls(_created_as_singleton=True),
                    os.getpid(),
                )
            return cls._singleton

    @classmethod
    def get_stub(
        cls,
        target: Endpoint,
        stub_type: Type[Stub],
        *,
        aio: bool,
        options: Tuple[Tuple[str, Any]] = (),
        channel_credentials: Optional[grpc.ChannelCredentials] = None,
        compression: Optional[grpc.Compression] = None,
    ) -> Stub:
        """
        Create a grpc channel with given options or reuse pre-existing one

        :param target: the recipient's address and port
        :param stub_type: a gRPC stub (client) to be instantiated
        :param aio: if True, returns grpc.Channel, otherwise returns grpc.aio.Channel
        :param options: see https://grpc.github.io/grpc/core/group__grpc__arg__keys.html
        :param channel_credentials: if specified, create a secure channel usin these credentials (default = insecure)
        :param compression: see https://github.com/grpc/grpc/tree/master/examples/python/compression


                           +----+
                           |    |
                           |    |
                           |    |
                           |    |
                   |------>|    |
                   |       |    |
        +----+     |       |    |
        |    |     |       +----+
        |    |-----+       +----+
        |    |             |    |
        |    |------------>|    |
        |    |             |    |
        |    |------+      |    |
        |    |      |      |    |
        +----+      |      |    |
                    |      |    |
                    |      +----+
                    |
                    |      +----+
                    |      |    |
                    |      |    |
                    v----->|    |
                           |    |
                           |    |
                           |    |
                           |    |
                           +----+


        """
        cache = cls.get_singleton()
        with cls._lock:
            key = ChannelInfo(
                target, aio, tuple(options), channel_credentials, compression
            )

            entry: ValueWithExpiration = super(cls, cache).get(key)

            if entry is not None:
                channel, stubs = entry.value
            else:
                channel = cls._create_channel(*key)
                stubs = {}

            channel._channel.check_connectivity_state(True)

            if stub_type not in stubs:
                stubs[stub_type] = stub_type(channel)

            # either cache channel or update expiration of an existing channel
            expiration_time = get_dht_time() + cls.EVICTION_PERIOD_SECONDS
            super(cls, cache).store(key, (channel, stubs), expiration_time)

            if expiration_time < cache._nearest_expiration_time:
                cache._nearest_expiration_time = expiration_time
                cls._update_eviction_evt.set()

            return stubs[stub_type]

    @classmethod
    def _create_channel(
        cls,
        target: Endpoint,
        aio: bool,
        extra_options: Tuple[Tuple[str, Any], ...],
        channel_credentials: Optional[grpc.ChannelCredentials],
        compression: Optional[grpc.Compression],
    ) -> Union[grpc.Channel, grpc.aio.Channel]:
        """
        功能: 原来可以如此精巧地构造 channel.

        3 args to init insecure channel which is the minimus.
        """

        namespace = grpc.aio if aio else grpc

        options = extra_options + GRPC_KEEPALIVE_OPTIONS

        if channel_credentials is None:
            logger.debug(
                f"Creating insecure {namespace} channel with options '{options}' "
                f"and compression '{compression}'"
            )
            return namespace.insecure_channel(
                target, options=options, compression=compression
            )
        else:
            logger.debug(
                f"Creating secure {namespace} channel with credentials '{channel_credentials}', "
                f"options '{options}' and compression '{compression}'"
            )
            return namespace.secure_channel(
                target,
                credentials=channel_credentials,
                options=options,
                compression=compression,
            )

    def _evict_stale_channels_in_background(self):
        """
        Summary: 不是很奏效, 我试图用不带具体变量名的语言来描述.

        while self._is_active is true, we first get the current time.
        Next, we calculate the time to wait. It must be larger than 0, and it is calculated by the difference between nearest_expiration time and the current time.
        The update eviction event is waiting/blocking for the waiting time we calculate above. If timeout, we will remove outdated channels.
        if

        we finally update the nearest expiration time. It is the same if the entry in the Timed Cache is not None.
        Otherwise we set

        === 对比一下.

        我觉得这个比较奏效.

        Summary: 周期性地进行 `_remove_outdated`, 周期是 wait() 内的时间, 否则就是永远 blocking 在 wait.

        While `self._is_active` is True, the `_evict_stale_channels_in_background` function runs forever.

        If we do not set `self._nearest_expiration_time`, the initial value is infinite,
        the timeout for `self._update_eviction_evt.wait()` is infinite, so wait forever.

        `self._update_eviction_evt` True or False?
        If `self._update_eviction_evt` is True,
        Reset the `self._update_eviction_evt` flag to False.

        Subsequently, threads calling `self._update_eviction_evt.wait()` will block until
        `self._update_eviction_evt.set()` is called to set the `self._update_eviction_evt` flag to True again.

        `self._update_eviction_evt.wait()` blocks until the `self._update_eviction_evt` flag is true.
        If the  `self._update_eviction_evt` flag is true on entry, return immediately.
        Otherwise, block until another thread calls `self._update_eviction_evt.set()` to set the flag
        `self._update_eviction_evt` to true, or until the optional timeout occurs.

        When the timeout argument `time_to_wait` is present and not None, it should be a floating point number
        specifying a timeout for the operation in seconds (or fractions thereof).

        `self._update_eviction_evt.wait()` method returns the internal flag `self._update_eviction_evt` on exit,
        so it will always return True except if a timeout `time_to_wait` is given and the operation times out.
        """
        while self._is_active:
            now = get_dht_time()
            time_to_wait = max(0.0, self._nearest_expiration_time - now)
            interrupted_early = self._update_eviction_evt.wait(
                time_to_wait if time_to_wait != float("inf") else None
            )
            if interrupted_early:
                self._update_eviction_evt.clear()
                continue

            with self._lock:
                self._remove_outdated()
                _, entry = super().top()
                self._nearest_expiration_time = (
                    entry.expiration_time if entry is not None else float("inf")
                )

    def _stop_background_thread(self):
        with self._lock:
            self._is_active = False
            self._update_eviction_evt.set()

    def store(self, *args, **kwargs) -> ValueError:
        raise ValueError(
            f"Please use {self.__class__.__name__}.get_stub to get or create stubs"
        )

    def get(self, *args, **kwargs) -> ValueError:
        raise ValueError(
            f"Please use {self.__class__.__name__}.get_stub to get or create stubs"
        )

    def top(self) -> ValueError:
        raise ValueError(
            f"Please use {self.__class__.__name__}.get_stub to get or create stubs"
        )


# STREAMING_CHUNK_SIZE_BYTES = 2 ** 16


# def split_for_streaming(
#     serialized_tensor: runtime_pb2.Tensor,
#     chunk_size_bytes: int = STREAMING_CHUNK_SIZE_BYTES,
# ) -> Iterator[runtime_pb2.Tensor]:
#     """Split serialized_tensor into multiple chunks for gRPC streaming"""
#     buffer = memoryview(serialized_tensor.buffer)
#     num_chunks = len(range(0, len(buffer), chunk_size_bytes))
#     yield runtime_pb2.Tensor(
#         compression=serialized_tensor.compression,
#         buffer=buffer[:chunk_size_bytes].tobytes(),
#         chunks=num_chunks,
#         size=serialized_tensor.size,
#         dtype=serialized_tensor.dtype,
#         requires_grad=serialized_tensor.requires_grad,
#     )
#     for chunk_start in range(chunk_size_bytes, len(buffer), chunk_size_bytes):
#         yield runtime_pb2.Tensor(buffer=buffer[chunk_start : chunk_start + chunk_size_bytes].tobytes())


# def combine_from_streaming(stream: Iterable[runtime_pb2.Tensor]) -> runtime_pb2.Tensor:
#     """Restore a result of split_into_chunks into a single serialized tensor"""
#     stream = iter(stream)
#     first_chunk = next(stream)
#     serialized_tensor = runtime_pb2.Tensor()
#     serialized_tensor.CopyFrom(first_chunk)
#     buffer_chunks = [first_chunk.buffer]
#     for tensor_part in stream:
#         buffer_chunks.append(tensor_part.buffer)
#     serialized_tensor.buffer = b"".join(buffer_chunks)
#     return serialized_tensor
