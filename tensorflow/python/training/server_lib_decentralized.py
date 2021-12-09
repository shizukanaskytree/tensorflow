"""A Python interface for creating TensorFlow servers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import device_filters_pb2
# from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.core.protobuf import tensorflow_server_decentralized_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import errors
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


# 写一个 class 也行.


@tf_export("distribute.DecentralizedServer")
class DecentralizedServer(object):
    """An in-process decentralized server, for use in distributed training.

    The server without a cluster info but can communicate with other servers
    joining
    """

    def __init__(
        self,
        server_or_cluster_def=None,
        job_name=None,
        task_index=None,
        protocol=None,
        config=None,
        start=True,
    ):

        self._server_def = tensorflow_server_decentralized_pb2.DecentralizedServerDef()
        self._server = c_api.TF_NewDecentralizedServer(self._server_def.SerializeToString())
        if start:
            self.start()
