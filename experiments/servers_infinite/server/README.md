# How to create a server without cluster requirement?

`tf.distribute.Server` 


```py
@tf_export("distribute.Server", v1=["distribute.Server", "train.Server"])
@deprecation.deprecated_endpoints("train.Server")
class Server(object):
  """An in-process TensorFlow server, for use in distributed training.

  A `tf.distribute.Server` instance encapsulates a set of devices and a
  `tf.compat.v1.Session` target that
  can participate in distributed training. A server belongs to a
  cluster (specified by a `tf.train.ClusterSpec`), and
  corresponds to a particular task in a named job. The server can
  communicate with any other server in the same cluster.
  """
```