tf.distribute.Server

API: https://www.tensorflow.org/api_docs/python/tf/distribute/Server


An in-process TensorFlow server, for use in distributed training.

```
tf.distribute.Server(
    server_or_cluster_def, job_name=None, task_index=None, protocol=None,
    config=None, start=True
)
```

Compat aliases for migration

See Migration guide for more details.

tf.compat.v1.distribute.Server, tf.compat.v1.train.Server

A `tf.distribute.Server` instance encapsulates a set of devices and a `tf.compat.v1.Session` target that can participate in distributed training. 

A server belongs to a cluster (specified by a `tf.train.ClusterSpec`), and corresponds to a particular task in a named job. 

**The server can communicate with any other server in the same cluster.**
- limitation: how to let other server participate?

Used in the notebooks:

Example 1: Migrate multi-worker CPU/GPU training
- https://www.tensorflow.org/guide/migrate/multi_worker_cpu_gpu_training

Example 2: Parameter server training with ParameterServerStrategy
- https://www.tensorflow.org/tutorials/distribute/parameter_server_training 


Args:

`server_or_cluster_def`
    describing the server to be created and/or the cluster of which it is a member.
    A `tf.train.ServerDef` or `tf.train.ClusterDef` protocol buffer, or a `tf.train.ClusterSpec` object,
    - `tf.train.ServerDef`, https://www.tensorflow.org/api_docs/python/tf/train/ServerDef
    - `tf.train.ClusterDef`, https://www.tensorflow.org/api_docs/python/tf/train/ClusterDef
    - `tf.train.ClusterSpec`, https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec



`config`: 
    (Options.) A `tf.compat.v1.ConfigProto` that specifies default configuration options for all sessions that run on this server.

`join`
    Blocks until the server has shut down. This method currently blocks forever.


# New findings

Module: tf.distribute.experimental.partitioners

https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/partitioners 