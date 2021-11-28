# import debugpy

# debugpy.listen(5678)
# debugpy.wait_for_client()
# debugpy.breakpoint()

"""
code: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/keras/distribute

batch process first.
process it next.

通过学习这个我可以了解 tf2 server 的用法.

- cluster
- server
- all worker
- graph partitioning

Parameter server training with ParameterServerStrategy

Overview

Parameter server training is a common data-parallel method to scale up model training on 
multiple machines.

A parameter server training cluster consists of workers and parameter servers. 
Variables are created on parameter servers and they are read and updated by workers 
in each step. By default, workers read and update these variables independently without 
synchronizing with each other. This is why sometimes parameter server-style training is 
called asynchronous training.

In TensorFlow 2, parameter server training is powered by the 
tf.distribute.experimental.ParameterServerStrategy class, which distributes the 
training steps to a cluster that scales up to thousands of workers (accompanied by 
parameter servers).

-------------------------------------------------------------------------------------

Supported training methods

There are two main supported training methods:

- The Keras Model.fit API, which is recommended when you prefer a high-level abstraction and 
handling of training.

- A custom training loop (you can refer to Custom training, Writing a training loop from 
    scratch and Custom training loop with Keras and MultiWorkerMirroredStrategy for more details.) 
    Custom loop training is recommended when you prefer to define the details of their training loop.

-------------------------------------------------------------------------------------

A cluster with jobs and tasks

Regardless of the API of choice (Model.fit or a custom training loop), distributed training 
in TensorFlow 2 involves: a 'cluster' with several 'jobs', and each of the jobs may have 
one or more 'tasks'.

When using parameter server training, it is recommended to have:

- One coordinator job (which has the job name chief)
- Multiple worker jobs (job name worker); and
- Multiple parameter server jobs (job name ps)

While the coordinator creates resources, dispatches training tasks, writes checkpoints, 
and deals with task failures, workers and parameter servers run tf.distribute.Server 
that listen for requests from the coordinator.

-------------------------------------------------------------------------------------

Parameter server training with Model.fit API

Parameter server training with the Model.fit API requires the coordinator to use a 
tf.distribute.experimental.ParameterServerStrategy object, and a 
tf.keras.utils.experimental.DatasetCreator as the input. Similar to Model.fit usage with 
no strategy, or with other strategies, the workflow involves creating and compiling the 
model, preparing the callbacks, followed by a Model.fit call.

-------------------------------------------------------------------------------------

Parameter server training with a custom training loop

With custom training loops, the tf.distribute.experimental.coordinator.ClusterCoordinator 
class is the key component used for the coordinator.

- The ClusterCoordinator class needs to work in conjunction with a tf.distribute.Strategy object.
- This tf.distribute.Strategy object is needed to provide the information of the cluster and 
    is used to define a training step, as demonstrated in Custom training with tf.distribute.Strategy.
- The ClusterCoordinator object then dispatches the execution of these training steps to remote workers.
- For parameter server training, the ClusterCoordinator needs to work with a tf.distribute.experimental.ParameterServerStrategy.

-------------------------------------------------------------------------------------

The most important API provided by the ClusterCoordinator object is schedule:

- The schedule API enqueues a tf.function and returns a future-like RemoteValue immediately.
- The queued functions will be dispatched to remote workers in background threads and their RemoteValues will be filled asynchronously.
- Since schedule doesn’t require worker assignment, the tf.function passed in can be executed on any available worker.
- If the worker it is executed on becomes unavailable before its completion, the function will be retried on another available worker.
- Because of this fact and the fact that function execution is not atomic, a function may be executed more than once.

-------------------------------------------------------------------------------------

In addition to dispatching remote functions, the ClusterCoordinator also helps to create datasets on all the workers and rebuild these datasets when a worker recovers from failure.

Tutorial setup
The tutorial will branch into Model.fit and custom training loop paths,
and you can choose the one that fits your needs. Sections other than "Training with X"
are applicable to both paths.
pip install portpicker

Cluster setup

As mentioned above, a parameter server training cluster requires a coordinator task that 
runs your training program, one or several workers and parameter server tasks that run 
TensorFlow servers—tf.distribute.Server—and possibly an additional evaluation task that 
runs side-car evaluation (see the side-car evaluation section below). 
The requirements to set them up are:

- The coordinator task needs to know the addresses and ports of all other TensorFlow servers except the evaluator.
- The workers and parameter servers need to know which port they need to listen to. For the sake of simplicity, you can usually pass in the complete cluster information when creating TensorFlow servers on these tasks.
- The evaluator task doesn’t have to know the setup of the training cluster. If it does, it should not attempt to connect to the training cluster.
- Workers and parameter servers should have task types as "worker" and "ps", respectively. The coordinator should use "chief" as the task type for legacy reasons.

wxf: 
* coordinator
* only workers

In this tutorial, you will create an in-process cluster so that the whole 
parameter server training can be run in Colab. You will learn how to set up real clusters 
in a later section.

----------------------------------------------------------------------------------

In-process cluster
核心: 在一个 process 里面构造了 N 个 TF server.

You will start by creating several TensorFlow servers in advance and connect to them later. 
Note that this is only for the purpose of this tutorial's demonstration, and in real training
the servers will be started on "worker" and "ps" machines.

"""

import multiprocessing
import os
import random
import portpicker

# tf 2
import tensorflow as tf


def create_in_process_cluster(num_workers, num_ps):
    """Creates and starts local servers and returns the cluster_resolver.

    num_workers: inter_op_parallelism_threads
    num_ps: if I set this to 0, it is also right for this function.
    """

    # for a range create a list.
    worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
    ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

    """
    cluster_dict is the args to the tf.train.ClusterSpec.
    tf.train.ClusterSpec: https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec

    To create a cluster with two jobs and five tasks, you specify
    the mapping from job names to lists of network addresses (typically hostname-port pairs).

    A tf.train.ClusterSpec represents the set of processes that participate in a distributed TensorFlow computation.

    `tf.train.ClusterSpec`: Represents a cluster as a set of "tasks", organized into "jobs".

    Every tf.distribute.Server is constructed in a particular cluster.

    This cluster has ps and workers
    It is like:
    cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                            "worker1.example.com:2222",
                                            "worker2.example.com:2222"],
                                    "ps": ["ps0.example.com:2222",
                                        "ps1.example.com:2222"]})
    """

    cluster_dict = {}
    cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
    if num_ps > 0:
        cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]
    cluster_spec = tf.train.ClusterSpec(cluster_dict)

    # Workers need some inter_ops threads to work properly.
    # We still use tf.compat.v1.ConfigProto() in the tf2 tutorial:
    # https://www.tensorflow.org/tutorials/distribute/parameter_server_training

    # tf.compat.v1.ConfigProto()
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/ConfigProto
    worker_config = tf.compat.v1.ConfigProto()
    if multiprocessing.cpu_count() < num_workers + 1:
        # assignment is OK for setting configuration.
        worker_config.inter_op_parallelism_threads = num_workers + 1

    for i in range(num_workers):
        # tf.distribute.Server is documented:
        # API: https://www.tensorflow.org/api_docs/python/tf/distribute/Server

        # Args: start, (Optional.) Boolean, indicating whether to start the server after creating it. Defaults to True.
        # So, the server starts.
        # https://www.tensorflow.org/tutorials/distribute/parameter_server_training
        tf.distribute.Server(
            cluster_spec,  # describing the server to be created and/or the cluster of which it is a member.
            job_name="worker",  # Specifies the name of the job of which the server is a member.
            task_index=i,  # Specifies the task index of the server in its job.
            config=worker_config,  # A tf.compat.v1.ConfigProto that specifies default configuration options for all sessions that run on this server.
            protocol="grpc",
        )

    for i in range(num_ps):
        tf.distribute.Server(cluster_spec, job_name="ps", task_index=i, protocol="grpc")

    # ClusterResolvers are a way for TensorFlow to communicate with various cluster management systems
    # (e.g. GCE, AWS, etc...) and gives TensorFlow necessary information to set up distributed training.

    # tf.distribute.cluster_resolver.SimpleClusterResolver
    # API and example: https://www.tensorflow.org/api_docs/python/tf/distribute/cluster_resolver/SimpleClusterResolver
    # Code: https://github.com/tensorflow/tensorflow/blob/v2.7.0/tensorflow/python/distribute/cluster_resolver/cluster_resolver.py#L293-L419

    # SimpleClusterResolver is based on ClusterResolver below.
    # tf.distribute.cluster_resolver.ClusterResolver
    # Doc: https://www.tensorflow.org/api_docs/python/tf/distribute/cluster_resolver/ClusterResolver

    # This library contains all implementations of ClusterResolvers. ClusterResolvers are a way of
    # specifying cluster information for distributed execution. Built on top of existing ClusterSpec
    # framework, ClusterResolvers are a way for TensorFlow to communicate with various cluster management
    # systems (e.g. GCE, AWS, etc...).
    cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec, rpc_layer="grpc"
    )
    return cluster_resolver


# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

"""
Experiment 1:
NUM_PS = 0
"""

# I create 3 worker servers, and 2 ps servers LOCALLY.
# Is this what I want? No.
# What I want to do is to instantiate 1 main TF server at each machine or for each GPU.
NUM_WORKERS = 3
NUM_PS = 2

# Creates and starts local servers and returns the cluster_resolver.
cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)

print("-" * 60)

"""
Experiment 2:
Print cluster_resolver attributes
"""
print(f"cluster_resolver.environment: {cluster_resolver.environment}")
print(f"cluster_resolver.rpc_layer: {cluster_resolver.rpc_layer}")


"""
----------------------------------------------------------------------------------

The in-process cluster setup is frequently used in unit testing, such as here.
Another option for local testing is to launch processes on the local machine—check out Multi-worker training with Keras for an example of this approach.

----------------------------------------------------------------------------------

Instantiate a ParameterServerStrategy
=====================================

Before you dive into the training code, let's instantiate a ParameterServerStrategy object. 
Note that this is needed regardless of whether you are proceeding with Model.fit or a custom training loop. 
The variable_partitioner argument will be explained in the Variable sharding section.


"""

print("*" * 60)
variable_partitioner = tf.distribute.experimental.partitioners.MinSizePartitioner(
    min_shard_bytes=(256 << 10), max_shards=NUM_PS
)
print(f"variable_partitioner : {variable_partitioner}")

strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver, variable_partitioner=variable_partitioner
)
print(f"strategy: {strategy}")

"""
variable_partitioner : <tensorflow.python.distribute.sharded_variable.MinSizePartitioner object at 0x7fccc9937f40>
2021-11-24 16:33:59.396418: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22289 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:01:00.0, compute capability: 8.6
2021-11-24 16:33:59.401434: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:chief/replica:0/task:0/device:GPU:0 with 22289 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:01:00.0, compute capability: 8.6
2021-11-24 16:33:59.413478: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job ps -> {0 -> localhost:20295, 1 -> localhost:16780}
2021-11-24 16:33:59.413500: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> localhost:15073, 1 -> localhost:19301, 2 -> localhost:16688}

2021-11-24 16:33:59.413504: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job chief -> {0 -> localhost:35127}
2021-11-24 16:33:59.428713: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job ps -> {0 -> localhost:20295, 1 -> localhost:16780}
2021-11-24 16:33:59.428738: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> localhost:15073, 1 -> localhost:19301, 2 -> localhost:16688}

2021-11-24 16:33:59.428743: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job chief -> {0 -> localhost:35127}
2021-11-24 16:33:59.428754: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job ps -> {0 -> localhost:20295, 1 -> localhost:16780}
2021-11-24 16:33:59.428772: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> localhost:15073, 1 -> localhost:19301, 2 -> localhost:16688}

2021-11-24 16:33:59.428777: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job chief -> {0 -> localhost:35127}
2021-11-24 16:33:59.428813: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job ps -> {0 -> localhost:20295, 1 -> localhost:16780}
2021-11-24 16:33:59.428829: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> localhost:15073, 1 -> localhost:19301, 2 -> localhost:16688}
2021-11-24 16:33:59.428835: I tensorflow/core/distributed_runtime/eager/eager_service_impl.cc:272] Creating sync eager service context with rendezvous_id on host protago-hp01-3090 /job:ps/replica:0/task:0

2021-11-24 16:33:59.428846: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job chief -> {0 -> localhost:35127}
2021-11-24 16:33:59.428855: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job ps -> {0 -> localhost:20295, 1 -> localhost:16780}
2021-11-24 16:33:59.428868: I tensorflow/core/distributed_runtime/eager/eager_service_impl.cc:272] Creating sync eager service context with rendezvous_id on host protago-hp01-3090 /job:ps/replica:0/task:1
2021-11-24 16:33:59.428878: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> localhost:15073, 1 -> localhost:19301, 2 -> localhost:16688}

2021-11-24 16:33:59.428884: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job chief -> {0 -> localhost:35127}
2021-11-24 16:33:59.428920: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job ps -> {0 -> localhost:20295, 1 -> localhost:16780}
2021-11-24 16:33:59.428932: I tensorflow/core/distributed_runtime/eager/eager_service_impl.cc:272] Creating sync eager service context with rendezvous_id on host protago-hp01-3090 /job:worker/replica:0/task:0
2021-11-24 16:33:59.428941: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> localhost:15073, 1 -> localhost:19301, 2 -> localhost:16688}

2021-11-24 16:33:59.428947: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job chief -> {0 -> localhost:35127}
2021-11-24 16:33:59.428952: I tensorflow/core/distributed_runtime/eager/eager_service_impl.cc:272] Creating sync eager service context with rendezvous_id on host protago-hp01-3090 /job:worker/replica:0/task:1
2021-11-24 16:33:59.429028: I tensorflow/core/distributed_runtime/eager/eager_service_impl.cc:272] Creating sync eager service context with rendezvous_id on host protago-hp01-3090 /job:worker/replica:0/task:2


2021-11-24 16:33:59.430414: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job ps -> {0 -> localhost:20295, 1 -> localhost:16780}
2021-11-24 16:33:59.430425: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> localhost:15073, 1 -> localhost:19301, 2 -> localhost:16688}
2021-11-24 16:33:59.430430: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job chief -> {0 -> localhost:35127}

2021-11-24 16:33:59.430599: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:427] Started server with target: grpc://localhost:35127
strategy: <tensorflow.python.distribute.parameter_server_strategy_v2.ParameterServerStrategyV2 object at 0x7fccc9937e20>
"""


"""
In order to use GPUs for training, allocate GPUs visible to each worker. 
ParameterServerStrategy will use all the available GPUs on each worker, 
with the restriction that all workers should have the same number of GPUs available.


Variable sharding
=================

Variable sharding refers to splitting a variable into multiple smaller variables, which are called shards. 
Variable sharding may be useful to distribute the network load when accessing these shards. 
It is also useful to distribute computation and storage of a normal variable across multiple parameter servers.

To enable variable sharding, you can pass in a variable_partitioner when constructing a ParameterServerStrategy object. 
The variable_partitioner will be invoked every time when a variable is created and it is expected to return 
the number of shards along each dimension of the variable. Some out-of-box variable_partitioners are 
provided such as tf.distribute.experimental.partitioners.MinSizePartitioner. 
It is recommended to use size-based partitioners like tf.distribute.experimental.partitioners.MinSizePartitioner 
to avoid partitioning small variables, which could have negative impact on model training speed.

When a variable_partitioner is passed in and if you create a variable directly under strategy.scope(), 
it will become a container type with a variables property which provides access to the list of shards. 
In most cases, this container will be automatically converted to a Tensor by concatenating all the shards. 
As a result, it can be used as a normal variable. On the other hand, some TensorFlow methods such as 
tf.nn.embedding_lookup provide efficient implementation for this container type and in these methods 
automatic concatenation will be avoided.

Please see the API docs of tf.distribute.experimental.ParameterServerStrategy for more details.

Training with Model.fit
=======================
Keras provides an easy-to-use training API via Model.fit that handles the training loop under the hood, 
with the flexibility of overridable train_step, and callbacks, which provide functionalities such as 
checkpoint saving or summary saving for TensorBoard.

With Model.fit, the same training code can be used for other strategies with a simple swap of the strategy object.

Input data
==========
Model.fit with parameter server training requires that the input data be provided in a callable 
that takes a single argument of type tf.distribute.InputContext, and returns a tf.data.Dataset. 
Then, create a tf.keras.utils.experimental.DatasetCreator object that takes such callable, 
and an optional tf.distribute.InputOptions object via input_options argument.

Note that it is recommended to shuffle and repeat the data with parameter server training, 
and specify steps_per_epoch in fit call so the library knows the epoch boundaries.

Please see the Distributed input tutorial for more information about the InputContext argument.
"""


def dataset_fn(input_context):
    global_batch_size = 64
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)

    x = tf.random.uniform((10, 10))
    y = tf.random.uniform((10,))

    dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
    dataset = dataset.shard(
        input_context.num_input_pipelines, input_context.input_pipeline_id
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)

    return dataset


dc = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

"""
The code in dataset_fn will be invoked on the input device, which is usually the CPU, on each of the worker machines.

Model construction and compiling
================================

Now, you will create a tf.keras.Model — a trivial tf.keras.models.Sequential model 
for demonstration purposes—followed by a Model.compile call to incorporate components, 
such as an optimizer, metrics, or parameters such as steps_per_execution:
"""
with strategy.scope():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])

model.compile(tf.keras.optimizers.SGD(), loss="mse", steps_per_execution=10)

"""
Callbacks and training
======================

Before you call model.fit for the actual training, let's prepare the needed callbacks for common tasks, such as:

ModelCheckpoint: to save the model weights.
BackupAndRestore: to make sure the training progress is automatically backed up, and recovered if the cluster experiences unavailability (such as abort or preemption); or
TensorBoard: to save the progress reports into summary files, which get visualized in TensorBoard tool.

Note: Due to performance consideration, custom callbacks cannot have batch level callbacks overridden when 
used with ParameterServerStrategy. Please modify your custom callbacks to make them epoch level calls, 
and adjust steps_per_epoch to a suitable value. In addition, steps_per_epoch is a required argument for 
Model.fit when used with ParameterServerStrategy.
"""

working_dir = "/tmp/my_working_dir"
log_dir = os.path.join(working_dir, "log")
ckpt_filepath = os.path.join(working_dir, "ckpt")
backup_dir = os.path.join(working_dir, "backup")

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath),
    tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_dir),
]

model.fit(dc, epochs=5, steps_per_epoch=20, callbacks=callbacks)
"""
Epoch 1/5
2021-11-24 17:04:14.734228: I tensorflow/stream_executor/cuda/cuda_blas.cc:1774] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2021-11-24 17:04:14.851884: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
2021-11-24 17:04:14.997126: I tensorflow/core/common_runtime/eager/kernel_and_device.cc:94] Ignoring error status when releasing multi-device function handle UNIMPLEMENTED: Releasing a multi-device component handle on a remote device is not yet implemented.
2021-11-24 17:04:15.083965: I tensorflow/core/common_runtime/eager/kernel_and_device.cc:94] Ignoring error status when releasing multi-device function handle UNIMPLEMENTED: Releasing a multi-device component handle on a remote device is not yet implemented.
20/20 - 3s - loss: 0.7441 - 3s/epoch - 136ms/step

Epoch 2/5
2021-11-24 17:04:15.363373: I tensorflow/core/common_runtime/eager/kernel_and_device.cc:94] Ignoring error status when releasing multi-device function handle UNIMPLEMENTED: Releasing a multi-device component handle on a remote device is not yet implemented.
2021-11-24 17:04:15.402817: I tensorflow/core/common_runtime/eager/kernel_and_device.cc:94] Ignoring error status when releasing multi-device function handle UNIMPLEMENTED: Releasing a multi-device component handle on a remote device is not yet implemented.
20/20 - 0s - loss: 0.6195 - 318ms/epoch - 16ms/step

Epoch 3/5
WARNING:tensorflow:5 out of the last 5 calls to <function MultiDeviceSaver.save.<locals>.tf_function_save at 0x7f184cf3eca0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
2021-11-24 17:04:15.594796: I tensorflow/core/common_runtime/eager/kernel_and_device.cc:94] Ignoring error status when releasing multi-device function handle UNIMPLEMENTED: Releasing a multi-device component handle on a remote device is not yet implemented.
WARNING:tensorflow:6 out of the last 6 calls to <function MultiDeviceSaver.save.<locals>.tf_function_save at 0x7f188c22c430> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
2021-11-24 17:04:15.635023: I tensorflow/core/common_runtime/eager/kernel_and_device.cc:94] Ignoring error status when releasing multi-device function handle UNIMPLEMENTED: Releasing a multi-device component handle on a remote device is not yet implemented.
20/20 - 0s - loss: 0.4829 - 231ms/epoch - 12ms/step

Epoch 4/5
2021-11-24 17:04:15.825682: I tensorflow/core/common_runtime/eager/kernel_and_device.cc:94] Ignoring error status when releasing multi-device function handle UNIMPLEMENTED: Releasing a multi-device component handle on a remote device is not yet implemented.
2021-11-24 17:04:15.867730: I tensorflow/core/common_runtime/eager/kernel_and_device.cc:94] Ignoring error status when releasing multi-device function handle UNIMPLEMENTED: Releasing a multi-device component handle on a remote device is not yet implemented.
20/20 - 0s - loss: 0.4135 - 231ms/epoch - 12ms/step

Epoch 5/5
2021-11-24 17:04:16.055826: I tensorflow/core/common_runtime/eager/kernel_and_device.cc:94] Ignoring error status when releasing multi-device function handle UNIMPLEMENTED: Releasing a multi-device component handle on a remote device is not yet implemented.
2021-11-24 17:04:16.095376: I tensorflow/core/common_runtime/eager/kernel_and_device.cc:94] Ignoring error status when releasing multi-device function handle UNIMPLEMENTED: Releasing a multi-device component handle on a remote device is not yet implemented.
20/20 - 0s - loss: 0.3408 - 226ms/epoch - 11ms/step
"""

"""
Direct usage with ClusterCoordinator (optional)
===============================================

Even if you choose the Model.fit training path, you can optionally instantiate a 
tf.distribute.experimental.coordinator.ClusterCoordinator object to schedule other 
functions you would like to be executed on the workers. See the Training with a 
custom training loop section for more details and examples.

Training with a custom training loop
====================================

Using custom training loops with tf.distribute.Strategy provides great flexibility to define training loops. With the ParameterServerStrategy defined above (as strategy), you will use a tf.distribute.experimental.coordinator.ClusterCoordinator to dispatch the execution of training steps to remote workers.

Then, you will create a model, define a dataset and a step function, as you have done in the training loop with other tf.distribute.Strategys. You can find more details in the Custom training with tf.distribute.Strategy tutorial.

To ensure efficient dataset prefetching, use the recommended distributed dataset creation APIs mentioned in the Dispatch training steps to remote workers section below. Also, make sure to call Strategy.run inside worker_fn to take full advantage of GPUs allocated to workers. The rest of the steps are the same for training with or without GPUs.

Let’s create these components in the following steps:


Set up the data
===============

First, write a function that creates a dataset that includes preprocessing logic implemented by Keras preprocessing layers.

You will create these layers outside the `dataset_fn` but apply the transformation inside the `dataset_fn`, since you will wrap the dataset_fn into a tf.function, which doesn't allow variables to be created inside it.

Note: tf.function doesn't allow variables to be created inside it.

Note: There is a known performance implication when using lookup table resources, which layers, such as tf.keras.layers.StringLookup, employ. Refer to the Known limitations section for more information.

"""

feature_vocab = [
    "avenger",
    "ironman",
    "batman",
    "hulk",
    "spiderman",
    "kingkong",
    "wonder_woman",
]
label_vocab = ["yes", "no"]

with strategy.scope():
    feature_lookup_layer = tf.keras.layers.StringLookup(
        vocabulary=feature_vocab, mask_token=None
    )
    label_lookup_layer = tf.keras.layers.StringLookup(
        vocabulary=label_vocab, num_oov_indices=0, mask_token=None
    )

    raw_feature_input = tf.keras.layers.Input(
        shape=(3,), dtype=tf.string, name="feature"
    )
    feature_id_input = feature_lookup_layer(raw_feature_input)

    feature_preprocess_stage = tf.keras.Model(
        {"features": raw_feature_input}, feature_id_input
    )

    raw_label_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="label")
    label_id_input = label_lookup_layer(raw_label_input)

    label_preprocess_stage = tf.keras.Model({"label": raw_label_input}, label_id_input)

"""
Generate toy examples in a dataset:
"""


def feature_and_label_gen(num_examples=200):
    examples = {"features": [], "label": []}
    for _ in range(num_examples):
        features = random.sample(feature_vocab, 3)
        label = ["yes"] if "avenger" in features else ["no"]
        examples["features"].append(features)
        examples["label"].append(label)
    return examples


examples = feature_and_label_gen()

"""Then, create the training dataset wrapped in a dataset_fn:"""


def dataset_fn(_):
    raw_dataset = tf.data.Dataset.from_tensor_slices(examples)

    train_dataset = (
        raw_dataset.map(
            lambda x: (
                {"features": feature_preprocess_stage(x["features"])},
                label_preprocess_stage(x["label"]),
            )
        )
        .shuffle(200)
        .batch(32)
        .repeat()
    )

    return train_dataset


"""
Build the model
===============

Next, create the model and other objects. Make sure to create all variables under strategy.scope.


"""

# These variables created under the `strategy.scope` will be placed on parameter
# servers in a round-robin fashion.
with strategy.scope():
    # Create the model. The input needs to be compatible with Keras processing layers.
    model_input = tf.keras.layers.Input(shape=(3,), dtype=tf.int64, name="model_input")

    emb_layer = tf.keras.layers.Embedding(
        input_dim=len(feature_lookup_layer.get_vocabulary()), output_dim=16384
    )

    emb_output = tf.reduce_mean(emb_layer(model_input), axis=1)
    dense_output = tf.keras.layers.Dense(units=1, activation="sigmoid")(emb_output)
    model = tf.keras.Model({"features": model_input}, dense_output)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)
    accuracy = tf.keras.metrics.Accuracy()

"""
Let's confirm that the use of FixedShardsPartitioner split all variables into two shards 
and each shard was assigned to different parameter servers:
"""
assert len(emb_layer.weights) == 2
assert emb_layer.weights[0].shape == (4, 16384)
assert emb_layer.weights[1].shape == (4, 16384)
assert emb_layer.weights[0].device == "/job:ps/replica:0/task:0/device:CPU:0"
assert emb_layer.weights[1].device == "/job:ps/replica:0/task:1/device:CPU:0"


"""
Define the training step

Third, create the training step wrapped into a tf.function:
"""


@tf.function
def step_fn(iterator):
    def replica_fn(batch_data, labels):
        with tf.GradientTape() as tape:
            pred = model(batch_data, training=True)

            per_example_loss = tf.keras.losses.BinaryCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE
            )(labels, pred)

            loss = tf.nn.compute_average_loss(per_example_loss)
            gradients = tape.gradient(loss, model.trainable_variables)

        # Comment: 你 apply 不 apply 我能改吗, 有其他效果吗? 这个灵活性有用吗?
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
        accuracy.update_state(labels, actual_pred)
        return loss

    batch_data, labels = next(iterator)
    losses = strategy.run(replica_fn, args=(batch_data, labels))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)


"""
@tf.function
def step_fn(iterator):

  def replica_fn(batch_data, labels):
    with tf.GradientTape() as tape:
      pred = model(batch_data, training=True)
      per_example_loss = tf.keras.losses.BinaryCrossentropy(
              reduction=tf.keras.losses.Reduction.NONE)(labels, pred)
      loss = tf.nn.compute_average_loss(per_example_loss)
      gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
    accuracy.update_state(labels, actual_pred)
    return loss

  batch_data, labels = next(iterator)
  losses = strategy.run(replica_fn, args=(batch_data, labels))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)
"""


"""
In the above training step function, calling Strategy.run and Strategy.reduce in 
the step_fn can support multiple GPUs per worker. If the workers have GPUs allocated, 
Strategy.run will distribute the datasets on multiple replicas.


Dispatch training steps to remote workers
=========================================

After all the computations are defined by ParameterServerStrategy, you will use 
the tf.distribute.experimental.coordinator.ClusterCoordinator class to create resources 
and distribute the training steps to remote workers.

Let’s first create a ClusterCoordinator object and pass in the strategy object:
"""
coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)

"""
Then, create a per-worker dataset and an iterator. In the per_worker_dataset_fn below, 
wrapping the dataset_fn into strategy.distribute_datasets_from_function is recommended 
to allow efficient prefetching to GPUs seamlessly.

"""


@tf.function
def per_worker_dataset_fn():
    return strategy.distribute_datasets_from_function(dataset_fn)


"""
More about dataset creation
===========================

The dataset in the above code is created using the ClusterCoordinator.create_per_worker_dataset API). 
It creates one dataset per worker and returns a container object. You can call the iter method on it to create a per-worker iterator. 
The per-worker iterator contains one iterator per worker and the corresponding slice of a worker will be substituted 
in the input argument of the function passed to the ClusterCoordinator.schedule method before the function 
is executed on a particular worker.

Currently, the ClusterCoordinator.schedule method assumes workers are equivalent and thus assumes the 
datasets on different workers are the same except they may be shuffled differently if they contain a 
Dataset.shuffle operation. Because of this, it is also recommended that the datasets to be repeated 
indefinitely and you schedule a finite number of steps instead of relying on the OutOfRangeError from a dataset.

Another important note is that tf.data datasets don’t support implicit serialization and deserialization 
across task boundaries. So it is important to create the whole dataset inside the function passed to 
ClusterCoordinator.create_per_worker_dataset.
"""
per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
per_worker_iterator = iter(per_worker_dataset)

"""
WARNING:tensorflow:Model was constructed with shape (None, 3) for input 
KerasTensor(type_spec=TensorSpec(shape=(None, 3), dtype=tf.string, name='feature'), name='feature', description="created by layer 'feature'"), 
but it was called on an input with incompatible shape (3,).

The final step is to distribute the computation to remote workers using ClusterCoordinator.schedule:
    The schedule method enqueues a tf.function and returns a future-like RemoteValue immediately. The queued functions will be dispatched to remote workers in background threads and the RemoteValue will be filled asynchronously.
    The join method (ClusterCoordinator.join) can be used to wait until all scheduled functions are executed.

"""

num_epoches = 4
steps_per_epoch = 5
for i in range(num_epoches):
    accuracy.reset_states()
    for _ in range(steps_per_epoch):
        coordinator.schedule(step_fn, args=(per_worker_iterator,))
    # Wait at epoch boundaries.
    coordinator.join()
    print("Finished epoch %d, accuracy is %f." % (i, accuracy.result().numpy()))

"""
Here is how you can fetch the result of a RemoteValue:
"""

loss = coordinator.schedule(step_fn, args=(per_worker_iterator,))
print("Final loss is %f" % loss.fetch())

"""
Alternatively, you can launch all steps and do something while waiting for completion:
"""

# Not complete code:
# for _ in range(total_steps):
#     coordinator.schedule(step_fn, args=(per_worker_iterator,))
# while not coordinator.done():
#     time.sleep(10)
#     # Do something like logging metrics or writing checkpoints.

"""
For the complete training and serving workflow for this particular example, please check out this test.

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/keras/distribute


More about dataset creation
===========================

The dataset in the above code is created using the ClusterCoordinator.create_per_worker_dataset API). It creates one dataset per worker and returns a container object. You can call the iter method on it to create a per-worker iterator. The per-worker iterator contains one iterator per worker and the corresponding slice of a worker will be substituted in the input argument of the function passed to the ClusterCoordinator.schedule method before the function is executed on a particular worker.

Currently, the ClusterCoordinator.schedule method assumes workers are equivalent and thus assumes the datasets on different workers are the same except they may be shuffled differently if they contain a Dataset.shuffle operation. Because of this, it is also recommended that the datasets to be repeated indefinitely and you schedule a finite number of steps instead of relying on the OutOfRangeError from a dataset.

Another important note is that tf.data datasets don’t support implicit serialization and deserialization across task boundaries. So it is important to create the whole dataset inside the function passed to ClusterCoordinator.create_per_worker_dataset.


Evaluation
==========

There is more than one way to define and run an evaluation loop in distributed training. 
Each has its own pros and cons as described below. The inline evaluation method is recommended if you don't have a preference.

未完待续.. 还差一点点
"""
