# 我突然觉得说可以把模型定义在不同的 worker 上
# 借助这一套来完成训练

# 参考的教程是:
# https://colab.research.google.com/drive/1oFfSIAnOmfohHcDsWWK6s21TP-L33-1y#scrollTo=GxypEyIthR0z

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "2"

# tensorflow/core/util/dump_graph.cc:134] Failed to dump after_grouping_2_139915407473008 because dump location is not  specified through either TF_DUMP_GRAPH_PREFIX environment variable or function argument.
os.environ["TF_DUMP_GRAPH_PREFIX"] = "/home/wxf/tf2/tensorflow/experiments/param_server/graph_dump"
import tensorflow as tf

# to log placement, eg: https://gist.github.com/shizukanaskytree/f8131342bc6475e1d92164f5da6819d9
tf.debugging.set_log_device_placement(True)

print(os.getpid())  # for gdb

import debugpy

debugpy.listen(5678)
debugpy.wait_for_client()

import multiprocessing
import random
import portpicker

tf.debugging.set_log_device_placement(True)


# Variables are created on parameter servers and they are read and updated by workers in each step.
# tf.distribute.experimental.ParameterServerStrategy class distributes the training steps to 
# a cluster that scales up to thousands of workers (accompanied by parameter servers).

# 在这个 cluster 里面有 5 个 server, coordinator 是自动被加入的, 我也不知道什么关系.
def create_in_process_cluster(num_workers, num_ps):
    """Creates and starts local servers and returns the cluster_resolver."""
    worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
    ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

    # a 'cluster' is with several 'jobs', and each of the jobs may have one or more 'tasks'.
    # When using parameter server training, it is recommended to have:
    # One coordinator job (which has the job name chief)
    # Multiple worker jobs (job name worker); and
    # Multiple parameter server jobs (job name ps)

    # While the coordinator creates resources, dispatches training tasks,
    # writes checkpoints, and deals with task failures, workers and
    # parameter servers run tf.distribute.Server that listen for requests
    # from the coordinator.

    cluster_dict = {}
    cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
    if num_ps > 0:
        cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

    cluster_spec = tf.train.ClusterSpec(cluster_dict)

    # Workers need some inter_ops threads to work properly.
    worker_config = tf.compat.v1.ConfigProto()
    if multiprocessing.cpu_count() < num_workers + 1:
        worker_config.inter_op_parallelism_threads = num_workers + 1

    for i in range(num_workers):
        tf.distribute.Server(
            cluster_spec,
            job_name="worker",
            task_index=i,
            config=worker_config,
            protocol="grpc",
        )

    for i in range(num_ps):
        tf.distribute.Server(cluster_spec, job_name="ps", task_index=i, protocol="grpc")

    cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec, rpc_layer="grpc"
    )
    return cluster_resolver


# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

NUM_WORKERS = 3
NUM_PS = 2
cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)

# print(dir(cluster_resolver))
# ['__abstractmethods__', '__class__', '__delattr__', '__dict__', '__dir__',
# '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__',
# '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__',
# '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__',
# '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl',
# '_cluster_spec', '_environment', '_master', '_num_accelerators', '_rpc_layer',
# '_task_id', '_task_type', '_tf_api_names', '_tf_api_names_v1', 'cluster_spec',
# 'environment', 'master', 'num_accelerators', 'rpc_layer', 'task_id', 'task_type']

print(cluster_resolver.cluster_spec())
# ClusterSpec({'ps': ['localhost:16898', 'localhost:18235'],
#              'worker': ['localhost:23511', 'localhost:23566', 'localhost:15823']})

# Instantiate a ParameterServerStrategy
# To enable variable sharding, you can pass in a variable_partitioner when
# constructing a ParameterServerStrategy object. The variable_partitioner
# will be invoked every time when a variable is created and it is expected
# to return the number of shards along each dimension of the variable. Some
# out-of-box variable_partitioners are provided such as tf.distribute.experimental.partitioners.MinSizePartitioner.
# It is recommended to use size-based partitioners like tf.distribute.experimental.partitioners.MinSizePartitioner
# to avoid partitioning small variables, which could have negative impact on model training speed.
variable_partitioner = tf.distribute.experimental.partitioners.MinSizePartitioner(
    min_shard_bytes=(256 << 10), max_shards=NUM_PS
)

# Parameter server training with the Model.fit API requires the coordinator to use a tf.distribute.experimental.ParameterServerStrategy object, and a tf.keras.utils.experimental.DatasetCreator as the input.
strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver, variable_partitioner=variable_partitioner
)


# Model.fit with parameter server training requires that the input data be
# provided in a callable that takes a single argument of type
# tf.distribute.InputContext, and returns a tf.data.Dataset.
# The code in dataset_fn will be invoked on the input device, which is usually the CPU, on each of the worker machines.
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


# Then, create a tf.keras.utils.experimental.DatasetCreator object that
# takes such callable, and an optional tf.distribute.InputOptions object
# via input_options argument.
dc = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

with strategy.scope():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])

model.compile(tf.keras.optimizers.SGD(), loss="mse", steps_per_execution=10)

# Callbacks and training
working_dir = "/tmp/my_working_dir"
log_dir = os.path.join(working_dir, "log")
ckpt_filepath = os.path.join(working_dir, "ckpt")
backup_dir = os.path.join(working_dir, "backup")

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath),
    tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_dir),
]

model.fit(dc, epochs=1, steps_per_epoch=20, callbacks=callbacks)

# =======================================================================
# This is the end of Model.fit tutorial, next is the "Training with a custom training loop"
# =======================================================================

# So, I comment all the code below.

# feature_vocab = [
#     "avenger",
#     "ironman",
#     "batman",
#     "hulk",
#     "spiderman",
#     "kingkong",
#     "wonder_woman",
# ]
# label_vocab = ["yes", "no"]

# with strategy.scope():
#     feature_lookup_layer = tf.keras.layers.StringLookup(
#         vocabulary=feature_vocab, mask_token=None
#     )
#     label_lookup_layer = tf.keras.layers.StringLookup(
#         vocabulary=label_vocab, num_oov_indices=0, mask_token=None
#     )

#     raw_feature_input = tf.keras.layers.Input(
#         shape=(3,), dtype=tf.string, name="feature"
#     )
#     feature_id_input = feature_lookup_layer(raw_feature_input)
#     feature_preprocess_stage = tf.keras.Model(
#         {"features": raw_feature_input}, feature_id_input
#     )

#     raw_label_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="label")
#     label_id_input = label_lookup_layer(raw_label_input)

#     label_preprocess_stage = tf.keras.Model({"label": raw_label_input}, label_id_input)


# def feature_and_label_gen(num_examples=200):
#     examples = {"features": [], "label": []}
#     for _ in range(num_examples):
#         features = random.sample(feature_vocab, 3)
#         label = ["yes"] if "avenger" in features else ["no"]
#         examples["features"].append(features)
#         examples["label"].append(label)
#     return examples


# examples = feature_and_label_gen()


# def dataset_fn(_):
#     raw_dataset = tf.data.Dataset.from_tensor_slices(examples)

#     train_dataset = (
#         raw_dataset.map(
#             lambda x: (
#                 {"features": feature_preprocess_stage(x["features"])},
#                 label_preprocess_stage(x["label"]),
#             )
#         )
#         .shuffle(200)
#         .batch(32)
#         .repeat()
#     )
#     return train_dataset


# # These variables created under the `strategy.scope` will be placed on parameter
# # servers in a round-robin fashion.
# with strategy.scope():
#     # Create the model. The input needs to be compatible with Keras processing layers.
#     model_input = tf.keras.layers.Input(shape=(3,), dtype=tf.int64, name="model_input")

#     emb_layer = tf.keras.layers.Embedding(
#         input_dim=len(feature_lookup_layer.get_vocabulary()), output_dim=16384
#     )
#     emb_output = tf.reduce_mean(emb_layer(model_input), axis=1)
#     dense_output = tf.keras.layers.Dense(units=1, activation="sigmoid")(emb_output)
#     model = tf.keras.Model({"features": model_input}, dense_output)

#     optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)
#     accuracy = tf.keras.metrics.Accuracy()

# assert len(emb_layer.weights) == 2
# assert emb_layer.weights[0].shape == (4, 16384)
# assert emb_layer.weights[1].shape == (4, 16384)
# assert emb_layer.weights[0].device == "/job:ps/replica:0/task:0/device:CPU:0"
# assert emb_layer.weights[1].device == "/job:ps/replica:0/task:1/device:CPU:0"

# # Third, create the training step wrapped into a tf.function:
# @tf.function
# def step_fn(iterator):
#     def replica_fn(batch_data, labels):
#         with tf.GradientTape() as tape:
#             pred = model(batch_data, training=True)
#             per_example_loss = tf.keras.losses.BinaryCrossentropy(
#                 reduction=tf.keras.losses.Reduction.NONE
#             )(labels, pred)
#             loss = tf.nn.compute_average_loss(per_example_loss)
#             gradients = tape.gradient(loss, model.trainable_variables)

#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#         actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
#         accuracy.update_state(labels, actual_pred)
#         return loss

#     batch_data, labels = next(iterator)
#     losses = strategy.run(replica_fn, args=(batch_data, labels))
#     return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)


# # With custom training loops, the tf.distribute.experimental.coordinator.ClusterCoordinator class is the key component used for the coordinator.
# # The ClusterCoordinator object then dispatches the execution of these training steps to remote workers.
# # For parameter server training, the ClusterCoordinator needs to work with a tf.distribute.experimental.ParameterServerStrategy.
# # The schedule API enqueues a tf.function and returns a future-like RemoteValue immediately.
# # The queued functions will be dispatched to remote workers in background threads and their RemoteValues will be filled asynchronously.
# # Since schedule doesn’t require worker assignment, the tf.function passed in can be executed on any available worker.
# # If the worker it is executed on becomes unavailable before its completion, the function will be retried on another available worker.
# # Because of this fact and the fact that function execution is not atomic, a function may be executed more than once.
# # In addition to dispatching remote functions, the ClusterCoordinator also helps to create datasets on all the workers and rebuild these datasets when a worker recovers from failure.

# # Even if you choose the Model.fit training path, you can optionally instantiate a
# # tf.distribute.experimental.coordinator.ClusterCoordinator object to schedule other
# # functions you would like to be executed on the workers.
# coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)

# # The final step is to distribute the computation to remote workers using ClusterCoordinator.schedule:
# @tf.function
# def per_worker_dataset_fn():
#     return strategy.distribute_datasets_from_function(dataset_fn)


# # More about dataset creation
# # The dataset in the above code is created using the ClusterCoordinator.create_per_worker_dataset API). It creates one dataset per worker and returns a container object. You can call the iter method on it to create a per-worker iterator. The per-worker iterator contains one iterator per worker and the corresponding slice of a worker will be substituted in the input argument of the function passed to the ClusterCoordinator.schedule method before the function is executed on a particular worker.
# # Currently, the ClusterCoordinator.schedule method assumes workers are equivalent and thus assumes the datasets on different workers are the same except they may be shuffled differently if they contain a Dataset.shuffle operation. Because of this, it is also recommended that the datasets to be repeated indefinitely and you schedule a finite number of steps instead of relying on the OutOfRangeError from a dataset.
# # Another important note is that tf.data datasets don’t support implicit serialization and deserialization across task boundaries. So it is important to create the whole dataset inside the function passed to ClusterCoordinator.create_per_worker_dataset.
# per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
# per_worker_iterator = iter(per_worker_dataset)

# num_epoches = 4
# steps_per_epoch = 5
# for i in range(num_epoches):
#     accuracy.reset_states()
#     for _ in range(steps_per_epoch):
#         # The schedule method enqueues a tf.function and returns a future-like RemoteValue immediately.
#         # The queued functions will be dispatched to remote workers in background threads and
#         # the RemoteValue will be filled asynchronously.
#         coordinator.schedule(step_fn, args=(per_worker_iterator,))

#     # The join method (ClusterCoordinator.join) can be used to wait until all scheduled functions are executed.
#     # Wait at epoch boundaries.
#     coordinator.join()
#     print("Finished epoch %d, accuracy is %f." % (i, accuracy.result().numpy()))

# # Here is how you can fetch the result of a RemoteValue:
# loss = coordinator.schedule(step_fn, args=(per_worker_iterator,))
# print("Final loss is %f" % loss.fetch())

# eval_dataset = (
#     tf.data.Dataset.from_tensor_slices(feature_and_label_gen(num_examples=16))
#     .map(
#         lambda x: (
#             {"features": feature_preprocess_stage(x["features"])},
#             label_preprocess_stage(x["label"]),
#         )
#     )
#     .batch(8)
# )

# eval_accuracy = tf.keras.metrics.Accuracy()

# for batch_data, labels in eval_dataset:
#     pred = model(batch_data, training=False)
#     actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
#     eval_accuracy.update_state(labels, actual_pred)

# print("Evaluation accuracy: %f" % eval_accuracy.result())

# with strategy.scope():
#     # Define the eval metric on parameter servers.
#     eval_accuracy = tf.keras.metrics.Accuracy()


# @tf.function
# def eval_step(iterator):
#     def replica_fn(batch_data, labels):
#         pred = model(batch_data, training=False)
#         actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
#         eval_accuracy.update_state(labels, actual_pred)

#     batch_data, labels = next(iterator)
#     strategy.run(replica_fn, args=(batch_data, labels))


# def eval_dataset_fn():
#     return (
#         tf.data.Dataset.from_tensor_slices(feature_and_label_gen(num_examples=16))
#         .map(
#             lambda x: (
#                 {"features": feature_preprocess_stage(x["features"])},
#                 label_preprocess_stage(x["label"]),
#             )
#         )
#         .shuffle(16)
#         .repeat()
#         .batch(8)
#     )


# per_worker_eval_dataset = coordinator.create_per_worker_dataset(eval_dataset_fn)
# per_worker_eval_iterator = iter(per_worker_eval_dataset)

# eval_steps_per_epoch = 2
# for _ in range(eval_steps_per_epoch):
#     coordinator.schedule(eval_step, args=(per_worker_eval_iterator,))
# coordinator.join()
# print("Evaluation accuracy: %f" % eval_accuracy.result())
