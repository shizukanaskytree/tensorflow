# # 我突然觉得说可以把模型定义在不同的 worker 上
# # 借助这一套来完成训练

# # 参考的教程是:
# # https://colab.research.google.com/drive/1oFfSIAnOmfohHcDsWWK6s21TP-L33-1y#scrollTo=GxypEyIthR0z

# import os
# import tensorflow_datasets as tfds

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "2"

# # tensorflow/core/util/dump_graph.cc:134] Failed to dump after_grouping_2_139915407473008 because dump location is not  specified through either TF_DUMP_GRAPH_PREFIX environment variable or function argument.
# os.environ[
#     "TF_DUMP_GRAPH_PREFIX"
# ] = "/home/wxf/tf2/tensorflow/experiments/param_server/graph_dump"
# import tensorflow as tf

# # to log placement, eg: https://gist.github.com/shizukanaskytree/f8131342bc6475e1d92164f5da6819d9
# tf.debugging.set_log_device_placement(True)

# print(os.getpid())  # for gdb

# import debugpy

# debugpy.listen(5678)
# debugpy.wait_for_client()

# import multiprocessing
# import random
# import portpicker

# tf.debugging.set_log_device_placement(True)


# # Variables are created on parameter servers and they are read and updated by workers in each step.
# # tf.distribute.experimental.ParameterServerStrategy class distributes the training steps to
# # a cluster that scales up to thousands of workers (accompanied by parameter servers).

# # 在这个 cluster 里面有 5 个 server, coordinator 是自动被加入的, 我也不知道什么关系.
# def create_in_process_cluster(num_workers, num_ps):
#     """Creates and starts local servers and returns the cluster_resolver."""
#     worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
#     ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

#     # a 'cluster' is with several 'jobs', and each of the jobs may have one or more 'tasks'.
#     # When using parameter server training, it is recommended to have:
#     # One coordinator job (which has the job name chief)
#     # Multiple worker jobs (job name worker); and
#     # Multiple parameter server jobs (job name ps)

#     # While the coordinator creates resources, dispatches training tasks,
#     # writes checkpoints, and deals with task failures, workers and
#     # parameter servers run tf.distribute.Server that listen for requests
#     # from the coordinator.

#     cluster_dict = {}

#     # 在 ip 下构造 devices
#     # ip:port
#     # ip is localhost
#     # 这里的 cluste name 到最后直接和构造的 device name  相互挂钩.
#     # 有几个 server 你心里没有 B 数吗?
#     # server 代号是什么
#     cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
#     if num_ps > 0:
#         cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

#     cluster_spec = tf.train.ClusterSpec(cluster_dict)

#     # Workers need some inter_ops threads to work properly.
#     worker_config = tf.compat.v1.ConfigProto()
#     if multiprocessing.cpu_count() < num_workers + 1:
#         worker_config.inter_op_parallelism_threads = num_workers + 1

#     for i in range(num_workers):
#         tf.distribute.Server(
#             cluster_spec,
#             job_name="worker",
#             task_index=i,
#             config=worker_config,
#             protocol="grpc",
#         )

#     for i in range(num_ps):
#         tf.distribute.Server(cluster_spec, job_name="ps", task_index=i, protocol="grpc")

#     cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
#         cluster_spec, rpc_layer="grpc"
#     )
#     return cluster_resolver


# # Set the environment variable to allow reporting worker and ps failure to the
# # coordinator. This is a workaround and won't be necessary in the future.
# os.environ["GRPC_FAIL_FAST"] = "use_caller"

# NUM_WORKERS = 3
# NUM_PS = 2
# cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)

# # print(dir(cluster_resolver))
# # ['__abstractmethods__', '__class__', '__delattr__', '__dict__', '__dir__',
# # '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__',
# # '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__',
# # '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__',
# # '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl',
# # '_cluster_spec', '_environment', '_master', '_num_accelerators', '_rpc_layer',
# # '_task_id', '_task_type', '_tf_api_names', '_tf_api_names_v1', 'cluster_spec',
# # 'environment', 'master', 'num_accelerators', 'rpc_layer', 'task_id', 'task_type']

# print(cluster_resolver.cluster_spec())
# # ClusterSpec({'ps': ['localhost:16898', 'localhost:18235'],
# #              'worker': ['localhost:23511', 'localhost:23566', 'localhost:15823']})

# # Instantiate a ParameterServerStrategy
# # To enable variable sharding, you can pass in a variable_partitioner when
# # constructing a ParameterServerStrategy object. The variable_partitioner
# # will be invoked every time when a variable is created and it is expected
# # to return the number of shards along each dimension of the variable. Some
# # out-of-box variable_partitioners are provided such as tf.distribute.experimental.partitioners.MinSizePartitioner.
# # It is recommended to use size-based partitioners like tf.distribute.experimental.partitioners.MinSizePartitioner
# # to avoid partitioning small variables, which could have negative impact on model training speed.
# variable_partitioner = tf.distribute.experimental.partitioners.MinSizePartitioner(
#     min_shard_bytes=(256 << 10), max_shards=NUM_PS
# )

# # Parameter server training with the Model.fit API requires the coordinator to use a tf.distribute.experimental.ParameterServerStrategy object, and a tf.keras.utils.experimental.DatasetCreator as the input.
# strategy = tf.distribute.experimental.ParameterServerStrategy(
#     cluster_resolver, variable_partitioner=variable_partitioner
# )


# # Model.fit with parameter server training requires that the input data be
# # provided in a callable that takes a single argument of type
# # tf.distribute.InputContext, and returns a tf.data.Dataset.
# # The code in dataset_fn will be invoked on the input device, which is usually the CPU, on each of the worker machines.
# def dataset_fn(input_context):
#     global_batch_size = 64
#     batch_size = input_context.get_per_replica_batch_size(global_batch_size)

#     x = tf.random.uniform((10, 10))
#     y = tf.random.uniform((10,))

#     dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
#     dataset = dataset.shard(
#         input_context.num_input_pipelines, input_context.input_pipeline_id
#     )
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.prefetch(2)

#     return dataset


# # Then, create a tf.keras.utils.experimental.DatasetCreator object that
# # takes such callable, and an optional tf.distribute.InputOptions object
# # via input_options argument.
# dc = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

# with strategy.scope():
#     model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])

# model.compile(tf.keras.optimizers.SGD(), loss="mse", steps_per_execution=10)

# # Callbacks and training
# working_dir = "/tmp/my_working_dir"
# log_dir = os.path.join(working_dir, "log")
# ckpt_filepath = os.path.join(working_dir, "ckpt")
# backup_dir = os.path.join(working_dir, "backup")

# callbacks = [
#     tf.keras.callbacks.TensorBoard(log_dir=log_dir),
#     tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath),
#     tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_dir),
# ]

# model.fit(dc, epochs=1, steps_per_epoch=20, callbacks=callbacks)

# # ====

# # Model to run on servers
# (ds_train, ds_test), ds_info = tfds.load(
#     "mnist",
#     split=["train", "test"],
#     shuffle_files=True,
#     as_supervised=True,
#     with_info=True,
# )


# def normalize_img(image, label):
#     """Normalizes images: `uint8` -> `float32`."""
#     return tf.cast(image, tf.float32) / 255.0, label


# ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
# ds_train = ds_train.cache()
# ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
# ds_train = ds_train.batch(128)
# ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
# ds_test = ds_test.batch(128)
# ds_test = ds_test.cache()
# ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# model = tf.keras.models.Sequential(
#     [
#         tf.keras.layers.Flatten(input_shape=(28, 28)),
#         tf.keras.layers.Dense(128, activation="relu"),
#         tf.keras.layers.Dense(10),
#     ]
# )
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.001),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
# )

# model.fit(
#     ds_train,
#     epochs=6,
#     validation_data=ds_test,
# )
