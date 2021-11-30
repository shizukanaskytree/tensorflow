import tensorflow as tf
import multiprocessing
import os
import json

# import portpicker

# # 在这个 cluster 里面有 N 个 server, 信息是由 DHT 维护的, 获取的.
# # coordinator 是自动被加入的, 我也不知道什么关系.
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
#     cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
#     if num_ps > 0:
#         cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

#     cluster_spec = tf.train.ClusterSpec(cluster_dict)

#     # Workers need some inter_ops threads to work properly.
#     worker_config = tf.compat.v1.ConfigProto()
#     if multiprocessing.cpu_count() < num_workers + 1:
#         # 因为 tf.compat.v1.ConfigProto() 里面 inter_op_parallelism_threads 默认最大值是 cpu_count
#         worker_config.inter_op_parallelism_threads = num_workers + 1

#     # 构造 worker servers
#     for i in range(num_workers):
#         tf.distribute.Server(
#             cluster_spec,
#             job_name="worker",
#             task_index=i,
#             config=worker_config,
#             protocol="grpc",
#         )

#     # 构造 ps servers
#     for i in range(num_ps):
#         tf.distribute.Server(cluster_spec, job_name="ps", task_index=i, protocol="grpc")

#     # 明确 cluster 情况.
#     cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
#         cluster_spec, rpc_layer="grpc"
#     )

#     return cluster_resolver


# https://www.tensorflow.org/guide/migrate/multi_worker_cpu_gpu_training

# Find ports that are available for the `'chief'` (the coordinator),
# `'worker'`s, and `'ps'` (parameter servers).
import portpicker

# 1 - chief
# 3 - worker
# 2 - ps
chief_port = portpicker.pick_unused_port()
worker_ports = [portpicker.pick_unused_port() for _ in range(3)]
ps_ports = [portpicker.pick_unused_port() for _ in range(2)]

# Dump the cluster information to `'TF_CONFIG'`.
tf_config = {
    # tell me your group
    "cluster": {
        "chief": ["localhost:%s" % chief_port],
        "worker": ["localhost:%s" % port for port in worker_ports],
        "ps": ["localhost:%s" % port for port in ps_ports],
    },
    # tell me your name
    "task": {"type": "chief", "index": 0},
}
os.environ["TF_CONFIG"] = json.dumps(tf_config)

# Use a cluster resolver to bridge the information to the strategy created below.
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()


# Workers need some inter_ops threads to work properly.
# This is only needed for this notebook to demo. Real servers
# should not need this.
worker_config = tf.compat.v1.ConfigProto()
worker_config.inter_op_parallelism_threads = 4


# 这里有硬编码的部分.

for i in range(3):
    tf.distribute.Server(
        cluster_resolver.cluster_spec(),
        job_name="worker",
        task_index=i,
        config=worker_config,
    )

for i in range(2):
    tf.distribute.Server(cluster_resolver.cluster_spec(), job_name="ps", task_index=i)

strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)


features = [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]
labels = [[0.3], [0.5], [0.7]]
eval_features = [[4.0, 4.5], [5.0, 5.5], [6.0, 6.5]]
eval_labels = [[0.8], [0.9], [1.0]]

dataset = (
    tf.data.Dataset.from_tensor_slices((features, labels))
    .shuffle(10)
    .repeat()
    .batch(64)
)

eval_dataset = (
    tf.data.Dataset.from_tensor_slices((eval_features, eval_labels)).repeat().batch(1)
)

with strategy.scope():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.05)
    model.compile(optimizer, "mse")

# train
model.fit(dataset, epochs=5, steps_per_epoch=10)

# eval
model.evaluate(eval_dataset, steps=10, return_dict=True)
