import tensorflow as tf

# paper section:
# 如何构造一个动态的cluster?


# ------------------------
# 有了 server 然后呢?
# 也就是对应了一堆的 device 的代号.
# 也就是只能在 ps 下验证存在.
# 我以前的想法其实只是说使用 tf 这边的图形切割
# 然后把 runtime 搞搞就算了, 没想到要碰这么多边边角角的东西.

# 然后我想着用最朴素的方法把它搞出来
# 虽然这个和原来的不符合.
# 好难搞.
# 懒惰
# 不做完, 也不做好.


cluster_dict = {}
cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]

# if num_ps > 0:
# cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

# Don't use this. use what is next.
# cluster_spec = tf.train.ClusterSpec(cluster_dict)

# template of resolve the server in this cluster.

# https://www.tensorflow.org/api_docs/python/tf/distribute/cluster_resolver/TFConfigClusterResolver
# 这个部分也要改.

# # Dump the cluster information to `'TF_CONFIG'`.
# tf_config = {
#     'cluster': {
#         'chief': ["localhost:%s" % chief_port],
#         'worker': ["localhost:%s" % port for port in worker_ports],
#         'ps':  ["localhost:%s" % port for port in ps_ports],
#     },
#     'task': {'type': 'chief', 'index': 0}
# }
# os.environ['TF_CONFIG'] = json.dumps(tf_config)

# # Use a cluster resolver to bridge the information to the strategy created below.
# cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

# tf.distribute.Server(
#     cluster_resolver.cluster_spec(),
#     job_name="worker",
#     task_index=i,
#     config=worker_config,
# )


# tf.distribute.Server(
#     cluster_spec, job_name="worker", task_index=i, config=worker_config, protocol="grpc"
# )

# cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
#     cluster_spec, rpc_layer="grpc"
# )
