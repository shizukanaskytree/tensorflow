# https://www.tensorflow.org/datasets/keras_example

import tensorflow as tf
import tensorflow_datasets as tfds
import portpicker

import debugpy

debugpy.listen(5678)
debugpy.wait_for_client()

# create two servers
# tf.distribute.Server
# 为什么构造一个 server 还有知道其他人呢?

# 初始的 cluster 是 ?
cluster_dict = {}
num_workers = 1
worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]

# init cluster
cluster_spec = tf.train.ClusterSpec(cluster_dict)

# 4 workers

# 如何才能构建这些东西呢?
# 我就只想构建一下 server 怎么就这么难呢?

server = tf.distribute.Server(
    cluster_spec,
    # job_name="worker",
    # task_index=i,
    # config=worker_config, # later todo
    protocol="grpc",
)
# Created device /job:worker/replica:0/task:0/device:GPU:3
# 这些可以得到的, 这样的话, 我就有了所有的 device 信息.


print(server.target)


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
