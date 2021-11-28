import tensorflow as tf
from typing import Tuple

# logger = logging.get_logger(__name__) # todo


def _setup_strategy() -> Tuple["tf.distribute.Strategy", int]:
    # self 在这里是原来 /home/wxf/anaconda3/envs/hm/lib/python3.8/site-packages/transformers/training_args_tf.py
    # 内的 `TFTrainingArguments`. 这里我觉得可以删除不用.

    # logger.info("Tensorflow: setting up strategy") # todo

    # if self.xla:
    #     tf.config.optimizer.set_jit(True)

    gpus = tf.config.list_physical_devices("GPU")

    strategy = tf.distribute.MirroredStrategy()

    return strategy

    # Set to float16 at first
    # if self.fp16:
    #     policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    #     tf.keras.mixed_precision.experimental.set_policy(policy)

    # if self.no_cuda:
    #     strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    # else:
    #     try:
    #         if self.tpu_name:
    #             tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
    #                 self.tpu_name, zone=self.tpu_zone, project=self.gcp_project
    #             )
    #         else:
    #             tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    #     except ValueError:
    #         if self.tpu_name:
    #             raise RuntimeError(f"Couldn't connect to TPU {self.tpu_name}!")
    #         else:
    #             tpu = None

    #     if tpu:
    #         # Set to bfloat16 in case of TPU
    #         if self.fp16:
    #             policy = tf.keras.mixed_precision.experimental.Policy("mixed_bfloat16")
    #             tf.keras.mixed_precision.experimental.set_policy(policy)

    #         tf.config.experimental_connect_to_cluster(tpu)
    #         tf.tpu.experimental.initialize_tpu_system(tpu)

    #         strategy = tf.distribute.TPUStrategy(tpu)

    #     elif len(gpus) == 0:
    #         strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    #     elif len(gpus) == 1:
    #         strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    #     elif len(gpus) > 1:
    #         # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
    #         strategy = tf.distribute.MirroredStrategy()
    #     else:
    #         raise ValueError(
    #             "Cannot find the proper strategy, please check your environment properties."
    #         )

    # return strategy
