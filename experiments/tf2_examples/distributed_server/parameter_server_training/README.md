# README

Regardless of the API of choice (Model.fit or a custom training loop), distributed training in TensorFlow 2 involves: a 'cluster' with several 'jobs', and each of the jobs may have one or more 'tasks'.

You will start by creating several TensorFlow servers in advance and connect to them later. Note that this is only for the purpose of this tutorial's demonstration, and in real training the servers will be started on "worker" and "ps" machines.
from: <https://www.tensorflow.org/tutorials/distribute/parameter_server_training>

In real-world distributed training, instead of starting all the tf.distribute.Servers on the coordinator, you will be using multiple machines, and the ones that are designated as "worker"s and "ps" (parameter servers) will each run a tf.distribute.Server. Refer to Clusters in the real world section in the Parameter server training tutorial for more details.
from: <https://www.tensorflow.org/guide/migrate/multi_worker_cpu_gpu_training>

The in-process cluster setup is frequently used in unit testing, such as here.
    <https://github.com/tensorflow/tensorflow/blob/7621d31921c2ed979f212da066631ddfda37adf5/tensorflow/python/distribute/coordinator/cluster_coordinator_test.py#L437>

Another option for local testing is to launch processes on the local machine—check out Multi-worker training with Keras for an example of this approach.
    <https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras>

Note:
<https://www.notion.so/xiaofengwu/e3c1269476e849978902f12fd26ebd66>

代码和文档一致化, 像素级学习.