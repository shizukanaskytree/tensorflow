import tensorflow as tf

# Define your list of IP address / port number combos
IP_ADDRESS1='127.0.0.1'
PORT1='2222'

IP_ADDRESS2='127.0.0.1'
PORT2='2224'

# Define cluster
cluster_spec = tf.train.ClusterSpec({'worker' : [(IP_ADDRESS1 + ":" + PORT1), (IP_ADDRESS2 + ":" + PORT2)]})

# Task index (integer) should correspond to the IP address of the machine that you are running this notebook on...

# For example, if you are running this notebook on (IP_ADDRESS2 + ":" + PORT2), task_idx=1 because it is 
# responsible for the second task of the job:worker based on how you defined cluster_spec above

# Define server for specific machine
task_idx = 1 # <--- This will be different for each non-chief machine you run this script on
server = tf.distribute.Server(cluster_spec, job_name='worker', task_index=task_idx)

# 2021-12-29 10:39:17.033329: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 2021-12-29 10:39:24.053722: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:worker/replica:0/task:0/device:GPU:0 with 29888 MB memory:  -> device: 0, name: Tesla V100-SXM2-32GB, pci bus id: 0000:61:00.0, compute capability: 7.0
# 2021-12-29 10:39:24.059357: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:worker/replica:0/task:0/device:GPU:1 with 29888 MB memory:  -> device: 1, name: Tesla V100-SXM2-32GB, pci bus id: 0000:62:00.0, compute capability: 7.0
# 2021-12-29 10:39:24.064652: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:worker/replica:0/task:0/device:GPU:2 with 29888 MB memory:  -> device: 2, name: Tesla V100-SXM2-32GB, pci bus id: 0000:89:00.0, compute capability: 7.0
# 2021-12-29 10:39:24.069728: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:worker/replica:0/task:0/device:GPU:3 with 29888 MB memory:  -> device: 3, name: Tesla V100-SXM2-32GB, pci bus id: 0000:8a:00.0, compute capability: 7.0
# 2021-12-29 10:39:24.103118: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:272] Initialize GrpcChannelCache for job worker -> {0 -> 127.0.0.1:2222, 1 -> 127.0.0.1:2224}
# 2021-12-29 10:39:24.119234: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:427] Started server with target: grpc://127.0.0.1:2222

# Server will run as long as the notebook is running
server.join()