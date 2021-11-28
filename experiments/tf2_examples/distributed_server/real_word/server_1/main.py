"""
The 2nd server in the cluster.
"""

import json
import os
import socket
import portpicker

import tensorflow as tf

# ip_server_0 = socket.gethostbyname(socket.gethostname())  # "192.168.1.162"
# port_server_0 = portpicker.pick_unused_port()

ip_server_0 = "192.168.1.162"
port_server_0 = "22224"
addr_0 = ip_server_0 + ":" + port_server_0

ip_server_1 = "192.168.1.160"
port_server_1 = "22226"
addr_1 = ip_server_1 + ":" + port_server_1

# Step 1.
# Define the role of this server in this cluseter.
# There should be no "ps" job except when using tf.distribute.experimental.ParameterServerStrategy.
# from: https://www.tensorflow.org/guide/distributed_training#TF_CONFIG
os.environ["TF_CONFIG"] = json.dumps(
    {
        "cluster": {
            "worker": [addr_0, addr_1],
        },
        "task": {"type": "worker", "index": 0},
    }
)


# Step 2.
# get attributes from the config setting
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()


# if cluster_resolver.task_type in ("worker"):
#     # Start a TensorFlow server and wait.
#     print("worker")
# elif cluster_resolver.task_type == "evaluator":
#     # Run side-car evaluation
#     print("")
# else:
#     # Run the coordinator.
#     print("")
