import tensorflow as tf

import tensorflow.compat.v1 as tf1

# Define your list of IP address / port number combos
# IP_ADDRESS1='ex1.ex1.ex1.ex1'
# PORT1='2222'
# IP_ADDRESS2='ex2.ex2.ex2.ex2'
# PORT2='2224'

IP_ADDRESS1='192.168.0.19' # test localhost 
PORT1='2222'
IP_ADDRESS2='192.168.0.19'
PORT2='2224'

# Define cluster
cluster_spec = tf1.train.ClusterSpec(
    {
        'worker' : 
            [
                (IP_ADDRESS1 + ":" + PORT1), 
                (IP_ADDRESS2 + ":" + PORT2)
            ]
    }
)

# Task index (integer) should correspond to the IP address of the machine 
# that you are running this notebook on...

# For example, if you are running this notebook on 
# (IP_ADDRESS2 + ":" + PORT2), task_idx=1 because it is 
# responsible for the second task of the job:worker based on 
# how you defined cluster_spec above

# Define server for specific machine
task_idx = 1 # <--- This will be different for each non-chief machine you run this script on
server = tf1.train.Server(cluster_spec, job_name='worker', task_index=task_idx)

# Server will run as long as the notebook is running
server.join()
