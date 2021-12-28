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
task_idx=1 #<--- This will be different for each non-chief machine you run this script on
server = tf.train.Server(cluster_spec, job_name='worker', task_index=task_idx)

# Server will run as long as the notebook is running
server.join()