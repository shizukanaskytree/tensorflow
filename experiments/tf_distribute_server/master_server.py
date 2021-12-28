import tensorflow as tf

# Create model

# Define devices that we wish to split our graph over
device0='/job:worker/task:0'
device1='/job:worker/task:1'
devices=(device0, device1)

tf.reset_default_graph() # Reset graph

# Set up cluster
IP_ADDRESS1='127.0.0.1'
PORT1='2222'
IP_ADDRESS2='127.0.0.1'
PORT2='2224'

# This line should match the same cluster definition in the Helper_Server.ipynb
cluster_spec = tf.train.ClusterSpec({'worker' : [(IP_ADDRESS1 + ":" + PORT1), (IP_ADDRESS2 + ":" + PORT2)]})

task_idx=0 # We have chosen this machine to be our chief (The first IPaddress:Port combo), so task_idx=0
server = tf.train.Server(cluster_spec, job_name='worker', task_index=task_idx)

# Check the server definition
print("server.server_def: ", server.server_def)


with tf.Session(server.target) as sess:  # <----- IMPORTANT: Pass the server target to the session definition
  ...



