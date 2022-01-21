import tensorflow as tf

# Define your list of IP address / port number combos
# IP_ADDRESS1='127.0.0.1'
# PORT1='2222'

# IP_ADDRESS2='127.0.0.1'
# PORT2='2224'

IP_ADDRESS3='127.0.0.1'
PORT3='2226'

# Define cluster
cluster_spec = tf.train.ClusterSpec(
  {'worker' : [
    # (IP_ADDRESS1 + ":" + PORT1),
    # (IP_ADDRESS2 + ":" + PORT2)
    (IP_ADDRESS3 + ":" + PORT3)
  ]}
)

# Define server for specific machine
task_idx = 0
server = tf.distribute.Server(cluster_spec, job_name='worker', task_index=task_idx)




# Server will run as long as the notebook is running
# server.join()