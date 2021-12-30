import tensorflow as tf
tf.config.run_functions_eagerly(False)

# ------------------------------------------------
# Define your list of IP address / port number combos
IP_ADDRESS1 = '127.0.0.1'
PORT1 = '2222'

IP_ADDRESS2 = '127.0.0.1'
PORT2 = '2224'

# Define cluster
cluster_spec = tf.train.ClusterSpec({'worker' : [(IP_ADDRESS1 + ":" + PORT1), (IP_ADDRESS2 + ":" + PORT2)]})

# Task index (integer) should correspond to the IP address of the machine that you are running this notebook on...

# For example, if you are running this notebook on (IP_ADDRESS2 + ":" + PORT2), task_idx=1 because it is
# responsible for the second task of the job:worker based on how you defined cluster_spec above

# Define server for specific machine
task_idx = 0 # <--- This will be different for each non-chief machine you run this script on
server = tf.distribute.Server(cluster_spec, job_name='worker', task_index=task_idx)

# Check the server definition
print(f"server.server_def: {server.server_def}")
print(f"server.target: {server.target}")
# ------------------------------------------------

# Define devices that we wish to split our graph over
# device0='/job:worker/task:0'
device0 = "/job:worker/replica:0/task:0/device:GPU:0"
# device1='/job:worker/task:1'
device1 = "/job:worker/replica:0/task:1/device:GPU:0"
devices=(device0, device1)

# Reset graph
# tf.compat.v1.reset_default_graph()

sess = tf.compat.v1.Session(server.target)

# Create model
with tf.device(devices[0]):
  x = tf.random.uniform(shape=[5, 6])

# x = tf.constant(5.0, shape=[5, 6])
# w = tf.constant([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

# # physical_devices = tf.config.list_physical_devices('GPU')
# # print(f"physical_devices: {physical_devices}")

# with tf.device(devices[0]):
#   xw = tf.multiply(x, w)

# with tf.device(devices[1]):
#   max_in_rows = tf.reduce_max(xw, 1)


# with tf.compat.v1.Session(server.target) as sess:  # <----- IMPORTANT: Pass the server target to the session definition
#   out = sess.run(max_in_rows)
#   print(out)
