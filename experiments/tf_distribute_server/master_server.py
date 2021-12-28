import tensorflow as tf

# Create model
x = tf.constant(5.0, shape=[5, 6])
w = tf.constant([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
xw = tf.multiply(x, w)
max_in_rows = tf.reduce_max(xw, 1)

sess = tf.Session()
print (sess.run(xw))
# ==> [[0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]]

print (sess.run(max_in_rows))
# ==> [25.0, 25.0, 25.0, 25.0, 25.0]


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



