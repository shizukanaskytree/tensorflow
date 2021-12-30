# Test the model

import tensorflow as tf

# Create model
x = tf.constant(5.0, shape=[5, 6])
w = tf.constant([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
xw = tf.multiply(x, w)

max_in_rows = tf.reduce_max(xw, 1)

# sess = tf.Session()
print (xw)
# ==> [[0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]]

print (max_in_rows)
# ==> [25.0, 25.0, 25.0, 25.0, 25.0]
