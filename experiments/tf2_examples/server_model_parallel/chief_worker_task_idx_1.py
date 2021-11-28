from  __future__  import division, print_function, absolute_import
import tensorflow as tf
import tensorflow.compat.v1 as tf1

import random
import numpy as np
from random import sample 
import time

# model op
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

tf.compat.v1.disable_eager_execution()

# Get data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

from tensorflow.keras.datasets import mnist

BATCH_SIZE = 128
num_classes = 10

# Read data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

tf.test.gpu_device_name()
# Some numbers
# batch_size = 128
# display_step = 10
# num_input = 784
# num_classes = 10

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#######################################################################
# model definition is finished.
#######################################################################
# 使用 tf2 重写模型
# code:
# https://github.com/AnkushMalaker/TF2-MNIST/blob/master/Mnist_Implimentation_on_TF2_Functional_API_Custom_Training.ipynb

# model definition
#       input layer
#       conv1 = conv1(input_layer)
#       ...
#       
input_layer = keras.Input(shape=(28, 28, 1)) 
# Convolution layer accepts input as (Batch_size, (x,y), channels). 
# We don't need to specify batch size. Channels are, forexample, colors, RGB. This is black and white, so its 1

ConvLayer1 = Conv2D(32, kernel_size=(3,3),activation='relu')(input_layer)

ConvLayer2 = Conv2D(64, (3,3), activation='relu')(ConvLayer1)

D1 = Dropout(0.25)(ConvLayer2)

Flatten_layer = Flatten()(D1)

Dense1 = Dense(128, activation='relu')(Flatten_layer)

D2 = Dropout(0.5)(Dense1)

Dense2 = Dense(10, activation='softmax')(D2)

# define the model
complete_model = keras.Model(inputs=input_layer, outputs = Dense2)

complete_model.summary()

#######################################################################
# model definition is finished.
#######################################################################

# plot the model but it's in the remote so I can't.
from tensorflow.keras.utils import plot_model
plot_model(complete_model, to_file='model.png')

# pack_batch is pack batch, pack 整理；把…打包
def pack_batch(features, targets, dataset_size, batch_size):
    x_batch = []
    y_batch = []
    
    for i in range(batch_size):
        index = random.randint(0, dataset_size-1)
        x_batch.append(features[index])
        y_batch.append(targets[index])
        
    return np.array(x_batch), np.array(y_batch)

x_batch, y_batch = pack_batch(x_train, y_train, 60000, BATCH_SIZE)
print("x_batch size: %s" %str(len(x_batch)))
print(x_batch.shape)


#############################################################
# optimizer
#############################################################
optimizer = tf.keras.optimizers.Adadelta()

def loss(model, inputs, targets):
    y_ = model(inputs)
    return keras.losses.categorical_crossentropy(targets, y_, from_logits=False, label_smoothing=0)

# Define gradients in the training
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

loss(complete_model, x_batch, y_batch)

# Used in the training.
def validate_accuracy(model, features, targets):
    epoch_validation_accuracy = tf.keras.metrics.CategoricalAccuracy()
    epoch_validation_accuracy.update_state(targets, model(features))
    return epoch_validation_accuracy.result()

train_loss_results = []
train_accuracy_results = []
validation_accuracy = []



# Set up cluster
# test in our localhost 
IP_ADDRESS1='192.168.0.19'
PORT1='2222'

IP_ADDRESS2='192.168.0.19' # we choose this IP:PORT as our worker server.
PORT2='2224'

# This line should match the same cluster definition in the Helper_Server.ipynb
cluster_spec = tf.train.ClusterSpec(
    {
        'worker': [
            (IP_ADDRESS1 + ":" + PORT1), 
            (IP_ADDRESS2 + ":" + PORT2)
        ]
    }
)


# task idx for this server is 1
# task idx for another server is 0
# We have chosen this machine to be our chief (The first IPaddress:Port combo), so task_idx=0
task_idx = 1

# !!!
# bear in mind: AttributeError: module 'tensorflow._api.v2.train' has no attribute 'Server'
server = tf1.train.Server(cluster_spec, job_name='worker', task_index=task_idx)

# Check the server definition
print(server.server_def)

# What is `server.target`?
print('===server.target:===')
print(server.target)
# print:
# grpc://192.168.0.19:2224




start_time = time.time()
epochs = 1
for epoch in range(epochs):
    # keras 封装了一个 epoch.
    
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
    
    # Training loop - using batches of 32
    x,y = pack_batch(x_train, y_train, 60000, BATCH_SIZE)
    
    # chain rule.
    loss_value, grads = grad(complete_model, x, y)
    
    # print(loss_value)
    if (epoch%100 == 0):        
        print("Running epoch %d, %d epochs left" %(epoch, epochs-epoch))
    
    # 
    optimizer.apply_gradients(zip(grads, complete_model.trainable_variables))
    
    x_validation, y_validation = pack_batch(x_test, y_test, 10000, 50)
    
    validation_accuracy.append(validate_accuracy(complete_model,x_validation,y_validation))
    
    # Track progress
    # Add current batch loss
    epoch_loss_avg.update_state(loss_value)
    
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    epoch_accuracy.update_state(y, complete_model(x, training=True))

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

print("Time taken: %d" %(time.time() - start_time))
# The end of the training model.


# def conv_layer(inputs, channels_in, channels_out, strides=1):
#     # Create variables
#     w=tf.Variable(tf1.random_normal([3, 3, channels_in, channels_out]))
#     b=tf.Variable(tf1.random_normal([channels_out]))
    
#     # We can double check the device that this variable was placed on
#     print(w.device) 
#     print(b.device)
    
#     # Define Ops
#     x = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding='SAME')
#     x = tf.nn.bias_add(x, b)
    
#     # Non-linear activation
#     return tf.nn.relu(x)

    
# def maxpool2d(x, k=2):
#     return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# # Create model
# def CNN(x, devices):
    
#     with tf.device(devices[0]): # <----------- Put first half of network on device 0

#         x = tf.reshape(x, shape=[-1, 28, 28, 1])

#         # Convolution Layer
#         conv1=conv_layer(x, 1, 32, strides=1)
#         pool1=maxpool2d(conv1)

#         # Convolution Layer
#         conv2=conv_layer(pool1, 32, 64, strides=1)
#         pool2=maxpool2d(conv2)

#     with tf.device(devices[1]):  # <----------- Put second half of network on device 1
#         # Fully connected layer
#         fc1 = tf.reshape(pool2, [-1, 7*7*64])
#         w1=tf.Variable(tf.random_normal([7*7*64, 1024]))
#         b1=tf.Variable(tf.random_normal([1024]))
#         fc1 = tf.add(tf.matmul(fc1,w1),b1)
#         fc1=tf.nn.relu(fc1)

#         # Output layer
#         w2=tf.Variable(tf.random_normal([1024, num_classes]))
#         b2=tf.Variable(tf.random_normal([num_classes]))
#         out = tf.add(tf.matmul(fc1,w2),b2)
        
#         # Check devices for good measure
#         print(w1.device)
#         print(b1.device)
#         print(w2.device)
#         print(b2.device)

#     return out


#############################################################
# start distributed training.
#############################################################

# Define devices that we wish to split our graph over
device0 = '/job:worker/task:0'
device1 = '/job:worker/task:1'

devices=(device0, device1)

tf1.reset_default_graph() # Reset graph

# # Construct model
# with tf.device(devices[0]):
#     X = tf1.placeholder(tf.float32, [None, 28,28,1]) # Input images feedable
#     # X = tf.placeholder(tf.float32, [None, num_input]) # Input images feedable
#     Y = tf1.placeholder(tf.float32, [None, num_classes]) # Ground truth feedable

# # logits = CNN(X, devices) # Unscaled probabilities

# with tf.device(devices[1]):
    
#     prediction = tf.nn.softmax(logits) # Class-wise probabilities
    
#     # Define loss and optimizer
#     loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
#     optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
#     train_op = optimizer.minimize(loss_op)

#     # Evaluate model
#     correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#     init = tf.global_variables_initializer()







# # deprecate it and we do not have it.
# - with tf.Session(server.target) as sess:
#   - this is the secret.
#   -  



# # Start training
# with tf.Session(server.target) as sess:  # <----- IMPORTANT: Pass the server target to the session definition

#     # Run the initializer
#     sess.run(init)

#     for step in range(100):
#         # batch_x, batch_y = mnist.train.next_batch(batch_size)
        
#         # Run optimization op (backprop)
#         # (x_train, y_train), (x_test, y_test) = mnist.load_data() # from data loader
#         sess.run(train_op, feed_dict={X: x_train, Y: y_train})
        
#         # if step % display_step == 0 or step == 1:
#             # Calculate batch loss and accuracy
#             # loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y : batch_y})
#             # print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

#     # Get test set accuracy
#     print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images[:256],Y: mnist.test.labels[:256]}))
