from __future__ import print_function
import os
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'

import tensorflow as tf
import numpy as np
import threading
import logging
#tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )


from tensorflow.examples.tutorials.mnist import input_data

def graph1_inference():
    np.random.seed(0)
    x_data = np.random.randn(5,10)
    w_data = np.random.randn(10,1)

    #import tensorflow as tf
    g_1 = tf.Graph()
    with g_1.as_default():
        x = tf.placeholder(tf.float32,shape=(5,10))
        w = tf.placeholder(tf.float32,shape=(10,1))
        #b = tf.fill((5,1),-1.)
        xw = tf.matmul(x,w)

        #xwb=xw+b
        xwb=xw
        s = tf.reduce_max(xwb)

        with tf.Session(graph=g_1) as sess:
            #outs_xwb = sess.run(xwb,feed_dict={x: x_data,w: w_data})
            outs = sess.run(s,feed_dict={x: x_data,w: w_data})
            print("outs of g1 = {}".format(outs))


def graph2_training():
    """ Neural Network.
    A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
    implementation with TensorFlow. This example is using the MNIST database
    of handwritten digits (http://yann.lecun.com/exdb/mnist/).
    Links:
        [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
    Author: Aymeric Damien
    Project: https://github.com/aymericdamien/TensorFlow-Examples/
    """


    # Import MNIST data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    import tensorflow as tf

    # Parameters
    learning_rate = 0.1
    num_steps = 500
    batch_size = 128
    display_step = 100

    # Network Parameters
    n_hidden_1 = 256  # 1st layer number of neurons
    n_hidden_2 = 256  # 2nd layer number of neurons
    num_input = 784  # MNIST data input (img shape: 28*28)
    num_classes = 10  # MNIST total classes (0-9 digits)

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # Create model
    def neural_net(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    # Construct model
    logits = neural_net(X)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for step in range(1, num_steps + 1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("G2 Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        print("G2 Optimization Finished!")

        # Calculate accuracy for MNIST test images
        print("G2 Testing Accuracy:", \
              sess.run(accuracy, feed_dict={X: mnist.test.images,
                                            Y: mnist.test.labels}))

num_infer_threads = 1000
infer_workers = []
for i in range(num_infer_threads):
    infer_workers.append(threading.Thread(name='infer worker {}'.format(num_infer_threads), target=graph1_inference))

for i in range(num_infer_threads):
    infer_workers[i].start()

g1_worker = threading.Thread(name='g1', target=graph2_training)
g2_worker = threading.Thread(name='g2', target=graph1_inference)

g1_worker.start()
g2_worker.start()




logging.debug('Waiting for worker threads')

main_thread = threading.currentThread()
for t in threading.enumerate():
    if t is not main_thread:
        t.join()

logging.debug('Done!')