import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'

import tensorflow as tf
import numpy as np
import threading
import logging

# logging doesn't work
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )


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
            outs = sess.run(s,feed_dict={x: x_data,w: w_data}) # 我感觉只要改这里就行了，Graph 2 的 Input 也 同时 feed 进去
            #print('out_xwb = \n {}'.format(outs_xwb))
            print("outs of g1 = {}".format(outs))
            #logging.debug("g1 outs = {}".format(outs))


def graph2_inference():
    np.random.seed(0)
    x_data = np.random.randn(5, 10)
    w_data = np.random.randn(10, 1)

    #import tensorflow as tf
    g_2 = tf.Graph()
    with g_2.as_default():
        x = tf.placeholder(tf.float32, shape=(5, 10))
        w = tf.placeholder(tf.float32, shape=(10, 1))
        # b = tf.fill((5,1),-1.)
        xw = tf.matmul(x, w)

        # xwb=xw+b
        xwb = xw
        s = tf.reduce_max(xwb)

        with tf.Session(graph=g_2) as sess:
            #outs_xwb = sess.run(xwb, feed_dict={x: x_data, w: w_data})
            outs = sess.run(s, feed_dict={x: x_data, w: w_data})  # 我感觉只要改这里就行了，Graph 2 的 Input 也 同时 feed 进去
            #print('out_xwb = \n {}'.format(outs_xwb))
            print("outs of g2 = {}".format(outs))
            #logging.debug("g2 outs = {}".format(outs))


g1_worker = threading.Thread(name='g1', target=graph1_inference)
g2_worker = threading.Thread(name='g2', target=graph2_inference)

g1_worker.start()
g2_worker.start()

logging.debug('Waiting for worker threads')

main_thread = threading.currentThread()
for t in threading.enumerate():
    if t is not main_thread:
        t.join()

#logging.debug('Done!')
print('Done!')
