import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'
import tensorflow as tf

#default_graph = tf.get_default_graph()
#print(default_graph)

g = tf.Graph()
g2 = tf.Graph()



#with g.as_default():
#    a = tf.constant(5, name='a')
#    b = tf.constant(2, name='b')
#    c = tf.multiply(a,b, name='c')
#    sess = tf.Session()
#    outs = sess.run(c)
#    #print(outs)
#    sess.close()

