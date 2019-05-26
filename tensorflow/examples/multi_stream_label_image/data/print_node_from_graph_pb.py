import tensorflow as tf
from tensorflow.python.platform import gfile

#GRAPH_PB_PATH = './inception_v3_2016_08_28_frozen.pb' #path to your .pb file
GRAPH_PB_PATH = './frozen_graph.pb' #path to your .pb file

with tf.Session() as sess:
  print("load graph")
  with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n.name for n in graph_def.node]
    print(graph_nodes)
