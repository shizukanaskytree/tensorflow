import tensorflow as tf

# 使用方式 1
# from tensorflow.python.distribute import collective_all_reduce_strategy_atom
# test = collective_all_reduce_strategy_atom.CollectiveAllReduceStrategyAtom()

# 使用方式 2: all fail
# test = tf.compat.v1.distribute.experimental.CollectiveAllReduceStrategyAtom()
# test = tf.distribute.CollectiveAllReduceStrategyAtom()
