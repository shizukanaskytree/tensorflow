# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for common methods in strategy classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceStrategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import nest


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
        ] + strategy_combinations.all_strategies,
        mode=['eager']))
class StrategyTest(test.TestCase, parameterized.TestCase):

  def testCaptureReplicaId(self, strategy):
    m = {}

    @def_function.function
    def f():
      return ds_context.get_replica_context().replica_id_in_sync_group

    @def_function.function
    def g():
      # Make g() a stateful function so it's traced twice.
      if m.get('v', None) is None:
        m['v'] = variables.Variable(0.)
      return strategy.run(f)

    g()


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.tpu_strategy
        ],
        mode=['graph', 'eager']))
class StrategyLocalResultTest(test.TestCase):

  def testLocalResultForDictionary(self, distribution):

    @def_function.function
    def model_fn():
      return {'a': constant_op.constant(1.), 'b': constant_op.constant(2.)}

    with distribution.scope():
      result = distribution.run(model_fn)
      got = self.evaluate(distribution.experimental_local_results(result))
      self.assertEqual(got, ({'a': 1., 'b': 2.}, {'a': 1., 'b': 2.}))

  def testLocalResultForList(self, distribution):

    @def_function.function
    def model_fn():
      return [constant_op.constant(1.), constant_op.constant(2.)]

    with distribution.scope():
      result = distribution.run(model_fn)
      got = self.evaluate(distribution.experimental_local_results(result))
      self.assertEqual(got, ([1., 2.], [1., 2.]))

  def testLocalResultForTuple(self, distribution):

    @def_function.function
    def model_fn():
      return (constant_op.constant(1.), constant_op.constant(2.),
              constant_op.constant(3.))

    with distribution.scope():
      result = distribution.run(model_fn)
      got = self.evaluate(distribution.experimental_local_results(result))
      self.assertEqual(got, ((1., 2., 3.), (1., 2., 3.)))

  def testLocalResultForNestedStruct(self, distribution):

    @def_function.function
    def model_fn():
      return ({
          'a': constant_op.constant(1.),
          'b': constant_op.constant(2.)
      }, {
          'a': constant_op.constant(4.),
          'b': constant_op.constant(6.)
      })

    with distribution.scope():
      result = distribution.run(model_fn)
      got = self.evaluate(distribution.experimental_local_results(result))
      self.assertEqual(got, (({
          'a': 1.,
          'b': 2.
      }, {
          'a': 4.,
          'b': 6.
      }), ({
          'a': 1.,
          'b': 2.
      }, {
          'a': 4.,
          'b': 6.
      })))

  def testLocalResultForNestedStructWithoutTensor(self, distribution):

    @def_function.function
    def model_fn():
      return {'a': 1., 'b': 2.}

    with distribution.scope():
      result = distribution.run(model_fn)
      v = self.evaluate(distribution.experimental_local_results(result))
      self.assertIsInstance(v, tuple)
      self.assertAllEqual(v, ({'a': 1., 'b': 2.}, {'a': 1., 'b': 2.}))

  def testLocalResultForScalarValue(self, distribution):

    @def_function.function
    def model_fn():
      return distribution.extended._get_local_replica_id(
          ds_context.get_replica_context().replica_id_in_sync_group)

    with distribution.scope():
      result = distribution.run(model_fn)
      v = self.evaluate(distribution.experimental_local_results(result))
      self.assertIsInstance(v, tuple)
      self.assertEqual(v, (0, 1))

  def testLocalResultForDictionaryDifferentReplicas(self, distribution):

    @def_function.function
    def model_fn():
      replica_id = distribution.extended._get_local_replica_id(
          ds_context.get_replica_context().replica_id_in_sync_group)
      return {
          'a': math_ops.cast(replica_id + 1, dtype=float),
          'b': math_ops.cast(replica_id + 2, dtype=float)
      }

    with distribution.scope():
      result = distribution.run(model_fn)
      got = self.evaluate(distribution.experimental_local_results(result))
      self.assertAllEqual(got, ({'a': 1., 'b': 2.}, {'a': 2., 'b': 3.}))

  def testLocalResultForTensor(self, distribution):

    @def_function.function
    def model_fn():
      return constant_op.constant([2., 3.])

    with distribution.scope():
      result = distribution.run(model_fn)
      v = self.evaluate(distribution.experimental_local_results(result))
      self.assertAllEqual(v, ([2., 3.], [2., 3.]))


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
        ] + strategy_combinations.all_strategies,
        mode=['eager']))
class ReduceTest(test.TestCase, parameterized.TestCase):

  def testBasic(self, strategy):
    per_replica_value = strategy.experimental_distribute_values_from_function(
        lambda _: array_ops.ones((), dtypes.float32))

    def fn_eager():

      return strategy.reduce(
          reduce_util.ReduceOp.SUM, value=per_replica_value, axis=None)

    fn_graph = def_function.function(fn_eager)
    # Run reduce under the strategy scope to explicitly enter
    # strategy default_device scope.
    with strategy.scope():
      self.assertEqual(fn_eager().numpy(), 1.0 * strategy.num_replicas_in_sync)
      self.assertEqual(fn_graph().numpy(), 1.0 * strategy.num_replicas_in_sync)

    # Run reduce without a strategy scope to implicitly enter
    # strategy default_device scope.
    self.assertEqual(fn_eager().numpy(), 1.0 * strategy.num_replicas_in_sync)
    self.assertEqual(fn_graph().numpy(), 1.0 * strategy.num_replicas_in_sync)

  def testAxis(self, strategy):

    @def_function.function
    def fn():
      return constant_op.constant([1., 2.])

    x = strategy.run(fn)

    x_m = strategy.reduce(reduce_util.ReduceOp.MEAN, x, axis=0)
    self.assertEqual(1.5, x_m)
    x_s = strategy.reduce(reduce_util.ReduceOp.SUM, x, axis=0)
    self.assertEqual(3 * strategy.num_replicas_in_sync, x_s)


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
            strategy_combinations.tpu_strategy,
        ] + strategy_combinations.strategies_minus_tpu,
        tf_function=[combinations.tf_function, combinations.no_tf_function],
        mode=['eager']))
class ReplicaCtxAllReduceTest(test.TestCase, parameterized.TestCase):

  def testBasic(self, strategy, tf_function):
    if (isinstance(strategy, tpu_strategy.TPUStrategy) and
        tf_function is combinations.no_tf_function):
      self.skipTest('Skip TPUStrategy + eager combination.')

    @tf_function
    def fn():

      def replica_fn():
        value = constant_op.constant(1.0)
        reduced = strategy.extended._replica_ctx_all_reduce('SUM', value)
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]
    self.assertEqual(got, 1.0 * strategy.num_replicas_in_sync)

  def testNestedInput(self, strategy, tf_function):
    # TODO(b/122840926): enable this test once cl/122840926 is submitted.
    self.skipTest('Enable after cl/353109164 is submitted.')

    if (isinstance(strategy, tpu_strategy.TPUStrategy) and
        tf_function is combinations.no_tf_function):
      self.skipTest('Skip TPUStrategy + eager combination.')

    @tf_function
    def fn():

      def replica_fn():
        value = (constant_op.constant(1.0), constant_op.constant(2.0))
        reduced = strategy.extended._replica_ctx_all_reduce('SUM', value)
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]
    self.assertEqual(got, (1.0 * strategy.num_replicas_in_sync,
                           2.0 * strategy.num_replicas_in_sync))


def _make_indexed_slices(values, indices, dense_shape):
  tensor = ops.IndexedSlices(
      values=constant_op.constant(values),
      indices=constant_op.constant(indices),
      dense_shape=constant_op.constant(dense_shape))
  return tensor


def _get_num_replicas_per_client(strategy):
  if isinstance(strategy, CollectiveAllReduceStrategy):
    resolver = strategy.cluster_resolver
    return max(nest.flatten(resolver.num_accelerators())[0], 1)
  else:
    return strategy.num_replicas_in_sync


def _is_tpu_strategy(strategy):
  return isinstance(strategy,
                    (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1,
                     tpu_strategy.TPUStrategyV2))


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
        ],
        mode=['eager']))
class DistributedCollectiveAllReduceStrategyTest(
    strategy_test_lib.DistributionTestBase,
    parameterized.TestCase):

  def testDatasetFromFunction(self, strategy):
    def dataset_fn(input_context):
      global_batch_size = 10
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      d = dataset_ops.DatasetV2.range(100).repeat().batch(batch_size)
      return d.shard(input_context.num_input_pipelines,
                     input_context.input_pipeline_id)

    expected_sum_on_workers = {'chief': 10, 'worker': 35}
    input_iterator = iter(
        strategy.distribute_datasets_from_function(dataset_fn))

    @def_function.function
    def run(iterator):
      return strategy.experimental_local_results(iterator.get_next())

    result = run(input_iterator)
    sum_value = math_ops.reduce_sum(result)
    self.assertEqual(
        sum_value.numpy(),
        expected_sum_on_workers[multi_worker_test_base.get_task_type()])

  def testSimpleInputFromDatasetLastPartialBatch(self, strategy):
    global_batch_size = 8
    dataset = dataset_ops.DatasetV2.range(14).batch(
        global_batch_size, drop_remainder=False)
    input_iterator = iter(strategy.experimental_distribute_dataset(dataset))

    @def_function.function
    def run(input_iterator):
      return strategy.run(lambda x: x, args=(next(input_iterator),))

    # Let the complete batch go.
    run(input_iterator)

    # `result` is an incomplete batch
    result = run(input_iterator)
    expected_data_on_workers = {'chief': [8, 9, 10], 'worker': [11, 12, 13]}
    self.assertAllEqual(
        expected_data_on_workers[multi_worker_test_base.get_task_type()],
        result.numpy(),
    )

  def testSimpleInputFromFnLastPartialBatch(self, strategy):

    def dataset_fn(input_context):
      global_batch_size = 8
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      dataset = dataset_ops.DatasetV2.range(14).batch(
          batch_size, drop_remainder=False)
      return dataset.shard(input_context.num_input_pipelines,
                           input_context.input_pipeline_id)

    input_iterator = iter(
        strategy.distribute_datasets_from_function(dataset_fn))

    @def_function.function
    def run(input_iterator):
      return strategy.run(lambda x: x, args=(next(input_iterator),))

    # Let the complete batch go.
    run(input_iterator)
    # `result` is an incomplete batch
    result = run(input_iterator)

    expected_data_on_worker = {'chief': [8, 9, 10, 11], 'worker': [12, 13]}
    self.assertAllEqual(
        expected_data_on_worker[multi_worker_test_base.get_task_type()],
        result.numpy())

  def testReduceHostTensor(self, strategy):
    reduced = strategy.reduce(
        reduce_util.ReduceOp.SUM, array_ops.identity(1.), axis=None)
    self.assertEqual(reduced.numpy(), 2.)

  def testReduceToHostTensor(self, strategy):
    value = array_ops.identity(1.)
    reduced = strategy.extended.reduce_to(reduce_util.ReduceOp.SUM, value,
                                          value)
    self.assertEqual(reduced.numpy(), 2.)

  def testBatchReduceToHostTensor(self, strategy):
    value = array_ops.identity(1.)
    reduced = strategy.extended.batch_reduce_to(reduce_util.ReduceOp.SUM,
                                                [(value, value),
                                                 (value, value)])
    self.assertAllEqual([2., 2.], reduced)

  def testReduceDeviceTensors(self, strategy):
    value = strategy.run(lambda: array_ops.identity(1.))
    reduced = strategy.reduce(reduce_util.ReduceOp.SUM, value, axis=None)
    self.assertEqual(reduced.numpy(), 2.)

  def testReduceToDeviceTensors(self, strategy):
    value = strategy.run(lambda: array_ops.identity(1.))
    reduced = strategy.extended.reduce_to(reduce_util.ReduceOp.SUM, value,
                                          value)
    self.assertEqual(reduced.numpy(), 2.)

  def testBatchReduceToDeviceTensors(self, strategy):
    value = strategy.run(lambda: array_ops.identity(1.))
    reduced = strategy.extended.batch_reduce_to(reduce_util.ReduceOp.SUM,
                                                [(value, value),
                                                 (value, value)])
    self.assertAllEqual([2., 2.], reduced)

  # TODO(crccw): add a test that mixes device and host tensors after multi
  # worker strategy combinations can run on a fixed number of GPUs.


class StrategyClusterResolverTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          strategy=[strategy_combinations.multi_worker_mirrored_2x1_cpu] +
          strategy_combinations.all_strategies,
          mode=['eager']))
  def testClusterResolverProperty(self, strategy):
    # CollectiveAllReduceStrategy and TPUStrategy must have a cluster resolver.
    # `None` otherwise.
    resolver = strategy.cluster_resolver
    if not isinstance(strategy, CollectiveAllReduceStrategy) and not isinstance(
        strategy, tpu_strategy.TPUStrategy):
      self.assertIsNone(resolver)
      return

    with strategy.scope():
      self.assertIs(strategy.cluster_resolver, resolver)

    self.assertTrue(hasattr(resolver, 'cluster_spec'))
    self.assertTrue(hasattr(resolver, 'master'))
    self.assertTrue(hasattr(resolver, 'num_accelerators'))
    self.assertTrue(hasattr(resolver, 'task_id'))
    self.assertTrue(hasattr(resolver, 'task_type'))
    if isinstance(strategy, CollectiveAllReduceStrategy):
      self.assertEqual(resolver.task_id, 0)
      self.assertAllInSet(resolver.task_type, ['chief', 'worker'])


if __name__ == '__main__':
  test_util.main()
