# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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
#
# This file has been modified by Graphcore Ltd.
# ==============================================================================

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from tensorflow.python import ipu
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.ipu.ops.f8_ops import create_metadata, Format
from tensorflow.python.framework import ops

import keras
from keras.engine.input_layer import Input

from ipu_tensorflow_addons.keras.layers import SerialDense, Dense, ConvertToF8, ConvertFromF8

TEST_CASES = ({
    'testcase_name': 'input_columns',
    'input_shape': [8, 16],
    'num_units': 5,
    'serialization_factor': 2,
    'serialization_dimension': 'input_columns',
}, {
    'testcase_name': 'input_rows_kernel_columns',
    'input_shape': [4, 21],
    'num_units': 8,
    'serialization_factor': 3,
    'serialization_dimension': 'input_rows_kernel_columns',
}, {
    'testcase_name': 'kernel_rows',
    'input_shape': [4, 4],
    'num_units': 8,
    'serialization_factor': 2,
    'serialization_dimension': 'kernel_rows',
})


def _getTestCases():
  from copy import deepcopy

  test_cases = list(TEST_CASES)
  # Add test cases with a batch dim for a.
  for case in deepcopy(TEST_CASES):
    case['testcase_name'] += "_batch_a"
    case['input_shape'] = [2] + case['input_shape']
    test_cases.append(case)
  return test_cases


# Note that in this test we expect small numerical differences as serializing
# means that some operations are done in a different order.
class SerialDenseTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0xDEADBEEF)

  @parameterized.named_parameters(*_getTestCases())
  @test_util.run_v2_only
  def testSerialDense(self, input_shape, num_units, serialization_factor,
                      serialization_dimension):
    input_val = np.random.normal(2.0, 2.0, input_shape)
    kernel_val = np.random.normal(2.0, 2.0, [input_shape[-1], num_units])

    def kernel_init(_shape, **_):
      return kernel_val

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      dense = keras.layers.Dense(num_units,
                                 kernel_initializer=kernel_init)(input_val)
      serial_dense = SerialDense(num_units,
                                 serialization_factor,
                                 serialization_dimension,
                                 kernel_initializer=kernel_init)(input_val)
      self.assertAllClose(dense, serial_dense, atol=1.e-05, rtol=1.e-05)

  @parameterized.named_parameters(*_getTestCases())
  @test_util.run_v2_only
  def testSerializedMatmulGrad(self, input_shape, num_units,
                               serialization_factor, serialization_dimension):
    input_val = np.random.normal(2.0, 2.0, input_shape)
    kernel_val = np.random.normal(2.0, 2.0, [input_shape[-1], num_units])

    def kernel_init(_shape, **_):
      return kernel_val

    def func(layer):
      with tf.GradientTape() as t:
        output = layer(input_val)
        # Not a real loss function, but good enough for testing backprop.
        loss = tf.reduce_sum(output)
      grads = t.gradient(loss, layer.weights)
      return grads

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      dense = keras.layers.Dense(num_units, kernel_initializer=kernel_init)
      serial_dense = SerialDense(num_units,
                                 serialization_factor,
                                 serialization_dimension,
                                 kernel_initializer=kernel_init)

      out_dense = func(dense)
      out_serial_dense = func(serial_dense)
    self.assertAllClose(out_dense, out_serial_dense, atol=1.e-05, rtol=1.e-05)


class DenseTest(tf.test.TestCase, parameterized.TestCase):
  """Test functionality of ipu_tensorflow_addons.keras.layers.Dense.
  """

  data = np.array([[1., 2.], [3., -1.], [-3., -5.]])
  """Sample data to be used in tests.
  """

  metadata = create_metadata(Format.F143, 0)
  """Sample metadata to be used in tests.
  """

  def assert_weights_equal(self, weights1, weights2):
    """Asserts that weights1 and weights2 are identical.
    """
    self.assertEqual(len(weights1), len(weights2))
    for w1, w2 in zip(weights1, weights2):
      self.assertAllEqual(w1, w2)

  def weights_equal(self, weights1, weights2):
    """Returns true if and only if weights1 and weights2 are identical.
    """
    if len(weights1) != len(weights2):
      return False
    for w1, w2 in zip(weights1, weights2):
      if (w1 != w2).any():
        return False
    return True

  def get_results_to_compare(self, arr_ipu, arr_keras):
    """Helper to get outputs of the 2 different Dense layers.
    """
    layer_dtype = "float16" if isinstance(arr_ipu, list) else arr_keras.dtype
    keras_layer = keras.layers.Dense(units=3, dtype=layer_dtype)

    ipu_layer = Dense(units=3, dtype=layer_dtype)

    result = ipu_layer(arr_ipu)
    weights = ipu_layer.get_weights()

    # Need a dummy initial call to be able to set the weights for the layer.
    _ = keras_layer(arr_keras)
    if isinstance(arr_ipu, list):
      kernel = ConvertFromF8()(weights[:2])
    else:
      kernel = weights[0]
    bias = weights[-1]
    keras_layer.set_weights([kernel, bias])

    expected = keras_layer(arr_keras)
    return result, expected

  @test_util.run_v2_only
  def test_dense_f8_set_get_weights(self):
    """Test that set_weights and get_weights functions work, and the layer
    produces a consistent result given the same weights and inputs.
    """
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      arr_ipu = ConvertToF8()(DenseTest.data, DenseTest.metadata)
      ipu_layer = Dense(units=3)
      # Dummy call so that the weights get initialised.
      expected = ipu_layer(arr_ipu)
      initial_weights = ipu_layer.get_weights()
      self.assertEqual(len(initial_weights), 3)

      # See that we can set new weights.
      new_layer = Dense(units=3)
      dummy_output = new_layer(arr_ipu)

      # These should not be equal since they have different weights.
      self.assertNotAllEqual(expected, dummy_output)

      old_weights = new_layer.get_weights()
      self.assertFalse(self.weights_equal(old_weights, initial_weights))
      new_layer.set_weights(initial_weights)
      new_weights = new_layer.get_weights()
      self.assert_weights_equal(new_weights, initial_weights)

      # Check that with the same weights we get identical results.
      result = new_layer(arr_ipu)
      self.assertAllEqual(expected, result)

  @test_util.run_v2_only
  def test_dense_f8(self):
    """Test that fp8 and fp16 inputs produce similar results.
    """
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      arr_ipu = ConvertToF8()(ops.convert_to_tensor(DenseTest.data),
                              DenseTest.metadata)
      arr_keras = ops.convert_to_tensor(DenseTest.data.astype("float16"))

      result, expected = self.get_results_to_compare(arr_ipu, arr_keras)

      # Check that the results are similar. Since we're taking weights from the
      # f8 model to the f16 one, these should basically be identical.
      self.assertAllClose(result, expected, rtol=1e-5)

  @parameterized.parameters("float16", "float32")
  @test_util.run_v2_only
  def test_dense_not_f8(self, tensor_type):
    """Test that the layer still works for non-f8 types.
    """
    strategy = ipu.ipu_strategy.IPUStrategyV1()

    with strategy.scope():
      arr = ops.convert_to_tensor(DenseTest.data.astype(tensor_type))
      result, expected = self.get_results_to_compare(arr, arr)

      # Check that the result is identical running on the two Dense layers.
      self.assertAllEqual(result, expected)

  def test_dense_f8_functional_model(self):
    """Test that we can create and use a model with an fp8 Dense layer.
    """
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      ipu_layer = Dense(units=3)
      inputs = Input(dtype="float16", shape=[2], batch_size=3)
      ipu_model = keras.Model(
          inputs, ipu_layer(ConvertToF8()(inputs, DenseTest.metadata)))

      arr_ipu = ops.convert_to_tensor(DenseTest.data.astype("float16"))
      # We expect using layers directly produce the
      # same result as using a model.
      expected = ipu_layer(ConvertToF8()(arr_ipu, DenseTest.metadata))
      result1 = ipu_model(arr_ipu)
      self.assertAllEqual(expected, result1)

      # Check that predict works and produces the same result.
      result2 = ipu_model.predict(arr_ipu, batch_size=3)
      self.assertAllEqual(expected, result2)

      # Check that we get the same result after compiling.
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      ipu_model.compile(opt, loss='mse')
      result3 = ipu_model.predict(arr_ipu, batch_size=3)
      self.assertAllEqual(expected, result3)

  @test_util.run_v2_only
  def test_dense_f8_submodel(self):
    """Test that we can use an f8 model as a submodel.

    This makes sure that layers work fine even with eager mode disabled.
    """
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      inputs = Input(dtype="float16", shape=[2], batch_size=3)

      class TestModel(keras.Model):  # pylint: disable=abstract-method

        def build(self, input_shape):
          self.metadata = DenseTest.metadata
          self.l = ConvertToF8()
          self.dense2 = Dense(units=4)
          super().build(input_shape)

        def call(self, inputs):  # pylint: disable=arguments-differ
          return self.dense2(self.l(inputs, self.metadata))

      arr_ipu = DenseTest.data.astype("float16")

      # Initialize the model and get the result of calling it.
      tmp_model = TestModel()
      tmp_model.build(input_shape=inputs)
      ipu_model = keras.Model(inputs=inputs, outputs=tmp_model(inputs))
      result1 = ipu_model(arr_ipu)

      # Set up an independent layer with the same weights
      # and check that it gives the same result.
      ipu_layer = Dense(units=4)
      layer_inputs = ConvertToF8()(arr_ipu, DenseTest.metadata)
      ipu_layer(layer_inputs)
      ipu_layer.set_weights(ipu_model.get_weights())
      expected = ipu_layer(layer_inputs)
      self.assertAllEqual(expected, result1)

      # Check that predict works and produces the same result.
      result2 = ipu_model.predict(arr_ipu, batch_size=3)
      self.assertAllEqual(expected, result2)

      # Check that we get the same result after compiling.
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      ipu_model.compile(opt, loss='mse')
      result3 = ipu_model.predict(arr_ipu, batch_size=3)
      self.assertAllEqual(expected, result3)


if __name__ == "__main__":
  googletest.main()
