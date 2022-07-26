# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

from unittest import mock
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow import keras
from absl.testing import parameterized
from tensorflow.python import ipu
from tensorflow.python.ipu import test_utils as tu
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.platform import googletest
from ipu_tensorflow_addons.keras import layers as ipu_layers


def create_n_replica_ipu_config(ipu_count):
  cfg = IPUConfig()
  cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10000
  cfg.auto_select_ipus = ipu_count
  tu.add_hw_ci_connection_options(cfg)

  return cfg


class ConditionalLayer(keras.layers.Layer):
  def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
    c = tf.constant(0, shape=inputs.shape, dtype=inputs.dtype)
    x = tf.reduce_all(tf.greater(inputs, c))
    y = tf.cond(x, lambda: ipu.cross_replica_ops.cross_replica_sum(inputs),
                lambda: tf.constant(0, shape=(2, 4), dtype=inputs.dtype))
    return y


class TestKerasAssumeEqual(test_util.TensorFlowTestCase,
                           parameterized.TestCase):
  # assume_equal_across_replicas supports copying or inplace operation depending
  # on the value of the inplace argument
  inplace_or_copy = [True, False]

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.run_v2_only
  def testNoDivergenceWithAssumeEqualLayer(self):

    cfg = create_n_replica_ipu_config(2)
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():

      input_layer = keras.layers.Input(shape=(32),
                                       dtype=np.single,
                                       batch_size=2)
      init = keras.initializers.Constant(0.1)

      dense_layer = keras.layers.Dense(4,
                                       name="layer0",
                                       kernel_initializer=init)(input_layer)

      assume_equals_layer = ipu_layers.AssumeEqualAcrossReplicas()(dense_layer)
      conditional_layer = ConditionalLayer()(assume_equals_layer)

      # Without the AssumeEqualAcrossReplicas layer we should get a Divergent
      # control flow compilation error coming from ConditionalLayer.
      m = keras.Model(input_layer, conditional_layer)
      m.compile('sgd', loss='mse', steps_per_execution=12)

      input_x = np.full([96, 32], 1.0, dtype=np.single)
      m.predict(input_x, batch_size=2)

  @parameterized.parameters(inplace_or_copy)
  @test_util.deprecated_graph_mode_only
  @mock.patch.object(ipu.ops.cross_replica_ops, "assume_equal_across_replicas")
  def testLayerUsesAssumeEqualOp(self, inplace, mock_op):
    placeholder = tf.compat.v1.placeholder(np.single, 32)
    ipu_layers.AssumeEqualAcrossReplicas(inplace)(placeholder)

    mock_op.assert_called_with(placeholder, inplace)

  @parameterized.parameters(inplace_or_copy)
  @tu.skip_on_hw
  @test_util.run_v2_only
  def testGetConfig(self, inplace):
    layer = ipu_layers.AssumeEqualAcrossReplicas(inplace)
    self.assertEqual(layer.get_config()["inplace"], inplace)


if __name__ == "__main__":
  googletest.main()
