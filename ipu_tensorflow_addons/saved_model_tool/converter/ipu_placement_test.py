# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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
"""Test for IPUPlacement Converter"""

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from ipu_tensorflow_addons.saved_model_tool.ipu_convert import IpuConversionParams
from ipu_tensorflow_addons.saved_model_tool.converter import IPUPlacement
from ipu_tensorflow_addons.saved_model_tool.saved_model_test_utils import \
    ModelForTest

tf.disable_v2_behavior()


class TestSavedmodel(ModelForTest):
  def create(self):
    x = tf.placeholder(np.float32, [8, 8], name="x")

    w0 = tf.get_variable("w0", shape=[8, 8], dtype=tf.float32)
    x = tf.matmul(w0, x)

    w1 = tf.get_variable("w1", shape=[8, 8], dtype=tf.float32)
    x = tf.matmul(w1, x)
    y = math_ops.reduce_sum(x)
    return y


class IpuPlacementTestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    self.test_model = TestSavedmodel(freeze=True)
    self.test_graph_def = self.test_model.graph_def
    self.test_signature_def = self.test_model.signature_def

  @staticmethod
  def _check_ipu_placement(node):
    if node.device != '/device:IPU:0':
      return False
    if '_XlaCompile' not in node.attr or \
       '_XlaScope' not in node.attr or \
       '_XlaSeparateCompiledGradients' not in node.attr:
      return False

    return True

  def test_ipu_placement(self):
    # check node with ipu placement
    params = IpuConversionParams()
    converter = IPUPlacement(params)
    graph_def, _ = converter.apply(self.test_graph_def,
                                   self.test_signature_def)

    self.assertIsNotNone(graph_def)
    self.assertGreater(len(graph_def.node), 0)

    for node in graph_def.node:
      if node.op == 'Placeholder':
        continue
      self.assertTrue(self._check_ipu_placement(node))

  def test_no_ipu_placement(self):
    # check node without ipu placement
    params = IpuConversionParams(ipu_placement=False)
    converter = IPUPlacement(params)
    graph_def, _ = converter.apply(self.test_graph_def,
                                   self.test_signature_def)
    self.assertIsNotNone(graph_def)
    self.assertGreater(len(graph_def.node), 0)

    for node in graph_def.node:
      if node.op == 'Placeholder':
        continue
      self.assertFalse(self._check_ipu_placement(node))

  def test_excluded_nodes(self):
    excluded_nodes = [
        '^MatMul$',
    ]
    params = IpuConversionParams(excluded_nodes=excluded_nodes)
    converter = IPUPlacement(params)
    graph_def, _ = converter.apply(self.test_graph_def,
                                   self.test_signature_def)
    self.assertIsNotNone(graph_def)
    self.assertGreater(len(graph_def.node), 0)
    for node in graph_def.node:
      if node.op == 'Placeholder':
        continue
      if node.name == 'MatMul':
        self.assertFalse(self._check_ipu_placement(node))
      else:
        self.assertTrue(self._check_ipu_placement(node))


if __name__ == '__main__':
  test.main()
