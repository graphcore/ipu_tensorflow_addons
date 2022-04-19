# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
import unittest
import re
import numpy
from tensorflow.python.client import session
from tensorflow import disable_v2_behavior
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.framework import dtypes, importer, ops, test_util
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework.ops import name_scope
from ipu_tensorflow_addons.saved_model_tool.ipu_convert import IpuConversionParams
from ipu_tensorflow_addons.saved_model_tool.converter import GeluReplacement
from ipu_tensorflow_addons.saved_model_tool.saved_model_test_utils import \
    ModelForTest

disable_v2_behavior()


class TestSavedmodel(ModelForTest):
  def create(self):
    def gelu(x):
      cdf = 0.5 * (1.0 + math_ops.tanh(
          (numpy.sqrt(2 / numpy.pi) * (x + 0.044715 * math_ops.pow(x, 3)))))
      return x * cdf

    x = array_ops.placeholder(dtypes.float32, [8, 8], name="x")
    w0 = variable_scope.get_variable("w0", shape=[8, 8], dtype=dtypes.float32)
    with name_scope("gelu/"):
      x = math_ops.add(x, w0, name='BiasAdd')
      x = gelu(x)
    with name_scope("gelu_end/"):
      y = math_ops.matmul(w0, x)
    return y


class TestSavedmodelParallelGelu(ModelForTest):
  def create(self):
    def gelu(x):
      cdf = 0.5 * (1.0 + math_ops.tanh(
          (numpy.sqrt(2 / numpy.pi) * (x + 0.044715 * math_ops.pow(x, 3)))))
      return x * cdf

    x = array_ops.placeholder(dtypes.float32, [8, 8], name="x")
    w0 = variable_scope.get_variable("w0", shape=[8, 8], dtype=dtypes.float32)
    w1 = variable_scope.get_variable("w1", shape=[8, 8], dtype=dtypes.float32)
    with name_scope("gelu0/"):
      x0 = math_ops.add(x, w0, name='BiasAdd')
      x0 = gelu(x0)
    with name_scope("gelu1/"):
      x1 = math_ops.add(x, w1, name='BiasAdd')
      x1 = gelu(x1)
    with name_scope("gelu0_end/"):
      y0 = math_ops.matmul(w0, x0)
    with name_scope("gelu1_end/"):
      y1 = math_ops.matmul(w1, x1)

    y = math_ops.add(y0, y1, name='result')
    return y


class TestSavedmodelConcurrentGelu(ModelForTest):
  def create(self):
    def gelu(x):
      cdf = 0.5 * (1.0 + math_ops.tanh(
          (numpy.sqrt(2 / numpy.pi) * (x + 0.044715 * math_ops.pow(x, 3)))))
      return x * cdf

    x = array_ops.placeholder(dtypes.float32, [8, 8], name="x")
    w0 = variable_scope.get_variable("w0", shape=[8, 8], dtype=dtypes.float32)
    w1 = variable_scope.get_variable("w1", shape=[8, 8], dtype=dtypes.float32)
    with name_scope("gelu0/"):
      x = math_ops.add(x, w0, name='BiasAdd')
      x = gelu(x)
    with name_scope("gelu_end1/"):
      x = math_ops.matmul(w0, x)

    with name_scope("gelu1/"):
      x = math_ops.add(x, w1, name='BiasAdd')
      x = gelu(x)
    with name_scope("gelu_end1/"):
      y = math_ops.matmul(w1, x)

    return y


class GeluReplacementTestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    self.test_model = TestSavedmodel(freeze=True)
    self.test_graph_def = self.test_model.graph_def
    self.test_signature_def = self.test_model.signature_def

    self.test_model_concurrent = TestSavedmodelConcurrentGelu(freeze=True)
    self.test_concurrent_graph_def = self.test_model_concurrent.graph_def
    self.test_concurrent_signature_def = \
      self.test_model_concurrent.signature_def

    self.test_model_parallel = TestSavedmodelParallelGelu(freeze=True)
    self.test_parallel_graph_def = self.test_model_parallel.graph_def
    self.test_parallel_signature_def = self.test_model_parallel.signature_def

  def graph2tensor(self, graph_def, feed_dict, output_name):
    with ops.Graph().as_default():
      importer.import_graph_def(graph_def, name="")
      with session.Session() as sess:
        return sess.run(output_name, feed_dict=feed_dict)

  def test_gelu_replace(self):
    graph_def = self.test_graph_def
    signature_def = self.test_signature_def

    gelu_replacement = dict()
    gelu_replacement["nodes"] = [
        "gelu/Pow$", "gelu/mul$", "gelu/add$", "gelu/mul_1$", "gelu/Tanh$",
        "gelu/add_1$", "gelu/mul_2$", "gelu/mul_3$"
    ]
    gelu_replacement["node_as_gelu_input"] = ["gelu/BiasAdd"]
    gelu_replacement["node_use_gelu_output"] = ["gelu_end/MatMul"]

    params = IpuConversionParams(gelu_replacement=gelu_replacement)
    graph_def, _ = GeluReplacement(params).apply(graph_def, signature_def)

    feed_dict = {"x:0": numpy.random.rand(8, 8)}
    output_name = [
        signature_def.outputs[t].name for t in signature_def.outputs
    ]
    result1 = self.graph2tensor(self.test_graph_def, feed_dict, output_name)
    result2 = self.graph2tensor(graph_def, feed_dict, output_name)
    for i, _ in enumerate(result1):
      max_diff = numpy.max(abs(result1[i] - result2[i]))
      self.assertTrue(max_diff < 1e-6)

    for node in graph_def.node:
      for pattern in gelu_replacement['nodes']:
        self.assertFalse(re.search(pattern, node.name))

    ipu_gelus = [n for n in graph_def.node if n.op == "IpuGelu"]
    self.assertTrue(len(ipu_gelus) == 1)

  def test_concurrent_gelu_replace(self):
    graph_def = self.test_model_concurrent.graph_def
    signature_def = self.test_model_concurrent.signature_def

    gelu_replacement = dict()
    gelu_replacement["nodes"] = [
        "Pow$", "mul$", "add$", "mul_1$", "Tanh$", "add_1$", "mul_2$", "mul_3$"
    ]
    gelu_replacement["node_as_gelu_input"] = ["BiasAdd"]
    gelu_replacement["node_use_gelu_output"] = ["MatMul"]

    params = IpuConversionParams(gelu_replacement=gelu_replacement)
    graph_def, _ = GeluReplacement(params).apply(graph_def, signature_def)

    with ops.Graph().as_default():
      importer.import_graph_def(graph_def)

  def test_parallel_gelu_replace(self):
    graph_def = self.test_model_parallel.graph_def
    signature_def = self.test_model_parallel.signature_def

    gelu_replacement = dict()
    gelu_replacement["nodes"] = [
        "Pow$", "mul$", "add$", "mul_1$", "Tanh$", "add_1$", "mul_2$", "mul_3$"
    ]
    gelu_replacement["node_as_gelu_input"] = ["BiasAdd"]
    gelu_replacement["node_use_gelu_output"] = ["MatMul"]

    params = IpuConversionParams(gelu_replacement=gelu_replacement)
    graph_def, _ = GeluReplacement(params).apply(graph_def, signature_def)

    with ops.Graph().as_default():
      importer.import_graph_def(graph_def)


if __name__ == '__main__':
  unittest.main()
