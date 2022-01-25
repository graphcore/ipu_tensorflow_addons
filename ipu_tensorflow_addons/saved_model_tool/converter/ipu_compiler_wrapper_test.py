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
import copy

from tensorflow import disable_v2_behavior
from tensorflow.python.framework import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import name_scope

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from ipu_tensorflow_addons.saved_model_tool.ipu_convert import IpuConversionParams
from ipu_tensorflow_addons.saved_model_tool.converter import IPUCompilerWrapper
from ipu_tensorflow_addons.saved_model_tool.converter import IPUPlacement
from ipu_tensorflow_addons.saved_model_tool.saved_model_test_utils import \
    ModelForTest
disable_v2_behavior()


class TestSavedmodel(ModelForTest):
  def create(self):
    x = array_ops.placeholder(dtypes.float32, [8, 8], name="x")
    with name_scope("sharding0"):
      w0 = variable_scope.get_variable("w0",
                                       shape=[8, 8],
                                       dtype=dtypes.float32)
      x = math_ops.matmul(w0, x, name='matmul0')

    with name_scope("sharding1"):
      w1 = variable_scope.get_variable("w1",
                                       shape=[8, 8],
                                       dtype=dtypes.float32)
      x = math_ops.matmul(w1, x, name='matmul1')
      x1 = math_ops.reduce_sum(x, name='reduce0')
    return x, x1


class IPUCompilerWrapperTestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    self.test_model = TestSavedmodel(freeze=True)
    self.test_graph_def = self.test_model.graph_def
    self.test_signature_def = self.test_model.signature_def

  def _check_ipu_placement(self, node):
    if node.op == "Placeholder":
      return True

    if node.device != '/device:IPU:0':
      return False
    attr_list = [
        '_XlaCompile',
        '_XlaScope',
        '_XlaSeparateCompiledGradients',
    ]
    if [attr for attr in attr_list if attr not in node.attr]:
      return False
    return True

  def test_ipu_compiler_wrapper(self):
    graph_def = copy.deepcopy(self.test_graph_def)
    signature_def = copy.deepcopy(self.test_signature_def)
    excluded_nodes = [
        '^sharding0',
        '^w0',
    ]
    params = IpuConversionParams(excluded_nodes=excluded_nodes)

    graph_def, signature_def = IPUPlacement(params).apply(
        graph_def, signature_def)
    graph_def, signature_def = IPUCompilerWrapper(params).apply(
        graph_def, signature_def)

    for node in graph_def.node:
      if node.name.startswith('sharding0'):
        self.assertFalse(self._check_ipu_placement(node))
      if node.name.startswith('sharding1'):
        self.assertTrue(self._check_ipu_placement(node))

    for k in self.test_signature_def.inputs:
      self.assertTrue(
          self.test_signature_def.inputs[k] == signature_def.inputs[k])
    for k in self.test_signature_def.outputs:
      self.assertTrue(
          self.test_signature_def.outputs[k] == signature_def.outputs[k])


if __name__ == '__main__':
  unittest.main()
