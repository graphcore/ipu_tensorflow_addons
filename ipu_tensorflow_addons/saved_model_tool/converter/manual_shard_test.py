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
"""
Tests for Manual Sharding Converter.
"""

from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes, ops, test_util
from tensorflow.python.ops import array_ops, math_ops, variable_scope
from tensorflow.python.platform import test
from tensorflow.python.ipu.scopes import ipu_shard
from ipu_tensorflow_addons.saved_model_tool import IpuConversionParams
from ipu_tensorflow_addons.saved_model_tool.converter import ManualSharding
from ipu_tensorflow_addons.saved_model_tool.saved_model_test_utils import ModelForTest


class TestGraphDefWithIPUInfo(ModelForTest):
  def create(self):
    x = array_ops.placeholder(dtypes.float32, [8, 8], name="x")

    with ipu_shard(0):
      with variable_scope.variable_scope("sharding0"):
        w0 = variable_scope.get_variable("w0",
                                         shape=[8, 8],
                                         dtype=dtypes.float32)
        with ops.device("/device:IPU:0"):
          x = math_ops.matmul(w0, x, name='matmul0')

    with ipu_shard(1):
      with variable_scope.variable_scope("sharding1"):
        w1 = variable_scope.get_variable("w1",
                                         shape=[8, 8],
                                         dtype=dtypes.float32)
        with ops.device("/device:IPU:0"):
          x = math_ops.matmul(w1, x, name='matmul1')
          y = math_ops.reduce_sum(x, name='reduce0')

    return y


class TestGraphDefWithoutIPUInfo(ModelForTest):
  def create(self):
    x = array_ops.placeholder(dtypes.float32, [8, 8], name="x")
    with variable_scope.variable_scope("sharding0"):
      w0 = variable_scope.get_variable("w0",
                                       shape=[8, 8],
                                       dtype=dtypes.float32)
      x = math_ops.matmul(w0, x, name='matmul0')

    with variable_scope.variable_scope("sharding1"):
      w1 = variable_scope.get_variable("w1",
                                       shape=[8, 8],
                                       dtype=dtypes.float32)
      x = math_ops.matmul(w1, x, name='matmul1')
      y = math_ops.reduce_sum(x, name='reduce0')

    return y


class ManualShardingConverterTestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    super().setUp()
    self.test_model_with_ipuinfo = TestGraphDefWithIPUInfo(freeze=True)
    self.test_model_without_ipuinfo = TestGraphDefWithoutIPUInfo(freeze=True)

  def _check_sharding_num(self, node, index):
    if '_XlaSharding' not in node.attr:
      print("_XlaSharding not in node {}".format(node))
      return False

    proto = xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.MAXIMAL,
                                    tile_assignment_devices=[index])
    attr_value = attr_value_pb2.AttrValue(s=proto.SerializeToString())

    if node.attr['_XlaSharding'] != attr_value:
      return False

    return True

  def test_manual_sharding(self):
    sharding_config = [[
        "^sharding0",
    ], [
        "^sharding1",
    ]]

    params = IpuConversionParams(num_ipus=2, manual_sharding=sharding_config)

    converter = ManualSharding(params)
    graph_def, _ = converter.apply(self.test_model_with_ipuinfo.graph_def,
                                   self.test_model_with_ipuinfo.signature_def)

    self.assertIsNotNone(graph_def)
    self.assertGreater(len(graph_def.node), 0)

    for node in graph_def.node:
      if node.name.startswith('sharding0') and node.op != "Const":
        self.assertTrue(self._check_sharding_num(node, 0))
        self.assertFalse(self._check_sharding_num(node, 1))
      if node.name.startswith('sharding1') and node.op != "Const":
        self.assertTrue(self._check_sharding_num(node, 1))
        self.assertFalse(self._check_sharding_num(node, 0))

  def test_manual_sharding_wo_ipu_device_info(self):
    sharding_config = [[
        "^sharding0",
    ], [
        "^sharding1",
    ]]

    params = IpuConversionParams(num_ipus=2, manual_sharding=sharding_config)

    converter = ManualSharding(params)
    with self.assertRaises(ValueError):
      converter.apply(self.test_model_without_ipuinfo.graph_def,
                      self.test_model_without_ipuinfo.signature_def)

  def test_wrong_type_manual_sharding(self):

    params = IpuConversionParams(manual_sharding=True,)

    with self.assertRaisesRegex(TypeError, '(.*)must be a list(.*)'):
      ManualSharding(params)

    params = IpuConversionParams(manual_sharding=["^sharding0"],)
    with self.assertRaisesRegex(TypeError,
                                '(.*)must only contain lists of strings(.*)'):
      ManualSharding(params)

    params = IpuConversionParams(
        manual_sharding={"shard_config": [["^sharding0"]]},)
    with self.assertRaisesRegex(TypeError, '(.*)must be a list(.*)'):
      ManualSharding(params)

    params = IpuConversionParams(manual_sharding=[],)
    ManualSharding(params)

  def test_not_equal_shards_and_num_of_ipu(self):
    sharding_config = [[
        "^sharding0",
    ], [
        "^sharding1",
    ]]
    params = IpuConversionParams(manual_sharding=sharding_config)

    with self.assertRaisesRegex(ValueError, '(.*)should be equal(.*)'):
      ManualSharding(params)


if __name__ == '__main__':
  test.main()
