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
from tensorflow import disable_v2_behavior
from tensorflow.nn import relu
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.core.framework import types_pb2

from tensorflow.python.ops import (array_ops, math_ops, variable_scope)
from ipu_tensorflow_addons.saved_model_tool.ipu_convert import IpuConversionParams
from ipu_tensorflow_addons.saved_model_tool.converter import Int64Conversion
from ipu_tensorflow_addons.saved_model_tool.saved_model_test_utils import \
    ModelForTest

disable_v2_behavior()


class TestSavedmodel(ModelForTest):
  def create(self):
    x = array_ops.placeholder(dtypes.int64, [8, 8], name="x")
    w0 = variable_scope.get_variable("w0", shape=[8, 8], dtype=dtypes.int64)
    x = math_ops.matmul(w0, x)

    w1 = variable_scope.get_variable("w1", shape=[8, 8], dtype=dtypes.int64)
    x = math_ops.matmul(w1, x)
    x = relu(x)
    y = math_ops.reduce_sum(x)
    return y


class LongIntConvertTestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    self.test_model = TestSavedmodel(freeze=True)
    self.test_graph_def = self.test_model.graph_def
    self.test_signature_def = self.test_model.signature_def
    self._ATTR_TYPE_ = ["T", 'SrcT', 'DstT', 'Tindices', 'TI', 'dtype']

  def test_longint_convert(self):
    params = IpuConversionParams()
    graph_def, _ = Int64Conversion(params).apply(self.test_graph_def,
                                                 self.test_signature_def)
    for node in graph_def.node:
      for _attr in node.attr:
        if _attr in self._ATTR_TYPE_ and node.attr[_attr].type:
          self.assertTrue(node.attr[_attr].type != types_pb2.DT_INT64)
        if _attr in "value" and node.attr[_attr].tensor.dtype:
          self.assertTrue(node.attr[_attr].tensor.dtype != types_pb2.DT_INT64)

  def test_no_longint_convert(self):
    params = IpuConversionParams(int64_to_int32_conversion=False)
    graph_def, _ = Int64Conversion(params).apply(self.test_graph_def,
                                                 self.test_signature_def)

    for node in graph_def.node:
      if node.name in ['x', 'w0', 'w1', 'MatMul', 'MatMul1']:
        for _attr in node.attr:
          if _attr in self._ATTR_TYPE_ and node.attr[_attr].type:
            self.assertTrue(node.attr[_attr].type == types_pb2.DT_INT64)
          if _attr in "value" and node.attr[_attr].tensor.dtype:
            self.assertTrue(
                node.attr[_attr].tensor.dtype == types_pb2.DT_INT64)


if __name__ == '__main__':
  test.main()
