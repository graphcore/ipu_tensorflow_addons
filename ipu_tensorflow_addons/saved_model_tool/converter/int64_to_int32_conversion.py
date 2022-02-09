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
Convert the precision of model from int64 to int32.
"""

import numpy as np
from tensorflow import make_ndarray
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util

from tensorflow.core.framework import attr_value_pb2, types_pb2
from ipu_tensorflow_addons.saved_model_tool.converter import Converter

_ATTR_TYPE = ["T", 'SrcT', 'DstT', 'Tindices', 'TI']


class Int64Conversion(Converter):
  def __init__(self, param):
    self.int64_to_int32_conversion = param.int64_to_int32_conversion

  def apply(self, graph_def, signature_def):
    if self.int64_to_int32_conversion:
      return self._do_int64_to_int32_conversion(graph_def), signature_def
    return graph_def, signature_def

  def _do_int64_to_int32_conversion(self, graph_def):
    for node in graph_def.node:
      for attr in node.attr:
        if attr in _ATTR_TYPE and node.attr[attr].type == types_pb2.DT_INT64:
          node.attr[attr].type = types_pb2.DT_INT32
        if attr == "dtype" and node.attr[attr].type == types_pb2.DT_INT64:
          node.attr['dtype'].CopyFrom(
              attr_value_pb2.AttrValue(type=types_pb2.DT_INT32))
        if attr == "value" and node.attr[
            'value'].tensor.dtype == types_pb2.DT_INT64:
          values = make_ndarray(node.attr['value'].tensor).astype(np.int32)
          node.attr['value'].tensor.CopyFrom(
              tensor_util.make_tensor_proto(values, dtype=dtypes.int32))
    return graph_def
