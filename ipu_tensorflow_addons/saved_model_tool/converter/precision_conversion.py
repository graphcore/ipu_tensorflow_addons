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
"""
Convert the precision of model between FP16 and FP32
Keep the precision of nodes in precision_conversion_excluded_nodes
"""
import re
from collections import defaultdict

from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import graph_pb2
from ipu_tensorflow_addons.saved_model_tool.converter import Converter
from ipu_tensorflow_addons.saved_model_tool.converter.utils import FLOAT_TYPE_LIST, ATTR_TYPE_LIST, NODES_TYPE_LIST, INPUT_NODES_TYPE_LIST
from ipu_tensorflow_addons.saved_model_tool.converter.utils import str_to_dtype


class PrecisionConversion(Converter):
  def __init__(self, param):
    self._precision_mode = param.precision_mode
    if param.precision_conversion_excluded_nodes is None:
      self._skip_list = list()
    elif not isinstance(param.precision_conversion_excluded_nodes, list):
      raise ValueError("precision_conversion_excluded_nodes should be a list")
    else:
      self._skip_list = param.precision_conversion_excluded_nodes

  def apply(self, graph_def, signature_def):
    # Firstly, convert those nodes in graph_def which have float/half type calculation and data
    # Second, for searching type-diff between nodes. travel the graph_def and insert cast op
    # In most cast, tensorflow op would store data-type in attr ['T', 'Tparam', 'dtype', 'DstT', 'SrcT']
    # For more details about tf.Node_def(), refer to https://www.tensorflow.org/api_docs/python/tf/compat/v1/NodeDef
    # and https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/supported_ops.html
    if self._precision_mode:
      graph_def = self._convert_precision_mode(graph_def)
      graph_def = self._insert_cast(graph_def)
    return graph_def, signature_def

  def _should_do_convert(self, node):
    if any(re.search(pattern, node.name) for pattern in self._skip_list):
      return False
    return True

  def _has_float_type(self, attr, node):
    if attr in ATTR_TYPE_LIST and node.attr[attr].type in FLOAT_TYPE_LIST:
      return True
    return False

  def _tensor_name_to_node_name(self, tensor_name):
    if ":" in tensor_name:
      tensor_name = tensor_name.spilt(":")[0]
    if "^" in tensor_name:
      tensor_name = tensor_name.replace('^', '')
    return tensor_name

  def _convert_precision_mode(self, graph_def):
    dst_dtype = str_to_dtype(self._precision_mode)
    target_graph_def = graph_pb2.GraphDef()
    target_graph_def.versions.CopyFrom(graph_def.versions)

    for node in graph_def.node:
      new_node = target_graph_def.node.add()
      new_node.op = node.op
      new_node.name = node.name
      new_node.input.extend(node.input)
      new_node.device = node.device

      for attr in node.attr:
        if (self._has_float_type(attr, node)
            and self._should_do_convert(node)):
          node.attr[attr].type = dst_dtype
        elif attr in ["value"]:
          tensor = node.attr[attr].tensor
          if (tensor.dtype in FLOAT_TYPE_LIST
              and self._should_do_convert(node)):
            float_val = tensor_util.MakeNdarray(node.attr[attr].tensor)
            new_node.attr[attr].tensor.CopyFrom(
                tensor_util.make_tensor_proto(float_val, dtype=dst_dtype))
            continue

        new_node.attr[attr].CopyFrom(node.attr[attr])

    return target_graph_def

  def _insert_cast(self, graph_def):

    node_dict = {node.name: node for node in graph_def.node}
    insert_dict = defaultdict(list)
    for node in graph_def.node:
      self_type = None
      input_type = None
      for attr in node.attr:
        if node.attr[attr].type and attr in NODES_TYPE_LIST:
          self_type = node.attr[attr].type
      for idx, input_name in enumerate(node.input):
        input_node = node_dict[self._tensor_name_to_node_name(input_name)]
        for attr in input_node.attr:
          if input_node.attr[attr].type and attr in INPUT_NODES_TYPE_LIST:
            input_type = input_node.attr[attr].type
        if input_type == types_pb2.DT_HALF and self_type == types_pb2.DT_FLOAT:
          insert_dict[node.name].append((idx, "FP16", "FP32"))
        if input_type == types_pb2.DT_FLOAT and self_type == types_pb2.DT_HALF:
          insert_dict[node.name].append((idx, "FP32", "FP16"))

    target_graph_def = graph_pb2.GraphDef()
    target_graph_def.versions.CopyFrom(graph_def.versions)

    for node in graph_def.node:
      new_node = target_graph_def.node.add()
      new_node.op = node.op
      new_node.name = node.name
      new_node.device = node.device
      new_node.input.extend(node.input)

      for attr in node.attr:
        new_node.attr[attr].CopyFrom(node.attr[attr])

      if node.name in insert_dict:
        for input_idx, src_type, dst_type in insert_dict[node.name]:
          cast_node = target_graph_def.node.add()
          cast_node.op = 'Cast'
          cast_node.name = node.name + '/CastInsertion_' + str(input_idx)
          cast_node.input.extend([new_node.input[input_idx]])
          cast_node.device = node.device
          cast_node.attr['SrcT'].type = str_to_dtype(src_type)
          cast_node.attr['DstT'].type = str_to_dtype(dst_type)
          # re-connect the corresponding node pair
          new_node.input[input_idx] = cast_node.name
    return target_graph_def
