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
Replace gelu function with 'IpuGelu' operator
"""
import re
from tensorflow.core.framework import graph_pb2
from ipu_tensorflow_addons.saved_model_tool.converter import Converter
from ipu_tensorflow_addons.saved_model_tool.converter.utils import NODES_TYPE_LIST


class GeluReplacement(Converter):
  def __init__(self, param):
    if param.gelu_replacement:
      if not isinstance(param.gelu_replacement, dict):
        raise ValueError("gelu_replacement should be a dict.")
      else:
        if any(key not in param.gelu_replacement for key in
               ["nodes", "node_as_gelu_input", "node_use_gelu_output"]):
          raise ValueError(
              "nodes, node_as_gelu_input, node_use_gelu_output must be set.")
        else:
          self._gelu_nodes = param.gelu_replacement['nodes']
          self._node_as_gelu_input = param.gelu_replacement[
              'node_as_gelu_input']
          self._node_use_gelu_output = param.gelu_replacement[
              'node_use_gelu_output']
    self._gelu_replacement = param.gelu_replacement

  def apply(self, graph_def, signature_def):
    if self._gelu_replacement:
      return self._do_replace_gelu(graph_def), signature_def
    return graph_def, signature_def

  def _do_replace_gelu(self, graph_def):
    target_graph_def = graph_pb2.GraphDef()
    target_graph_def.versions.CopyFrom(graph_def.versions)

    # Copy every node except the nodes from the original gelu function.
    for node in graph_def.node:
      gelu_node_found = any(re.search(p, node.name) for p in self._gelu_nodes)
      if gelu_node_found:
        continue
      new_node = target_graph_def.node.add()
      new_node.op = node.op
      new_node.name = node.name
      new_node.input.extend(node.input)
      for attr in list(node.attr):
        new_node.attr[attr].CopyFrom(node.attr[attr])

    # Insert Gelu node.
    ipu_gelu_node_name = None
    replaced_gelu = list()
    for node in target_graph_def.node:
      for node_pattern in self._node_as_gelu_input:
        if re.search(node_pattern,
                     node.name) and node.name not in replaced_gelu:
          ipu_gelu_node = target_graph_def.node.add()
          ipu_gelu_node.op = 'IpuGelu'
          ipu_gelu_node.name = node.name + "/IpuGelu"
          replaced_gelu.append(ipu_gelu_node.name)
          ipu_gelu_node.input.extend([node.name])
          node_type = None
          for attr in node.attr:
            if node.attr[attr].type and attr in NODES_TYPE_LIST:
              node_type = node.attr[attr].type
          if node_type is None:
            raise ValueError(
                f"{ipu_gelu_node.name} can not get type from {node.name}.")
          ipu_gelu_node.attr["dtype"].type = node_type
          ipu_gelu_node_name = ipu_gelu_node.name
          break

    # Connect Gelu node.
    for node in target_graph_def.node:
      for node_pattern in self._node_use_gelu_output:
        if re.search(node_pattern, node.name):
          for idx, _input in enumerate(node.input):
            if any(re.search(pattern, _input) for pattern in self._gelu_nodes):
              node.input[idx] = ipu_gelu_node_name
          break
    return target_graph_def
