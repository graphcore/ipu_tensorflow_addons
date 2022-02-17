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
"""
Add IPU device and XLA annotations for nodes not in excluded_nodes list
"""
import re
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.saved_model import utils
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.saved_model import signature_def_utils
from ipu_tensorflow_addons.saved_model_tool.converter.converter import Converter
from ipu_tensorflow_addons.saved_model_tool.converter.utils import add_ipu_scope, get_edge_tensor, split_graph_by_device_placement
from ipu_tensorflow_addons.saved_model_tool.converter.utils import tensor_name_to_placehoder_name


class IPUPlacement(Converter):
  def __init__(self, param):
    if param.excluded_nodes is None:
      self._excluded_nodes = list()
    elif not isinstance(param.excluded_nodes, list):
      raise ValueError("excluded_nodes should be a list.")
    else:
      self._excluded_nodes = param.excluded_nodes
    self._ipu_placement = param.ipu_placement
    self._remove_excluded_nodes = param.remove_excluded_nodes

  def apply(self, graph_def, signature_def):
    if self._ipu_placement:
      graph_def = self._do_ipu_placement(graph_def)
      if self._remove_excluded_nodes:
        ipu_graph_def, cpu_graph_def = split_graph_by_device_placement(
            graph_def)
        edge_tensors = get_edge_tensor(cpu_graph_def, ipu_graph_def, True)
        return self._do_remove_excluded_nodes(edge_tensors, signature_def,
                                              ipu_graph_def)
    return graph_def, signature_def

  def _modify_signature_inputs(self, org_signature_def, input_keys,
                               input_tensors):
    signature_inputs = {
        n: utils.build_tensor_info(t)
        for n, t in zip(input_keys, input_tensors)
    }
    return signature_def_utils.build_signature_def(
        signature_inputs, org_signature_def.outputs,
        org_signature_def.method_name)

  def _do_remove_excluded_nodes(self, edge_tensors_dict, org_signature_def,
                                ipu_graph_def):
    # Build new graph.
    with session.Session(graph=ops.Graph()) as sess:
      sess.graph.as_default()
      placehoder_name = []
      placehoder_tensor = []
      # Build placeholder.
      for name, tensor in edge_tensors_dict.items():
        _name = tensor_name_to_placehoder_name(name)
        _tensor = array_ops.placeholder(tensor.dtype,
                                        shape=tensor.shape,
                                        name=_name)
        placehoder_name.append(_name)
        placehoder_tensor.append(_tensor)

      input_map = dict(zip(placehoder_name, placehoder_tensor))
      output_tensor_names = list()
      for key in org_signature_def.outputs:
        output_tensor_names.append(org_signature_def.outputs[key].name)
      importer.import_graph_def(ipu_graph_def,
                                name="",
                                input_map=input_map,
                                return_elements=output_tensor_names)

      new_signature_def = self._modify_signature_inputs(
          org_signature_def, placehoder_name, placehoder_tensor)
      new_graph_def = ops.get_default_graph().as_graph_def()

    for node in new_graph_def.node:
      if node.op != "Placeholder":
        add_ipu_scope(node, '/device:IPU:0')

    return new_graph_def, new_signature_def

  def _do_ipu_placement(self, graph_def):
    for _, node in enumerate(graph_def.node):
      if self._should_do_placement(node):
        add_ipu_scope(node, '/device:IPU:0')

    return graph_def

  def _should_do_placement(self, node):
    if any(re.search(pattern, node.name) for pattern in self._excluded_nodes):
      return False
    if hasattr(node, 'device') and node.op != 'Placeholder':
      return True

    return False
