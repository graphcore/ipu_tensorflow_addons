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
Wrap graph by IPU compiler.
"""
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from ipu_tensorflow_addons.saved_model_tool.converter.converter import Converter
from ipu_tensorflow_addons.saved_model_tool.converter.utils import add_ipu_scope, get_edge_tensor, split_graph_by_device_placement
from ipu_tensorflow_addons.saved_model_tool.converter.utils import node_name_from_tensor_name, input_name_to_node_name


class IPUCompilerWrapper(Converter):
  def __init__(self, param):
    self._excluded_nodes = param.excluded_nodes
    self._remove_excluded_nodes = param.remove_excluded_nodes
    self._merge_subgraphs = param.merge_subgraphs
    self._embedded_runtime_save_config = param.embedded_runtime_save_config
    self._validate_param(param)

  def _validate_param(self, param):
    if not isinstance(param.excluded_nodes, list):
      raise TypeError("excluded_nodes must be a list.")

    if not isinstance(param.remove_excluded_nodes, bool):
      raise TypeError("remove_excluded_nodes must be a bool.")

  def _should_do_ipu_wrapper(self):
    return self._merge_subgraphs and not self._embedded_runtime_save_config

  def apply(self, graph_def, signature_def):
    if self._should_do_ipu_wrapper():
      ipu_graph_def, cpu_graph_def = split_graph_by_device_placement(graph_def)
      edge_tensors = get_edge_tensor(cpu_graph_def, ipu_graph_def, False)
      return self._wrap_graph_def(edge_tensors, signature_def, cpu_graph_def,
                                  ipu_graph_def)
    return graph_def, signature_def

  def _use_ipu_compiler(self, sess, ipu_graph_def, edge_tensor_names,
                        edge_tensors, input_signature_def):
    outputs = input_signature_def.outputs
    output_tensor_names = [outputs[key].name for key in outputs]

    output_tensor_names.sort()

    def _ipu_imported_graph_builder(edge_tensor_names, edge_tensors,
                                    output_tensor_names, ipu_graph_def):
      def _model(edge_tensors):
        input_map = dict(zip(edge_tensor_names, edge_tensors))
        return importer.import_graph_def(ipu_graph_def,
                                         name="",
                                         input_map=input_map,
                                         return_elements=output_tensor_names)

      operation = ipu.ipu_compiler.compile(_model, [edge_tensors])
      return operation

    # Wrap the IPU part of the graph in ipu_compiler.compile().
    with ops.device('/device:IPU:0'):
      results = _ipu_imported_graph_builder(edge_tensor_names, edge_tensors,
                                            output_tensor_names, ipu_graph_def)
    # Get the name of the tensors returned by ipu_compiler.compile().
    new_output_names = [t.name.split(":")[0] for t in results]
    self._output_tensors = results
    return graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(add_shapes=True), new_output_names)

  def _add_output_identity(self, graph_def, input_signature_def):
    node_dict = {node.name: node for node in graph_def.node}
    output_tensor_name = [t.name for t in input_signature_def.outputs.values()]
    output_tensor_name.sort()
    node_name_list = [
        node_name_from_tensor_name(t) for t in output_tensor_name
    ]
    # Replace name in node's input with [name]/wrapped.
    for node in node_dict.values():
      for idx, input_name in enumerate(node.input):
        node_name = input_name_to_node_name(input_name)
        if node_name in node_name_list:
          node.input[idx] = input_name.replace(node_name,
                                               node_name + '/wrapped')

    # Change the name of model output nodes to '[name]/wrapped'.
    for name in node_name_list:
      merge_name = name + '/wrapped'
      node_dict[merge_name] = node_dict[name]
      node_dict[merge_name].name = merge_name

    # To synchronize with signature.outputs, add new identity ops
    # behind the outputs created by _use_ipu_compiler.
    for idx, name in enumerate(output_tensor_name):
      node_name = node_name_from_tensor_name(name)
      node = node_dict["output" + str(idx)]
      new_indentity = graph_def.node.add()
      new_indentity.op = 'Identity'
      new_indentity.name = node_name
      new_indentity.input.extend(["output_" + str(idx)])
      add_ipu_scope(new_indentity, '/device:IPU:0')
      for attr in node.attr:
        new_indentity.attr[attr].CopyFrom(node.attr[attr])

  def _wrap_graph_def(self, edge_tensors_dict, input_signature_def,
                      cpu_graph_def, ipu_graph_def):
    # The outputs from the CPU part are the inputs to the IPU part.
    with session.Session(graph=ops.Graph()) as sess:
      sess.graph.as_default()
      edge_tensor_names = [
          edge_tensors_dict[key].name for key in edge_tensors_dict
      ]
      # Import the CPU graph.
      edge_tensor = importer.import_graph_def(
          cpu_graph_def, name="", return_elements=edge_tensor_names)

      # Import and wrap the IPU part.
      merge_graph_def = self._use_ipu_compiler(sess, ipu_graph_def,
                                               edge_tensor_names, edge_tensor,
                                               input_signature_def)
      # Because the output tensors in signature.outputs would disappear after
      # calling _use_ipu_compiler, add output tensors to the end of the
      # graph_def by referring to signature.outputs.
      self._add_output_identity(merge_graph_def, input_signature_def)

    return merge_graph_def, input_signature_def
