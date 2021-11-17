# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
Add IPU device and XLA annotations for nodes not in excluded_nodes list
"""
import re
from tensorflow.core.framework import attr_value_pb2
from ipu_tensorflow_addons.saved_model_tool.converter import Converter


class IPUPlacement(Converter):
  def __init__(self, param):
    if param.excluded_nodes is None:
      self._excluded_nodes = list()
    elif not isinstance(param.excluded_nodes, list):
      raise ValueError("excluded_nodes should be a list")
    else:
      self._excluded_nodes = param.excluded_nodes
    self._ipu_placement = param.ipu_placement

  def apply(self, graph_def, signature_def):
    if self._ipu_placement:
      return self._do_ipu_placement(graph_def), signature_def
    return graph_def, signature_def

  @staticmethod
  def _add_ipu_scope(node):
    node.device = '/device:IPU:0'
    node.attr['_XlaCompile'].CopyFrom(attr_value_pb2.AttrValue(b=True))
    node.attr['_XlaScope'].CopyFrom(
        attr_value_pb2.AttrValue(s='jit_scope_ipu_0'.encode()))
    node.attr['_XlaSeparateCompiledGradients'].CopyFrom(
        attr_value_pb2.AttrValue(b=False))

  def _do_ipu_placement(self, graph_def):
    for _, node in enumerate(graph_def.node):
      if self._should_do_placement(node):
        self._add_ipu_scope(node)

    return graph_def

  def _should_do_placement(self, node):
    if any(re.search(pattern, node.name) for pattern in self._excluded_nodes):
      return False
    if hasattr(node, 'device') and node.op != 'Placeholder':
      return True

    return False
