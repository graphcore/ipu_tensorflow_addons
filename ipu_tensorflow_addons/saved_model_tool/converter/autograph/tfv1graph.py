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
import json
from collections import defaultdict
from itertools import chain, cycle
from typing import Dict, Set, List
from abc import ABCMeta, abstractmethod

import networkx as nx

from tensorflow.python import ops
from tensorflow.python.framework.importer import import_graph_def
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.utils import (
    analyze_pb_inputs_outputs, import_from_graph, tf_type_to_str_type)
from ipu_tensorflow_addons.saved_model_tool.converter.utils import (
    node_name_from_tensor_name, get_tensor_shape)

DEFAULT_EXCLUDE_OPTYPE_SET = {"Const", "Identity", "Pack"}


class TensorInfo(object):
  def __init__(self, name="ops", shape=(), dtype="float32"):
    self.name = name
    self.shape = shape
    self.dtype = dtype

  def __str__(self):
    return (f"<TensorInfo: name={self.name}, "
            f"shape={self.shape}, dtype={self.dtype}>")

  def __repr__(self):
    return (f"<TensorInfo: name={self.name}, "
            f"shape={self.shape}, dtype={self.dtype}>")


class Input(TensorInfo):
  pass


class Output(TensorInfo):
  pass


class Node(object):
  def __init__(self,
               name="ops",
               inputs: List[Input] = None,
               outputs: List[Output] = None,
               op_type: str = "",
               **attributes):

    self.inputs = inputs if inputs else []
    self.outputs = outputs if outputs else []
    self.op_type = op_type
    self.name = name
    self.add_attr(**attributes)

  def __repr__(self):
    strs = '<Node: ' + ','.join([
        f"{an}={av}"
        for an, av in self.__dict__.items() if an not in ('inputs', 'outputs')
    ]) + ">"
    return strs

  def __str__(self):
    strs = '<Node: ' + ','.join([
        f"{an}={av}"
        for an, av in self.__dict__.items() if an not in ('inputs', 'outputs')
    ]) + ">"
    return strs

  def add_attr(self, **attr):
    for a, v in attr.items():
      setattr(self, a, v)


class Graph(metaclass=ABCMeta):
  def __init__(self):
    self.nxg = nx.DiGraph()
    self._inputs = None
    self._outputs = None

  @abstractmethod
  def pipelined(self, num_of_ipu=None):
    pass

  @property
  def inputs(self):
    return self._inputs

  @property
  def outputs(self):
    return self._outputs

  def topology_sort(self, key=None):  # pylint: disable=unused-argument
    for node_name in nx.topological_sort(self.nxg):
      yield self.nxg.nodes[node_name]


class TFv1Graph(Graph):
  def __init__(self, pb_tf_graph: ops.Graph, signature_def=None):
    super().__init__()
    self.node_list = []
    self.graph = import_from_graph(pb_tf_graph)
    self._get_input_output(signature_def)
    self._construct_nxgraph()
    self.amp = 0.6
    self.device_info = None
    self._stage_list = None
    self._nodes_dict = self._get_nodes_dict()

  def _get_input_output(self, signature_def=None):
    if not signature_def:
      input_ops, output_ops = analyze_pb_inputs_outputs(self.graph)
      inputs_list = list(chain(*[list(ops.outputs) for ops in input_ops]))
      outputs_list = list(chain(*[list(ops.outputs) for ops in output_ops]))
    else:
      input_tensors_names = sorted(
          [i.name for i in signature_def.inputs.values()])
      output_tensors_names = sorted(
          [i.name for i in signature_def.outputs.values()])
      inputs_list = [
          self.graph.get_tensor_by_name(n) for n in input_tensors_names
      ]
      outputs_list = [
          self.graph.get_tensor_by_name(n) for n in output_tensors_names
      ]

    self._inputs = [
        Input(name=i.name,
              shape=self._parse_shape(i),
              dtype=self._parse_dtype(i)) for i in inputs_list
    ]
    self._outputs = [
        Output(name=i.name,
               shape=self._parse_shape(i),
               dtype=self._parse_dtype(i)) for i in outputs_list
    ]

  def _construct_node_list(self):
    for op in self.graph.get_operations():
      inputs = [
          Input(name=i.name,
                shape=self._parse_shape(i),
                dtype=self._parse_dtype(i)) for i in op.inputs
      ]
      outputs = [
          Output(name=i.name,
                 shape=self._parse_shape(i),
                 dtype=self._parse_dtype(i)) for i in op.outputs
      ]
      self.node_list.append(
          Node(
              name=op.name,
              inputs=inputs,
              outputs=outputs,
              op_type=op.type,
              node_def=op.node_def,
              op_def=op.op_def,
              control_inputs=op.control_inputs,
              device=op.device,
          ))

  def _gen_input2node_mapping(self,
                              exclude_op_type: Set = None,
                              include_op_type: Set = None):
    if not exclude_op_type:
      exclude_op_type = set()
    if not include_op_type:
      include_op_type = set()
    input_mapping = defaultdict(list)
    for n in self.node_list:
      if n.op_type not in exclude_op_type:
        if not include_op_type:
          for inp in n.inputs:
            input_mapping[inp.name].append(n.name)
        elif n.op_type in include_op_type:
          for inp in n.inputs:
            input_mapping[inp.name].append(n.name)

    return input_mapping

  def _gen_output2node_mapping(self,
                               exclude_op_type: Set = None,
                               include_op_type: Set = None):
    if not exclude_op_type:
      exclude_op_type = set()
    if not include_op_type:
      include_op_type = set()
    output_mapping = {}
    for n in self.node_list:
      if n.op_type not in exclude_op_type:
        if not include_op_type:
          for out in n.outputs:
            output_mapping[out.name] = n.name
        elif n.op_type in include_op_type:
          for out in n.outputs:
            output_mapping[out.name] = n.name
    return output_mapping

  def _gen_name2node_mapping(self,
                             exclude_op_type: Set = None,
                             include_op_type: Set = None):
    if not exclude_op_type:
      exclude_op_type = set()
    if not include_op_type:
      include_op_type = set()
    name_mapping = {}
    for n in self.node_list:
      if n.op_type not in exclude_op_type:
        if not include_op_type:
          name_mapping[n.name] = n
        elif n.op_type in include_op_type:
          name_mapping[n.name] = n
    return name_mapping

  def _parse_shape(self, tensor):
    return None if not tensor.shape else get_tensor_shape(tensor)

  def _parse_dtype(self, tensor):
    return tf_type_to_str_type(tensor.dtype)

  def _construct_nxgraph(self):
    self._construct_node_list()
    input2node = self._gen_input2node_mapping()
    self.nxg.add_nodes_from([(n.name, {"node": n}) for n in self.node_list])
    for node_name, node_value in self.nxg.nodes.items():
      for o in node_value["node"].outputs:
        if input2node.get(o.name, []):
          for in_node_name in input2node[o.name]:
            self.nxg.add_edge(node_name, in_node_name, name=in_node_name)

  def _get_nodes_dict(self, exclude_op_type: Set = None):
    if not exclude_op_type:
      exclude_op_type = DEFAULT_EXCLUDE_OPTYPE_SET
    return {
        node_name: node_value["node"]
        for node_name, node_value in self.nxg.nodes.items()
        if node_value["node"].op_type not in exclude_op_type
    }

  @property
  def nodes(self) -> Dict[str, Node]:
    """Nodes without `Const`, `Identity`, and `Pack`

    Returns:
      Dict[str, Node]: Dictionary of name and node.
    """
    return self._nodes_dict

  def _check_pipeline_stage_attr(self, exclude_op_type: Set = None):
    if not exclude_op_type:
      exclude_op_type = DEFAULT_EXCLUDE_OPTYPE_SET
    for node_name, node_value in self._get_nodes_dict(
        exclude_op_type=exclude_op_type).items():
      if not hasattr(node_value, "pipeline_stage"):
        raise AttributeError(
            f"The node `{node_name}` of {self.__class__.__name__} "
            "is not fully set with `pipeline_stage`.")
    return True

  def _package_all_const(self, node: Node, pipeline_stage: int):
    """Find the indentity constant input of node and set them to same pipeline stage id.

    Args:
      node (Node): Operation of graph node.
      pipeline_stage (int): Pipeline stage id.
    """
    output2node = self._gen_output2node_mapping()

    def _package_all_const_help(node, pipeline_stage):
      if node.op_type not in DEFAULT_EXCLUDE_OPTYPE_SET:
        return

      if node.op_type == "Const":
        node.add_attr(pipeline_stage=pipeline_stage)
        return

      node.add_attr(pipeline_stage=pipeline_stage)

      for inp in node.inputs:
        _package_all_const_help(self.nxg.nodes[output2node[inp.name]]["node"],
                                pipeline_stage)

    for inp in node.inputs:
      _package_all_const_help(self.nxg.nodes[output2node[inp.name]]["node"],
                              pipeline_stage)

  def _package_indentity_from_output(self, output_tensor):
    output2node = self._gen_output2node_mapping()
    indentity_node_cur = self.nxg.nodes[output2node[
        output_tensor.name]]["node"]
    if indentity_node_cur.op_type != "Identity":
      return
    identity_nodes_names = []
    while not hasattr(indentity_node_cur, "pipeline_stage"):
      identity_nodes_names.append(indentity_node_cur.name)
      indentity_node_cur = self.nxg.nodes[output2node[
          indentity_node_cur.inputs[0].name]]["node"]

    pipeline_stage = indentity_node_cur.pipeline_stage
    for id_name in identity_nodes_names:
      self.nxg.nodes[id_name]["node"].add_attr(pipeline_stage=pipeline_stage)

  def _package_all_identity_from_output(self):
    for out in self._outputs:
      self._package_indentity_from_output(out)

  def _package_const_ops_with_node(self):
    self._check_pipeline_stage_attr()
    for _, node_value in self._get_nodes_dict().items():
      self._package_all_const(node_value, node_value.pipeline_stage)
    self._package_all_identity_from_output()
    self._check_pipeline_stage_attr(exclude_op_type=set())

  def _group_by_pipline_stage(self):
    # pylint: disable=line-too-long
    """Group operations by pipeline stage.

    Returns:
      list: A list containing the list of operation names for each pipeline stage. [[a, b], [c, d], ...]
    """
    stages = defaultdict(list)
    self._package_const_ops_with_node()
    for node_name, node_value in self.nxg.nodes.items():
      stages[node_value["node"].pipeline_stage].append(node_name)

    self._stage_list = [stages[l] for l in range(len(stages))]

  def set_pipeline_device_info(self, device_info):
    if self._stage_list and len(device_info) != len(self._stage_list):
      raise ValueError(f"The length of device_info ({len(device_info)}) must "
                       f"be equal to the number of pipeline stages "
                       f"({len(self._stage_list)}).")
    self.device_info = device_info

  def set_amp(self, amp):
    self.amp = amp

  def name_mapping4pb_node(self):
    return {node.name: node for node in self.graph.as_graph_def().node}

  def _subgraph(self, node_names):
    # pylint:disable=line-too-long
    r"""Generate subgraph of TensorFlow protobuf.

    If the outdegree of a certain node is more than one and one of those span the 2 subgraphs,
    the output list will consider 2 different output.

    For example:
      If the graph is like this:

    ```
        A
        | \
        |  |  subgraph A
        v  |
        B  |
      --+--+--
        |  |
        D  |
        |  |
        |  |  subgraph B
        v /
        C

      This function will consider the output of the subgraph A is ["A:0", "B:0"], which means:

        A
        | \
        |  |  subgraph A
        v  |
        B  A'   <-- output list

      --------

        D  A'
        |  |
        |  |  subgraph B
        v /
        C       <-- output list
    ```

    Args:
      node_names (List[str]): The node name list.

    Returns:
      protobuf: Subgraph protobuf.
      set: The input arguments of subgraph.
      set: The names of non-placeholder nodes in the subgraph.
           This is for inferring the inputs and outputs of each pipeline stage.
      set: The names of placeholder nodes in the subgraph.
           This is for inferring `return_elements` in `tf.import_graph_def`.
    """

    intput_set = set()
    output_set = set()
    new_graph_def = ops.Graph().as_graph_def()
    subgraph_node_name_without_placeholder = set()
    subgraph_node_name_set = set()

    for name in node_names:
      if self.nxg.nodes[name]["node"].op_type != "Placeholder":
        intput_set.update(
            set(inp.name for inp in self.nxg.nodes[name]["node"].inputs))
        output_set.update(
            set(out.name for out in self.nxg.nodes[name]["node"].outputs))
        subgraph_node_name_without_placeholder.add(name)
      subgraph_node_name_set.add(name)
      add_ptr = new_graph_def.node.add()
      add_ptr.CopyFrom(self.nxg.nodes[name]["node"].node_def)

    intput_set -= output_set
    return new_graph_def, intput_set, subgraph_node_name_without_placeholder, subgraph_node_name_set

  def _inputs_and_outputs_info_for_tf_function(self, subgraph_inputs_set_list,
                                               subgraph_node_set_list,
                                               output_expected_for_last_stage):
    # pylint:disable=line-too-long
    r"""Complete the input and output info for every pipeline stage function.

    Although the subgraph has the input tensor, the input and output info for the pipeline stages may still be incomplete.
    For example:

    ```text
        A
        | \
        |  |  subgraph A
        v  |
        B  |
      --+--+--
        |  |
        E  |  subgraph B
        |  |
      --+--+--
        |  |
        D  |
        |  |
        |  |  subgraph C
        v /
        C

        ||
        ||
        \/

        A
        | \
        |  |  subgraph A
        v  |
        B  A'
      --+--+--
        B' A'
        |  |
        |  |  subgraph B
        |  |
        E  A'
      --+--+--
        E  A'
        |  |  subgraph C
        D  A'
        |  |
        v /
        C
    ```

    The outputs for subgraph A are [B, A], which the `subgraph` function can give the correct answer for.
    The outputs for subgraph B are [E, A], which the `subgraph` function can't give the correct answer for.
    It will not consider operation A because it is unused.

    The inputs for subgraph C are [D, A].

    This function determines the complete inputs and outputs for each pipeline stage.

    Args:
      subgraph_inputs_set_list (List[set[str]]): The set of inputs for each stage.
      subgraph_node_set_list (List[set[str]]): The set of nodes in the subgraph for each stage.
      output_expected_for_last_stage (List[str]): The list of the expected outputs for the final stage.

    Returns:
      List[List[List[str]]]: the inputs for each stage, in the form [[[<unused_inputs_in_subgraph>], [<used_inputs_in_subgraph>]], ...].
      List[List[List[str]]]: the outputs for each stage, in the form [[[<unused_outputs_in_subgraph>], [<used_outputs_in_subgraph>]], ...].
    """
    stage_inputs_packed_list = []

    def unsed_nodes(node):
      return (
          node_name_from_tensor_name(node) not in subgraph_node_set_list[-1]
          and node not in subgraph_inputs_set_list[-1])

    unused_in_subgraph_last_stage = list(
        filter(unsed_nodes, output_expected_for_last_stage))

    subgraph_output_last_stage = [
        i for i in output_expected_for_last_stage
        if i not in unused_in_subgraph_last_stage
    ]

    subgraph_node_set_list_reversed = reversed(subgraph_node_set_list)
    subgraph_inputs_set_list_reversed = reversed(subgraph_inputs_set_list)

    previous_stage_output_list = output_expected_for_last_stage
    for subgraph_nodes, subgraph_input_set in zip(
        subgraph_node_set_list_reversed, subgraph_inputs_set_list_reversed):
      unused = [
          out for out in previous_stage_output_list
          if node_name_from_tensor_name(out) not in subgraph_nodes
          and node_name_from_tensor_name(out) not in subgraph_input_set
      ]
      stage_inputs_packed = [unused, list(subgraph_input_set)]
      stage_inputs_packed_list.append(stage_inputs_packed)
      previous_stage_output_list = unused + list(subgraph_input_set)

    stage_inputs_packed_list.reverse()

    stage_output_packed_list = stage_inputs_packed_list[1:] + [[
        unused_in_subgraph_last_stage, subgraph_output_last_stage
    ]]

    return stage_inputs_packed_list, stage_output_packed_list

  def pipeline_stage_funcs(self):
    # pylint:disable=line-too-long
    """Creates a list of pipeline stage functions to be used in pipeline_ops.

    Generate subgraphs in the form [[<node_names>, <node_names>], [...], ...] from `list_of_stages`.
    Determine the inputs and outputs for each stage from the subgraphs.
    Generate TensorFlow functions from the inputs, outputs, and subgraphs.

    Returns:
      List[Callable]: computational stage.
    """
    input_data_feed_name = [inp.name for inp in self.inputs]
    expected_output_name = [out.name for out in self.outputs]

    stage_py_func_list = []
    subgraphs_node_set_for_stage_inputs_completion_list = []
    subgraph_inputs_set_list = []
    subgraphs_list = []
    subgraph_node_set_list = []

    for stage in self._stage_list:
      (new_graph_def, subgraph_inputs_set,
       subgraph_node_set_for_stage_input_completion,
       subgraph_node_set) = self._subgraph(stage)

      subgraphs_node_set_for_stage_inputs_completion_list.append(
          subgraph_node_set_for_stage_input_completion)
      subgraphs_list.append(new_graph_def)
      subgraph_inputs_set_list.append(subgraph_inputs_set)
      subgraph_node_set_list.append(subgraph_node_set)

    (stage_inputs_list,
     stage_output_list) = \
      self._inputs_and_outputs_info_for_tf_function(
          subgraph_inputs_set_list,
          subgraphs_node_set_for_stage_inputs_completion_list,
          expected_output_name
      )

    untopology_node = set(
        chain(*stage_inputs_list[0])) - set(input_data_feed_name)
    if untopology_node:
      raise ValueError(
          f"The nodes {untopology_node} are not in topology order "
          "or can not be found in the stage list.")

    first_stage_input = input_data_feed_name
    for subgraph, stage_inputs_packed, stage_outputs_packed, node_set in zip(
        subgraphs_list, stage_inputs_list, stage_output_list,
        subgraph_node_set_list):

      stage_outputs = list(chain(*stage_outputs_packed))
      input_unused_in_subgraph, subgraph_inputs = stage_inputs_packed
      subgraph_outputs = [
          o for o in stage_outputs if o.split(":")[0] in node_set
      ]
      stage_inputs = (input_unused_in_subgraph + subgraph_inputs
                      if not first_stage_input else first_stage_input)
      tf_fun = stage_function_factory(subgraph, stage_inputs, subgraph_outputs,
                                      subgraph_inputs, stage_outputs)

      first_stage_input = None

      stage_py_func_list.append(tf_fun)

    return stage_py_func_list

  def _gen_device_info_auto(self, num_of_ipu):
    ipu_num_gen = cycle(range(num_of_ipu))
    if self._stage_list:
      device_info = [next(ipu_num_gen) for _ in range(len(self._stage_list))]
    else:
      device_info = [next(ipu_num_gen) for _ in range(num_of_ipu)]
    return device_info

  def auto_assign_ipu_number(self, num_of_ipu):
    """Generate a ordered list of integers used as `device_mapping`.

    Args:
        num_of_ipu (int): The ipu number to be used.

    Raises:
        ValueError: if stage_list is empty.
    """
    if self.device_info:
      return
    if not self._stage_list:
      raise ValueError("The `stage_list` should not be empty.")
    self.device_info = self._gen_device_info_auto(num_of_ipu)

  def pipelined(self, num_of_ipu: int = None):
    # pylint: disable=line-too-long
    """Pipelined a graph to a list of functions.

    Args:
        num_of_ipu (int, optional): The number of ipu used. Defaults to None.

    Raises:
        ValueError: If missing device info for pipelined graph.
        ValueError: If the length of specified device info is not equal to pipeline stages.

    Returns:
        List[Callable]: The computational function for each pipeline stage.
        List[Int]: The device mapping for each pipeline stage.
    """
    if not self.device_info and not num_of_ipu:
      raise ValueError(
          "No pipline device info specified. "
          "Please set it using `TFv1Graph.set_pipeline_device_info`.")

    self._group_by_pipline_stage()

    stage_of_func_list = self.pipeline_stage_funcs()

    if self.device_info and len(self.device_info) != len(stage_of_func_list):
      raise ValueError(
          f"The length of the device info given ({len(self.device_info)}) must "
          f"be equal to the number of pipeline stages "
          f"({len(stage_of_func_list)}).")

    if not self.device_info:
      self.auto_assign_ipu_number(num_of_ipu)

    self.amp = ([self.amp for _ in range(len(stage_of_func_list))]
                if isinstance(self.amp, float) else self.amp)

    return stage_of_func_list, self.device_info

  def topology_sort(self, key=None):
    # pylint: disable=line-too-long
    """Return a topology sorted list of nodes.

    Args:
        key (Callable, optional): The key function used to sort. Defaults to None.

    Returns:
        List[Node]: The topology sorted list.
    """
    topo_list = []
    for node_name in nx.algorithms.dag.topological_sort(self.nxg):
      if self.nxg.nodes[node_name][
          "node"].op_type not in DEFAULT_EXCLUDE_OPTYPE_SET:
        topo_list.append(self.nxg.nodes[node_name]["node"])
    return topo_list

  def total_size_of_nodes(self):
    return len(self.nxg.nodes)

  def size_of_nodes(self):
    # pylint: disable=line-too-long
    """The total number of nodes in the graph (excluding `Pack`, `Indentity`, and `Const` ops).

    Returns:
        int: The total number of nodes in the graph.
    """
    return len(self._get_nodes_dict())

  def _save_nodes_to_json(self, file_name, device_info=None):
    with open(file_name, "w") as file:
      pipeline_info_dict = {
          op_name: op_value_dict['node'].pipeline_stage if hasattr(
              op_value_dict['node'], 'pipeline_stage') else -1
          for op_name, op_value_dict in self.nxg.nodes.items()
      }

      device_info_list = device_info or self.device_info
      if not device_info_list:
        num_of_ipu = max(pipeline_info_dict.values()) + 1
        device_info_list = self._gen_device_info_auto(num_of_ipu)

      json.dump(
          {
              "pipeline_mapping": pipeline_info_dict,
              "device_mapping": device_info_list
          },
          file,
          indent=2)

  def _load_nodes_from_json(self, file_name):
    with open(file_name, "r") as file:
      pipe_config = json.load(file)
      if "pipeline_mapping" not in pipe_config:
        raise ValueError(
            "`pipeline_mapping` must be specified in pipeconfig file.")
      pipeline_info_dict = pipe_config["pipeline_mapping"]  # required
      device_info_list = pipe_config.get("device_mapping", None)
      if not device_info_list:
        num_of_ipu = max(pipeline_info_dict.values()) + 1
        device_info_list = self._gen_device_info_auto(num_of_ipu)

      for node_name, stageId in pipeline_info_dict.items():
        self.nxg.nodes[node_name]["node"].pipeline_stage = stageId

      self.device_info = device_info_list

  def save_pipeline_config(self, file_name, device_info=None):
    # pylint: disable=line-too-long
    """Save the pipeline configuration and device mapping to JSON.

    Args:
        file_name (str): [description]
        device_info (List[int], optional): Device mapping used in `pipelining_op`. Defaults to None.
    """
    self._package_const_ops_with_node()
    self._save_nodes_to_json(file_name, device_info)

  def read_pipeline_config(self, file_name):
    """Reading the pipeline configuration.

    Args:
        file_name (str): File name of the configuration.
    """
    self._load_nodes_from_json(file_name)

  def clear_device_info(self):
    """Clear device info."""
    self.device_info = None


def stage_function_factory(
    tfgraph,
    stage_inputs,
    subgraph_outputs,
    subgraph_inputs=None,
    stage_outputs=None,
):
  # pylint: disable=line-too-long
  """Produce a TensorFlow function for a computational stage with a python wrapper.

  Args:
    tfgraph (protobuf): The subgraph to be executed by the function.
    stage_inputs (list): All inputs to the stage (including unused inputs).
    subgraph_outputs (list): Outputs from the subgraph.
    subgraph_inputs (list, optional): Inputs which are used in the subgraph. Defaults to be equal to `stage_inputs`.
    stage_outputs (list, optional): All outputs of the stage (including unused inputs which are directly returned). Defaults to be equal to `subgraph_outputs`.
  """
  if subgraph_inputs is None:
    subgraph_inputs = stage_inputs
  if stage_outputs is None:
    stage_outputs = subgraph_outputs

  def tf_functions(*args):
    stage_inputs_map = dict(zip(stage_inputs, args))
    subgraph_input_map = {
        nam: v
        for nam, v in stage_inputs_map.items() if nam in subgraph_inputs
    }

    output = import_graph_def(tfgraph,
                              name="",
                              input_map=subgraph_input_map,
                              return_elements=subgraph_outputs)

    subgraph_output_map = dict(zip(subgraph_outputs, output))
    subgraph_output_map.update(stage_inputs_map)

    return tuple(subgraph_output_map[o] for o in stage_outputs)

  return tf_functions
