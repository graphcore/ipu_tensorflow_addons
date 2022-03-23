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
from itertools import accumulate, chain

import numpy as np
from networkx.classes import function as F
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python import layers
from tensorflow.python.client import session
from tensorflow.python.framework import (constant_op, dtypes, ops, test_util)
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.ops import (array_ops, init_ops, nn, variable_scope,
                                   variables)
from tensorflow.python.platform import test
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python import ipu
from tensorflow.python.saved_model.utils import build_tensor_info

from ipu_tensorflow_addons.saved_model_tool.converter.autograph import (
    Node, TFv1Graph)
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.utils import \
    convert_graph_def_to_graph
from ipu_tensorflow_addons.saved_model_tool.converter.utils import \
    get_tensor_shape, tf_type_to_numpy
from ipu_tensorflow_addons.saved_model_tool.saved_model_test_utils import \
    ModelForTest


def bucket_idx(bucket_range_list, ord_idx):
  for idx, b in enumerate(bucket_range_list):
    if ord_idx < b:
      return idx
  return 0


def get_ticket_for_ops(solutions):
  return [*accumulate(solutions, lambda x, y: x + y)]


def add_stage_info(graph: TFv1Graph, bucket_list):
  for idx, node in enumerate(graph.topology_sort()):
    node.add_attr(pipeline_stage=bucket_idx(bucket_list, idx))


def first_try(graph: TFv1Graph):
  pipline_stage_num = 2
  quanters = graph.size_of_nodes() // pipline_stage_num + 1
  prev_split_pos = [quanters] * pipline_stage_num
  bucket_range_list = get_ticket_for_ops(prev_split_pos)
  add_stage_info(graph, bucket_range_list)


def mapping_strategy_to_node(pipeline_strategy_list, agraph: TFv1Graph):
  for stage, node_list in enumerate(pipeline_strategy_list):
    for n in node_list:
      agraph.nxg.nodes[n]["node"].add_attr(pipeline_stage=stage)


def concat_computational_stages(computational_stages, *args):
  x = args
  for call_fun in computational_stages:
    x = call_fun(*x)
  return x


def evaluate_by_cpu(graph_def, feed_dict, output_names):
  with ops.Graph().as_default():
    out_phs = import_graph_def(graph_def,
                               name="",
                               return_elements=output_names)
    with session.Session() as sess:
      out_nd_array = sess.run(out_phs, feed_dict)
  return out_nd_array


def run_by_ipu(graph_def, computational_stages, device_mapping,
               input_tensor_names, batch_size):
  tfgraph = convert_graph_def_to_graph(graph_def)
  inputs_shape_and_dtype = [(
      name,
      get_tensor_shape(tfgraph.get_tensor_by_name(name))[1:],
      tf_type_to_numpy(tfgraph.get_tensor_by_name(name).dtype),
  ) for name in input_tensor_names]
  cfg = ipu.config.IPUConfig()
  cfg.auto_select_ipus = len(set(device_mapping))
  cfg.configure_ipu_system()

  dataset = Dataset.from_tensors(
      tuple([
          np.random.randint(10, size=shape).astype(dtype)
          for _, shape, dtype, in inputs_shape_and_dtype
      ]))
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
  infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
  outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
  pipeline_op = ipu.pipelining_ops.pipeline(
      computational_stages=computational_stages,
      device_mapping=device_mapping,
      gradient_accumulation_count=len(computational_stages),
      repeat_count=1,
      inputs=[],
      infeed_queue=infeed_queue,
      outfeed_queue=outfeed_queue,
      name='pipeline_op')

  with ops.device("/device:IPU:0"):
    r = ipu.ipu_compiler.compile(lambda: pipeline_op, inputs=[])

  with session.Session() as sess:
    sess.run(variables.global_variables_initializer())
    sess.run(infeed_queue.initializer)
    sess.run(r)
    out_nd_array = sess.run(outfeed_queue.dequeue())
  return out_nd_array


# pylint: disable=abstract-method
class ModifySignatureModel(ModelForTest):
  def add_output_to_signature(self, tensor_names):
    tensors = [self._graph.get_tensor_by_name(name) for name in tensor_names]
    signature_added_output = meta_graph_pb2.SignatureDef()
    signature_added_output.CopyFrom(self.signature_def)
    for tensor in tensors:
      signature_added_output.outputs[tensor.name].CopyFrom(
          build_tensor_info(tensor))
    return signature_added_output


class InputNeedPassedOnModel(ModifySignatureModel):
  def create(self):
    o1 = array_ops.placeholder(shape=[None, 2],
                               dtype=dtypes.float32)  # shape (2, )
    o2 = array_ops.placeholder(shape=[None, 5],
                               dtype=dtypes.float32)  # shape (5, )
    o3 = array_ops.placeholder(shape=[None, 4],
                               dtype=dtypes.float32)  # shape (4, )
    o4 = o3

    for i in range(5):
      with variable_scope.variable_scope(f"left/unit_{i}"):
        o1 = layers.dense(
            inputs=o1,
            units=2,
            activation=nn.relu,
            kernel_initializer=init_ops.truncated_normal_initializer(),
            bias_initializer=init_ops.ones_initializer())

    for i in range(3):
      with variable_scope.variable_scope(f"right/unit_{i}"):
        o2 = layers.dense(
            inputs=o2,
            units=2,
            activation=nn.relu,
            kernel_initializer=init_ops.truncated_normal_initializer(),
            bias_initializer=init_ops.ones_initializer())

    for i in range(6):
      with variable_scope.variable_scope(f"middle/unit_{i}"):
        o3 = layers.dense(
            inputs=o3,
            units=4,
            activation=nn.relu,
            kernel_initializer=init_ops.truncated_normal_initializer(),
            bias_initializer=init_ops.ones_initializer())

    ok = array_ops.concat([o1, o2, o3], axis=1, name='concat')
    o = ok

    with variable_scope.variable_scope("res"):
      for i in range(3):
        with variable_scope.variable_scope(f"unit_{i}"):
          o = layers.dense(
              inputs=o,
              units=4,
              activation=nn.relu,
              kernel_initializer=init_ops.truncated_normal_initializer(),
              bias_initializer=init_ops.ones_initializer())
      o = o + o4
    return o


class MultiInputNeedPassedOn(ModifySignatureModel):
  def create(self):
    o1 = array_ops.placeholder(
        shape=[None, 2],
        dtype=dtypes.float32,
    )  # shape (2, )
    o2 = array_ops.placeholder(
        shape=[None, 5],
        dtype=dtypes.float32,
    )  # shape (5, )
    o3 = array_ops.placeholder(
        shape=[None, 4],
        dtype=dtypes.float32,
    )  # shape (4, )
    o4 = o3
    unused = array_ops.placeholder(shape=[None, 4],
                                   dtype=dtypes.float32,
                                   name="unused")  # shape (4, )

    for i in range(5):
      with variable_scope.variable_scope(f"left/unit_{i}"):
        o1 = layers.dense(
            inputs=o1,
            units=2,
            activation=nn.relu,
            kernel_initializer=init_ops.truncated_normal_initializer(),
            bias_initializer=init_ops.ones_initializer())

    for i in range(3):
      with variable_scope.variable_scope(f"right/unit_{i}"):
        o2 = layers.dense(
            inputs=o2,
            units=2,
            activation=nn.relu,
            kernel_initializer=init_ops.truncated_normal_initializer(),
            bias_initializer=init_ops.ones_initializer())

    for i in range(6):
      with variable_scope.variable_scope(f"middle/unit_{i}"):
        o3 = layers.dense(
            inputs=o3,
            units=4,
            activation=nn.relu,
            kernel_initializer=init_ops.truncated_normal_initializer(),
            bias_initializer=init_ops.ones_initializer())

    ok = array_ops.concat([o1, o2, o3], axis=1, name='concat')
    o = ok

    with variable_scope.variable_scope("res"):
      for i in range(3):
        with variable_scope.variable_scope(f"unit_{i}"):
          o = layers.dense(
              inputs=o,
              units=8,
              activation=nn.relu,
              kernel_initializer=init_ops.truncated_normal_initializer(),
              bias_initializer=init_ops.ones_initializer())
          o = o + ok
      with variable_scope.variable_scope(f"down"):
        o = layers.dense(
            inputs=o,
            units=4,
            activation=nn.relu,
            kernel_initializer=init_ops.truncated_normal_initializer(),
            bias_initializer=init_ops.ones_initializer())

      o = o + o4
    return o, unused


class TFv1GraphTestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    super().setUp()
    self.test_model = InputNeedPassedOnModel(freeze=True)
    self.tfgraph = convert_graph_def_to_graph(self.test_model.graph_def)
    self.agGraph = TFv1Graph(self.tfgraph,
                             signature_def=self.test_model.signature_def)
    self.tempfile_path = self.get_temp_dir()

  def test_construct_graph(self):
    self.assertFalse(F.is_empty(self.agGraph.nxg))
    self.assertTrue(F.is_directed(self.agGraph.nxg))
    self.assertEqual(
        self.agGraph.nxg.nodes["res/unit_1/dense/bias"]["node"].name,
        "res/unit_1/dense/bias")
    self.assertEqual(
        self.agGraph.nxg.nodes["res/unit_1/dense/bias"]["node"].op_type,
        "Const")
    self.assertListEqual(
        self.agGraph.nxg.nodes["res/unit_1/dense/bias"]["node"].inputs, [])
    self.assertEqual(
        self.agGraph.nxg.nodes["res/unit_1/dense/bias"]
        ["node"].outputs[0].name, 'res/unit_1/dense/bias:0')
    self.assertEqual(
        self.agGraph.nxg.nodes["res/unit_1/dense/bias"]
        ["node"].outputs[0].dtype, 'float32')
    self.assertEqual(
        len(self.agGraph.nxg.nodes["res/unit_1/dense/bias"]["node"].outputs),
        1)
    self.assertListEqual(
        self.agGraph.nxg.nodes["res/unit_1/dense/bias"]
        ["node"].outputs[0].shape, [4])
    self.assertListEqual(list(self.agGraph.nxg["res/unit_1/dense/bias/read"]),
                         ['res/unit_1/dense/BiasAdd'])

  def test__gen_input2node_mapping(self):
    # pylint: disable=protected-access
    all_node_dict = self.agGraph._gen_input2node_mapping()
    no_identity_node_dict = self.agGraph._gen_input2node_mapping(
        exclude_op_type={
            "Identity",
        })
    only_include_matmul_node_dict = self.agGraph._gen_input2node_mapping(
        include_op_type={
            "MatMul",
        })

    self.assertIn("res/unit_2/dense/bias/read",
                  set(chain(*list(all_node_dict.values()))))
    self.assertNotIn("res/unit_2/dense/bias/read",
                     set(chain(*list(no_identity_node_dict.values()))))
    self.assertIn("left/unit_3/dense/MatMul",
                  set(chain(*list(only_include_matmul_node_dict.values()))))
    self.assertNotIn("left/unit_1/dense/BiasAdd",
                     set(chain(*list(only_include_matmul_node_dict.values()))))
    self.assertListEqual(all_node_dict['right/unit_0/dense/Relu:0'],
                         ['right/unit_1/dense/MatMul'])

  def test__gen_output2node_mapping(self):
    # pylint: disable=protected-access
    all_node_dict = self.agGraph._gen_output2node_mapping()
    no_identity_node_dict = self.agGraph._gen_output2node_mapping(
        exclude_op_type={
            "Identity",
        })
    only_include_matmul_node_dict = self.agGraph._gen_output2node_mapping(
        include_op_type={
            "MatMul",
        })
    self.assertIn("res/unit_2/dense/bias/read",
                  set(chain(list(all_node_dict.values()))))
    self.assertNotIn("res/unit_2/dense/bias/read",
                     set(chain(list(no_identity_node_dict.values()))))
    self.assertIn("left/unit_3/dense/MatMul",
                  set(chain(list(only_include_matmul_node_dict.values()))))
    self.assertNotIn("left/unit_1/dense/BiasAdd",
                     set(chain(list(only_include_matmul_node_dict.values()))))
    self.assertEqual(all_node_dict['right/unit_0/dense/Relu:0'],
                     'right/unit_0/dense/Relu')

  def test__gen_name2node_mapping(self):
    # pylint: disable=protected-access
    all_node_dict = self.agGraph._gen_name2node_mapping()
    no_identity_node_dict = self.agGraph._gen_name2node_mapping(
        exclude_op_type={
            "Identity",
        })
    only_include_matmul_node_dict = self.agGraph._gen_name2node_mapping(
        include_op_type={
            "MatMul",
        })
    self.assertIn("res/unit_2/dense/bias/read", all_node_dict)
    self.assertNotIn("res/unit_2/dense/bias/read", no_identity_node_dict)
    self.assertIn("left/unit_3/dense/MatMul", only_include_matmul_node_dict)
    self.assertNotIn("left/unit_1/dense/BiasAdd",
                     only_include_matmul_node_dict)
    self.assertIsInstance(all_node_dict['right/unit_0/dense/Relu'], Node)
    self.assertEqual(all_node_dict['right/unit_0/dense/Relu'].name,
                     'right/unit_0/dense/Relu')

  def test_nodes(self):
    self.assertNotIn("res/unit_2/dense/bias/read", self.agGraph.nodes)
    self.assertIn("res/unit_2/dense/BiasAdd", self.agGraph.nodes)

  def test_set_amp(self):
    self.agGraph.set_amp(0.3)
    self.assertEqual(self.agGraph.amp, 0.3)
    self.agGraph.set_amp(0.6)
    self.assertEqual(self.agGraph.amp, 0.6)

  def test_name_mapping4pb_node(self):
    untested = self.agGraph.name_mapping4pb_node()
    self.assertIn("res/unit_2/dense/bias/read", untested)

  def test_subgraph(self):
    subgraph_namestr = [
        'Placeholder',
        'Placeholder_1',
        'left/unit_0/dense/kernel',
        'left/unit_0/dense/kernel/read',
        'left/unit_0/dense/bias',
        'left/unit_0/dense/bias/read',
        'left/unit_0/dense/MatMul',
        'left/unit_0/dense/BiasAdd',
        'left/unit_0/dense/Relu',
    ]
    (new_graph_def, input_list_final, subgraph_node_name_wo_placeholder,
     subgraph_node_name_w_placeholder) = \
      self.agGraph._subgraph(subgraph_namestr) # pylint: disable=protected-access

    self.assertSetEqual(input_list_final, {'Placeholder:0'})
    for sub_node_str in subgraph_namestr:
      self.assertIn(sub_node_str, subgraph_node_name_w_placeholder)
      if not sub_node_str.startswith("Placeholder"):
        self.assertIn(sub_node_str, subgraph_node_name_wo_placeholder)
      else:
        self.assertNotIn(sub_node_str, subgraph_node_name_wo_placeholder)

    self.assertNotIn("left/unit_3/dense/MatMul",
                     subgraph_node_name_w_placeholder)
    self.assertNotIn("left/unit_3/dense/MatMul",
                     subgraph_node_name_wo_placeholder)

    for node in new_graph_def.node:
      self.assertProtoEquals(
          node, self.agGraph.nxg.nodes[node.name]["node"].node_def)

  def test_pipelined_num_of_ipus(self):
    first_try(self.agGraph)
    stage_of_func_list, device_info = self.agGraph.pipelined(num_of_ipu=2)
    input_0 = constant_op.constant([[1., 2.]])
    input_1 = constant_op.constant([[1., 2., 3., 4., 5.]])
    input_2 = constant_op.constant([[1., 2., 3., 4.]])
    outcome = stage_of_func_list[1](*stage_of_func_list[0]
                                    (input_0, input_1, input_2))
    self.assertEqual(len(outcome), 1)
    self.assertListEqual(outcome[0].shape.as_list(), [1, 4])
    self.assertEqual(outcome[0].name, 'res/add:0')
    self.assertListEqual(device_info, [0, 1])

  def test_pipelined_wo_num_of_ipus(self):
    first_try(self.agGraph)
    with self.assertRaisesRegex(ValueError,
                                "No pipline device info specified"):
      _, _ = self.agGraph.pipelined()

  def test_save_pipeline_config(self):
    first_try(self.agGraph)
    self.agGraph.save_pipeline_config(
        f"{self.tempfile_path}/first_try.pipeconfig")
    with open(f"{self.tempfile_path}/first_try.pipeconfig", 'r') as f:
      _pipeline_info_dict = json.load(f)
    _pipeline_info = _pipeline_info_dict["pipeline_mapping"]
    _device_info = _pipeline_info_dict["device_mapping"]
    for node_dict in self.agGraph.nxg.nodes.values():
      self.assertEqual(node_dict["node"].pipeline_stage,
                       _pipeline_info[node_dict["node"].name])
    self.assertListEqual(_device_info, [0, 1])

  def test_read_pipeline_config(self):
    first_try(self.agGraph)
    self.agGraph.save_pipeline_config(
        f"{self.tempfile_path}/first_try.pipeconfig")
    new_graph = TFv1Graph(self.agGraph.graph)
    new_graph.read_pipeline_config(
        f"{self.tempfile_path}/first_try.pipeconfig")
    for name, node_dict in self.agGraph.nxg.nodes.items():
      self.assertEqual(node_dict["node"].pipeline_stage,
                       new_graph.nxg.nodes[name]["node"].pipeline_stage)
    self.assertListEqual(new_graph.device_info, [0, 1])

  def test_read_pipeline_config_wo_device_info(self):
    first_try(self.agGraph)
    self.agGraph.save_pipeline_config(
        f"{self.tempfile_path}/first_try.pipeconfig")
    new_graph = TFv1Graph(self.agGraph.graph)
    with open(f"{self.tempfile_path}/first_try.pipeconfig") as file:
      _info_dict = json.load(file)
      _info_dict.pop('device_mapping')
    with open(f"{self.tempfile_path}/first_try.pipeconfig", "w") as file:
      json.dump(_info_dict, file)
    new_graph.read_pipeline_config(
        f"{self.tempfile_path}/first_try.pipeconfig")
    for name, node_dict in self.agGraph.nxg.nodes.items():
      self.assertEqual(node_dict["node"].pipeline_stage,
                       new_graph.nxg.nodes[name]["node"].pipeline_stage)
    self.assertListEqual(new_graph.device_info, [0, 1])

  def test_set_and_clear_pipeline_device_info(self):
    first_try(self.agGraph)
    self.agGraph.pipelined(num_of_ipu=2)
    self.assertListEqual(self.agGraph.device_info, [0, 1])
    with self.assertRaises(ValueError):
      self.agGraph.set_pipeline_device_info([
          0,
      ])
    self.agGraph.set_pipeline_device_info([0, 0])
    self.assertListEqual(self.agGraph.device_info, [0, 0])
    self.agGraph.clear_device_info()
    self.assertIsNone(self.agGraph.device_info)


class TFv1GraphInputNeedPassedOnModelTestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    super().setUp()
    self.input_need_passed_on_model = InputNeedPassedOnModel(freeze=True)
    self.pipeline_strategy_list = [
        [
            'Placeholder_1', 'Placeholder_2', 'right/unit_0/dense/kernel',
            'right/unit_0/dense/kernel/read', 'right/unit_0/dense/bias',
            'right/unit_0/dense/bias/read', 'right/unit_0/dense/MatMul',
            'right/unit_0/dense/BiasAdd', 'right/unit_0/dense/Relu',
            'right/unit_1/dense/kernel', 'right/unit_1/dense/kernel/read',
            'right/unit_1/dense/bias', 'right/unit_1/dense/bias/read',
            'right/unit_1/dense/MatMul', 'right/unit_1/dense/BiasAdd',
            'right/unit_1/dense/Relu', 'right/unit_2/dense/kernel',
            'right/unit_2/dense/kernel/read', 'right/unit_2/dense/bias',
            'right/unit_2/dense/bias/read', 'right/unit_2/dense/MatMul',
            'right/unit_2/dense/BiasAdd', 'middle/unit_0/dense/kernel',
            'middle/unit_0/dense/kernel/read', 'middle/unit_0/dense/bias',
            'middle/unit_0/dense/bias/read', 'middle/unit_0/dense/MatMul',
            'middle/unit_0/dense/BiasAdd', 'middle/unit_0/dense/Relu',
            'middle/unit_1/dense/kernel', 'middle/unit_1/dense/kernel/read',
            'middle/unit_1/dense/bias', 'middle/unit_1/dense/bias/read',
            'middle/unit_1/dense/MatMul', 'middle/unit_1/dense/BiasAdd',
            'middle/unit_1/dense/Relu', 'middle/unit_2/dense/kernel',
            'middle/unit_2/dense/kernel/read', 'middle/unit_2/dense/bias',
            'middle/unit_2/dense/bias/read', 'middle/unit_2/dense/MatMul',
            'middle/unit_2/dense/BiasAdd', 'middle/unit_2/dense/Relu',
            'middle/unit_3/dense/kernel', 'middle/unit_3/dense/kernel/read',
            'middle/unit_3/dense/bias', 'middle/unit_3/dense/bias/read',
            'middle/unit_3/dense/MatMul', 'middle/unit_3/dense/BiasAdd',
            'middle/unit_3/dense/Relu', 'middle/unit_4/dense/kernel',
            'middle/unit_4/dense/kernel/read', 'middle/unit_4/dense/bias',
            'middle/unit_4/dense/bias/read', 'middle/unit_4/dense/MatMul',
            'middle/unit_4/dense/BiasAdd', 'middle/unit_4/dense/Relu',
            'middle/unit_5/dense/kernel', 'middle/unit_5/dense/kernel/read',
            'middle/unit_5/dense/bias', 'middle/unit_5/dense/bias/read',
            'middle/unit_5/dense/MatMul', 'middle/unit_5/dense/BiasAdd',
            'middle/unit_5/dense/Relu'
        ],
        [
            'Placeholder', 'Placeholder_2', 'left/unit_0/dense/kernel',
            'left/unit_0/dense/kernel/read', 'left/unit_0/dense/bias',
            'left/unit_0/dense/bias/read', 'left/unit_0/dense/MatMul',
            'left/unit_0/dense/BiasAdd', 'left/unit_0/dense/Relu',
            'left/unit_1/dense/kernel', 'left/unit_1/dense/kernel/read',
            'left/unit_1/dense/bias', 'left/unit_1/dense/bias/read',
            'left/unit_1/dense/MatMul', 'left/unit_1/dense/BiasAdd',
            'left/unit_1/dense/Relu', 'left/unit_2/dense/kernel',
            'left/unit_2/dense/kernel/read', 'left/unit_2/dense/bias',
            'left/unit_2/dense/bias/read', 'left/unit_2/dense/MatMul',
            'left/unit_2/dense/BiasAdd', 'left/unit_2/dense/Relu',
            'left/unit_3/dense/kernel', 'left/unit_3/dense/kernel/read',
            'left/unit_3/dense/bias', 'left/unit_3/dense/bias/read',
            'left/unit_3/dense/MatMul', 'left/unit_3/dense/BiasAdd',
            'left/unit_3/dense/Relu', 'left/unit_4/dense/kernel',
            'left/unit_4/dense/kernel/read', 'left/unit_4/dense/bias',
            'left/unit_4/dense/bias/read', 'left/unit_4/dense/MatMul',
            'left/unit_4/dense/BiasAdd', 'left/unit_4/dense/Relu',
            'right/unit_2/dense/Relu', 'concat/axis', 'concat',
            'res/unit_0/dense/kernel', 'res/unit_0/dense/kernel/read',
            'res/unit_0/dense/bias', 'res/unit_0/dense/bias/read',
            'res/unit_0/dense/MatMul', 'res/unit_0/dense/BiasAdd',
            'res/unit_0/dense/Relu', 'res/unit_1/dense/kernel',
            'res/unit_1/dense/kernel/read', 'res/unit_1/dense/bias',
            'res/unit_1/dense/bias/read', 'res/unit_1/dense/MatMul',
            'res/unit_1/dense/BiasAdd', 'res/unit_1/dense/Relu',
            'res/unit_2/dense/kernel', 'res/unit_2/dense/kernel/read',
            'res/unit_2/dense/bias', 'res/unit_2/dense/bias/read',
            'res/unit_2/dense/MatMul', 'res/unit_2/dense/BiasAdd',
            'res/unit_2/dense/Relu'
        ], ['res/add']
    ]

  def test_pipelined(self):
    tfgraph = convert_graph_def_to_graph(
        self.input_need_passed_on_model.graph_def)
    agGraph = TFv1Graph(
        tfgraph, signature_def=self.input_need_passed_on_model.signature_def)
    mapping_strategy_to_node(self.pipeline_strategy_list, agGraph)
    computational_stages, device_mapping = agGraph.pipelined(
        num_of_ipu=len(self.pipeline_strategy_list))
    self.assertListEqual(device_mapping, [0, 1, 2])

    with ops.Graph().as_default() as compute_graph:
      batch_size = 3
      input_ph1 = array_ops.placeholder(shape=[batch_size, 2],
                                        dtype=dtypes.float32)  # shape (2, )
      input_ph2 = array_ops.placeholder(shape=[batch_size, 5],
                                        dtype=dtypes.float32)  # shape (5, )
      input_ph3 = array_ops.placeholder(shape=[batch_size, 4],
                                        dtype=dtypes.float32)  # shape (4, )
      res_add = concat_computational_stages(computational_stages, input_ph1,
                                            input_ph2, input_ph3)
      compute_graph_def = compute_graph.as_graph_def()
      feed_dict = {
          input_ph1.name:
          np.random.randint(10, size=input_ph1.shape).astype(np.float32),
          input_ph2.name:
          np.random.randint(10, size=input_ph2.shape).astype(np.float32),
          input_ph3.name:
          np.random.randint(10, size=input_ph3.shape).astype(np.float32),
      }
      self.assertListEqual(get_tensor_shape(res_add[0]), [batch_size, 4])

    compute_out_ndarray = evaluate_by_cpu(compute_graph_def, feed_dict,
                                          ["res/add:0"])
    out_ndarray = evaluate_by_cpu(self.input_need_passed_on_model.graph_def,
                                  feed_dict, ["res/add:0"])
    self.assertAllClose(out_ndarray, compute_out_ndarray)
    run_by_ipu(self.input_need_passed_on_model.graph_def, computational_stages,
               [0, 1, 0], list(feed_dict.keys()), batch_size)

  def test_middle_output(self):
    middle_output_tensor = ["middle/unit_1/dense/Relu:0"]
    tfgraph = convert_graph_def_to_graph(
        self.input_need_passed_on_model.graph_def)
    agGraph = TFv1Graph(
        tfgraph,
        signature_def=self.input_need_passed_on_model.add_output_to_signature(
            middle_output_tensor))
    mapping_strategy_to_node(self.pipeline_strategy_list, agGraph)
    computational_stages, device_mapping = agGraph.pipelined(
        num_of_ipu=len(self.pipeline_strategy_list))
    self.assertListEqual(device_mapping, [0, 1, 2])

    with ops.Graph().as_default() as compute_graph:
      batch_size = 3
      input_ph1 = array_ops.placeholder(shape=[batch_size, 2],
                                        dtype=dtypes.float32)  # shape (2, )
      input_ph2 = array_ops.placeholder(shape=[batch_size, 5],
                                        dtype=dtypes.float32)  # shape (5, )
      input_ph3 = array_ops.placeholder(shape=[batch_size, 4],
                                        dtype=dtypes.float32)  # shape (4, )
      res_add = concat_computational_stages(computational_stages, input_ph1,
                                            input_ph2, input_ph3)
      compute_graph_def = compute_graph.as_graph_def()
      feed_dict = {
          input_ph1.name:
          np.random.randint(10, size=input_ph1.shape).astype(np.float32),
          input_ph2.name:
          np.random.randint(10, size=input_ph2.shape).astype(np.float32),
          input_ph3.name:
          np.random.randint(10, size=input_ph3.shape).astype(np.float32),
      }
      self.assertListEqual(get_tensor_shape(res_add[0]), [batch_size, 4])
      self.assertListEqual(get_tensor_shape(res_add[1]), [batch_size, 4])

    actual_output_tensor_names = ["res/add:0"] + middle_output_tensor
    compute_out_ndarray = evaluate_by_cpu(compute_graph_def, feed_dict,
                                          actual_output_tensor_names)
    out_ndarray = evaluate_by_cpu(self.input_need_passed_on_model.graph_def,
                                  feed_dict, actual_output_tensor_names)
    self.assertAllClose(out_ndarray, compute_out_ndarray)
    output_from_ipu = run_by_ipu(self.input_need_passed_on_model.graph_def,
                                 computational_stages, [0, 1, 0],
                                 list(feed_dict.keys()), batch_size)
    self.assertEqual(len(output_from_ipu), len(actual_output_tensor_names))


class TFv1GraphMultiInputNeedPassedOnTestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    super().setUp()
    self.multi_input_need_passed_on_model = MultiInputNeedPassedOn(freeze=True)
    self.pipeline_strategy_list = [
        [
            'Placeholder_1', 'Placeholder_2', 'right/unit_0/dense/kernel',
            'right/unit_0/dense/kernel/read', 'right/unit_0/dense/bias',
            'right/unit_0/dense/bias/read', 'right/unit_0/dense/MatMul',
            'right/unit_0/dense/BiasAdd', 'right/unit_0/dense/Relu',
            'right/unit_1/dense/kernel', 'right/unit_1/dense/kernel/read',
            'right/unit_1/dense/bias', 'right/unit_1/dense/bias/read',
            'right/unit_1/dense/MatMul', 'right/unit_1/dense/BiasAdd',
            'right/unit_1/dense/Relu', 'right/unit_2/dense/kernel',
            'right/unit_2/dense/kernel/read', 'right/unit_2/dense/bias',
            'right/unit_2/dense/bias/read', 'right/unit_2/dense/MatMul',
            'right/unit_2/dense/BiasAdd', 'middle/unit_0/dense/kernel',
            'middle/unit_0/dense/kernel/read', 'middle/unit_0/dense/bias',
            'middle/unit_0/dense/bias/read', 'middle/unit_0/dense/MatMul',
            'middle/unit_0/dense/BiasAdd', 'middle/unit_0/dense/Relu',
            'middle/unit_1/dense/kernel', 'middle/unit_1/dense/kernel/read',
            'middle/unit_1/dense/bias', 'middle/unit_1/dense/bias/read',
            'middle/unit_1/dense/MatMul', 'middle/unit_1/dense/BiasAdd',
            'middle/unit_1/dense/Relu', 'middle/unit_2/dense/kernel',
            'middle/unit_2/dense/kernel/read', 'middle/unit_2/dense/bias',
            'middle/unit_2/dense/bias/read', 'middle/unit_2/dense/MatMul',
            'middle/unit_2/dense/BiasAdd', 'middle/unit_2/dense/Relu',
            'middle/unit_3/dense/kernel', 'middle/unit_3/dense/kernel/read',
            'middle/unit_3/dense/bias', 'middle/unit_3/dense/bias/read',
            'middle/unit_3/dense/MatMul', 'middle/unit_3/dense/BiasAdd',
            'middle/unit_3/dense/Relu', 'middle/unit_4/dense/kernel',
            'middle/unit_4/dense/kernel/read', 'middle/unit_4/dense/bias',
            'middle/unit_4/dense/bias/read', 'middle/unit_4/dense/MatMul',
            'middle/unit_4/dense/BiasAdd', 'middle/unit_4/dense/Relu',
            'middle/unit_5/dense/kernel', 'middle/unit_5/dense/kernel/read',
            'middle/unit_5/dense/bias', 'middle/unit_5/dense/bias/read',
            'middle/unit_5/dense/MatMul', 'middle/unit_5/dense/BiasAdd',
            'middle/unit_5/dense/Relu'
        ],
        [
            'Placeholder',
            'Placeholder_2',
            "unused",
            'left/unit_0/dense/kernel',
            'left/unit_0/dense/kernel/read',
            'left/unit_0/dense/bias',
            'left/unit_0/dense/bias/read',
            'left/unit_0/dense/MatMul',
            'left/unit_0/dense/BiasAdd',
            'left/unit_0/dense/Relu',
            'left/unit_1/dense/kernel',
            'left/unit_1/dense/kernel/read',
            'left/unit_1/dense/bias',
            'left/unit_1/dense/bias/read',
            'left/unit_1/dense/MatMul',
            'left/unit_1/dense/BiasAdd',
            'left/unit_1/dense/Relu',
            'left/unit_2/dense/kernel',
            'left/unit_2/dense/kernel/read',
            'left/unit_2/dense/bias',
            'left/unit_2/dense/bias/read',
            'left/unit_2/dense/MatMul',
            'left/unit_2/dense/BiasAdd',
            'left/unit_2/dense/Relu',
            'left/unit_3/dense/kernel',
            'left/unit_3/dense/kernel/read',
            'left/unit_3/dense/bias',
            'left/unit_3/dense/bias/read',
            'left/unit_3/dense/MatMul',
            'left/unit_3/dense/BiasAdd',
            'left/unit_3/dense/Relu',
            'left/unit_4/dense/kernel',
            'left/unit_4/dense/kernel/read',
            'left/unit_4/dense/bias',
            'left/unit_4/dense/bias/read',
            'left/unit_4/dense/MatMul',
            'left/unit_4/dense/BiasAdd',
            'left/unit_4/dense/Relu',
            'right/unit_2/dense/Relu',
            'concat/axis',
            'concat',
            'res/unit_0/dense/kernel',
            'res/unit_0/dense/kernel/read',
            'res/unit_0/dense/bias',
            'res/unit_0/dense/bias/read',
            'res/unit_0/dense/MatMul',
            'res/unit_0/dense/BiasAdd',
            'res/unit_0/dense/Relu',
            'res/unit_0/add',
        ],
        [
            'res/unit_1/dense/kernel',
            'res/unit_1/dense/kernel/read',
            'res/unit_1/dense/bias',
            'res/unit_1/dense/bias/read',
            'res/unit_1/dense/MatMul',
            'res/unit_1/dense/BiasAdd',
            'res/unit_1/dense/Relu',
            'res/unit_1/add',
            'left/unit_3/dense/bias/read',
        ],
        [
            'res/unit_2/dense/kernel', 'res/unit_2/dense/kernel/read',
            'res/unit_2/dense/bias', 'res/unit_2/dense/bias/read',
            'res/unit_2/dense/MatMul', 'res/unit_2/dense/BiasAdd',
            'res/unit_2/dense/Relu', 'res/unit_2/add', 'res/down/dense/kernel',
            'res/down/dense/kernel/read', 'res/down/dense/bias',
            'res/down/dense/bias/read', 'res/down/dense/MatMul',
            'res/down/dense/BiasAdd', 'res/down/dense/Relu', 'res/add'
        ]
    ]

  def test_pipelined(self):
    middle_output_tensor = ["left/unit_4/dense/BiasAdd:0"]
    tfgraph = convert_graph_def_to_graph(
        self.multi_input_need_passed_on_model.graph_def)
    agGraph = TFv1Graph(tfgraph,
                        signature_def=self.multi_input_need_passed_on_model.
                        add_output_to_signature(middle_output_tensor))
    mapping_strategy_to_node(self.pipeline_strategy_list, agGraph)
    computational_stages, device_mapping = agGraph.pipelined(
        num_of_ipu=len(self.pipeline_strategy_list))
    self.assertListEqual(device_mapping, [0, 1, 2, 3])

    with ops.Graph().as_default() as compute_graph:
      batch_size = 3
      input_ph1 = array_ops.placeholder(shape=[batch_size, 2],
                                        dtype=dtypes.float32)  # shape (2, )
      input_ph2 = array_ops.placeholder(shape=[batch_size, 5],
                                        dtype=dtypes.float32)  # shape (5, )
      input_ph3 = array_ops.placeholder(shape=[batch_size, 4],
                                        dtype=dtypes.float32)  # shape (4, )
      unused = array_ops.placeholder(shape=[batch_size, 4],
                                     dtype=dtypes.float32,
                                     name="unused")  # shape (4, )
      res_add = concat_computational_stages(computational_stages, input_ph1,
                                            input_ph2, input_ph3, unused)
      compute_graph_def = compute_graph.as_graph_def()
      feed_dict = {
          input_ph1.name:
          np.random.randint(10, size=input_ph1.shape).astype(np.float32),
          input_ph2.name:
          np.random.randint(10, size=input_ph2.shape).astype(np.float32),
          input_ph3.name:
          np.random.randint(10, size=input_ph3.shape).astype(np.float32),
          input_ph3.name:
          np.random.randint(10, size=input_ph3.shape).astype(np.float32),
          unused.name:
          np.random.randint(10, size=input_ph3.shape).astype(np.float32),
      }
      self.assertListEqual(get_tensor_shape(res_add[0]), [batch_size, 2])
      self.assertListEqual(get_tensor_shape(res_add[1]), [batch_size, 4])
      self.assertListEqual(get_tensor_shape(res_add[2]), [batch_size, 4])

    actual_output_tensor_names = (["res/add:0", "unused:0"] +
                                  middle_output_tensor)
    compute_out_ndarray = evaluate_by_cpu(compute_graph_def, feed_dict,
                                          actual_output_tensor_names)
    out_ndarray = evaluate_by_cpu(
        self.multi_input_need_passed_on_model.graph_def, feed_dict,
        actual_output_tensor_names)
    self.assertAllClose(out_ndarray, compute_out_ndarray)
    output_from_ipu = run_by_ipu(
        self.multi_input_need_passed_on_model.graph_def, computational_stages,
        device_mapping, list(feed_dict.keys()), batch_size)
    self.assertEqual(len(output_from_ipu), len(actual_output_tensor_names))


if __name__ == '__main__':
  test.main()
