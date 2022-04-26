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
Test cases for utils
"""
import os

from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.contrib.learn.python.learn.estimators import constants

from ipu_tensorflow_addons.saved_model_tool import IpuConversionParams
from ipu_tensorflow_addons.saved_model_tool.converter.utils import str_to_dtype
from ipu_tensorflow_addons.saved_model_tool.converter.utils import add_ipu_scope
from ipu_tensorflow_addons.saved_model_tool.converter.utils import input_name_to_node_name
from ipu_tensorflow_addons.saved_model_tool.converter.utils import input_name_to_tensor_name
from ipu_tensorflow_addons.saved_model_tool.converter.utils import input_name_to_placeholder_name
from ipu_tensorflow_addons.saved_model_tool.converter.utils import tensor_name_to_placehoder_name
from ipu_tensorflow_addons.saved_model_tool.converter.utils import node_name_from_tensor_name
from ipu_tensorflow_addons.saved_model_tool.converter.utils import tensor_name_from_node_name
from ipu_tensorflow_addons.saved_model_tool.converter.utils import split_graph_by_device_placement
from ipu_tensorflow_addons.saved_model_tool.converter.utils import get_edge_tensor
from ipu_tensorflow_addons.saved_model_tool.converter.utils import casted_input_from_signature
from ipu_tensorflow_addons.saved_model_tool.converter.utils import casted_output_from_signature
from ipu_tensorflow_addons.saved_model_tool.converter.utils import extract_emb_setting_from_param


class TestUtils(test_util.TensorFlowTestCase):
  def test_str_to_dtype(self):
    self.assertEqual(str_to_dtype('FP16'), types_pb2.DT_HALF)
    self.assertEqual(str_to_dtype("FP32"), types_pb2.DT_FLOAT)

  def test_add_ipu_scope(self):
    node = node_def_pb2.NodeDef()
    add_ipu_scope(node, "/device:IPU:0")
    self.assertEqual(node.device, "/device:IPU:0")

  def test_input_name_to_node_name(self):
    self.assertEqual(input_name_to_node_name("node:0"), "node")
    self.assertEqual(input_name_to_node_name("node"), "node")

  def test_input_name_to_tensor_name(self):
    self.assertEqual(input_name_to_tensor_name("node:0"), "node:0")
    self.assertEqual(input_name_to_tensor_name("node"), "node:0")

  def test_input_name_to_placeholder_name(self):
    self.assertEqual(input_name_to_placeholder_name("node:0"), "node_0")

  def test_tensor_name_to_placehoder_name(self):
    self.assertEqual(tensor_name_to_placehoder_name("node:0"), "node")
    self.assertEqual(tensor_name_to_placehoder_name("node:1"), "node_1")

  def test_node_name_from_tensor_name(self):
    self.assertEqual(node_name_from_tensor_name("node:1"), "node")

  def test_tensor_name_from_node_name(self):
    self.assertEqual(tensor_name_from_node_name("node", 1), "node:1")

  def test_split_graph_by_device_placement(self):
    with ops.Graph().as_default() as graph:
      with ops.device("cpu"):
        a = array_ops.placeholder(dtypes.float32)
        b = array_ops.placeholder(dtypes.float32)
      with ops.device("/device:IPU:0"):
        _ = a + b
    graph_def = graph.as_graph_def()
    ipu_graph_def, cpu_graph_def = split_graph_by_device_placement(graph_def)
    self.assertEqual(ipu_graph_def.node[0].device, "/device:IPU:0")
    for n in cpu_graph_def.node:
      self.assertEqual(n.device, "/device:CPU:0")

  def test_get_edge_tensor(self):
    with ops.Graph().as_default() as graph:
      with ops.device("cpu"):
        a = array_ops.placeholder(dtypes.float32)
        b = array_ops.placeholder(dtypes.float32)
      with ops.device("/device:IPU:0"):
        _ = a + b
    graph_def = graph.as_graph_def()
    ipu_graph_def, cpu_graph_def = split_graph_by_device_placement(graph_def)
    edge_tensors = get_edge_tensor(cpu_graph_def, ipu_graph_def, False)
    self.assertIn('Placeholder_1:0', edge_tensors)
    self.assertIn('Placeholder:0', edge_tensors)
    self.assertNotIn('add:0', edge_tensors)

  def signature_def_examples(self):
    with ops.Graph().as_default():
      input_tensors = {
          "input_1":
          array_ops.placeholder(dtypes.float32, [None, 4], name="input_1"),
          "input_2":
          array_ops.placeholder(dtypes.float32, [None, 4], name="input_2")
      }
      output_tensors = {
          "probabilities":
          array_ops.placeholder(dtypes.float32, [None, 2],
                                name="probabilities"),
          "logits":
          array_ops.placeholder(dtypes.float32, [None, 2], name="logits"),
      }
      problem_type = constants.ProblemType.UNSPECIFIED
      signature_without_cast = (
          saved_model_export_utils.build_standardized_signature_def(
              input_tensors, output_tensors, problem_type))

    with ops.Graph().as_default():
      input_tensors = {
          "input_1":
          array_ops.placeholder(dtypes.float16, [None, 4], name="input_1"),
          "input_2":
          array_ops.placeholder(dtypes.float16, [None, 4], name="input_2")
      }
      output_tensors = {
          "probabilities":
          array_ops.placeholder(dtypes.float16, [None, 2],
                                name="probabilities"),
          "logits":
          array_ops.placeholder(dtypes.float16, [None, 2], name="logits"),
      }
      problem_type = constants.ProblemType.UNSPECIFIED
      signature_with_cast = (
          saved_model_export_utils.build_standardized_signature_def(
              input_tensors, output_tensors, problem_type))
    return signature_without_cast, signature_with_cast

  def test_casted_input_from_signature(self):
    signature_without_cast, signature_with_cast = self.signature_def_examples()
    inputs_shape_and_dtype = [("input_1:0", [None, 4], dtypes.float32),
                              ("input_2:0", [None, 4], dtypes.float32)]
    with ops.Graph().as_default() as actual_graph:
      casted_input_from_signature(inputs_shape_and_dtype,
                                  signature_without_cast,
                                  batch_size=3)

    nodes_op_type = [node.op for node in actual_graph.as_graph_def().node]
    self.assertNotIn("Cast", nodes_op_type)

    with ops.Graph().as_default() as actual_graph:
      casted_input_from_signature(inputs_shape_and_dtype,
                                  signature_with_cast,
                                  batch_size=3)

    nodes_op_type = [node.op for node in actual_graph.as_graph_def().node]
    self.assertIn("Cast", nodes_op_type)

  def test_casted_output_from_signature(self):
    signature_without_cast, signature_with_cast = self.signature_def_examples()
    outputs_shape_and_dtype = [("probabilities:0", [None, 2], dtypes.float32),
                               ("logits:0", [None, 2], dtypes.float32)]
    with ops.Graph().as_default() as actual_graph:
      _ = array_ops.placeholder(dtypes.float32, [None, 2],
                                name="probabilities")
      _ = array_ops.placeholder(dtypes.float32, [None, 2], name="logits")
      output_tensors = [
          actual_graph.get_tensor_by_name(name)
          for name in ("probabilities:0", "logits:0")
      ]
      casted_output_from_signature(output_tensors, outputs_shape_and_dtype,
                                   signature_without_cast)

    nodes_op_type = [node.op for node in actual_graph.as_graph_def().node]
    self.assertNotIn("Cast", nodes_op_type)

    with ops.Graph().as_default() as actual_graph:
      _ = array_ops.placeholder(dtypes.float32, [None, 2],
                                name="probabilities")
      _ = array_ops.placeholder(dtypes.float32, [None, 2], name="logits")
      output_tensors = [
          actual_graph.get_tensor_by_name(name)
          for name in ("probabilities:0", "logits:0")
      ]
      casted_output_from_signature(output_tensors, outputs_shape_and_dtype,
                                   signature_with_cast)

    nodes_op_type = [node.op for node in actual_graph.as_graph_def().node]
    self.assertIn("Cast", nodes_op_type)

  def test_extract_emb_setting_from_param(self):
    params = IpuConversionParams(
        embedded_runtime_save_config={
            "embedded_runtime_exec_cachedir": "poplar_exec",
            "runtime_api_timeout_us": 1000
        })
    (embedded_runtime_exec_cachedir, poplar_exec_filepath,
     runtime_api_timeout_us,
     batch_per_step) = extract_emb_setting_from_param(params)
    self.assertEqual(embedded_runtime_exec_cachedir, "poplar_exec")
    self.assertEqual(runtime_api_timeout_us, 1000)
    self.assertEqual(poplar_exec_filepath,
                     os.path.join("poplar_exec", "application.poplar_exec"))
    self.assertEqual(batch_per_step, 1)


if __name__ == '__main__':
  test.main()
