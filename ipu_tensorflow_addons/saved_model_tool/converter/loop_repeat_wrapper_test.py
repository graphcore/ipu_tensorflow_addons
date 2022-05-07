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
Tests for LoopRepeatWrapper Converter.
"""
from tensorflow import disable_v2_behavior
from tensorflow.core.framework import types_pb2
from tensorflow.python import layers
from tensorflow.python.framework import dtypes, importer, ops, test_util
from tensorflow.python.ops import (array_ops, init_ops, math_ops, nn,
                                   variable_scope)
from tensorflow.python.platform import test
from tensorflow.python.ipu.sharding_utils import set_ipu_shard
from tensorflow.python.ipu import test_utils as tu

from ipu_tensorflow_addons.saved_model_tool.converter import LoopRepeatWrapper
from ipu_tensorflow_addons.saved_model_tool.ipu_convert import IpuConversionParams
from ipu_tensorflow_addons.saved_model_tool.saved_model_test_utils import ModelForTest, declare_signature

disable_v2_behavior()


def dtype_same_as_(graph_def, signature, assertFunc):

  name_and_dtype_from_signature = {
      ten.name: ten.dtype
      for ten in signature.inputs.values()
  }
  name_and_dtype_from_signature.update(
      {ten.name: ten.dtype
       for ten in signature.outputs.values()})

  with ops.Graph().as_default() as tfgraph:
    importer.import_graph_def(graph_def, name='')
    for tensor_name, tensor_type in name_and_dtype_from_signature.items():
      assertFunc(tfgraph.get_tensor_by_name(tensor_name).dtype, tensor_type)


class TestGraphDef(ModelForTest):
  @declare_signature(input_name_keys=["Input"], output_name_keys=["Output"])
  def create(self):
    x = array_ops.placeholder(dtypes.float32, [None, 8], name="x")
    with variable_scope.variable_scope("sharding0"):
      w0 = variable_scope.get_variable("w0",
                                       shape=[8, 8],
                                       dtype=dtypes.float32)
      x = math_ops.matmul(x, w0, name='matmul0')

    with variable_scope.variable_scope("sharding1"):
      w1 = variable_scope.get_variable("w1",
                                       shape=[8, 8],
                                       dtype=dtypes.float32)
      x = math_ops.matmul(x, w1, name='matmul1')
      y = math_ops.reduce_sum(x, name='reduce0')
    return y

  def modify_io_float16(self):
    self.signature_def.inputs["Input"].dtype = types_pb2.DT_HALF
    self.signature_def.outputs["Output"].dtype = types_pb2.DT_HALF

  def set_ipu_shard(self):
    with ops.Graph().as_default() as shard_graph:
      importer.import_graph_def(self.graph_def, name="")
      for op in shard_graph.get_operations():
        if 'IPU' in op.device:
          if op.name.startswith("sharding1"):
            set_ipu_shard(op, 1)
          else:
            set_ipu_shard(op, 0)
    return shard_graph.as_graph_def()


class TestGraphDefMultiOutput(ModelForTest):
  @declare_signature(output_name_keys=["Output1", "Output2"])
  def create(self):
    input_1 = array_ops.placeholder(shape=[None, 2],
                                    dtype=dtypes.float32,
                                    name="input_1")
    input_2 = array_ops.placeholder(shape=[None, 5],
                                    dtype=dtypes.float32,
                                    name="input_2")
    input_3 = array_ops.placeholder(shape=[None, 4],
                                    dtype=dtypes.float32,
                                    name="input_3")
    o1 = input_1  # shape (2, )
    o2 = input_2  # shape (5, )
    o3 = input_3  # shape (4, )
    o4 = input_3
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
      with variable_scope.variable_scope("down"):
        o = layers.dense(
            inputs=o,
            units=4,
            activation=nn.relu,
            kernel_initializer=init_ops.truncated_normal_initializer(),
            bias_initializer=init_ops.ones_initializer())

      o = o + o4
    return o, ok

  def modify_signature(self):
    self.signature_def.inputs["input_1"].dtype = types_pb2.DT_INT64
    self.signature_def.inputs["input_2"].dtype = types_pb2.DT_HALF
    self.signature_def.inputs["input_3"].dtype = types_pb2.DT_INT32
    self.signature_def.outputs["Output1"].dtype = types_pb2.DT_HALF
    self.signature_def.outputs["Output2"].dtype = types_pb2.DT_INT64


class LoopRepeatWrapperTestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    self.poplar_exec_path = self.get_temp_dir()
    self.model = TestGraphDef(freeze=True)
    self.multiple_output_model = TestGraphDefMultiOutput(freeze=True)

    self.converter = LoopRepeatWrapper(
        IpuConversionParams(batch_size=1,
                            int64_to_int32_conversion=True,
                            embedded_runtime_save_config={
                                "embedded_runtime_exec_cachedir":
                                self.poplar_exec_path,
                                "runtime_api_timeout_us": 5000
                            }))
    self.shard_converter = LoopRepeatWrapper(
        IpuConversionParams(batch_size=1,
                            num_ipus=2,
                            int64_to_int32_conversion=True,
                            embedded_runtime_save_config={
                                "embedded_runtime_exec_cachedir":
                                self.poplar_exec_path,
                                "runtime_api_timeout_us": 5000
                            }))

  def test_validate_params(self):
    with self.assertRaisesRegex(TypeError,
                                '(.*)must be a dictionary containing(.*)'):
      LoopRepeatWrapper(
          IpuConversionParams(
              batch_size=1,
              num_ipus=2,
              int64_to_int32_conversion=True,
              embedded_runtime_save_config=["ApplicationCall", "Runtime"]))

    with self.assertRaisesRegex(ValueError, '(.*)merge_subgraphs=False(.*)'):
      LoopRepeatWrapper(
          IpuConversionParams(batch_size=1,
                              excluded_nodes=["^sharding0"],
                              merge_subgraphs=True,
                              int64_to_int32_conversion=True,
                              embedded_runtime_save_config={
                                  "embedded_runtime_exec_cachedir":
                                  self.poplar_exec_path,
                                  "runtime_api_timeout_us": 5000
                              }))

    with self.assertRaisesRegex(ValueError,
                                '(.*)int64_to_int32_conversion=True(.*)'):
      LoopRepeatWrapper(
          IpuConversionParams(batch_size=1,
                              int64_to_int32_conversion=False,
                              embedded_runtime_save_config={
                                  "embedded_runtime_exec_cachedir":
                                  self.poplar_exec_path,
                                  "runtime_api_timeout_us": 5000
                              }))

  def test_batch_per_step_is_zero(self):
    converter = LoopRepeatWrapper(
        IpuConversionParams(batch_size=1,
                            int64_to_int32_conversion=True,
                            batch_per_step=0))
    self.multiple_output_model.modify_signature()
    graph_def, signature = converter.apply(
        self.multiple_output_model.graph_def,
        self.multiple_output_model.signature_def)

    self.assertProtoEquals(graph_def, self.multiple_output_model.graph_def)
    self.assertProtoEquals(signature, self.multiple_output_model.signature_def)

  @tu.test_uses_ipus(1)
  def test_not_add_cast_apply(self):
    graph_def, signature = self.converter.apply(self.model.graph_def,
                                                self.model.signature_def)
    op_type_list = [n.op for n in graph_def.node]
    self.assertNotIn("Cast", op_type_list)
    self.assertIn("ApplicationRuntime", op_type_list)
    self.assertIn("ApplicationCall", op_type_list)
    self.assertProtoEquals(signature, self.model.signature_def)

  @tu.test_uses_ipus(1)
  def test_add_cast_to_io_apply(self):
    self.model.modify_io_float16()

    graph_def, signature = self.converter.apply(self.model.graph_def,
                                                self.model.signature_def)
    op_type_list = [n.op for n in graph_def.node]
    self.assertIn("Cast", op_type_list)
    self.assertIn("ApplicationRuntime", op_type_list)
    self.assertIn("ApplicationCall", op_type_list)
    self.assertProtoEquals(signature, self.model.signature_def)

  @tu.test_uses_ipus(2)
  def test_sharded_model_apply(self):
    sharded_graph_def = self.model.set_ipu_shard()

    graph_def, signature = self.shard_converter.apply(sharded_graph_def,
                                                      self.model.signature_def)
    dtype_same_as_(graph_def, signature, self.assertEqual)

  @tu.test_uses_ipus(1)
  def test_multiple_input_and_outputs(self):
    self.multiple_output_model.modify_signature()
    graph_def, signature = self.converter.apply(
        self.multiple_output_model.graph_def,
        self.multiple_output_model.signature_def)

    dtype_same_as_(graph_def, signature, self.assertEqual)

  @tu.test_uses_ipus(1)
  def test_mutual_exclusion_with_ipuwrapper(self):
    converter = LoopRepeatWrapper(
        IpuConversionParams(batch_size=1,
                            int64_to_int32_conversion=True,
                            embedded_runtime_save_config={
                                "embedded_runtime_exec_cachedir":
                                self.poplar_exec_path,
                                "runtime_api_timeout_us": 5000
                            }))

    graph_def, signature = converter.apply(self.model.graph_def,
                                           self.model.signature_def)

    self.assertNotEqual(graph_def, self.model.graph_def)
    self.assertEqual(signature, self.model.signature_def)

    converter = LoopRepeatWrapper(
        IpuConversionParams(batch_size=1,
                            excluded_nodes=["^sharding0"],
                            remove_excluded_nodes=False,
                            int64_to_int32_conversion=True,
                            embedded_runtime_save_config={
                                "embedded_runtime_exec_cachedir":
                                self.poplar_exec_path,
                                "runtime_api_timeout_us": 5000
                            }))
    graph_def, signature = converter.apply(self.model.graph_def,
                                           self.model.signature_def)

    self.assertNotEqual(graph_def, self.model.graph_def)
    self.assertEqual(signature, self.model.signature_def)

  @tu.test_uses_ipus(1)
  def test_unique_engine_name(self):
    def find_app_call(graph_def):
      for node in graph_def.node:
        if node.op == "ApplicationCall":
          return node.attr["engine_name"].s
      return b""

    converter = LoopRepeatWrapper(
        IpuConversionParams(batch_size=1,
                            int64_to_int32_conversion=True,
                            embedded_runtime_save_config={
                                "embedded_runtime_exec_cachedir":
                                self.poplar_exec_path,
                                "runtime_api_timeout_us": 5000
                            }))

    graph_def_first, _ = converter.apply(self.model.graph_def,
                                         self.model.signature_def)
    first_engine_name = find_app_call(graph_def_first)
    converter = LoopRepeatWrapper(
        IpuConversionParams(batch_size=2,
                            int64_to_int32_conversion=True,
                            embedded_runtime_save_config={
                                "embedded_runtime_exec_cachedir":
                                self.poplar_exec_path,
                                "runtime_api_timeout_us": 5000
                            }))

    graph_def_second, _ = converter.apply(self.model.graph_def,
                                          self.model.signature_def)
    second_engine_name = find_app_call(graph_def_second)
    self.assertNotEqual(first_engine_name, second_engine_name)

  def test_skip_with_pipeline_cfg(self):
    converter = LoopRepeatWrapper(
        IpuConversionParams(batch_size=1,
                            excluded_nodes=["^sharding0"],
                            remove_excluded_nodes=False,
                            int64_to_int32_conversion=True,
                            pipeline_cfg={
                                "converter": "auto",
                                "fine_tune_iter": 5,
                                "ipu_model": True,
                                "max_ipu_quantity": 64,
                                "min_ipu_quantity": 2,
                                "priority": "cycle",
                                "profiling_root_dir": "profiling",
                                "solution_dir": "solution"
                            },
                            embedded_runtime_save_config={
                                "embedded_runtime_exec_cachedir":
                                self.poplar_exec_path,
                                "runtime_api_timeout_us": 5000
                            }))
    self.assertFalse(converter._should_do_loop_repeat_ipu_wrapper())  # pylint: disable=protected-access


if __name__ == '__main__':
  test.main()
