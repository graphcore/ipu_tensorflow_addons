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
Tests for AutoPipelineImpl and AutoPipeline Converter.
"""
import os
import time

from tensorflow import disable_v2_behavior
from tensorflow.python import layers
from tensorflow.python.framework import dtypes, test_util
from tensorflow.python.ops import (array_ops, init_ops, nn, variable_scope)
from tensorflow.python.platform import test
from tensorflow.python.ipu.test_utils import test_uses_ipus
from ipu_tensorflow_addons.saved_model_tool.ipu_convert import IpuConversionParams
from ipu_tensorflow_addons.saved_model_tool.converter import AutoPipeline
from ipu_tensorflow_addons.saved_model_tool.converter.auto_pipeline import get_latest_pipeconf, AutoPipelineImpl
from ipu_tensorflow_addons.saved_model_tool.saved_model_test_utils import ModelForTest, declare_signature

disable_v2_behavior()


class TestAutoGraphDef(ModelForTest):
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
    return o


class AutoPipelineImplTestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    self.test_model = TestAutoGraphDef(freeze=True)
    self.ipu_compile_params = IpuConversionParams(
        int64_to_int32_conversion=True)
    self.profiling_root_dir = self.get_temp_dir()
    self.solution_dir = self.get_temp_dir()

    self.ipu_compile_params.pipeline_cfg = {
        "converter": "auto",
        "fine_tune_iter": 5,
        "ipu_model": True,
        "max_ipu_quantity": 64,
        "min_ipu_quantity": 2,
        "priority": "cycle",
        "profiling_root_dir": self.profiling_root_dir,
        "solution_dir": self.solution_dir,
    }
    self.converter = AutoPipelineImpl(
        self.ipu_compile_params.batch_size, self.ipu_compile_params.matmul_amp,
        self.ipu_compile_params.matmul_partial_type,
        self.ipu_compile_params.conv_amp,
        self.ipu_compile_params.conv_partial_type, 5, True, 64, 2, "cycle",
        self.profiling_root_dir, self.solution_dir)

  def test_get_latest_pipeconfig(self):
    temp_file_path = self.get_temp_dir()
    with open(os.path.join(temp_file_path, "bb"), "w"):
      pass
    time.sleep(1)
    with open(os.path.join(temp_file_path, "cc"), "w"):
      pass
    time.sleep(1)
    with open(os.path.join(temp_file_path, "dd"), "w"):
      pass
    paths = get_latest_pipeconf(temp_file_path)
    self.assertEqual(paths, os.path.join(temp_file_path, "dd"))

  def test_auto_pipeline_impl_apply(self):
    num_of_ipus = self.converter.apply(self.test_model.graph_def,
                                       self.test_model.signature_def)
    self.assertEqual(num_of_ipus, 2)


class AutoPipelineConverterTestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    self.test_model = TestAutoGraphDef(freeze=True)
    self.ipu_compile_params = IpuConversionParams(
        int64_to_int32_conversion=True)
    self.profiling_root_dir = self.get_temp_dir()
    self.solution_dir = self.get_temp_dir()
    self.poplar_exec_path = self.get_temp_dir()

    self.ipu_compile_params.pipeline_cfg = {
        "converter": "auto",
        "fine_tune_iter": 5,
        "ipu_model": True,
        "max_ipu_quantity": 64,
        "min_ipu_quantity": 2,
        "priority": "cycle",
        "profiling_root_dir": self.profiling_root_dir,
        "solution_dir": self.solution_dir,
    }

    self.ipu_compile_params.embedded_runtime_save_config = {
        "embedded_runtime_exec_cachedir": self.poplar_exec_path,
        "runtime_api_timeout_us": 5000
    }

    self.converter = AutoPipeline(self.ipu_compile_params)

  @test_uses_ipus(2)
  def test_auto_pipeline_converter_apply(self):
    _, _ = self.converter.apply(self.test_model.graph_def,
                                self.test_model.signature_def)

  def test__do_auto_pipline(self):
    # pylint: disable=protected-access
    num_ipus, solution_path = self.converter._do_auto_pipline(
        self.test_model.graph_def, self.test_model.signature_def)
    self.assertEqual(num_ipus, 2)
    self.assertEqual(
        solution_path,
        os.path.join(self.solution_dir, "greedy_search_solutions.pipeconf"))


if __name__ == '__main__':
  test.main()
