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
import json

from tensorflow import disable_v2_behavior
from tensorflow.python import layers
from tensorflow.python.framework import dtypes, test_util
from tensorflow.python.ops import (array_ops, init_ops, nn, variable_scope)
from tensorflow.python.platform import test
from tensorflow.python.ipu.test_utils import test_uses_ipus
from ipu_tensorflow_addons.saved_model_tool.ipu_convert import IpuConversionParams
from ipu_tensorflow_addons.saved_model_tool.converter.pipeconf_loader import PipelineConfProfiler, PipelineConfLoader
from ipu_tensorflow_addons.saved_model_tool.saved_model_test_utils import ModelForTest, declare_signature

disable_v2_behavior()


def write_pipeconf_to_file(solution_path):
  pipeconf_dict = {
      "device_mapping": [0, 1],
      "pipeline_mapping": {
          "input_1": 0,
          "input_2": 0,
          "input_3": 0,
          "left/unit_0/dense/kernel": 0,
          "left/unit_0/dense/kernel/read": 0,
          "left/unit_0/dense/bias": 1,
          "left/unit_0/dense/bias/read": 1,
          "left/unit_0/dense/MatMul": 0,
          "left/unit_0/dense/BiasAdd": 1,
          "left/unit_0/dense/Relu": 1,
          "left/unit_1/dense/kernel": 1,
          "left/unit_1/dense/kernel/read": 1,
          "left/unit_1/dense/bias": 1,
          "left/unit_1/dense/bias/read": 1,
          "left/unit_1/dense/MatMul": 1,
          "left/unit_1/dense/BiasAdd": 1,
          "left/unit_1/dense/Relu": 1,
          "left/unit_2/dense/kernel": 1,
          "left/unit_2/dense/kernel/read": 1,
          "left/unit_2/dense/bias": 1,
          "left/unit_2/dense/bias/read": 1,
          "left/unit_2/dense/MatMul": 1,
          "left/unit_2/dense/BiasAdd": 1,
          "left/unit_2/dense/Relu": 1,
          "left/unit_3/dense/kernel": 1,
          "left/unit_3/dense/kernel/read": 1,
          "left/unit_3/dense/bias": 1,
          "left/unit_3/dense/bias/read": 1,
          "left/unit_3/dense/MatMul": 1,
          "left/unit_3/dense/BiasAdd": 1,
          "left/unit_3/dense/Relu": 1,
          "left/unit_4/dense/kernel": 1,
          "left/unit_4/dense/kernel/read": 1,
          "left/unit_4/dense/bias": 1,
          "left/unit_4/dense/bias/read": 1,
          "left/unit_4/dense/MatMul": 1,
          "left/unit_4/dense/BiasAdd": 1,
          "left/unit_4/dense/Relu": 1,
          "right/unit_0/dense/kernel": 0,
          "right/unit_0/dense/kernel/read": 0,
          "right/unit_0/dense/bias": 0,
          "right/unit_0/dense/bias/read": 0,
          "right/unit_0/dense/MatMul": 0,
          "right/unit_0/dense/BiasAdd": 0,
          "right/unit_0/dense/Relu": 0,
          "right/unit_1/dense/kernel": 0,
          "right/unit_1/dense/kernel/read": 0,
          "right/unit_1/dense/bias": 0,
          "right/unit_1/dense/bias/read": 0,
          "right/unit_1/dense/MatMul": 0,
          "right/unit_1/dense/BiasAdd": 0,
          "right/unit_1/dense/Relu": 0,
          "right/unit_2/dense/kernel": 0,
          "right/unit_2/dense/kernel/read": 0,
          "right/unit_2/dense/bias": 0,
          "right/unit_2/dense/bias/read": 0,
          "right/unit_2/dense/MatMul": 0,
          "right/unit_2/dense/BiasAdd": 0,
          "right/unit_2/dense/Relu": 0,
          "middle/unit_0/dense/kernel": 0,
          "middle/unit_0/dense/kernel/read": 0,
          "middle/unit_0/dense/bias": 0,
          "middle/unit_0/dense/bias/read": 0,
          "middle/unit_0/dense/MatMul": 0,
          "middle/unit_0/dense/BiasAdd": 0,
          "middle/unit_0/dense/Relu": 0,
          "middle/unit_1/dense/kernel": 0,
          "middle/unit_1/dense/kernel/read": 0,
          "middle/unit_1/dense/bias": 0,
          "middle/unit_1/dense/bias/read": 0,
          "middle/unit_1/dense/MatMul": 0,
          "middle/unit_1/dense/BiasAdd": 0,
          "middle/unit_1/dense/Relu": 0,
          "middle/unit_2/dense/kernel": 0,
          "middle/unit_2/dense/kernel/read": 0,
          "middle/unit_2/dense/bias": 0,
          "middle/unit_2/dense/bias/read": 0,
          "middle/unit_2/dense/MatMul": 0,
          "middle/unit_2/dense/BiasAdd": 0,
          "middle/unit_2/dense/Relu": 0,
          "middle/unit_3/dense/kernel": 0,
          "middle/unit_3/dense/kernel/read": 0,
          "middle/unit_3/dense/bias": 0,
          "middle/unit_3/dense/bias/read": 0,
          "middle/unit_3/dense/MatMul": 0,
          "middle/unit_3/dense/BiasAdd": 0,
          "middle/unit_3/dense/Relu": 0,
          "middle/unit_4/dense/kernel": 0,
          "middle/unit_4/dense/kernel/read": 0,
          "middle/unit_4/dense/bias": 0,
          "middle/unit_4/dense/bias/read": 0,
          "middle/unit_4/dense/MatMul": 0,
          "middle/unit_4/dense/BiasAdd": 0,
          "middle/unit_4/dense/Relu": 0,
          "middle/unit_5/dense/kernel": 0,
          "middle/unit_5/dense/kernel/read": 0,
          "middle/unit_5/dense/bias": 0,
          "middle/unit_5/dense/bias/read": 0,
          "middle/unit_5/dense/MatMul": 0,
          "middle/unit_5/dense/BiasAdd": 0,
          "middle/unit_5/dense/Relu": 0,
          "concat/axis": 1,
          "concat": 1,
          "res/unit_0/dense/kernel": 1,
          "res/unit_0/dense/kernel/read": 1,
          "res/unit_0/dense/bias": 1,
          "res/unit_0/dense/bias/read": 1,
          "res/unit_0/dense/MatMul": 1,
          "res/unit_0/dense/BiasAdd": 1,
          "res/unit_0/dense/Relu": 1,
          "res/unit_0/add": 1,
          "res/unit_1/dense/kernel": 1,
          "res/unit_1/dense/kernel/read": 1,
          "res/unit_1/dense/bias": 1,
          "res/unit_1/dense/bias/read": 1,
          "res/unit_1/dense/MatMul": 1,
          "res/unit_1/dense/BiasAdd": 1,
          "res/unit_1/dense/Relu": 1,
          "res/unit_1/add": 1,
          "res/unit_2/dense/kernel": 1,
          "res/unit_2/dense/kernel/read": 1,
          "res/unit_2/dense/bias": 1,
          "res/unit_2/dense/bias/read": 1,
          "res/unit_2/dense/MatMul": 1,
          "res/unit_2/dense/BiasAdd": 1,
          "res/unit_2/dense/Relu": 1,
          "res/unit_2/add": 1,
          "res/down/dense/kernel": 1,
          "res/down/dense/kernel/read": 1,
          "res/down/dense/bias": 1,
          "res/down/dense/bias/read": 1,
          "res/down/dense/MatMul": 1,
          "res/down/dense/BiasAdd": 1,
          "res/down/dense/Relu": 1,
          "res/add": 1,
      }
  }
  with open(solution_path, "w") as solution_file:
    json.dump(pipeconf_dict, solution_file)


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


class PipelineConfProfilerTestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    self.test_model = TestAutoGraphDef(freeze=True)
    self.ipu_compile_params = IpuConversionParams(
        num_ipus=2, int64_to_int32_conversion=True)
    self.profiling_root_dir = self.get_temp_dir()
    self.solution_path = os.path.join(self.get_temp_dir(), "solution.pipeconf")
    write_pipeconf_to_file(self.solution_path)
    self.ipu_compile_params.pipeline_cfg = {
        "converter": "load",
        "ipu_model": True,
        "profiling_root_dir": self.profiling_root_dir,
        "solution_path": self.solution_path,
        "profiling_enable": True
    }
    self.converter = PipelineConfProfiler(
        self.ipu_compile_params.batch_size, self.ipu_compile_params.matmul_amp,
        self.ipu_compile_params.num_ipus,
        self.ipu_compile_params.matmul_partial_type,
        self.ipu_compile_params.conv_amp,
        self.ipu_compile_params.conv_partial_type,
        self.ipu_compile_params.pipeline_cfg["ipu_model"],
        self.ipu_compile_params.pipeline_cfg["profiling_root_dir"],
        self.ipu_compile_params.pipeline_cfg["solution_path"],
        self.ipu_compile_params.pipeline_cfg["profiling_enable"])

  def test_pipeline_conf_profiler_apply(self):
    self.converter.apply(self.test_model.graph_def,
                         self.test_model.signature_def)


class PipelineConfLoaderTestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    self.test_model = TestAutoGraphDef(freeze=True)
    self.ipu_compile_params = IpuConversionParams(
        num_ipus=2, int64_to_int32_conversion=True)
    self.profiling_root_dir = self.get_temp_dir()
    self.solution_path = os.path.join(self.get_temp_dir(), "solution.pipeconf")
    self.poplar_exec_path = self.get_temp_dir()

    write_pipeconf_to_file(self.solution_path)
    self.ipu_compile_params.pipeline_cfg = {
        "converter": "load",
        "ipu_model": True,
        "profiling_root_dir": self.profiling_root_dir,
        "solution_path": self.solution_path,
        "profiling_enable": True
    }
    self.ipu_compile_params.embedded_runtime_save_config = {
        "embedded_runtime_exec_cachedir": self.poplar_exec_path,
        "runtime_api_timeout_us": 5000
    }
    self.converter = PipelineConfLoader(self.ipu_compile_params)

  @test_uses_ipus(2)
  def test_pipeline_conf_loader_converter_apply(self):
    _, _ = self.converter.apply(self.test_model.graph_def,
                                self.test_model.signature_def)

  def test__take_profile(self):
    # pylint: disable=protected-access
    self.converter._take_profile(self.test_model.graph_def,
                                 self.test_model.signature_def)


if __name__ == '__main__':
  test.main()
