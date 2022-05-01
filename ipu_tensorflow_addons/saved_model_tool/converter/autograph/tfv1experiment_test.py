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
Test cases for Tfv1Experiment
"""
import os
import tempfile
import unittest

from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test
from tensorflow.python import layers
from tensorflow.python.ops import (array_ops, init_ops, nn, variable_scope)
from ipu_tensorflow_addons.saved_model_tool.converter.autograph import TFv1Experiment
from ipu_tensorflow_addons.saved_model_tool.converter.autograph import RunConfig
from ipu_tensorflow_addons.saved_model_tool.converter.autograph import TFv1Graph
from ipu_tensorflow_addons.saved_model_tool.saved_model_test_utils import ModelForTest
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.utils import convert_graph_def_to_graph


class TFv1ExperimentModel(ModelForTest):
  def create(self):
    o1 = array_ops.placeholder(dtypes.float32, [None, 2], name="input_1")
    o2 = array_ops.placeholder(dtypes.float32, [None, 5], name="input_2")

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

    ok = array_ops.concat([o1, o2], axis=1, name='concat')
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
      o = o + ok
    return o


class TFv1ExperimentTest(unittest.TestCase):
  def setUp(self):
    stage_list = [
        [
            'input_1', 'input_2', 'left/unit_0/dense/kernel',
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
            'right/unit_0/dense/kernel', 'right/unit_0/dense/kernel/read',
            'right/unit_0/dense/bias', 'right/unit_0/dense/bias/read',
            'right/unit_0/dense/MatMul'
        ],
        [
            'right/unit_0/dense/BiasAdd', 'right/unit_0/dense/Relu',
            'right/unit_1/dense/kernel', 'right/unit_1/dense/kernel/read',
            'right/unit_1/dense/bias', 'right/unit_1/dense/bias/read',
            'right/unit_1/dense/MatMul', 'right/unit_1/dense/BiasAdd',
            'right/unit_1/dense/Relu', 'right/unit_2/dense/kernel',
            'right/unit_2/dense/kernel/read', 'right/unit_2/dense/bias',
            'right/unit_2/dense/bias/read', 'right/unit_2/dense/MatMul',
            'right/unit_2/dense/BiasAdd', 'right/unit_2/dense/Relu',
            'concat/axis', 'concat', 'res/unit_0/dense/kernel',
            'res/unit_0/dense/kernel/read', 'res/unit_0/dense/bias',
            'res/unit_0/dense/bias/read', 'res/unit_0/dense/MatMul',
            'res/unit_0/dense/BiasAdd', 'res/unit_0/dense/Relu',
            'res/unit_1/dense/kernel', 'res/unit_1/dense/kernel/read',
            'res/unit_1/dense/bias', 'res/unit_1/dense/bias/read',
            'res/unit_1/dense/MatMul', 'res/unit_1/dense/BiasAdd',
            'res/unit_1/dense/Relu', 'res/unit_2/dense/kernel',
            'res/unit_2/dense/kernel/read', 'res/unit_2/dense/bias',
            'res/unit_2/dense/bias/read', 'res/unit_2/dense/MatMul',
            'res/unit_2/dense/BiasAdd', 'res/unit_2/dense/Relu', 'res/add'
        ]
    ]
    self.model = TFv1ExperimentModel(freeze=True)
    self.graph = convert_graph_def_to_graph(self.model.graph_def)
    self.tfv1graph = TFv1Graph(pb_tf_graph=self.graph,
                               signature_def=self.model.signature_def)
    self.profiling_root_dir = tempfile.mkdtemp(dir=test.get_temp_dir())

    pipeconf_dict = {}
    for idx, node_names in enumerate(stage_list):
      for node_name in node_names:
        pipeconf_dict[node_name] = idx

    for node_name, stageId in pipeconf_dict.items():
      self.tfv1graph.nxg.nodes[node_name]["node"].pipeline_stage = stageId
    self.tfv1graph.set_pipeline_device_info(device_info=[0, 1])

    self.ipu_model_run_cfg = RunConfig(num_required_ipus=2, ipu_model=True)
    self.ipu_run_cfg = RunConfig(num_required_ipus=2, ipu_model=False)

    self.exp = TFv1Experiment(run_config=self.ipu_model_run_cfg,
                              profiling=True,
                              profiling_path=self.profiling_root_dir)

  def test_init_and_disengage(self):
    exp = TFv1Experiment(run_config=RunConfig(num_required_ipus=2,
                                              ipu_model=True),
                         profiling=True,
                         profiling_path=self.profiling_root_dir)
    exp.initialize()
    self.assertIn("TF_POPLAR_FLAGS", os.environ)
    self.assertIn("--use_ipu_model", os.environ.get("TF_POPLAR_FLAGS"))
    self.assertNotIn("--executable_cache_path",
                     os.environ.get("TF_POPLAR_FLAGS"))
    self.assertIn("POPLAR_ENGINE_OPTIONS", os.environ)
    exp.disengage()

    exp = TFv1Experiment(run_config=RunConfig(num_required_ipus=2,
                                              ipu_model=False),
                         profiling=False,
                         profiling_path=self.profiling_root_dir)
    exp.initialize()
    self.assertNotIn("--use_ipu_model", os.environ.get("TF_POPLAR_FLAGS"))
    self.assertNotIn("POPLAR_ENGINE_OPTIONS", os.environ)
    exp.disengage()

  def test_run(self):
    self.assertIsNone(self.exp.run(self.tfv1graph))
    pa = self.exp.get_profile_analysis(self.tfv1graph)
    self.assertTrue(pa.is_parsed())


if __name__ == "__main__":
  unittest.main()
