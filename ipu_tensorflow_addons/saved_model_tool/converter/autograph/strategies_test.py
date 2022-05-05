# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
# ==============================================================================
"""
Test for manual and greedy search strategies.
"""
import os
import json
import unittest

from tensorflow.python import layers
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops, init_ops, nn, variable_scope

from ipu_tensorflow_addons.saved_model_tool.converter.autograph import RunConfig, TFv1Graph, ProfileAnalyzer
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.utils import load_tf_graph
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.strategies import find_opt
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.strategies import ManualPipelineStrategy
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.strategies import GreedySolveStrategy
from ipu_tensorflow_addons.saved_model_tool.saved_model_test_utils import ModelForTest

from ipu_tensorflow_addons.saved_model_tool.converter.autograph.profile_test import dict2obj


class ToyModel(ModelForTest):
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


class StrategyUtilsTestCase(unittest.TestCase):
  def test_find_opt(self):
    self.assertListEqual(find_opt([1, 2, 3, 4, 4, 3, 2, 1], 2),
                         [[[1, 2, 3, 4], [4, 3, 2, 1]]])
    self.assertListEqual(
        find_opt([840, 570, 314, 294, 483, 828, 607, 139, 236, 440], 2),
        [[[840, 570, 314, 294, 483], [828, 607, 139, 236, 440]]])
    self.assertListEqual(
        find_opt([
            1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1200, 1, 1, 1, 1, 1, 1, 34, 56,
            74, 39, 26, 49
        ], 4), [[[1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1200],
                 [1, 1, 1, 1, 1, 1, 34, 56, 74], [39, 26, 49]]])
    self.assertListEqual(find_opt([1, 1, 2, 4, 90, 0, 45, 45], 4),
                         [[[1, 1, 2, 4], [90, 0], [45], [45]]])
    self.assertListEqual(
        find_opt([1, 2, 3, 4, 5, 6, 90, 110, 100, 120, 100, 100, 100, 100], 8),
        [[[1, 2, 3, 4, 5, 6, 90], [110], [100], [120], [100], [100], [100],
          [100]]])

    # The optional solution should be [[20], [40], [30], [10, 10, 10], [10, 10], [50], [70], [30]].
    # we only find the suboptimal solution due to the fact that the problem is a NP-hard problem.
    self.assertListEqual(
        find_opt([20, 40, 30, 10, 10, 10, 10, 10, 50, 70, 30], 8),
        [[[20], [40], [30, 10], [10, 10, 10], [10], [50], [70], [30]]])


class ManualPipelineStrategyTestCase(unittest.TestCase):
  def setUp(self):
    self.toy_model = ToyModel(freeze=True, save=True)
    self.manual_pipe_conf = [
        [
            "Placeholder_2",
            "^middle/unit_0",
            '^middle/unit_1',
            '^middle/unit_2/',
            '^middle/unit_3',
            '^middle/unit_4',
        ],
        [
            '^middle/unit_5',
            'Placeholder$',
            'Placeholder_1$',
            '^right/unit_0',
            '^right/unit_1',
            '^right/unit_2',
            '^left/unit_0',
        ],
        [
            '^left/unit_1',
            '^left/unit_2',
            '^left/unit_3',
            '^left/unit_4',
            'concat',
            '^res/unit_0/',
        ],
        [
            '^res/unit_1',
            '^res/unit_2',
            '^res/down/',
            '^res/down',
            '^res/add',
        ],
    ]

  def test_chop(self):
    tfgraph, _ = load_tf_graph(self.toy_model.model_path)
    graph = TFv1Graph(tfgraph, self.toy_model.signature_def)
    self.manual_pipeline_strategy = ManualPipelineStrategy(
        RunConfig(num_required_ipus=4))
    self.manual_pipeline_strategy.chop(graph, self.manual_pipe_conf)
    self.assertEqual(graph.nodes["Placeholder_2"].pipeline_stage, 0)
    self.assertEqual(graph.nodes["middle/unit_3/dense/Relu"].pipeline_stage, 0)
    self.assertEqual(graph.nodes["Placeholder"].pipeline_stage, 1)
    self.assertEqual(graph.nodes["left/unit_0/dense/Relu"].pipeline_stage, 1)
    self.assertEqual(graph.nodes["res/unit_0/dense/Relu"].pipeline_stage, 2)
    self.assertEqual(graph.nodes["left/unit_1/dense/Relu"].pipeline_stage, 2)
    self.assertEqual(graph.nodes["res/down/dense/BiasAdd"].pipeline_stage, 3)


class GreedySolveStrategyTestCase(unittest.TestCase):
  def setUp(self):
    self.toy_model = ToyModel(freeze=True, save=True)
    self.tfgraph, _ = load_tf_graph(self.toy_model.model_path)
    self.greedy_strategy = GreedySolveStrategy(RunConfig(num_required_ipus=2))
    self.report_root = os.path.join("ipu_tensorflow_addons",
                                    "saved_model_tool", "testdata",
                                    "profile_test")
    self.pa = ProfileAnalyzer(self.report_root)

  def test_first_try(self):
    graph = TFv1Graph(self.tfgraph, self.toy_model.signature_def)
    self.greedy_strategy.first_try(graph)

    self.assertEqual(graph.nodes["Placeholder_2"].pipeline_stage, 0)
    self.assertEqual(graph.nodes["middle/unit_3/dense/Relu"].pipeline_stage, 0)
    self.assertEqual(graph.nodes["Placeholder"].pipeline_stage, 0)
    self.assertEqual(graph.nodes["left/unit_0/dense/Relu"].pipeline_stage, 1)
    self.assertEqual(graph.nodes["res/unit_0/dense/Relu"].pipeline_stage, 1)
    self.assertEqual(graph.nodes["left/unit_1/dense/Relu"].pipeline_stage, 1)
    self.assertEqual(graph.nodes["res/down/dense/BiasAdd"].pipeline_stage, 1)

  def test_greedy_search_solutions(self):
    graph = TFv1Graph(self.tfgraph, self.toy_model.signature_def)
    with open(os.path.join(self.report_root, "profile.json"), "r") as f:
      report = json.load(f, object_hook=dict2obj)
    self.pa.report = report
    self.pa.num_ipus = self.pa.report.compilation.target.numIPUs
    self.pa.MembytesPerTile = self.pa.report.compilation.target.bytesPerTile
    self.pa.MembytesPerIPU = self.pa.report.compilation.target.bytesPerIPU
    self.pa.totalMemory = self.pa.report.compilation.target.totalMemory
    self.pa.clockFrequency = self.pa.report.compilation.target.clockFrequency
    self.pa.tilesPerIpu = self.pa.report.compilation.target.tilesPerIpu
    self.pa.numTiles = self.pa.report.compilation.target.numTiles

    self.pa.replicas = self.pa.report.compilation.target.numReplicas
    self.pa.ipusPerReplica = self.pa.report.compilation.target.ipusPerReplica
    self.pa.tilesPerReplica = self.pa.report.compilation.target.tilesPerReplica
    self.pa.memoryPerReplica = (
        self.pa.report.compilation.target.memoryPerReplica)

    self.pa.which_ipu_oom()
    self.pa.mem_state_per_tile()
    self.pa.activate_tile_balance_per_ipu()
    self.pa.tile_balance_per_ipu()
    self.pa.ops_memory_info(graph)
    self.pa.update_other_ops_alwayslive_mem_info()
    self.pa.ops_cycle_info(graph)

    _ = self.pa.mem_overflow_per_ipu()

    self.greedy_strategy.greedy_search_solutions(graph, self.pa)

    sum_list = [0] * self.pa.num_ipus
    for node in graph.nodes.values():
      mem_size_for_node = self.pa.meminfo_dict[
          node.name].alwayslive + self.pa.meminfo_dict[node.name].nonalwayslive
      sum_list[node.pipeline_stage] += mem_size_for_node

    # This is the best solution minimize the stdev of the group sum
    # [
    #    [0, 192, 16, 0, 64, 0, 0, 64, 0, 0, 64, 0, 0, 64, 0, 0, 64, 0, 0, 0, 184, 0, 0, 16, 0, 0, 16, 0, 0, 0, 214, 32, 0],
    #    [262, 0, 0, 16, 0, 0, 16, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 544, 0, 0, 0, 256, 0, 0, 0, 128, 0, 0, 0]
    # ]
    # score: 175.36248173426378
    # the sum is [990, 1238]

    self.assertListEqual(sum_list, [990, 1238])

  def test_moves(self):
    best_solution = [[
        0, 192, 16, 0, 64, 0, 0, 64, 0, 0, 64, 0, 0, 64, 0, 0, 64, 0, 0, 0,
        184, 0, 0, 16, 0, 0, 16, 0, 0, 0, 214, 32, 0
    ],
                     [
                         262, 0, 0, 16, 0, 0, 16, 0, 0, 16, 0, 0, 0, 0, 0, 0,
                         0, 544, 0, 0, 0, 256, 0, 0, 0, 128, 0, 0, 0
                     ]]
    mem_list = [
        0, 192, 16, 0, 64, 0, 0, 64, 0, 0, 64, 0, 0, 64, 0, 0, 64, 0, 0, 0,
        184, 0, 0, 16, 0, 0, 16, 0, 0, 0, 214, 32, 0, 262, 0, 0, 16, 0, 0, 16,
        0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 544, 0, 0, 0, 256, 0, 0, 0, 128, 0, 0, 0
    ]
    split_pos_list = [30, 32]
    mem_free_per_ipu = [246, -246]
    move_after = self.greedy_strategy.translate_to_grouped_mem_list(
        split_pos_list, mem_list)
    self.greedy_strategy.move_ops_to_left(move_after, 1, mem_free_per_ipu)
    self.assertListEqual(move_after, best_solution)

    best_solution = [[
        0, 192, 16, 0, 64, 0, 0, 64, 0, 0, 64, 0, 0, 64, 0, 0, 64, 0, 0, 0,
        184, 0, 0, 16, 0, 0, 16, 0, 0, 0, 214, 32
    ],
                     [
                         0, 262, 0, 0, 16, 0, 0, 16, 0, 0, 16, 0, 0, 0, 0, 0,
                         0, 0, 544, 0, 0, 0, 256, 0, 0, 0, 128, 0, 0, 0
                     ]]
    split_pos_list = [37, 25]
    mem_free_per_ipu = [-278, 278]
    move_after = self.greedy_strategy.translate_to_grouped_mem_list(
        split_pos_list, mem_list)
    self.greedy_strategy.move_ops_to_right(move_after, 0, mem_free_per_ipu)
    self.assertListEqual(move_after, best_solution)


if __name__ == "__main__":
  googletest.main()
