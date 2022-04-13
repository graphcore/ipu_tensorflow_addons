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
#
# This file has been modified by Graphcore Ltd.
# ==============================================================================
import unittest
import json
import os
from statistics import mean, stdev
from types import SimpleNamespace

from tensorflow.python import layers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops, init_ops, nn, variable_scope

import pva

from ipu_tensorflow_addons.saved_model_tool.converter.autograph import Node, TFv1Graph, Input, Output, ProfileAnalyzer
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.profile import OPTypeCycleEstimateKey, TrieDictionary
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.utils import load_tf_graph
from ipu_tensorflow_addons.saved_model_tool.saved_model_test_utils import ModelForTest


class PvaNamespace(SimpleNamespace):
  def __hash__(self):
    if hasattr(self, "name"):
      if self.name in pva.Program.Type.__members__:
        return hash(getattr(pva.Program.Type, self.name))
    return super().__hash__()


def dict2obj(d):
  return PvaNamespace(**d)


class ToyModel(ModelForTest):
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


class OPTypeCycleEstimateKeyTestCase(unittest.TestCase):
  def test_estimate_key_is_equal(self):
    node_a = Node("bert/encoder",
                  inputs=[Input("A", shape=[8, 8]),
                          Input("B", shape=[8, 8])],
                  outputs=[Output("O", shape=[8, 8])],
                  op_type="MatMulV2")
    node_same_shape_as_a = Node(
        "bert/encoder/layers",
        inputs=[Input("A1", shape=[8, 8]),
                Input("B1", shape=[8, 8])],
        outputs=[Output("O1", shape=[8, 8])],
        op_type="MatMulV2")
    node_not_same_shape_as_a = Node(
        "bert/encoder/layers",
        inputs=[Input("A2", shape=[8, 9]),
                Input("B2", shape=[9, 8])],
        outputs=[Output("O2", shape=[8, 8])],
        op_type="MatMulV2")

    self.assertEqual(OPTypeCycleEstimateKey(node_a),
                     OPTypeCycleEstimateKey(node_same_shape_as_a))
    self.assertNotEqual(OPTypeCycleEstimateKey(node_a),
                        OPTypeCycleEstimateKey(node_not_same_shape_as_a))


class TrieDictionaryTestCase(unittest.TestCase):
  def setUp(self):
    self.tries = TrieDictionary(sep='/')

  def test_add_fun(self):
    self.tries.add("bert/encoder/layer_0/self/query/matmul")
    self.assertEqual(self.tries.root, {
        'bert': {
            'encoder': {
                'layer_0': {
                    'self': {
                        'query': {
                            'matmul': True
                        }
                    }
                }
            }
        }
    })

  def test_top_match(self):
    self.tries.add("bert/encoder/layer_0/self/query/matmul")
    self.tries.add("bert/encoder/layer_0/self/query/matmul_1")
    self.assertFalse(self.tries.top_match("berts"))
    self.assertEqual(self.tries.top_match("bert"), "bert")
    self.assertEqual(self.tries.top_match("bert/encoder/layer_0/self/query/"),
                     "bert/encoder/layer_0/self/query")
    self.assertEqual(
        self.tries.top_match("bert/encoder/layer_0/self/query/matmul"),
        "bert/encoder/layer_0/self/query/matmul")
    self.assertEqual(
        self.tries.top_match("bert/encoder/layer_0/self/query/matmul/xxx/yy"),
        "bert/encoder/layer_0/self/query/matmul")
    self.assertEqual(
        self.tries.top_match(
            "bert/encoder/layer_0/self/query/matmul_1/xxx/yy"),
        "bert/encoder/layer_0/self/query/matmul_1")
    self.assertNotEqual(
        self.tries.top_match(
            "bert/encoder/layer_0/self/query/matmul_1/xxx/yy"),
        "bert/encoder/layer_0/self/query/matmul")


class ProfileAnalyzerTestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    self.toy_model = ToyModel(freeze=True, save=True)
    self.report_root = os.path.join("ipu_tensorflow_addons",
                                    "saved_model_tool", "testdata",
                                    "profile_test")
    self.pa = ProfileAnalyzer(self.report_root)

  def test_profile_analyzer(self):
    tfgraph, _ = load_tf_graph(self.toy_model.model_path)
    graph = TFv1Graph(tfgraph, self.toy_model.signature_def)
    with open(os.path.join(self.report_root, "profile.json"), "r") as f:
      report = json.load(f, object_hook=dict2obj)
    self.pa.report = report
    self.assertTrue(self.pa.is_parsed())

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

    mem_overflow_list = self.pa.mem_overflow_per_ipu()

    self.assertFalse(self.pa.check_if_oom())
    self.assertListEqual(self.pa.which_ipu_oom(), [])
    self.assertEqual(self.pa.totalMemory, 1881145344)
    self.assertEqual(self.pa.tilesPerReplica, 2944)
    self.assertEqual(self.pa.tilesPerIpu, 1472)
    self.assertListEqual(self.pa.activate_tile_balance,
                         [0.686243834801846, 0.6385870807657626])
    self.assertEqual(self.pa.clockFrequency, 1330000000.0)
    self.assertEqual(self.pa.num_ipus, 2)
    self.assertEqual(self.pa.MembytesPerTile, 638976)
    self.assertEqual(self.pa.MembytesPerIPU, 940572672)
    self.assertEqual(self.pa.numTiles, 2944)
    self.assertEqual(self.pa.replicas, 1)
    self.assertEqual(self.pa.ipusPerReplica, 2)
    self.assertEqual(self.pa.memoryPerReplica, 1881145344)

    self.assertListEqual(self.pa.ipu_oom_list, [0, 0])
    self.assertEqual(mean(self.pa.mem_info_per_tile[0]), 157.87907608695653)
    self.assertEqual(stdev(self.pa.mem_info_per_tile[0]), 91.05000633907912)
    self.assertEqual(mean(self.pa.mem_info_per_tile[1]), 161.7180706521739)
    self.assertEqual(stdev(self.pa.mem_info_per_tile[1]), 121.52682643789272)

    self.assertEqual(
        self.pa.ipu_other_ops['instrumentationResults'].alwayslive, 96832)
    self.assertEqual(self.pa.ipu_other_ops['vertexInstanceState'].alwayslive,
                     55640)
    self.assertEqual(
        self.pa.ipu_other_ops["hostExchangePacketHeader"].alwayslive, 24448)
    self.assertEqual(self.pa.ipu_other_ops["vertexFieldData"].alwayslive, 1296)
    self.assertEqual(self.pa.ipu_other_ops["vectorListDescriptor"].alwayslive,
                     288)
    self.assertEqual(self.pa.ipu_other_ops["copyDescriptor"].alwayslive, 83)
    self.assertEqual(
        self.pa.ipu_other_ops["globalExchangePacketHeader"].alwayslive, 120)
    self.assertEqual(self.pa.ipu_other_ops["controlId"].alwayslive, 12)
    self.assertEqual(len(self.pa.max_mem_non_alwayslive_var), 0)
    self.assertEqual(
        self.pa.meminfo_dict["res/unit_2/dense/MatMul"].alwayslive, 256)
    self.assertEqual(
        self.pa.meminfo_dict["left/unit_1/dense/MatMul"].alwayslive, 262)
    self.assertListEqual(self.pa.tile_balance,
                         [0.04927401475621688, 0.03806170894611768])
    self.assertEqual(
        self.pa.cycle_info_dict['left/unit_0/dense/MatMul'].cycles, 1177)
    self.assertEqual(
        self.pa.cycle_info_dict['left/unit_2/dense/MatMul'].cycles, 1176)
    self.assertEqual(self.pa.namescope_sep, '/')

    self.assertListEqual(mem_overflow_list, [940340274, 940334623])


if __name__ == '__main__':
  googletest.main()
