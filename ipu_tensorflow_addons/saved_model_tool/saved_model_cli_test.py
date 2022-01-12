# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
Tests for SavedModelCLI tool.
"""

import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from ipu_tensorflow_addons.saved_model_tool import saved_model_cli
from ipu_tensorflow_addons.saved_model_tool.saved_model_test_utils import ModelForTest


class SavedModelCLITestModel(ModelForTest):
  def create(self):
    """
    Create a simple SavedModel on the fly.
    t = x + x
    y = t * t
    return y

    Placeholder -> AddV2 -> Mul
    """
    in_tensor = array_ops.placeholder(shape=[1],
                                      dtype=dtypes.float32,
                                      name="x")
    tmp_tensor = in_tensor + in_tensor
    out_tensor = tmp_tensor * tmp_tensor
    return out_tensor


class SavedModelCLITestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    super().setUp()
    self.model = SavedModelCLITestModel(freeze=True, save=True)

  def testRunCommandInputExprs(self):
    parser = saved_model_cli.create_parser()
    output_tensor_dir = os.path.join(self.get_temp_dir(), 'out_dir')
    args = parser.parse_args([
        'run', '--dir', self.model.model_path, '--tag_set',
        tag_constants.SERVING, '--signature_def',
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY, '--input_exprs',
        'x=np.ones((1))', '--outdir', output_tensor_dir
    ])
    saved_model_cli.run(args)
    y_actual = np.load(os.path.join(output_tensor_dir, 'mul.npy'))
    y_expected = np.array([4.0])
    self.assertAllEqual(y_expected, y_actual)

  def testSimpleConvert(self):
    parser = saved_model_cli.create_parser()
    converted_model_path = os.path.join(self.get_temp_dir(),
                                        'converted_savedmodel')
    convert_args = parser.parse_args([
        'convert',
        '--dir',
        self.model.model_path,
        '--output_dir',
        converted_model_path,
        '--tag_set',
        tag_constants.SERVING,
        'ipu',
    ])
    saved_model_cli.convert_with_ipu(convert_args)
    self.assertTrue(
        os.path.isfile(os.path.join(converted_model_path, 'saved_model.pb')))

    # Make sure all nodes are placed on IPU
    with session.Session(graph=ops.Graph()) as sess:
      meta_graph_def = loader.load(sess, [tag_constants.SERVING],
                                   converted_model_path)
      graph_def = meta_graph_def.graph_def
      self.assertEqual(len(graph_def.node), 3)
      for node in graph_def.node:
        if node.op == 'Placeholder':
          continue
        self.assertTrue(node.device == '/device:IPU:0')

    # Run converted model on IPU
    output_tensor_dir = os.path.join(self.get_temp_dir(), 'out_dir')
    run_args = parser.parse_args([
        'run',
        '--dir',
        converted_model_path,
        '--tag_set',
        tag_constants.SERVING,
        '--signature_def',
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
        '--input_exprs',
        'x=np.ones((1))',
        '--outdir',
        output_tensor_dir,
        '--init_ipu',
        '--matmul_amp',
        '0.4',
        '--conv_amp',
        '0.3',
    ])
    saved_model_cli.run(run_args)
    y_actual = np.load(os.path.join(output_tensor_dir, 'mul.npy'))
    y_expected = np.array([4.0])
    self.assertAllEqual(y_expected, y_actual)

  def testConvertWithExcludedNodes(self):
    parser = saved_model_cli.create_parser()
    converted_model_path = os.path.join(self.get_temp_dir(),
                                        'converted_savedmodel')
    convert_args = parser.parse_args([
        'convert', '--dir', self.model.model_path, '--output_dir',
        converted_model_path, '--tag_set', tag_constants.SERVING, 'ipu',
        '--excluded_nodes', '^add$'
    ])
    saved_model_cli.convert_with_ipu(convert_args)

    with session.Session(graph=ops.Graph()) as sess:
      meta_graph_def = loader.load(sess, [tag_constants.SERVING],
                                   converted_model_path)
      graph_def = meta_graph_def.graph_def
      node_names = [node.name for node in graph_def.node]
      self.assertEqual(len(graph_def.node), 3)
      self.assertTrue('add' in node_names)

      for node in graph_def.node:
        if node.op == 'Placeholder':
          continue
        if node.name == 'add':
          self.assertTrue(node.device != '/device:IPU:0')
        else:
          self.assertTrue(node.device == '/device:IPU:0')

  def testConvertWithNoIpuPlacement(self):
    parser = saved_model_cli.create_parser()
    converted_model_path = os.path.join(self.get_temp_dir(),
                                        'converted_savedmodel')
    convert_args = parser.parse_args([
        'convert',
        '--dir',
        self.model.model_path,
        '--output_dir',
        converted_model_path,
        '--tag_set',
        tag_constants.SERVING,
        'ipu',
        '--no_ipu_placement',
    ])
    saved_model_cli.convert_with_ipu(convert_args)

    with session.Session(graph=ops.Graph()) as sess:
      meta_graph_def = loader.load(sess, [tag_constants.SERVING],
                                   converted_model_path)
      graph_def = meta_graph_def.graph_def
      self.assertEqual(len(graph_def.node), 3)
      for node in graph_def.node:
        self.assertTrue(node.device != '/device:IPU:0')

  def testPrecisionConversion(self):
    parser = saved_model_cli.create_parser()
    converted_model_path = os.path.join(self.get_temp_dir(),
                                        'converted_savedmodel')
    convert_args = parser.parse_args([
        'convert', '--dir', self.model.model_path, '--output_dir',
        converted_model_path, '--tag_set', tag_constants.SERVING, 'ipu',
        '--precision_mode', 'FP16', '--precision_conversion_excluded_nodes',
        '^add$'
    ])
    saved_model_cli.convert_with_ipu(convert_args)

    with session.Session(graph=ops.Graph()) as sess:
      meta_graph_def = loader.load(sess, [tag_constants.SERVING],
                                   converted_model_path)
      graph_def = meta_graph_def.graph_def
      node_dict = {node.name: node for node in graph_def.node}
      self.assertEqual(node_dict['add'].attr['T'].type, tf.float32)
      self.assertEqual(node_dict['mul'].attr['T'].type, tf.float16)

  def testPrecisionConversionWithConfigFile(self):
    parser = saved_model_cli.create_parser()
    converted_model_path = os.path.join(self.get_temp_dir(),
                                        'converted_savedmodel')
    cfg_data = {
        "precision_mode": "FP16",
        "precision_conversion_excluded_nodes": [
            "^add$",
        ]
    }
    cfg_file = os.path.join(self.get_temp_dir(), 'cfg.json')
    with open(cfg_file, 'w') as f:
      json.dump(cfg_data, f)

    convert_args = parser.parse_args([
        'convert',
        '--dir',
        self.model.model_path,
        '--output_dir',
        converted_model_path,
        '--tag_set',
        tag_constants.SERVING,
        'ipu',
        '--config_file',
        cfg_file,
    ])
    saved_model_cli.convert_with_ipu(convert_args)

    with session.Session(graph=ops.Graph()) as sess:
      meta_graph_def = loader.load(sess, [tag_constants.SERVING],
                                   converted_model_path)
      graph_def = meta_graph_def.graph_def
      node_dict = {node.name: node for node in graph_def.node}
      self.assertEqual(node_dict['add'].attr['T'].type, tf.float32)
      self.assertEqual(node_dict['mul'].attr['T'].type, tf.float16)

  def testConvertWithConfigFile(self):
    cfg_data = {
        "excluded_nodes": [
            "^mul$",
        ]
    }
    cfg_file = os.path.join(self.get_temp_dir(), 'cfg.json')
    with open(cfg_file, 'w') as f:
      json.dump(cfg_data, f)

    parser = saved_model_cli.create_parser()
    converted_model_path = os.path.join(self.get_temp_dir(),
                                        'converted_savedmodel')
    convert_args = parser.parse_args([
        'convert',
        '--dir',
        self.model.model_path,
        '--output_dir',
        converted_model_path,
        '--tag_set',
        tag_constants.SERVING,
        'ipu',
        '--config_file',
        cfg_file,
    ])
    saved_model_cli.convert_with_ipu(convert_args)

    with session.Session(graph=ops.Graph()) as sess:
      meta_graph_def = loader.load(sess, [tag_constants.SERVING],
                                   converted_model_path)
      graph_def = meta_graph_def.graph_def
      node_names = [node.name for node in graph_def.node]
      self.assertEqual(len(graph_def.node), 3)
      self.assertTrue('mul' in node_names)

      for node in graph_def.node:
        if node.op == 'Placeholder':
          continue
        if node.name == 'mul':
          self.assertTrue(node.device != '/device:IPU:0')
        else:
          self.assertTrue(node.device == '/device:IPU:0')


if __name__ == '__main__':
  test.main()
