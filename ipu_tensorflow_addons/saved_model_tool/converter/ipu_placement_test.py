# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest
import tempfile
import os
import shutil
import numpy as np

from tensorflow.python.ops import math_ops
from tensorflow.saved_model.signature_def_utils import predict_signature_def
from tensorflow.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import tensorflow.compat.v1 as tf
from ipu_tensorflow_addons.saved_model_tool.ipu_convert import IpuGraphConverter, IpuConversionParams
tf.disable_v2_behavior()


class TestSavedmodel(object):
  def __init__(self):
    # Create temp directory for saving savedmodel
    self.base_path = tempfile.mkdtemp()
    print("Temp directory {} created.".format(self.base_path))
    self.model_path = self._create_test_savedmodel(self.base_path)

  def __del__(self):
    # Remove temp directory created by __init__()
    print("Remove temp directory {}.".format(self.base_path))
    shutil.rmtree(self.base_path)

  @staticmethod
  def _create_test_savedmodel(base_path):
    def basic_graph(x):
      w0 = tf.get_variable("w0", shape=[8, 8], dtype=tf.float32)
      x = tf.matmul(w0, x)

      w1 = tf.get_variable("w1", shape=[8, 8], dtype=tf.float32)
      x = tf.matmul(w1, x)
      y = math_ops.reduce_sum(x)
      return y

    model_path = os.path.join(base_path, "test_savedmodel")

    with tf.Graph().as_default():
      x = tf.placeholder(np.float32, [8, 8], name="x")
      y = basic_graph(x)

      with tf.Session() as sess:
        tf.global_variables_initializer().run()
        builder = tf.saved_model.builder.SavedModelBuilder(model_path)
        signature = predict_signature_def(inputs={'Input': x},
                                          outputs={'Output': y})
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature
            })
        builder.save()

    return model_path


class IpuPlacementTestCase(unittest.TestCase):
  def setUp(self):
    self.test_model = TestSavedmodel()

  @staticmethod
  def _check_ipu_placement(node):
    if node.device != '/device:IPU:0':
      return False
    if '_XlaCompile' not in node.attr or \
       '_XlaScope' not in node.attr or \
       '_XlaSeparateCompiledGradients' not in node.attr:
      return False

    return True

  def test_ipu_placement(self):
    # check node with ipu placement
    converter = IpuGraphConverter(
        input_saved_model_dir=self.test_model.model_path,
        conversion_params=IpuConversionParams())
    graph_def = converter.convert()
    self.assertIsNotNone(graph_def)
    self.assertGreater(len(graph_def.node), 0)

    for node in graph_def.node:
      if node.op == 'Placeholder':
        continue
      self.assertTrue(self._check_ipu_placement(node))

  def test_no_ipu_placement(self):
    # check node without ipu placement
    converter = IpuGraphConverter(
        input_saved_model_dir=self.test_model.model_path,
        conversion_params=IpuConversionParams(ipu_placement=False))
    graph_def = converter.convert()
    self.assertIsNotNone(graph_def)
    self.assertGreater(len(graph_def.node), 0)

    for node in graph_def.node:
      if node.op == 'Placeholder':
        continue
      self.assertFalse(self._check_ipu_placement(node))

  def test_excluded_nodes(self):
    excluded_nodes = [
        '^MatMul$',
    ]
    converter = IpuGraphConverter(
        input_saved_model_dir=self.test_model.model_path,
        conversion_params=IpuConversionParams(excluded_nodes=excluded_nodes))
    graph_def = converter.convert()
    self.assertIsNotNone(graph_def)
    self.assertGreater(len(graph_def.node), 0)
    for node in graph_def.node:
      if node.op == 'Placeholder':
        continue
      if node.name == 'MatMul':
        self.assertFalse(self._check_ipu_placement(node))
      else:
        self.assertTrue(self._check_ipu_placement(node))


if __name__ == '__main__':
  unittest.main()
