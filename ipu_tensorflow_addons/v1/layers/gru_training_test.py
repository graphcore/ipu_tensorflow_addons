# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

# Naive GRU to learn three-char time steps to one-char mapping

import pva
from absl.testing import parameterized
import numpy as np
from tensorflow.compat import v1 as tf
from tensorflow.python import ipu
from tensorflow.python.ipu import test_utils as tu
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.platform import googletest
from ipu_tensorflow_addons.v1 import layers

DATA_TYPE = np.float32

SEQ_LEN = 3
BATCH_SIZE = 40 - SEQ_LEN
INPUT_SIZE = 1
NUM_HIDDEN = 64
NUM_TRAINING_STEPS = 100
LEARNING_RATE = 10


# pylint: disable=unused-argument
def _PopnnGRU(x,
              initial_state,
              y,
              sequence_len=None,
              num_hidden=NUM_HIDDEN,
              **kwargs):
  gru_cell = layers.PopnnGRU(
      num_hidden,
      dtype=DATA_TYPE,
      weights_initializer=tf.zeros_initializer(dtype=DATA_TYPE),
      bias_initializer=tf.zeros_initializer(dtype=DATA_TYPE),
      reset_after=False,
      **kwargs)
  outputs, _ = gru_cell(x, initial_state=initial_state, training=True)
  softmax = tf.nn.softmax_cross_entropy_with_logits_v2(
      logits=outputs[-1], labels=tf.stop_gradient(y))
  loss = tf.reduce_mean(softmax)
  train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
  return [loss, train]


# pylint: disable=unused-argument
def _PopnnGRU_DynamicGRU(x,
                         initial_state,
                         y,
                         sequence_len=None,
                         num_hidden=NUM_HIDDEN,
                         **kwargs):
  gru_cell = layers.PopnnDynamicGRU(
      num_hidden,
      dtype=DATA_TYPE,
      weights_initializer=tf.zeros_initializer(dtype=DATA_TYPE),
      bias_initializer=tf.zeros_initializer(dtype=DATA_TYPE),
      reset_after=False,
      **kwargs)
  outputs, _ = gru_cell(x,
                        sequence_len,
                        initial_state=initial_state,
                        training=True)

  softmax = tf.nn.softmax_cross_entropy_with_logits_v2(
      logits=outputs[-1], labels=tf.stop_gradient(y))
  loss = tf.reduce_mean(softmax)
  train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
  return [loss, train]


# pylint: disable=unused-argument
def _PopnnGRU_ResetAfter(x,
                         initial_state,
                         y,
                         sequence_len=None,
                         num_hidden=NUM_HIDDEN,
                         **kwargs):
  gru_cell = layers.PopnnGRU(
      num_hidden,
      dtype=DATA_TYPE,
      weights_initializer=tf.zeros_initializer(dtype=DATA_TYPE),
      bias_initializer=tf.zeros_initializer(dtype=DATA_TYPE),
      reset_after=True,
      **kwargs)
  outputs, _ = gru_cell(x, initial_state=initial_state, training=True)
  softmax = tf.nn.softmax_cross_entropy_with_logits_v2(
      logits=outputs[-1], labels=tf.stop_gradient(y))
  loss = tf.reduce_mean(softmax)
  train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
  return [loss, train]


def _tfGRU(x,
           initial_state,
           y,
           sequence_len=None,
           num_hidden=NUM_HIDDEN,
           **kwargs):
  gru_cell = tf.nn.rnn_cell.GRUCell(
      num_hidden,
      name='gru_cell',
      kernel_initializer=tf.zeros_initializer(dtype=DATA_TYPE),
      bias_initializer=tf.zeros_initializer(dtype=DATA_TYPE),
      **kwargs)
  outputs, _ = tf.nn.dynamic_rnn(gru_cell,
                                 x,
                                 sequence_length=sequence_len,
                                 dtype=DATA_TYPE,
                                 initial_state=initial_state,
                                 time_major=True)

  softmax = tf.nn.softmax_cross_entropy_with_logits_v2(
      logits=outputs[-1], labels=tf.stop_gradient(y))
  loss = tf.reduce_mean(softmax)
  train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
  return [loss, train]


def get_one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def _generate_inputs(seq_len=SEQ_LEN,
                     batch_size=BATCH_SIZE,
                     input_size=INPUT_SIZE):
  n = seq_len * batch_size * input_size
  inputs = np.arange(0, n)
  X = np.reshape(inputs, (seq_len, batch_size, input_size))
  # normalize
  return X / n


def _generate_outputs(batch_size=BATCH_SIZE, num_hidden=NUM_HIDDEN):
  labels = np.zeros([batch_size, num_hidden], dtype=DATA_TYPE)
  labels[:, 0] = 1.

  return labels


def _total_tile_memory(report):
  return sum(tile.memory.total.excludingGaps
             for tile in report.compilation.tiles)


class GRUTrainingTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  def _configure_ipu(self):
    opts = IPUConfig()
    opts.ipu_model.compile_ipu_code = False
    opts.configure_ipu_system()

  def _run_layer(self,
                 layer_func,
                 x,
                 y,
                 s=None,
                 batch_size=BATCH_SIZE,
                 num_hidden=NUM_HIDDEN,
                 num_training_steps=NUM_TRAINING_STEPS,
                 **kwargs):
    with self.session() as sess:
      with tf.device('cpu'):
        px = tf.placeholder(DATA_TYPE, shape=x.shape)
        pi_state = tf.placeholder(DATA_TYPE, shape=[batch_size, num_hidden])
        py = tf.placeholder(DATA_TYPE, shape=y.shape)
        compile_inputs = [px, pi_state, py]
        fd = {px: x, pi_state: np.zeros(pi_state.shape), py: y}

        if s is not None:
          ps = tf.placeholder(np.int32, shape=s.shape)
          compile_inputs.append(ps)
          fd[ps] = s

      def wrapped_layer_func(px, pi_state, py, ps=None):
        return layer_func(px, pi_state, py, ps, num_hidden, **kwargs)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(wrapped_layer_func, inputs=compile_inputs)

      utils.move_variable_initialization_to_cpu()
      sess.run(tf.global_variables_initializer())
      losses = []
      for _ in range(0, num_training_steps):
        loss = sess.run(r, fd)
        losses.append(loss)
    return losses

  # Check that the loss goes down (and is identical to reference version).
  @test_util.deprecated_graph_mode_only
  def testTraining(self):
    np.random.seed(42)
    # prepare the dataset of input to output pairs encoded as integers
    X = _generate_inputs()

    # geneate a target
    labels = _generate_outputs()

    custom_losses = self._run_layer(_PopnnGRU, X, labels)
    # Check the loss goes down
    self.assertTrue(custom_losses[0] > custom_losses[-1])
    # Check that the loss is the same for the reference as well
    ref_losses = self._run_layer(_tfGRU, X, labels)
    self.assertTrue(ref_losses[0] > ref_losses[-1])
    self.assertAllClose(custom_losses, ref_losses, rtol=0.05)

  @test_util.deprecated_graph_mode_only
  def testTrainingWithSeqLen(self):
    self._configure_ipu()
    X = _generate_inputs()
    S = np.array([(i % SEQ_LEN) + 1 for i in range(BATCH_SIZE)])

    # Generate a target
    labels = _generate_outputs()

    custom_losses = self._run_layer(_PopnnGRU_DynamicGRU, X, labels, s=S)

    # Check the loss goes down
    self.assertTrue(custom_losses[0] > custom_losses[-1])
    # Check that the loss is the same for the reference as well
    ref_losses = self._run_layer(_tfGRU, X, labels, s=S)
    self.assertTrue(ref_losses[0] > ref_losses[-1])
    self.assertAllClose(custom_losses, ref_losses, rtol=0.05)

  @test_util.deprecated_graph_mode_only
  def testTraining_resetAfter(self):
    self._configure_ipu()
    X = _generate_inputs()
    labels = _generate_outputs()

    custom_losses = self._run_layer(_PopnnGRU_ResetAfter, X, labels)
    # Check the loss goes down
    self.assertTrue(custom_losses[0] > custom_losses[-1])

    # TF GRU does not support reset_after so no reference comparison
    # is done here.

  @parameterized.parameters((True,), (False,))
  @test_util.deprecated_graph_mode_only
  def testGRUWithAvailableMemoryProportionFwd(self, valid_value):
    self._configure_ipu()

    def run_gru(options):
      X = _generate_inputs()
      labels = _generate_outputs()

      return lambda: self._run_layer(
          _PopnnGRU, X, labels, options=options, options_bwd=options)

    if valid_value:
      run_gru({'availableMemoryProportion': 0.8})()
    else:
      self.assertRaisesRegex(tf.errors.InternalError,
                             "Value must be greater than or equal to 0",
                             run_gru({'availableMemoryProportion': -123.}))

  @test_util.deprecated_graph_mode_only
  def testGRUGreaterAvailableMemoryProportionFwdMeansGreaterTotalTileMemory(
      self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 32
    cfg.configure_ipu_system()

    name = "availableMemoryProportion"
    seq_len = 1
    batch_size = 256
    input_size = 256
    num_hidden = 256

    X = _generate_inputs(seq_len=seq_len,
                         batch_size=batch_size,
                         input_size=input_size)
    labels = _generate_outputs(batch_size=batch_size, num_hidden=num_hidden)

    self._run_layer(_PopnnGRU,
                    X,
                    labels,
                    name=name,
                    batch_size=batch_size,
                    num_hidden=num_hidden,
                    options={'availableMemoryProportion': 0.8},
                    options_bwd={'availableMemoryProportion': 0.8})
    self._run_layer(_PopnnGRU,
                    X,
                    labels,
                    name=name,
                    batch_size=batch_size,
                    num_hidden=num_hidden,
                    options={'availableMemoryProportion': 0.1},
                    options_bwd={'availableMemoryProportion': 0.1})

    report_paths = report_helper.find_reports()
    self.assertEqual(len(report_paths), 2)
    reports = [pva.openReport(report) for report in report_paths]

    self.assertGreater(_total_tile_memory(reports[0]),
                       _total_tile_memory(reports[1]))

  def _run_single_gru_training_step(self,
                                    name=None,
                                    batch_size=BATCH_SIZE,
                                    input_size=INPUT_SIZE,
                                    num_hidden=NUM_HIDDEN,
                                    options_bwd=None):
    X = _generate_inputs(batch_size=batch_size, input_size=input_size)
    labels = _generate_outputs(batch_size=batch_size, num_hidden=num_hidden)

    self._run_layer(_PopnnGRU,
                    X,
                    labels,
                    name=name,
                    batch_size=batch_size,
                    num_hidden=num_hidden,
                    num_training_steps=1,
                    options_bwd=options_bwd)

  @parameterized.parameters((True), (False))
  @test_util.deprecated_graph_mode_only
  def testGRUWithAvailableMemoryProportionBwd(self, valid_value):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    name = ("" if valid_value else "in") + "validAvailableMemoryProportionBwd"

    with self.session():
      if valid_value:
        self._run_single_gru_training_step(
            name=name, options_bwd={'availableMemoryProportion': 0.7})
      else:
        with self.assertRaisesRegex(
            tf.errors.InternalError,
            "Value must be greater than or equal to 0"):
          self._run_single_gru_training_step(
              name=name, options_bwd={'availableMemoryProportion': -123.})

  @test_util.deprecated_graph_mode_only
  def testGRUGreaterAvailableMemoryProportionBwdMeansGreaterTotalTileMemory(
      self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 32
    cfg.configure_ipu_system()

    name = "availableMemoryProportion"
    batch_size = 256
    input_size = 256
    num_hidden = 256

    self._run_single_gru_training_step(
        name=name,
        batch_size=batch_size,
        input_size=input_size,
        num_hidden=num_hidden,
        options_bwd={'availableMemoryProportion': 0.8})
    self._run_single_gru_training_step(
        name=name,
        batch_size=batch_size,
        input_size=input_size,
        num_hidden=num_hidden,
        options_bwd={'availableMemoryProportion': 0.1})

    report_paths = report_helper.find_reports()
    self.assertEqual(len(report_paths), 2)
    reports = [pva.openReport(report) for report in report_paths]

    self.assertGreater(_total_tile_memory(reports[0]),
                       _total_tile_memory(reports[1]))


if __name__ == "__main__":
  googletest.main()
