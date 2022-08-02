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

# Naive LSTM to learn three-char time steps to one-char mapping
from absl.testing import parameterized
import numpy as np
import pva
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
def _PopnnLSTM(x, h, c, y, sequence_len=None, num_hidden=NUM_HIDDEN, **kwargs):
  lstm_cell = layers.PopnnLSTM(
      num_hidden,
      dtype=DATA_TYPE,
      weights_initializer=tf.zeros_initializer(dtype=DATA_TYPE),
      bias_initializer=tf.zeros_initializer(dtype=DATA_TYPE),
      **kwargs)
  state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
  outputs, _ = lstm_cell(x, initial_state=state, training=True)
  softmax = tf.nn.softmax_cross_entropy_with_logits_v2(
      logits=outputs[-1], labels=tf.stop_gradient(y))
  loss = tf.reduce_mean(softmax)
  train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
  return [loss, train]


def _PopnnLSTM_DynamicLSTM(x,
                           h,
                           c,
                           y,
                           sequence_len=None,
                           num_hidden=NUM_HIDDEN,
                           **kwargs):
  lstm_cell = layers.PopnnDynamicLSTM(
      num_hidden,
      dtype=DATA_TYPE,
      weights_initializer=tf.zeros_initializer(dtype=DATA_TYPE),
      bias_initializer=tf.zeros_initializer(dtype=DATA_TYPE),
      **kwargs)
  state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
  outputs, _ = lstm_cell(x, sequence_len, initial_state=state, training=True)
  softmax = tf.nn.softmax_cross_entropy_with_logits_v2(
      logits=outputs[-1], labels=tf.stop_gradient(y))
  loss = tf.reduce_mean(softmax)
  train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
  return [loss, train]


def _tfLSTM(x, h, c, y, sequence_len=None, num_hidden=NUM_HIDDEN, **kwargs):
  lstm_cell = tf.nn.rnn_cell.LSTMCell(
      num_hidden,
      name='basic_lstm_cell',
      forget_bias=0.,
      initializer=tf.zeros_initializer(dtype=DATA_TYPE),
      **kwargs)
  state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
  outputs, _ = tf.nn.dynamic_rnn(lstm_cell,
                                 x,
                                 sequence_length=sequence_len,
                                 dtype=DATA_TYPE,
                                 initial_state=state,
                                 time_major=True)
  softmax = tf.nn.softmax_cross_entropy_with_logits_v2(
      logits=outputs[-1], labels=tf.stop_gradient(y))
  loss = tf.reduce_mean(softmax)
  train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
  return [loss, train]


def _generate_inputs(seq_len=SEQ_LEN,
                     batch_size=BATCH_SIZE,
                     input_size=INPUT_SIZE):
  # prepare the dataset of input to output pairs encoded as integers
  n = seq_len * batch_size * input_size
  inputs = np.arange(0, n)
  X = np.reshape(inputs, (seq_len, batch_size, input_size))
  # normalize
  return X / n


def _generate_outputs(batch_size=BATCH_SIZE, num_hidden=NUM_HIDDEN):
  # Generate a target
  labels = np.zeros([batch_size, num_hidden], dtype=DATA_TYPE)
  labels[:, 0] = 1.
  return labels


def _total_tile_memory(report):
  return sum(tile.memory.total.excludingGaps
             for tile in report.compilation.tiles)


def get_one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


class LstmTrainingTest(test_util.TensorFlowTestCase, parameterized.TestCase):
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
        ph = tf.placeholder(DATA_TYPE, shape=[batch_size, num_hidden])
        pc = tf.placeholder(DATA_TYPE, shape=[batch_size, num_hidden])
        py = tf.placeholder(DATA_TYPE, shape=y.shape)

        compile_inputs = [px, ph, pc, py]
        fd = {px: x, ph: np.ones(ph.shape), pc: np.ones(pc.shape), py: y}
        if s is not None:
          ps = tf.placeholder(np.int32, shape=s.shape)
          compile_inputs.append(ps)
          fd[ps] = s

      def wrapped_layer_func(px, ph, pc, py, ps=None):
        return layer_func(px, ph, pc, py, ps, num_hidden, **kwargs)

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
    self._configure_ipu()
    nums = np.arange(BATCH_SIZE + SEQ_LEN)
    # prepare the dataset of input to output pairs encoded as integers
    inputs = []
    one_hot = []
    for i in range(0, len(nums) - SEQ_LEN):
      sequence = nums[i:i + SEQ_LEN]
      output = nums[i + SEQ_LEN]
      inputs.append(sequence)
      one_hot.append(output)
    X = np.reshape(inputs, (SEQ_LEN, BATCH_SIZE, INPUT_SIZE))
    # normalize
    X = X / float(len(nums))
    # one hot encode the output variable
    y = get_one_hot(nums[SEQ_LEN:], nums.size)
    labels = np.zeros([BATCH_SIZE, NUM_HIDDEN], dtype=DATA_TYPE)
    labels[:y.shape[0], :y.shape[1]] = y

    custom_losses = self._run_layer(_PopnnLSTM, X, labels)
    # Check the loss goes down
    self.assertTrue(custom_losses[0] > custom_losses[-1])
    # Check that the loss is the same for the reference as well
    ref_losses = self._run_layer(_tfLSTM, X, labels)
    self.assertAllClose(custom_losses, ref_losses, atol=0.01)

  @test_util.deprecated_graph_mode_only
  def testTrainingWithSeqLen(self):
    self._configure_ipu()
    X = _generate_inputs()
    S = np.array([(i % SEQ_LEN) + 1 for i in range(BATCH_SIZE)])

    labels = _generate_outputs()

    custom_losses = self._run_layer(_PopnnLSTM_DynamicLSTM, X, labels, s=S)

    # Check the loss goes down
    self.assertTrue(custom_losses[0] > custom_losses[-1])
    # Check that the loss is the same for the reference as well
    ref_losses = self._run_layer(_tfLSTM, X, labels, s=S)
    self.assertTrue(ref_losses[0] > ref_losses[-1])
    self.assertAllClose(custom_losses, ref_losses, rtol=0.05)

  @parameterized.parameters((True,), (False,))
  @test_util.deprecated_graph_mode_only
  def testLSTMWithAvailableMemoryProportionFwd(self, valid_value):
    self._configure_ipu()

    def run_lstm(options):
      X = _generate_inputs()
      labels = _generate_outputs()

      return lambda: self._run_layer(
          _PopnnLSTM, X, labels, options=options, options_bwd=options)

    if valid_value:
      run_lstm({'availableMemoryProportion': 0.8})()
    else:
      self.assertRaisesRegex(tf.errors.InternalError,
                             "Value must be greater than or equal to 0",
                             run_lstm({'availableMemoryProportion': -123.}))

  @test_util.deprecated_graph_mode_only
  def testLSTMGreaterAvailableMemoryProportionFwdMeansGreaterTotalTileMemory(
      self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 32
    cfg.configure_ipu_system()

    seq_len = 1
    batch_size = 256
    input_size = 256
    num_hidden = 256

    X = _generate_inputs(seq_len=seq_len,
                         batch_size=batch_size,
                         input_size=input_size)
    labels = _generate_outputs(batch_size=batch_size, num_hidden=num_hidden)

    self._run_layer(_PopnnLSTM,
                    X,
                    labels,
                    batch_size=batch_size,
                    num_hidden=num_hidden,
                    options={'availableMemoryProportion': 0.8},
                    options_bwd={'availableMemoryProportion': 0.8})
    self._run_layer(_PopnnLSTM,
                    X,
                    labels,
                    batch_size=batch_size,
                    num_hidden=num_hidden,
                    options={'availableMemoryProportion': 0.1},
                    options_bwd={'availableMemoryProportion': 0.1})

    report_paths = report_helper.find_reports()
    self.assertEqual(len(report_paths), 2)
    reports = [pva.openReport(report) for report in report_paths]

    self.assertGreater(_total_tile_memory(reports[0]),
                       _total_tile_memory(reports[1]))

  def _run_lstm_single_training_step(self,
                                     name,
                                     batch_size=BATCH_SIZE,
                                     input_size=INPUT_SIZE,
                                     num_hidden=NUM_HIDDEN,
                                     options_bwd=None):
    X = _generate_inputs(batch_size=batch_size, input_size=input_size)
    labels = _generate_outputs(batch_size=batch_size, num_hidden=num_hidden)

    self._run_layer(_PopnnLSTM,
                    X,
                    labels,
                    name=name,
                    batch_size=batch_size,
                    num_hidden=num_hidden,
                    num_training_steps=1,
                    options_bwd=options_bwd)

  @parameterized.parameters((True,), (False,))
  @test_util.deprecated_graph_mode_only
  def testLSTMWithAvailableMemoryProportionBwd(self, valid_value):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    name = ("" if valid_value else "in") + "validAvailableMemoryProportionBwd"

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      if valid_value:
        self._run_lstm_single_training_step(
            name=name, options_bwd={'availableMemoryProportion': 0.7})
      else:
        with self.assertRaisesRegex(
            tf.errors.InternalError,
            "Value must be greater than or equal to 0"):
          self._run_lstm_single_training_step(
              name=name, options_bwd={'availableMemoryProportion': -123.})

  @test_util.deprecated_graph_mode_only
  def testLSTMGreaterAvailableMemoryProportionBwdMeansGreaterTotalTileMemory(
      self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 32
    cfg.configure_ipu_system()

    name = "availableMemoryProportionBwd"
    batch_size = 256
    input_size = 256
    num_hidden = 256

    self._run_lstm_single_training_step(
        name=name,
        batch_size=batch_size,
        input_size=input_size,
        num_hidden=num_hidden,
        options_bwd={'availableMemoryProportion': 0.8})
    self._run_lstm_single_training_step(
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
