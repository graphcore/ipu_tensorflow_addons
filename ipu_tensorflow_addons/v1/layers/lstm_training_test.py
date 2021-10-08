# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Naive LSTM to learn three-char time steps to one-char mapping
import numpy as np
from tensorflow.python import ipu
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.ipu.config import IPUConfig

from ipu_tensorflow_addons.v1 import layers

dataType = np.float32

seq_len = 3
batch_size = 40 - seq_len
input_size = 1
num_hidden = 64
num_training_steps = 100
lr = 10


# pylint: disable=unused-argument
def _PopnnLSTM(x, h, c, y, sequence_len=None):
  lstm_cell = layers.PopnnLSTM(
      num_hidden,
      dtype=dataType,
      weights_initializer=init_ops.zeros_initializer(dtype=dataType),
      bias_initializer=init_ops.zeros_initializer(dtype=dataType))
  state = rnn_cell.LSTMStateTuple(c, h)
  outputs, _ = lstm_cell(x, initial_state=state, training=True)
  softmax = nn.softmax_cross_entropy_with_logits_v2(
      logits=outputs[-1], labels=array_ops.stop_gradient(y))
  loss = math_ops.reduce_mean(softmax)
  train = gradient_descent.GradientDescentOptimizer(lr).minimize(loss)
  return [loss, train]


def _PopnnLSTM_DynamicLSTM(x, h, c, y, sequence_len=None):
  lstm_cell = layers.PopnnDynamicLSTM(
      num_hidden,
      dtype=dataType,
      weights_initializer=init_ops.zeros_initializer(dtype=dataType),
      bias_initializer=init_ops.zeros_initializer(dtype=dataType))
  state = rnn_cell.LSTMStateTuple(c, h)
  outputs, _ = lstm_cell(x, sequence_len, initial_state=state, training=True)
  softmax = nn.softmax_cross_entropy_with_logits_v2(
      logits=outputs[-1], labels=array_ops.stop_gradient(y))
  loss = math_ops.reduce_mean(softmax)
  train = gradient_descent.GradientDescentOptimizer(lr).minimize(loss)
  return [loss, train]


def _tfLSTM(x, h, c, y, sequence_len=None):
  lstm_cell = rnn_cell.LSTMCell(
      num_hidden,
      name='basic_lstm_cell',
      forget_bias=0.,
      initializer=init_ops.zeros_initializer(dtype=dataType))
  state = rnn_cell.LSTMStateTuple(c, h)
  outputs, _ = rnn.dynamic_rnn(lstm_cell,
                               x,
                               sequence_length=sequence_len,
                               dtype=dataType,
                               initial_state=state,
                               time_major=True)
  softmax = nn.softmax_cross_entropy_with_logits_v2(
      logits=outputs[-1], labels=array_ops.stop_gradient(y))
  loss = math_ops.reduce_mean(softmax)
  train = gradient_descent.GradientDescentOptimizer(lr).minimize(loss)
  return [loss, train]


def get_one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


class LstmTrainingTest(test_util.TensorFlowTestCase):
  def _RunLayer(self, layer_func, x, y, s=None):
    with self.session() as sess:
      with ops.device('cpu'):
        px = array_ops.placeholder(dataType, shape=x.shape)
        ph = array_ops.placeholder(dataType, shape=[batch_size, num_hidden])
        pc = array_ops.placeholder(dataType, shape=[batch_size, num_hidden])
        py = array_ops.placeholder(dataType, shape=y.shape)

        compile_inputs = [px, ph, pc, py]
        fd = {px: x, ph: np.ones(ph.shape), pc: np.ones(pc.shape), py: y}
        if s is not None:
          ps = array_ops.placeholder(np.int32, shape=s.shape)
          compile_inputs.append(ps)
          fd[ps] = s

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(layer_func, inputs=compile_inputs)

      opts = IPUConfig()
      opts._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      opts._profiling.use_poplar_text_report = True  # pylint: disable=protected-access
      opts.ipu_model.compile_ipu_code = False
      opts.configure_ipu_system()

      sess.run(variables.global_variables_initializer())
      losses = []
      for _ in range(0, num_training_steps):
        loss = sess.run(r, fd)
        losses.append(loss)
    return losses

  # Check that the loss goes down (and is identical to reference version).
  @test_util.deprecated_graph_mode_only
  def testTraining(self):
    np.random.seed(42)
    nums = np.arange(batch_size + seq_len)
    # prepare the dataset of input to output pairs encoded as integers
    inputs = []
    one_hot = []
    for i in range(0, len(nums) - seq_len):
      sequence = nums[i:i + seq_len]
      output = nums[i + seq_len]
      inputs.append(sequence)
      one_hot.append(output)
    X = np.reshape(inputs, (seq_len, batch_size, input_size))
    # normalize
    X = X / float(len(nums))
    # one hot encode the output variable
    y = get_one_hot(nums[seq_len:], nums.size)
    labels = np.zeros([batch_size, num_hidden], dtype=dataType)
    labels[:y.shape[0], :y.shape[1]] = y

    custom_losses = self._RunLayer(_PopnnLSTM, X, labels)
    # Check the loss goes down
    self.assertTrue(custom_losses[0] > custom_losses[-1])
    # Check that the loss is the same for the reference as well
    ref_losses = self._RunLayer(_tfLSTM, X, labels)
    self.assertAllClose(custom_losses, ref_losses, atol=0.01)

  @test_util.deprecated_graph_mode_only
  def testTrainingWithSeqLen(self):
    np.random.seed(42)
    nums = np.arange(batch_size + seq_len)
    # prepare the dataset of input to output pairs encoded as integers
    inputs = []
    for i in range(0, len(nums) - seq_len):
      sequence = nums[i:i + seq_len]
      inputs.append(sequence)
    X = np.reshape(inputs, (seq_len, batch_size, input_size))
    S = np.array([(i % seq_len) + 1 for i in range(batch_size)])
    # normalize
    X = X / float(len(nums))

    # Generate a target
    labels = np.zeros([batch_size, num_hidden], dtype=dataType)
    labels[:, 0] = 1.

    custom_losses = self._RunLayer(_PopnnLSTM_DynamicLSTM, X, labels, s=S)

    # Check the loss goes down
    self.assertTrue(custom_losses[0] > custom_losses[-1])
    # Check that the loss is the same for the reference as well
    ref_losses = self._RunLayer(_tfLSTM, X, labels, s=S)
    self.assertTrue(ref_losses[0] > ref_losses[-1])
    self.assertAllClose(custom_losses, ref_losses, rtol=0.05)


if __name__ == "__main__":
  googletest.main()