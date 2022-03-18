# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the final state of RNN cells."""

import numpy as np
from tensorflow.compat.v1 import disable_v2_behavior
from tensorflow.python import keras
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ipu import config
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import scopes
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

from ipu_tensorflow_addons.v1.layers import rnn_ops

_BATCH_SIZE = 6
_MAX_TIME = 5
_INPUT_SIZE = 2
_NUM_UNITS = 3


class RnnTest(test.TestCase):
  def make_inputs(self):
    inputs = np.random.normal(0, 1, (_MAX_TIME, _BATCH_SIZE, _INPUT_SIZE))
    return inputs.astype(np.float32)

  def make_seq_lens(self):
    seq_lens = np.arange(_BATCH_SIZE) % _MAX_TIME + 1
    np.random.shuffle(seq_lens)
    return seq_lens.astype(np.int32)

  def get_inputs_mask(self, seq_lens):
    mask = np.array([[t < seq_lens[b] for b in range(_BATCH_SIZE)]
                     for t in range(_MAX_TIME)])
    return mask

  def get_outputs_mask(self, inputs_mask):
    mask = np.tile(inputs_mask[:, :, None], [1, 1, _NUM_UNITS])
    return mask


class LstmTest(RnnTest):
  def call_lstm_ipu(self, inputs):
    layer = rnn_ops.PopnnLSTM(_NUM_UNITS,
                              weights_initializer=keras.initializers.Ones(),
                              bias_initializer=keras.initializers.Ones())

    with scopes.ipu_scope("/device:IPU:0"):
      result = ipu_compiler.compile(layer, inputs=(inputs,))

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      outputs, (c_state, h_state) = sess.run(result)

    return outputs, h_state, c_state

  def call_dynamic_lstm_ipu(self, inputs, seq_lens):
    layer = rnn_ops.PopnnDynamicLSTM(
        _NUM_UNITS,
        weights_initializer=keras.initializers.Ones(),
        bias_initializer=keras.initializers.Ones())

    with scopes.ipu_scope("/device:IPU:0"):
      result = ipu_compiler.compile(layer, inputs=(inputs, seq_lens))

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      outputs, (c_state, h_state) = sess.run(result)

    return outputs, h_state, c_state

  def call_lstm_cpu(self, inputs, mask=None):
    layer = recurrent_v2.LSTM(_NUM_UNITS,
                              return_sequences=True,
                              return_state=True,
                              kernel_initializer=keras.initializers.Ones(),
                              recurrent_initializer=keras.initializers.Ones(),
                              bias_initializer=keras.initializers.Ones(),
                              dropout=0.0,
                              unit_forget_bias=False,
                              stateful=False,
                              time_major=True)
    with ops.device('cpu'):
      kwargs = {}
      if mask is not None:
        kwargs['mask'] = constant_op.constant(mask)
      result = layer(inputs, **kwargs)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      outputs, h_state, c_ctate = sess.run(result, feed_dict={})

    return outputs, h_state, c_ctate

  def test_lstm(self):
    inputs = self.make_inputs()

    cpu_outputs, cpu_h_state, cpu_c_ctate = self.call_lstm_cpu(inputs)
    ipu_outputs, ipu_h_state, ipu_c_ctate = self.call_lstm_ipu(inputs)

    self.assertAllClose(cpu_outputs, ipu_outputs)
    self.assertAllClose(cpu_h_state, ipu_h_state)
    self.assertAllClose(cpu_c_ctate, ipu_c_ctate)

  def test_dynamic_lstm(self):
    inputs = self.make_inputs()
    seq_lens = self.make_seq_lens()
    inputs_mask = self.get_inputs_mask(seq_lens)
    outputs_mask = self.get_outputs_mask(inputs_mask)

    cpu_outputs, cpu_h_state, cpu_c_ctate = self.call_lstm_cpu(  # pylint: disable=unused-variable
        inputs, inputs_mask)
    ipu_outputs, ipu_h_state, ipu_c_ctate = self.call_dynamic_lstm_ipu(  # pylint: disable=unused-variable
        inputs, seq_lens)

    # In the CPU LSTM layer implementation, the last valid output gets copied
    # for the remaining time-steps. Below, these values get zeroed so that the
    # output can be compared against the IPU output.
    cpu_outputs = array_ops.where_v2(outputs_mask, cpu_outputs, 0)

    self.assertAllEqual(outputs_mask, ipu_outputs != 0)
    self.assertAllClose(cpu_outputs, ipu_outputs)
    self.assertAllClose(cpu_h_state, ipu_h_state)
    # TODO(T41670): Enable this case.
    # self.assertAllClose(cpu_c_ctate, ipu_c_ctate)


class GruTest(RnnTest):
  def make_attention_score(self):
    score = np.random.normal(0, 1, (_MAX_TIME, _BATCH_SIZE))
    return score.astype(np.float32)

  def call_gru_ipu(self, inputs):
    layer = rnn_ops.PopnnGRU(
        _NUM_UNITS,
        weights_initializer=keras.initializers.Ones(),
        bias_initializer=keras.initializers.Ones(),
    )

    with scopes.ipu_scope("/device:IPU:0"):
      result = ipu_compiler.compile(layer, inputs=(inputs,))

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      outputs, h_state = sess.run(result)

    return outputs, h_state

  def call_dynamic_gru_ipu(self, inputs, seq_lens):
    layer = rnn_ops.PopnnDynamicGRU(
        _NUM_UNITS,
        weights_initializer=keras.initializers.Ones(),
        bias_initializer=keras.initializers.Ones(),
    )

    with scopes.ipu_scope("/device:IPU:0"):
      result = ipu_compiler.compile(layer, inputs=(inputs, seq_lens))

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      outputs, h_state = sess.run(result)

    return outputs, h_state

  def call_augru_ipu(self, inputs, seq_lens, attention_score):
    layer = rnn_ops.PopnnAUGRU(
        _NUM_UNITS,
        weights_initializer=keras.initializers.Ones(),
        bias_initializer=keras.initializers.Ones(),
    )

    with scopes.ipu_scope("/device:IPU:0"):
      result = ipu_compiler.compile(layer,
                                    inputs=(inputs, seq_lens, attention_score))

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      outputs, h_state = sess.run(result)

    return outputs, h_state

  def call_gru_cpu(self, inputs, mask=None):
    layer = recurrent_v2.GRU(_NUM_UNITS,
                             return_sequences=True,
                             return_state=True,
                             kernel_initializer=keras.initializers.Ones(),
                             recurrent_initializer=keras.initializers.Ones(),
                             bias_initializer=keras.initializers.Ones(),
                             dropout=0.0,
                             stateful=False,
                             time_major=True,
                             reset_after=False)

    with ops.device('cpu'):
      kwargs = {}
      if mask is not None:
        kwargs['mask'] = constant_op.constant(mask)
      result = layer(inputs, **kwargs)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      outputs, h_state = sess.run(result, feed_dict={})

    return outputs, h_state

  def test_gru(self):
    inputs = self.make_inputs()

    cpu_outputs, cpu_h_state = self.call_gru_cpu(inputs)
    ipu_outputs, ipu_h_state = self.call_gru_ipu(inputs)

    self.assertAllClose(cpu_outputs, ipu_outputs)
    self.assertAllClose(cpu_h_state, ipu_h_state)

  def test_dynamic_gru(self):
    inputs = self.make_inputs()
    seq_lens = self.make_seq_lens()
    inputs_mask = self.get_inputs_mask(seq_lens)
    outputs_mask = self.get_outputs_mask(inputs_mask)

    cpu_outputs, cpu_h_state = self.call_gru_cpu(inputs, inputs_mask)
    ipu_outputs, ipu_h_state = self.call_dynamic_gru_ipu(inputs, seq_lens)

    # In the CPU GRU layer implementation, the last valid output gets copied for
    # the remaining time-steps. Below, these values get zeroed so that the
    # output can be compared against the IPU output.
    cpu_outputs = array_ops.where_v2(outputs_mask, cpu_outputs, 0)

    self.assertAllEqual(outputs_mask, ipu_outputs != 0)
    self.assertAllClose(cpu_outputs, ipu_outputs)
    self.assertAllClose(cpu_h_state, ipu_h_state)

  def test_augru(self):
    inputs = self.make_inputs()
    seq_lens = self.make_seq_lens()
    inputs_mask = self.get_inputs_mask(seq_lens)
    outputs_mask = self.get_outputs_mask(inputs_mask)
    attention_score = self.make_attention_score()

    ipu_outputs, ipu_h_state = self.call_augru_ipu(inputs, seq_lens,
                                                   attention_score)

    self.assertAllEqual(outputs_mask, ipu_outputs != 0)
    self.assertNotAllEqual(ipu_h_state, 0)


if __name__ == '__main__':
  disable_v2_behavior()

  # Configure IPUs
  cfg = config.IPUConfig()
  cfg.auto_select_ipus = 1
  cfg.configure_ipu_system()

  test.main()
