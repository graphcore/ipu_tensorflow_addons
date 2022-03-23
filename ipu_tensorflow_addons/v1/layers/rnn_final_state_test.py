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
from absl.testing import parameterized
from tensorflow.compat.v1 import disable_v2_behavior
from tensorflow.python import keras
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ipu import config
from tensorflow.python.ipu import scopes
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

from ipu_tensorflow_addons.v1.layers import rnn_ops

_BATCH_SIZE = 6
_MAX_TIME = 5
_INPUT_SIZE = 2
_NUM_UNITS = 3


def as_list(value):
  if isinstance(value, (list, tuple)):
    return list(value)
  return [value]


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


@parameterized.named_parameters({
    'testcase_name': f'_{return_state}',
    'return_state': return_state,
} for return_state in (True, False))
class LstmTest(RnnTest, parameterized.TestCase):
  def call_lstm_ipu(self, inputs, return_state):
    layer = rnn_ops.PopnnLSTM(
        _NUM_UNITS,
        weights_initializer=keras.initializers.Ones(),
        bias_initializer=keras.initializers.Ones(),
        return_state=return_state,
    )

    with scopes.ipu_scope("/device:IPU:0"):
      result = layer(inputs)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      return as_list(sess.run(result))

  def call_dynamic_lstm_ipu(self, inputs, seq_lens, return_state):
    layer = rnn_ops.PopnnDynamicLSTM(
        _NUM_UNITS,
        weights_initializer=keras.initializers.Ones(),
        bias_initializer=keras.initializers.Ones(),
        return_state=return_state,
    )

    with scopes.ipu_scope("/device:IPU:0"):
      result = layer(inputs, seq_lens)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      return as_list(sess.run(result))

  def call_lstm_cpu(self, inputs, mask, return_state):
    layer = recurrent_v2.LSTM(_NUM_UNITS,
                              return_sequences=True,
                              return_state=return_state,
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
      return as_list(sess.run(result))

  def test_lstm(self, return_state):
    inputs = self.make_inputs()

    cpu_result = self.call_lstm_cpu(inputs, None, return_state)
    ipu_result = self.call_lstm_ipu(inputs, return_state)

    if return_state:  # Flatten the output from the PopNN implementation.
      ipu_result = [ipu_result[0], ipu_result[1][1], ipu_result[1][0]]

    self.assertAllClose(cpu_result, ipu_result)

  def test_dynamic_lstm(self, return_state):
    inputs = self.make_inputs()
    seq_lens = self.make_seq_lens()
    inputs_mask = self.get_inputs_mask(seq_lens)
    outputs_mask = self.get_outputs_mask(inputs_mask)

    cpu_result = self.call_lstm_cpu(inputs, inputs_mask, return_state)
    ipu_result = self.call_dynamic_lstm_ipu(inputs, seq_lens, return_state)

    if return_state:  # Flatten the output from the PopNN implementation.
      ipu_result = [ipu_result[0], ipu_result[1][1], ipu_result[1][0]]

    # In the CPU LSTM layer implementation, the last valid output gets copied
    # for the remaining time-steps. Below, these values get zeroed so that the
    # output can be compared against the IPU output.
    cpu_result[0] = np.where(outputs_mask, cpu_result[0], 0)

    self.assertAllEqual(outputs_mask, ipu_result[0] != 0)
    self.assertAllClose(cpu_result, ipu_result)


@parameterized.named_parameters({
    'testcase_name': f'_{return_state}',
    'return_state': return_state,
} for return_state in (True, False))
class GruTest(RnnTest, parameterized.TestCase):
  def make_attention_score(self):
    score = np.random.normal(0, 1, (_MAX_TIME, _BATCH_SIZE))
    return score.astype(np.float32)

  def call_gru_ipu(self, inputs, return_state):
    layer = rnn_ops.PopnnGRU(
        _NUM_UNITS,
        weights_initializer=keras.initializers.Ones(),
        bias_initializer=keras.initializers.Ones(),
        return_state=return_state,
    )

    with scopes.ipu_scope("/device:IPU:0"):
      result = layer(inputs)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      return as_list(sess.run(result))

  def call_dynamic_gru_ipu(self, inputs, seq_lens, return_state):
    layer = rnn_ops.PopnnDynamicGRU(
        _NUM_UNITS,
        weights_initializer=keras.initializers.Ones(),
        bias_initializer=keras.initializers.Ones(),
        return_state=return_state,
    )

    with scopes.ipu_scope("/device:IPU:0"):
      result = layer(inputs, seq_lens)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      return as_list(sess.run(result))

  def call_augru_ipu(self, inputs, seq_lens, attention_score, return_state):
    layer = rnn_ops.PopnnAUGRU(
        _NUM_UNITS,
        weights_initializer=keras.initializers.Ones(),
        bias_initializer=keras.initializers.Ones(),
        return_state=return_state,
    )

    with scopes.ipu_scope("/device:IPU:0"):
      result = layer(inputs, seq_lens, attention_score)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      return as_list(sess.run(result))

  def call_gru_cpu(self, inputs, mask, return_state):
    layer = recurrent_v2.GRU(_NUM_UNITS,
                             return_sequences=True,
                             return_state=return_state,
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
      return as_list(sess.run(result))

  def test_gru(self, return_state):
    inputs = self.make_inputs()

    cpu_result = self.call_gru_cpu(inputs, None, return_state)
    ipu_result = self.call_gru_ipu(inputs, return_state)

    self.assertAllClose(cpu_result, ipu_result)

  def test_dynamic_gru(self, return_state):
    inputs = self.make_inputs()
    seq_lens = self.make_seq_lens()
    inputs_mask = self.get_inputs_mask(seq_lens)
    outputs_mask = self.get_outputs_mask(inputs_mask)

    cpu_result = self.call_gru_cpu(inputs, inputs_mask, return_state)
    ipu_result = self.call_dynamic_gru_ipu(inputs, seq_lens, return_state)

    # In the CPU GRU layer implementation, the last valid output gets copied for
    # the remaining time-steps. Below, these values get zeroed so that the
    # output can be compared against the IPU output.
    cpu_result[0] = np.where(outputs_mask, cpu_result[0], 0)

    self.assertAllEqual(outputs_mask, ipu_result[0] != 0)
    self.assertAllClose(cpu_result, ipu_result)

  def test_augru(self, return_state):
    inputs = self.make_inputs()
    seq_lens = self.make_seq_lens()
    inputs_mask = self.get_inputs_mask(seq_lens)
    outputs_mask = self.get_outputs_mask(inputs_mask)
    attention_score = self.make_attention_score()

    ipu_result = self.call_augru_ipu(inputs, seq_lens, attention_score,
                                     return_state)

    self.assertAllEqual(outputs_mask, ipu_result[0] != 0)
    if return_state:
      self.assertNotAllEqual(ipu_result[1], 0)


if __name__ == '__main__':
  disable_v2_behavior()

  # Configure IPUs
  cfg = config.IPUConfig()
  cfg.ipu_model.compile_ipu_code = False
  cfg.configure_ipu_system()

  test.main()
