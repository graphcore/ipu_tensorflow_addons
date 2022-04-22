# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pva
import numpy as np
import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ipu import test_utils as tu
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from ipu_tensorflow_addons import layers

DATA_TYPE = np.float16

BATCH_SIZE = 1
INPUT_SIZE = 28
TIME_STEPS = 5
NUM_HIDDEN = 512


def _PopnnLSTM(x, h, c):
  lstm_cell = layers.PopnnLSTM(
      NUM_HIDDEN,
      dtype=DATA_TYPE,
      weights_initializer=tf.zeros_initializer(dtype=DATA_TYPE),
      bias_initializer=tf.zeros_initializer(dtype=DATA_TYPE))
  state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
  return lstm_cell(x, initial_state=state, training=False)


def _tfLSTM(x, h, c):
  lstm_cell = tf.nn.rnn_cell.LSTMCell(
      NUM_HIDDEN,
      name='basic_lstm_cell',
      forget_bias=0.,
      initializer=tf.zeros_initializer(dtype=DATA_TYPE))
  state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
  return tf.nn.dynamic_rnn(lstm_cell,
                           x,
                           dtype=DATA_TYPE,
                           initial_state=state,
                           time_major=True)


class LstmSizeTest(test_util.TensorFlowTestCase):
  def RunLayer(self, layer_func, x):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with tf.device('cpu'):
        px = tf.placeholder(DATA_TYPE, shape=x.shape)
        ph = tf.placeholder(DATA_TYPE, shape=[BATCH_SIZE, NUM_HIDDEN])
        pc = tf.placeholder(DATA_TYPE, shape=[BATCH_SIZE, NUM_HIDDEN])
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(layer_func, inputs=[px, ph, pc])

      sess.run(tf.global_variables_initializer())
      report_helper.clear_reports()
      result = sess.run(r, {
          px: x,
          ph: np.ones(ph.shape),
          pc: np.ones(pc.shape)
      })

    report = pva.openReport(report_helper.find_report())
    size = sum(tile.memory.total.excludingGaps
               for tile in report.compilation.tiles)
    return (size, result)

  # Test which verifies that:
  # 1. Custom op uses less memory
  # 2. Custom op and Tf op return the same result
  @test_util.deprecated_graph_mode_only
  def testCustomOpIsSmaller(self):
    np.random.seed(42)
    x = np.random.rand(TIME_STEPS, BATCH_SIZE, INPUT_SIZE).astype(DATA_TYPE)
    size_custom_op, result_custom_op = self.RunLayer(_PopnnLSTM, x)
    size_tf, result_tf = self.RunLayer(_tfLSTM, x)
    self.assertAllClose(result_custom_op, result_tf)
    self.assertTrue(size_custom_op < size_tf)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
