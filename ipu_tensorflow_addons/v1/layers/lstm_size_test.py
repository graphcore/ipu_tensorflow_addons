# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
import pva

from ipu_tensorflow_addons import test_utils as tu
from ipu_tensorflow_addons.v1 import layers

DATA_TYPE = np.float16

BATCH_SIZE = 1
INPUT_SIZE = 28
TIME_STEPS = 5
NUM_HIDDEN = 512


def _PopnnLSTM(x, h, c):
  lstm_cell = layers.PopnnLSTM(
      NUM_HIDDEN,
      dtype=DATA_TYPE,
      weights_initializer=init_ops.zeros_initializer(dtype=DATA_TYPE),
      bias_initializer=init_ops.zeros_initializer(dtype=DATA_TYPE))
  state = rnn_cell.LSTMStateTuple(c, h)
  return lstm_cell(x, initial_state=state, training=False)


def _tfLSTM(x, h, c):
  lstm_cell = rnn_cell.LSTMCell(
      NUM_HIDDEN,
      name='basic_lstm_cell',
      forget_bias=0.,
      initializer=init_ops.zeros_initializer(dtype=DATA_TYPE))
  state = rnn_cell.LSTMStateTuple(c, h)
  return rnn.dynamic_rnn(lstm_cell,
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
      with ops.device('cpu'):
        px = array_ops.placeholder(DATA_TYPE, shape=x.shape)
        ph = array_ops.placeholder(DATA_TYPE, shape=[BATCH_SIZE, NUM_HIDDEN])
        pc = array_ops.placeholder(DATA_TYPE, shape=[BATCH_SIZE, NUM_HIDDEN])
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(layer_func, inputs=[px, ph, pc])

      sess.run(variables.global_variables_initializer())
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