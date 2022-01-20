# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for IPU LSTM layers."""

import numpy as np
from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

from ipu_tensorflow_addons.keras import layers

# Test hyperparameters.
batch_size = 1
num_input = 3
timesteps = 4
num_hidden = 5
data_type = np.float32


def test_language_dataset(length=None):
  constant_d = constant_op.constant(1, shape=[32], dtype=np.int32)
  constant_l = constant_op.constant(2, shape=[32], dtype=np.int32)

  ds = dataset_ops.Dataset.from_tensors((constant_d, constant_l))
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


def _getLSTMLayer(keras_layer=None,
                  return_state=True,
                  return_sequences=False,
                  time_major=False,
                  dropout=0.,
                  unit_forget_bias=False,
                  stateful=False,
                  kernel_initializer=None,
                  recurrent_initializer=None,
                  bias_initializer=None,
                  **kwargs):
  kernel_initializer = (kernel_initializer if kernel_initializer else
                        init_ops.constant_initializer(0.1, data_type))
  recurrent_initializer = (recurrent_initializer if recurrent_initializer else
                           init_ops.constant_initializer(0.2, data_type))
  bias_initializer = (bias_initializer if bias_initializer else
                      init_ops.constant_initializer(0.3, data_type))
  return keras_layer(num_hidden,
                     dtype=data_type,
                     kernel_initializer=kernel_initializer,
                     recurrent_initializer=recurrent_initializer,
                     bias_initializer=bias_initializer,
                     recurrent_activation="sigmoid",
                     dropout=dropout,
                     time_major=time_major,
                     return_sequences=return_sequences,
                     return_state=return_state,
                     unit_forget_bias=unit_forget_bias,
                     stateful=stateful,
                     **kwargs)


def _kerasLSTMImpl(instance,
                   x_vals,
                   h_val,
                   c_val,
                   keras_layer=None,
                   device="cpu",
                   training=True,
                   return_state=True,
                   return_sequences=False,
                   time_major=False,
                   dropout=0.,
                   unit_forget_bias=False,
                   stateful=False):

  with ops.device(device):
    x = array_ops.placeholder(x_vals[0].dtype, x_vals[0].shape)
    h = array_ops.placeholder(h_val.dtype, h_val.shape)
    c = array_ops.placeholder(c_val.dtype, c_val.shape)

    state = None if stateful else rnn_cell.LSTMStateTuple(c, h)

    layer = _getLSTMLayer(keras_layer, return_state, return_sequences,
                          time_major, dropout, unit_forget_bias, stateful)
    output = layer(inputs=x, initial_state=state, training=training)
    shapes = [w.shape for w in layer.get_weights()]

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    outputs = []

    # Run the op and any updates.
    to_run = [output, [layer.updates]] if layer.updates else output
    for x_val in x_vals:
      r = sess.run(to_run, {x: x_val, h: h_val, c: c_val})
      r = r[0] if layer.updates else r
      outputs.append(r)
    return (outputs, shapes)


def _lstmIPU(*args, **kwargs):
  return _kerasLSTMImpl(*args,
                        **kwargs,
                        keras_layer=layers.PopnnLSTM,
                        device='/device:IPU:0')


def _lstmCPU(*args, **kwargs):
  return _kerasLSTMImpl(*args, **kwargs, keras_layer=recurrent_v2.LSTM)


class IpuLstmTest(test.TestCase):
  def _get_random_inputs(self, time_major=False, num_samples=1):
    np.random.seed(42)
    h = np.random.rand(batch_size, num_hidden).astype(data_type)
    c = np.random.rand(batch_size, num_hidden).astype(data_type)
    xs = []
    for _ in range(num_samples):
      shape = [timesteps, batch_size, num_input] \
              if time_major else [batch_size, timesteps, num_input]
      xs.append(np.random.rand(*shape).astype(data_type))
    return xs, h, c

  @test_util.deprecated_graph_mode_only
  def test_lstm(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, h, c = self._get_random_inputs()

    cpu_result = _lstmCPU(self, x, h, c)
    ipu_result = _lstmIPU(self, x, h, c)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_lstm_time_major(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, h, c = self._get_random_inputs(time_major=True)

    cpu_result = _lstmCPU(self, x, h, c, time_major=True)
    ipu_result = _lstmIPU(self, x, h, c, time_major=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_lstm_unit_forget_bias(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, h, c = self._get_random_inputs()

    cpu_result = _lstmCPU(self, x, h, c, unit_forget_bias=True)
    ipu_result = _lstmIPU(self, x, h, c, unit_forget_bias=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_lstm_all_seq(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, h, c = self._get_random_inputs()

    ipu_result = _lstmIPU(self, x, h, c, return_sequences=True)
    cpu_result = _lstmCPU(self, x, h, c, return_sequences=True)
    self.assertAllClose(ipu_result, cpu_result)

    self.assertEqual(ipu_result[0][0][0].shape,
                     (batch_size, timesteps, num_hidden))

  @test_util.deprecated_graph_mode_only
  def test_lstm_no_state(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, h, c = self._get_random_inputs()

    ipu_result = _lstmIPU(self, x, h, c, return_state=False)
    self.assertTrue(isinstance(ipu_result[0][0], np.ndarray))

  @test_util.deprecated_graph_mode_only
  def test_class_alias(self):
    self.assertTrue(isinstance(layers.LSTM, type))
    self.assertEqual(layers.PopnnLSTM, layers.LSTM)

  @test_util.deprecated_graph_mode_only
  def test_lstm_dropout(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, h, c = self._get_random_inputs()

    dropout_none_result = _lstmIPU(self,
                                   x,
                                   h,
                                   c,
                                   return_state=False,
                                   dropout=0.)
    dropout_most_result = _lstmIPU(self,
                                   x,
                                   h,
                                   c,
                                   return_state=False,
                                   dropout=0.9)

    self.assertNotAllClose(dropout_none_result, dropout_most_result)

  @test_util.run_v2_only
  def test_can_call_without_state_change(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, h, c = self._get_random_inputs()

    layer = layers.PopnnLSTM(
        num_hidden,
        dtype=data_type,
        kernel_initializer=init_ops.random_uniform_initializer(
            seed=42, dtype=data_type),
        recurrent_initializer=init_ops.random_uniform_initializer(
            seed=42, dtype=data_type),
        bias_initializer=init_ops.zeros_initializer(dtype=data_type))
    layer.build(x[0].shape)

    @def_function.function
    def impl(x, c, h):
      state = rnn_cell.LSTMStateTuple(c, h)
      return layer(inputs=x, initial_state=state)

    self.assertEqual(layer.kernel.shape, [num_input, num_hidden * 4])
    _ = impl(x[0], c, h)
    self.assertEqual(layer.kernel.shape, [num_input, num_hidden * 4])
    _ = impl(x[0], c, h)

  @test_util.deprecated_graph_mode_only
  def test_lstm_stateful(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, h, c = self._get_random_inputs(num_samples=10)

    cpu_result = _lstmCPU(self, x, h, c, stateful=True)
    ipu_result = _lstmIPU(self, x, h, c, stateful=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_lstm_stateful_time_major(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, h, c = self._get_random_inputs(time_major=True, num_samples=10)

    cpu_result = _lstmCPU(self, x, h, c, stateful=True, time_major=True)
    ipu_result = _lstmIPU(self, x, h, c, stateful=True, time_major=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.run_v2_only
  def test_lstm_save_load_weights(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    xs, _, _ = self._get_random_inputs()
    x = xs[0]
    # Run on CPU
    layer_cpu = _getLSTMLayer(recurrent_v2.LSTM,
                              kernel_initializer='truncated_normal',
                              recurrent_initializer='normal',
                              bias_initializer='truncated_normal')
    cpu_result = layer_cpu(x, training=True)

    # Create IPU layer, build it, and get the weights from the cpu layer.
    layer_ipu = _getLSTMLayer(layers.PopnnLSTM)
    layer_ipu.build((batch_size, timesteps, num_input))
    layer_ipu.set_weights(layer_cpu.get_weights())

    ipu_result = layer_ipu(x, training=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.run_v2_only
  def test_weight_type(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    layer_ipu = layers.PopnnLSTM(num_hidden)
    layer_ipu.build((batch_size, timesteps, num_input))
    self.assertTrue(all(w.dtype == dtypes.float32 for w in layer_ipu.weights))

    layer_ipu = layers.PopnnLSTM(num_hidden, dtype=dtypes.float16)
    layer_ipu.build((batch_size, timesteps, num_input))
    self.assertTrue(all(w.dtype == dtypes.float16 for w in layer_ipu.weights))

    keras.backend.set_floatx('float16')
    layer_ipu = layers.PopnnLSTM(num_hidden)
    layer_ipu.build((batch_size, timesteps, num_input))
    self.assertTrue(all(w.dtype == dtypes.float16 for w in layer_ipu.weights))
    keras.backend.set_floatx('float32')

  @test_util.run_v2_only
  def test_upstream_layer(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    # Prepare Data
    xs, h, c = self._get_random_inputs(num_samples=2)
    x_fit, x_predict = xs[0], xs[1]
    y = np.random.rand(batch_size, num_hidden).astype(data_type)

    # Setup CPU LSTM layer
    layer_cpu = _getLSTMLayer(recurrent_v2.LSTM,
                              kernel_initializer='truncated_normal',
                              recurrent_initializer='normal',
                              bias_initializer='truncated_normal',
                              return_state=False)
    layer_cpu.build((batch_size, timesteps, num_input))
    initial_weights = layer_cpu.get_weights()

    # Create CPU graph
    initial_h_cpu = keras.Input(batch_shape=(batch_size, num_hidden))
    initial_c_cpu = keras.Input(batch_shape=(batch_size, num_hidden))
    inputs_cpu = keras.Input(batch_shape=(batch_size, timesteps, num_input))
    outputs_cpu = layer_cpu(inputs_cpu,
                            initial_state=(initial_h_cpu, initial_c_cpu))

    # Create, fit, and make prediction with CPU model
    model_cpu = keras.Model(inputs=(inputs_cpu, initial_h_cpu, initial_c_cpu),
                            outputs=outputs_cpu)
    model_cpu.compile(loss='categorical_crossentropy', optimizer='adam')
    model_cpu.fit((x_fit, h, c), y, batch_size=batch_size)
    results_cpu = model_cpu.predict((x_predict, h, c), batch_size=batch_size)
    weights_cpu = layer_cpu.get_weights()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      layer_ipu = _getLSTMLayer(recurrent_v2.LSTM, return_state=False)
      layer_ipu.build((batch_size, timesteps, num_input))
      layer_ipu.set_weights(initial_weights)

      # Create IPU graph
      initial_h_ipu = keras.Input(batch_shape=(batch_size, num_hidden))
      initial_c_ipu = keras.Input(batch_shape=(batch_size, num_hidden))
      inputs_ipu = keras.Input(batch_shape=(batch_size, timesteps, num_input))
      outputs_ipu = layer_ipu(inputs_ipu,
                              initial_state=(initial_h_ipu, initial_c_ipu))

      # Create, fit, and make prediction with IPU model
      model_ipu = keras.Model(inputs=(inputs_ipu, initial_h_ipu,
                                      initial_c_ipu),
                              outputs=outputs_ipu)
      model_ipu.compile(loss='categorical_crossentropy', optimizer='adam')
      model_ipu.fit((x_fit, h, c), y, batch_size=batch_size)
      results_ipu = model_ipu.predict((x_predict, h, c), batch_size=batch_size)
      weights_ipu = layer_ipu.get_weights()

    self.assertAllClose(results_ipu, results_cpu)
    self.assertAllClose(weights_ipu, weights_cpu)

  @test_util.run_v2_only
  def test_get_config(self):
    layer = _getLSTMLayer(layers.PopnnLSTM)
    config = layer.get_config()
    layer2 = layers.PopnnLSTM.from_config(config)
    self.assertEqual(config, layer2.get_config())

  @test_util.run_v2_only
  def testTrainPipelineWithLstm(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32),
                                       dtype=dtypes.int32,
                                       batch_size=batch_size)

      with ipu.keras.PipelineStage(0):
        x = layers.Embedding(8000, 128)(input_layer)

      with ipu.keras.PipelineStage(1):
        x = layers.PopnnLSTM(128, dropout=0.2)(x)
        x = keras.layers.Dense(1, activation='sigmoid')(x)

      m = keras.Model(input_layer, x)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=24)
      m.compile('sgd', loss='mse', steps_per_execution=48)

      # Fit the weights to the dataset
      history = m.fit(test_language_dataset(length=96), epochs=3, verbose=0)

      losses = history.history['loss']
      self.assertTrue(losses[0] > losses[-1])

  @test_util.run_v2_only
  def testTrainSequentialPipelineWithLstm(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential([
          layers.Embedding(8000, 128),
          layers.PopnnLSTM(128, dropout=0.2),
          keras.layers.Dense(1, activation='sigmoid')
      ])
      m.set_pipeline_stage_assignment([0, 1, 1])
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=8)
      m.compile('sgd', loss='mse', steps_per_execution=16)

      # Fit the weights to the dataset
      history = m.fit(test_language_dataset(length=96), epochs=3, verbose=0)

      losses = history.history['loss']
      self.assertTrue(losses[0] > losses[-1])

  def test_options(self):
    # Prepare Data
    xs, h, c = self._get_random_inputs(num_samples=2)
    x_fit, _ = xs[0], xs[1]
    y = np.random.rand(batch_size, num_hidden).astype(data_type)

    def run_layer(options=None, options_bwd=None):
      strategy = ipu.ipu_strategy.IPUStrategy()
      with strategy.scope():
        layer = _getLSTMLayer(layers.PopnnLSTM,
                              options=options,
                              options_bwd=options_bwd)
        layer.build((batch_size, timesteps, num_input))

        # Create IPU graph.
        initial_h = keras.Input(batch_shape=(batch_size, num_hidden))
        initial_c = keras.Input(batch_shape=(batch_size, num_hidden))
        inputs = keras.Input(batch_shape=(batch_size, timesteps, num_input))
        outputs = layer(inputs, initial_state=(initial_h, initial_c))

        # Create, and fit with an IPU model.
        model = keras.Model(inputs=(inputs, initial_h, initial_c),
                            outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit((x_fit, h, c), y, batch_size=batch_size)

    error_msg = r'\[Poplar\]\[Build graph\] invalid_option:.*'
    with self.assertRaisesRegex(errors.InternalError, error_msg):
      run_layer(options={'availableMemoryProportion': -273.15})
    with self.assertRaisesRegex(errors.InternalError, error_msg):
      run_layer(options_bwd={'availableMemoryProportion': -273.15})
    run_layer(options={'availableMemoryProportion': 0.42})
    run_layer(options_bwd={'availableMemoryProportion': 0.42})

  # TODO(T54285): Delete this test.
  def test_options_with_amp(self):
    layer = _getLSTMLayer(layers.PopnnLSTM,
                          available_memory_proportion_fwd=0.1)
    self.assertTrue(
        layer._options_with_amp['availableMemoryProportion'] == 0.1)  # pylint: disable=protected-access
    self.assertTrue(
        layer._options_bwd_with_amp['availableMemoryProportion'] == 0.1)  # pylint: disable=protected-access

    layer = _getLSTMLayer(layers.PopnnLSTM,
                          available_memory_proportion_fwd=0.1,
                          available_memory_proportion_bwd=0.2)
    self.assertTrue(
        layer._options_with_amp['availableMemoryProportion'] == 0.1)  # pylint: disable=protected-access
    self.assertTrue(
        layer._options_bwd_with_amp['availableMemoryProportion'] == 0.2)  # pylint: disable=protected-access

    layer = _getLSTMLayer(layers.PopnnLSTM,
                          available_memory_proportion_fwd=0.1,
                          available_memory_proportion_bwd=0.2,
                          options={'availableMemoryProportion': 0.3})
    self.assertTrue(
        layer._options_with_amp['availableMemoryProportion'] == 0.3)  # pylint: disable=protected-access
    self.assertTrue(
        layer._options_bwd_with_amp['availableMemoryProportion'] == 0.2)  # pylint: disable=protected-access

    layer = _getLSTMLayer(layers.PopnnLSTM,
                          available_memory_proportion_fwd=0.1,
                          available_memory_proportion_bwd=0.2,
                          options={'availableMemoryProportion': 0.3},
                          options_bwd={'availableMemoryProportion': 0.4})
    self.assertTrue(
        layer._options_with_amp['availableMemoryProportion'] == 0.3)  # pylint: disable=protected-access
    self.assertTrue(
        layer._options_bwd_with_amp['availableMemoryProportion'] == 0.4)  # pylint: disable=protected-access


def _getGRULayer(keras_layer=None,
                 return_state=True,
                 return_sequences=False,
                 time_major=False,
                 dropout=0.,
                 stateful=False,
                 reset_after=False,
                 kernel_initializer=None,
                 recurrent_initializer=None,
                 bias_initializer=None,
                 **kwargs):
  kernel_initializer = (kernel_initializer
                        or init_ops.constant_initializer(0.1, data_type))
  recurrent_initializer = (recurrent_initializer
                           or init_ops.constant_initializer(0.2, data_type))
  bias_initializer = (bias_initializer
                      or init_ops.constant_initializer(0.3, data_type))
  return keras_layer(num_hidden,
                     dtype=data_type,
                     kernel_initializer=kernel_initializer,
                     recurrent_initializer=recurrent_initializer,
                     bias_initializer=bias_initializer,
                     recurrent_activation="sigmoid",
                     dropout=dropout,
                     time_major=time_major,
                     return_sequences=return_sequences,
                     return_state=return_state,
                     reset_after=reset_after,
                     stateful=stateful,
                     **kwargs)


def _kerasGRUImpl(instance,
                  x_vals,
                  init_val,
                  keras_layer=None,
                  device="cpu",
                  training=True,
                  return_state=True,
                  return_sequences=False,
                  time_major=False,
                  dropout=0.,
                  stateful=False,
                  reset_after=False):

  with ops.device(device):
    x = array_ops.placeholder(x_vals[0].dtype, x_vals[0].shape)
    init_ph = array_ops.placeholder(init_val.dtype, init_val.shape)

    init = None if stateful else init_ph

    layer = _getGRULayer(keras_layer, return_state, return_sequences,
                         time_major, dropout, stateful, reset_after)
    output = layer(inputs=x, initial_state=init, training=training)

  with instance.test_session() as sess:
    sess.run(variables.global_variables_initializer())
    outputs = []

    # Run the op and any updates.
    to_run = [output, [layer.updates]] if layer.updates else output
    for x_val in x_vals:
      r = sess.run(to_run, {x: x_val, init_ph: init_val})
      r = r[0] if layer.updates else r
      outputs.append(r)
    return outputs


def _gruIPU(*args, **kwargs):
  return _kerasGRUImpl(*args,
                       **kwargs,
                       keras_layer=layers.PopnnGRU,
                       device='/device:IPU:0')


def _gruCPU(*args, **kwargs):
  return _kerasGRUImpl(*args, **kwargs, keras_layer=recurrent_v2.GRU)


class IpuGruTest(test.TestCase):
  def _get_random_inputs(self, time_major=False, num_samples=1):
    np.random.seed(43)
    init = np.random.rand(batch_size, num_hidden).astype(data_type)
    xs = []
    for _ in range(num_samples):
      shape = [timesteps, batch_size, num_input] \
              if time_major else [batch_size, timesteps, num_input]
      xs.append(np.random.rand(*shape).astype(data_type))
    return xs, init

  @test_util.deprecated_graph_mode_only
  def test_gru(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, init = self._get_random_inputs()

    cpu_result = _gruCPU(self, x, init)
    ipu_result = _gruIPU(self, x, init)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_gru_seq_major(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, init = self._get_random_inputs(True)

    ipu_result = _gruIPU(self, x, init, time_major=True)
    cpu_result = _gruCPU(self, x, init, time_major=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_gru_all_seq(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, init = self._get_random_inputs()

    ipu_result = _gruIPU(self, x, init, return_sequences=True)
    cpu_result = _gruCPU(self, x, init, return_sequences=True)

    self.assertAllClose(ipu_result, cpu_result)
    self.assertEqual(ipu_result[0][0].shape,
                     (batch_size, timesteps, num_hidden))

  @test_util.deprecated_graph_mode_only
  def test_gru_no_state(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, init = self._get_random_inputs()

    ipu_result = _gruIPU(self, x, init, return_state=False)
    self.assertTrue(isinstance(ipu_result[0], np.ndarray))

  @test_util.deprecated_graph_mode_only
  def test_class_alias(self):
    self.assertTrue(isinstance(layers.GRU, type))
    self.assertEqual(layers.PopnnGRU, layers.GRU)

  @test_util.deprecated_graph_mode_only
  def test_gru_dropout(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, init = self._get_random_inputs()

    dropout_none_result = _gruIPU(self,
                                  x,
                                  init,
                                  dropout=0.,
                                  return_state=False,
                                  return_sequences=True)
    dropout_most_result = _gruIPU(self,
                                  x,
                                  init,
                                  dropout=0.9,
                                  return_state=False,
                                  return_sequences=True)

    self.assertNotAllClose(dropout_none_result, dropout_most_result)

  @test_util.deprecated_graph_mode_only
  def test_gru_stateful(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, init = self._get_random_inputs(num_samples=10)

    cpu_result = _gruCPU(self, x, init, stateful=True)
    ipu_result = _gruIPU(self, x, init, stateful=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_gru_stateful_time_major(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, init = self._get_random_inputs(time_major=True, num_samples=10)

    cpu_result = _gruCPU(self, x, init, stateful=True, time_major=True)
    ipu_result = _gruIPU(self, x, init, stateful=True, time_major=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.deprecated_graph_mode_only
  def test_gru_reset_after(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    x, init = self._get_random_inputs(num_samples=10)

    cpu_result = _gruCPU(self, x, init, reset_after=True)
    ipu_result = _gruIPU(self, x, init, reset_after=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.run_v2_only
  def test_gru_save_load_weights(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    xs, _ = self._get_random_inputs()
    x = xs[0]

    # Run on CPU
    layer_cpu = _getGRULayer(recurrent_v2.GRU,
                             kernel_initializer='truncated_normal',
                             recurrent_initializer='normal',
                             bias_initializer='truncated_normal')
    cpu_result = layer_cpu(x, training=True)

    # Create IPU layer, build it, and get the weights from the cpu layer.
    layer_ipu = _getGRULayer(layers.PopnnGRU)
    layer_ipu.build((batch_size, timesteps, num_input))
    layer_ipu.set_weights(layer_cpu.get_weights())

    ipu_result = layer_ipu(x, training=True)
    self.assertAllClose(ipu_result, cpu_result)

  @test_util.run_v2_only
  def test_upstream_layer(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    # Prepare Data
    xs, init = self._get_random_inputs(num_samples=2)
    x_fit, x_predict = xs[0], xs[1]
    y = np.random.rand(batch_size, num_hidden).astype(data_type)

    # Setup CPU GRU layer
    layer_cpu = _getGRULayer(recurrent_v2.GRU,
                             kernel_initializer='truncated_normal',
                             recurrent_initializer='normal',
                             bias_initializer='truncated_normal',
                             reset_after=True,
                             return_state=False)
    layer_cpu.build((batch_size, timesteps, num_input))
    initial_weights = layer_cpu.get_weights()

    # Create CPU graph
    initial_state_cpu = keras.Input(batch_shape=(batch_size, num_hidden))
    inputs_cpu = keras.Input(batch_shape=(batch_size, timesteps, num_input))
    outputs_cpu = layer_cpu(inputs_cpu, initial_state=initial_state_cpu)

    # Create, fit, and make prediction with CPU model
    model_cpu = keras.Model(inputs=(inputs_cpu, initial_state_cpu),
                            outputs=outputs_cpu)
    model_cpu.compile(loss='categorical_crossentropy', optimizer='adam')
    model_cpu.fit((x_fit, init), y, batch_size=batch_size)
    results_cpu = model_cpu.predict((x_predict, init), batch_size=batch_size)
    weights_cpu = layer_cpu.get_weights()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      # Setup IPU GRU layer
      layer_ipu = _getGRULayer(recurrent_v2.GRU,
                               reset_after=True,
                               return_state=False)
      layer_ipu.build((batch_size, timesteps, num_input))
      layer_ipu.set_weights(initial_weights)

      # Create IPU graph
      initial_state_ipu = keras.Input(batch_shape=(batch_size, num_hidden))
      inputs_ipu = keras.Input(batch_shape=(batch_size, timesteps, num_input))
      outputs_ipu = layer_ipu(inputs_ipu, initial_state=initial_state_ipu)

      # Create, fit, and make prediction with IPU model
      model_ipu = keras.Model(inputs=(inputs_ipu, initial_state_ipu),
                              outputs=outputs_ipu)
      model_ipu.compile(loss='categorical_crossentropy', optimizer='adam')
      model_ipu.fit((x_fit, init), y, batch_size=batch_size)
      results_ipu = model_ipu.predict((x_predict, init), batch_size=batch_size)
      weights_ipu = layer_ipu.get_weights()

    self.assertAllClose(results_ipu, results_cpu)
    self.assertAllClose(weights_ipu, weights_cpu)

  @test_util.run_v2_only
  def test_weight_type(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    layer_ipu = layers.PopnnGRU(num_hidden)
    layer_ipu.build((batch_size, timesteps, num_input))
    self.assertTrue(all(w.dtype == dtypes.float32 for w in layer_ipu.weights))

    layer_ipu = layers.PopnnGRU(num_hidden, dtype=dtypes.float16)
    layer_ipu.build((batch_size, timesteps, num_input))
    self.assertTrue(all(w.dtype == dtypes.float16 for w in layer_ipu.weights))

    keras.backend.set_floatx('float16')
    layer_ipu = layers.PopnnGRU(num_hidden)
    layer_ipu.build((batch_size, timesteps, num_input))
    self.assertTrue(all(w.dtype == dtypes.float16 for w in layer_ipu.weights))
    keras.backend.set_floatx('float32')

  @test_util.run_v2_only
  def test_gru_ipu_vs_cpu_results_reset_after(self):
    # Configure
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.configure_ipu_system()

    # Prepare Data
    xs, init = self._get_random_inputs(num_samples=2)
    x_fit, x_predict = xs[0], xs[1]
    y = np.random.rand(batch_size, num_hidden).astype(data_type)

    # Setup CPU GRU layer
    layer_cpu = _getGRULayer(recurrent_v2.GRU,
                             kernel_initializer='truncated_normal',
                             recurrent_initializer='normal',
                             bias_initializer='truncated_normal',
                             reset_after=True,
                             return_state=False)
    layer_cpu.build((batch_size, timesteps, num_input))
    initial_weights = layer_cpu.get_weights()

    # Create CPU graph
    initial_state_cpu = keras.Input(batch_shape=(batch_size, num_hidden))
    inputs_cpu = keras.Input(batch_shape=(batch_size, timesteps, num_input))
    outputs_cpu = layer_cpu(inputs_cpu, initial_state=initial_state_cpu)

    # Create, fit, and make prediction with CPU model
    model_cpu = keras.Model(inputs=(inputs_cpu, initial_state_cpu),
                            outputs=outputs_cpu)
    model_cpu.compile(loss='categorical_crossentropy', optimizer='adam')
    model_cpu.fit((x_fit, init), y, batch_size=batch_size)
    results_cpu = model_cpu.predict((x_predict, init), batch_size=batch_size)
    weights_cpu = layer_cpu.get_weights()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      layer_ipu = _getGRULayer(layers.PopnnGRU,
                               reset_after=True,
                               return_state=False)
      layer_ipu.build((batch_size, timesteps, num_input))
      layer_ipu.set_weights(initial_weights)

      # Create IPU graph
      initial_state_ipu = keras.Input(batch_shape=(batch_size, num_hidden))
      inputs_ipu = keras.Input(batch_shape=(batch_size, timesteps, num_input))
      outputs_ipu = layer_ipu(inputs_ipu, initial_state=initial_state_ipu)

      # Create, fit, and make prediction with IPU model
      model_ipu = keras.Model(inputs=(inputs_ipu, initial_state_ipu),
                              outputs=outputs_ipu)
      model_ipu.compile(loss='categorical_crossentropy', optimizer='adam')
      model_ipu.fit((x_fit, init), y, batch_size=batch_size)
      results_ipu = model_ipu.predict((x_predict, init), batch_size=batch_size)
      weights_ipu = layer_ipu.get_weights()

    self.assertAllClose(results_ipu, results_cpu)
    self.assertAllClose(weights_ipu, weights_cpu)

  @test_util.run_v2_only
  def test_get_config(self):
    layer = _getGRULayer(layers.PopnnGRU)
    config = layer.get_config()
    layer2 = layers.PopnnGRU.from_config(config)
    self.assertEqual(config, layer2.get_config())

  def test_options(self):
    # Prepare Data
    xs, init = self._get_random_inputs(num_samples=2)
    x_fit, _ = xs[0], xs[1]
    y = np.random.rand(batch_size, num_hidden).astype(data_type)

    def run_layer(options=None, options_bwd=None):
      strategy = ipu.ipu_strategy.IPUStrategy()
      with strategy.scope():
        layer = _getGRULayer(layers.PopnnGRU,
                             options=options,
                             options_bwd=options_bwd)
        layer.build((batch_size, timesteps, num_input))

        # Create IPU graph.
        initial_state = keras.Input(batch_shape=(batch_size, num_hidden))
        inputs = keras.Input(batch_shape=(batch_size, timesteps, num_input))
        outputs = layer(inputs, initial_state=initial_state)

        # Create, and fit with an IPU model.
        model = keras.Model(inputs=(inputs, initial_state), outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit((x_fit, init), y, batch_size=batch_size)

    error_msg = r'\[Poplar\]\[Build graph\] invalid_option:.*'
    with self.assertRaisesRegex(errors.InternalError, error_msg):
      run_layer(options={'availableMemoryProportion': -273.15})
    with self.assertRaisesRegex(errors.InternalError, error_msg):
      run_layer(options_bwd={'availableMemoryProportion': -273.15})
    run_layer(options={'availableMemoryProportion': 0.42})
    run_layer(options_bwd={'availableMemoryProportion': 0.42})

  # TODO(T54285): Delete this test.
  def test_options_with_amp(self):
    layer = _getGRULayer(layers.PopnnGRU, available_memory_proportion_fwd=0.1)
    self.assertTrue(
        layer._options_with_amp['availableMemoryProportion'] == 0.1)  # pylint: disable=protected-access
    self.assertTrue(
        layer._options_bwd_with_amp['availableMemoryProportion'] == 0.1)  # pylint: disable=protected-access

    layer = _getGRULayer(layers.PopnnGRU,
                         available_memory_proportion_fwd=0.1,
                         available_memory_proportion_bwd=0.2)
    self.assertTrue(
        layer._options_with_amp['availableMemoryProportion'] == 0.1)  # pylint: disable=protected-access
    self.assertTrue(
        layer._options_bwd_with_amp['availableMemoryProportion'] == 0.2)  # pylint: disable=protected-access

    layer = _getGRULayer(layers.PopnnGRU,
                         available_memory_proportion_fwd=0.1,
                         available_memory_proportion_bwd=0.2,
                         options={'availableMemoryProportion': 0.3})
    self.assertTrue(
        layer._options_with_amp['availableMemoryProportion'] == 0.3)  # pylint: disable=protected-access
    self.assertTrue(
        layer._options_bwd_with_amp['availableMemoryProportion'] == 0.2)  # pylint: disable=protected-access

    layer = _getGRULayer(layers.PopnnGRU,
                         available_memory_proportion_fwd=0.1,
                         available_memory_proportion_bwd=0.2,
                         options={'availableMemoryProportion': 0.3},
                         options_bwd={'availableMemoryProportion': 0.4})
    self.assertTrue(
        layer._options_with_amp['availableMemoryProportion'] == 0.3)  # pylint: disable=protected-access
    self.assertTrue(
        layer._options_bwd_with_amp['availableMemoryProportion'] == 0.4)  # pylint: disable=protected-access


if __name__ == '__main__':
  test.main()
