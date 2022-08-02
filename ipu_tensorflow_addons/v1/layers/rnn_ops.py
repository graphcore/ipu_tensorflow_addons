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
"""
Popnn recurrent neural network operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import json
import logging

from tensorflow.compat import v1 as tf
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ipu.ops import op_util
from tensorflow.python.util import deprecation

POPNN_LSTM = "lstm"
POPNN_GRU = "gru"
POPNN_DYNAMIC_GRU = "dynamic_gru"
POPNN_AUGRU = "augru"

POPNN_LSTM_NUM_GATES = 4
POPNN_GRU_NUM_GATES = 3
POPNN_DYNAMIC_GRU_NUM_GATES = 3
POPNN_AUGRU_NUM_GATES = 3

__all__ = ["PopnnLSTM", "PopnnGRU", "PopnnDynamicGRU", "PopnnAUGRU"]


class _PopnnRNN(base_layer.Layer):  #pylint: disable=W0223
  """Base class for implementing XLA and Popnn compatible RNN layers.
  """

  def __init__(
      self,
      num_units,
      dtype=tf.float32,
      partials_dtype=tf.float32,
      seed=None,
      weights_initializer=None,
      bias_initializer=None,
      activation='tanh',
      recurrent_activation='sigmoid',
      return_state=True,
      name=None,
      options=None,
      options_bwd=None,
  ):
    """Creates a _PopnnRNN model from model spec.

    Args:
      num_units: the number of units within the RNN model.
      dtype: tf.float16 or tf.float32
      partials_dtype: the type used by Popnn to perform partial calculations.
        Either tf.float16 or tf.float32.
      seed: A Python integer. Used to create the default Glorot uniform
        initializer weights_initializer.
      weights_initializer: starting value to initialize the weight
        (default is all zeros).
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      activation: Activation function. Defaults to "tanh".
        Accepted values: "tanh", "relu", "softmax", "sigmoid", "hard_sigmoid".
      recurrent_activation: Recurrent activation function. Defaults to
        "sigmoid". Must generate output in the [0,1] range.
        Accepted values: "tanh", "softmax", "sigmoid", "hard_sigmoid".
      return_state: Boolean. Whether to return the last state in addition to the
        output. Default: `True`.
      name: VariableScope for the created subgraph; defaults to class name.
        This only serves the default scope if later no scope is specified when
        invoking ``__call__()``.
      options: A Python dictionary.
        Implementation or debug options for the forward LSTM cell in PopLibs.
        See the LSTM documentation in the PopLibs API reference for the full
        list of options.
      options_bwd: A Python dictionary.
        Implementation or debug options for the backward LSTM cell in PopLibs.
        See the LSTM documentation in the PopLibs API reference for the full
        list of options.
    """
    super().__init__(dtype=dtype, name=name)

    if dtype not in [tf.float16, tf.float32]:
      raise ValueError("Only support float16, float32, provided %s" % dtype)
    # Layer self.dtype is type name, the original DType object is kept here.
    self._plain_dtype = dtype
    self._partials_dtype = partials_dtype
    self._num_layers = 1
    self._num_units = num_units
    self._weights_initializer = weights_initializer
    self._bias_initializer = bias_initializer
    self._seed = seed
    # Init input_size to None, which will be set after build().
    self._input_size = None
    self._saveable = None

    activation = op_util.get_activation_name(activation)
    recurrent_activation = op_util.get_activation_name(recurrent_activation)

    self._activation = activation
    self._recurrent_activation = recurrent_activation
    self._return_state = return_state
    self._options = dict() if options is None else options
    self._options_bwd = dict() if options_bwd is None else options_bwd

  @property
  def num_layers(self):
    return self._num_layers

  @property
  def num_units(self):
    return self._num_units

  @property
  def input_size(self):
    if not self._input_size:
      raise ValueError(
          "\'input_size\' is unknown since layer has not been built.")
    return self._input_size

  @property
  def saveable(self):
    raise NotImplementedError(
        "This cell does not yet support object-based saving. File a feature "
        "request if this limitation bothers you.")

  @property
  def canonical_weight_shape(self):
    """Shapes of Popnn canonical weight tensors."""
    if not self._input_size:
      raise RuntimeError(
          "%s.canonical_weight_shape invoked before input shape is known" %
          type(self).__name__)

    return self._canonical_weight_shape(0)

  @property
  def canonical_bias_shapes(self):
    """Shapes of Popnn canonical bias tensors."""
    return self._canonical_bias_shape(0)

  def build(self, input_shape):
    del input_shape
    raise ValueError("This method needs to be overridden.")

  def _build(self, input_shape):
    """Create variables of the Popnn RNN.

    It can be called manually before `__call__()` or automatically through
    `__call__()`. In the former case, any subsequent `__call__()` will skip
    creating variables.

    Args:
      input_shape: a TensorShape object with 3 dimensions.

    Raises:
      ValueError: if input_shape has wrong dimension or unknown 3rd dimension.
    """
    if self.built:  # pylint: disable=access-member-before-definition
      return

    input_shape = tf.TensorShape(input_shape)
    if input_shape.ndims != 3:
      raise ValueError("Expecting input_shape with 3 dims, got %d" %
                       input_shape.ndims)
    input_shape = input_shape.as_list()
    if input_shape[-1] is None:
      raise ValueError("The last dimension of the inputs to `_PopnnRNN` "
                       "should be defined. Found `None`.")
    self._input_size = input_shape[-1]
    self.input_spec = base_layer.InputSpec(ndim=3, axes={-1: self._input_size})

    # Create the variables
    with tf.variable_scope(self._scope, reuse=self.built):  # pylint: disable=access-member-before-definition
      if self._weights_initializer is None:
        self._weights_initializer = tf.glorot_uniform_initializer(
            self._seed, dtype=self._plain_dtype)
      if self._bias_initializer is None:
        self._bias_initializer = tf.constant_initializer(
            0.0, dtype=self._plain_dtype)
      self.kernel = tf.get_variable("kernel",
                                    dtype=self._plain_dtype,
                                    initializer=self._weights_initializer,
                                    shape=self.canonical_weight_shape)
      self.biases = tf.get_variable("biases",
                                    dtype=self._plain_dtype,
                                    initializer=self._bias_initializer,
                                    shape=self.canonical_bias_shapes)

    self.built = True

  # pylint: disable=unused-argument
  # pylint: disable=arguments-differ
  def call(self, inputs, initial_state=None, training=True):
    raise ValueError("This method needs to be overridden.")

  def state_shape(self, batch_size):
    raise ValueError("This method needs to be overridden.")

  def _zero_state(self, batch_size):
    raise ValueError("This method needs to be overridden.")

  def _canonical_weight_shape(self, layer):
    """Shapes of Popnn canonical weight tensors for given layer."""
    if layer < 0 or layer >= self._num_layers:
      raise ValueError("\'layer\' is not valid, got %s, expecting [%d, %d]" %
                       (layer, 0, self._num_layers - 1))
    if not self._input_size:
      raise RuntimeError(
          "%s._canonical_weight_shape invoked before input shape is known" %
          type(self).__name__)

    input_size = self._input_size
    num_units = self._num_units
    num_gates = self._num_gates_per_layer

    if layer == 0:
      tf_wts = [input_size, num_units * num_gates]
    else:
      #TODO we only support one layer.
      tf_wts = [num_units, num_units * num_gates]
    tf_wts[0] += num_units
    return tf_wts

  def _canonical_bias_shape(self, unused_layer):
    """Shapes of Popnn canonical bias tensors for given layer."""
    return [self._num_gates_per_layer, self._num_units]

  def _extract_final_state(self, outputs, seq_len=None):
    time_len = tf.shape(outputs)[0]
    batch_size = tf.shape(outputs)[1]

    if seq_len is not None:
      indices = tf.add(seq_len,
                       tf.range(0, time_len * batch_size, delta=time_len)) - 1
      state = tf.transpose(outputs, perm=(1, 0, 2))
      state = tf.reshape(state, [-1, self._num_units])
      state = tf.gather(state, indices)
    else:
      state = outputs[-1, :, :]

    return state


class PopnnLSTM(_PopnnRNN):
  # pylint:disable=line-too-long
  """XLA compatible, time-major Popnn implementation of an LSTM layer.

  Below is a typical workflow:

  .. code-block:: python

    with tf.Graph().as_default():
      lstm = PopnnLSTM(num_units, ...)

      outputs, output_states = lstm(inputs, initial_states, training=True)

  """
  # pylint:enable=line-too-long
  _rnn_mode = POPNN_LSTM
  _num_gates_per_layer = POPNN_LSTM_NUM_GATES

  def __init__(self,
               num_units,
               dtype=tf.float32,
               partials_dtype=tf.float32,
               seed=None,
               weights_initializer=None,
               bias_initializer=None,
               activation='tanh',
               recurrent_activation='sigmoid',
               return_state=True,
               name=None,
               options=None,
               options_bwd=None):
    """Creates a PopnnLSTM model from model spec.

    Args:
      num_units: the number of units within the LSTM model.
      dtype: tf.float16 or tf.float32
      partials_dtype: the type used by Popnn to perform partial calculations.
        Either tf.float16 or tf.float32.
      seed: A Python integer. Used to create the default Glorot uniform
        initializer weights_initializer.
      weights_initializer: starting value to initialize the weights
        (default is Glorot uniform initializer).
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      activation: Activation function. Defaults to "tanh".
        Accepted values: "tanh", "relu", "softmax", "sigmoid", "hard_sigmoid".
      recurrent_activation: Recurrent activation function. Defaults to
        "sigmoid". Must generate output in the [0,1] range.
        Accepted values: "tanh", "softmax", "sigmoid", "hard_sigmoid".
      return_state: Boolean. Whether to return the last state in addition to the
        output. Default: `True`.
      name: VariableScope for the created subgraph; defaults to class name.
        This only serves the default scope if later no scope is specified when
        invoking ``__call__()``.
      options: A Python dictionary.
        Implementation or debug options for the forward LSTM cell in PopLibs.
        See the LSTM documentation in the PopLibs API reference for the full
        list of options.
      options_bwd: A Python dictionary.
        Implementation or debug options for the backward LSTM cell in PopLibs.
        See the LSTM documentation in the PopLibs API reference for the full
        list of options.
    """
    super().__init__(num_units=num_units,
                     dtype=dtype,
                     partials_dtype=partials_dtype,
                     seed=seed,
                     weights_initializer=weights_initializer,
                     bias_initializer=bias_initializer,
                     activation=activation,
                     recurrent_activation=recurrent_activation,
                     return_state=return_state,
                     name=name,
                     options=options,
                     options_bwd=options_bwd)

  def build(self, input_shape):
    """Create variables of the PopnnLSTM.

    It can be called manually before `__call__()` or automatically through
    `__call__()`. In the former case, any subsequent `__call__()` will skip
    creating variables.

    Args:
      input_shape: a TensorShape object with 3 dimensions.

    Raises:
      ValueError: if input_shape has wrong dimension or unknown 3rd dimension.
    """
    self._build(input_shape)

  def call(self, inputs, initial_state=None, training=True):
    """Runs the forward step for the LSTM model.

    Args:
      inputs: 3D tensor with shape [time_len, batch_size, input_size].
      initial_state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size, num_units]`. If not provided, the state is
        initialized to zeros.
      training: Set to False to use the LSTM model in inference mode.

    Returns:
      A tuple of output and output state.

      * output: a tensor of shape [time_len, batch_size, num_units].
      * output_state: An `LSTMStateTuple` of the same shape and structure as
        initial_state.

    Raises:
      ValueError: if initial_state is not valid.

    """

    dtype = self.dtype
    inputs = tf.convert_to_tensor(inputs, dtype=dtype)

    batch_size = tf.shape(inputs)[1]

    if initial_state is not None and not isinstance(
        initial_state, tf.nn.rnn_cell.LSTMStateTuple):
      raise ValueError("Invalid initial_state type: `%s`, expecting "
                       "`LSTMStateTuple`." % type(initial_state))

    if initial_state is None:
      # Create a zero state.
      initial_state = self._zero_state(batch_size)

    c, h = initial_state
    h = tf.convert_to_tensor(h, dtype=dtype)
    c = tf.convert_to_tensor(c, dtype=dtype)

    options = json.dumps(self._options)
    options_bwd = json.dumps(self._options_bwd)

    popnn_result = gen_popnn_ops.popnn_lstm_layer(
        inputs=inputs,
        num_channels=self._num_units,
        kernel=self.kernel,
        biases=self.biases,
        input_h_state=h,
        input_c_state=c,
        is_training=training,
        partials_dtype=self._partials_dtype,
        activation=self._activation,
        recurrent_activation=self._recurrent_activation,
        name=self._name,
        options=options,
        options_bwd=options_bwd)

    return self._make_call_result(popnn_result)

  def state_shape(self, batch_size):
    """Shape of Popnn LSTM states.

    Shape is a 2-element tuple. Each is [batch_size, num_units]

    Args:
      batch_size: an int

    Returns:
      a tuple of Python arrays.
    """
    return ([batch_size, self.num_units], [batch_size, self.num_units])

  def _zero_state(self, batch_size):
    res = []
    for sp in self.state_shape(batch_size):
      res.append(tf.zeros(sp, dtype=self.dtype))
    return tf.nn.rnn_cell.LSTMStateTuple(*res)

  def _make_call_result(self, popnn_result, seq_len=None):
    """Takes the result from the popnn call and converts it to the correct
    format.
    """
    outputs, c_state, _ = popnn_result
    if self._return_state:
      h_state = self._extract_final_state(outputs, seq_len)
      state = tf.nn.rnn_cell.LSTMStateTuple(c_state, h_state)
      return outputs, state
    return outputs

  @property
  def saveable(self):
    return False


class PopnnDynamicLSTM(PopnnLSTM):
  #pylint: disable=W0223
  def call(self, inputs, seq_len, initial_state=None, training=True):
    #pylint: disable=W0221
    """Runs the forward step for the LSTM model.

    Args:
      inputs: 3D tensor with shape [time_len, batch_size, input_size].
      seq_len: 1-D tensor with the sequence length of samples in each batch.
      initial_state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size, num_units]`. If not provided, the state is
        initialized to zeros.
      training: Set to False to use the LSTM model in inference mode.

    Returns:
      A tuple of output and output state.

      * output: a tensor of shape [time_len, batch_size, num_units].
      * output_state: An `LSTMStateTuple` of the same shape and structure as
        initial_state.

    Raises:
      ValueError: if initial_state is not valid.

    """

    dtype = self.dtype
    inputs = tf.convert_to_tensor(inputs, dtype=dtype)

    batch_size = tf.shape(inputs)[1]

    if initial_state is not None and not isinstance(
        initial_state, tf.nn.rnn_cell.LSTMStateTuple):
      raise ValueError("Invalid initial_state type: `%s`, expecting "
                       "`LSTMStateTuple`." % type(initial_state))

    if initial_state is None:
      # Create a zero state.
      initial_state = self._zero_state(batch_size)

    c, h = initial_state
    h = tf.convert_to_tensor(h, dtype=dtype)
    c = tf.convert_to_tensor(c, dtype=dtype)

    options = json.dumps(self._options)
    options_bwd = json.dumps(self._options_bwd)

    popnn_result = gen_popnn_ops.popnn_dynamic_lstm_layer(
        inputs=inputs,
        seq_len=seq_len,
        num_channels=self._num_units,
        kernel=self.kernel,
        biases=self.biases,
        input_h_state=h,
        input_c_state=c,
        is_training=training,
        partials_dtype=self._partials_dtype,
        activation=self._activation,
        recurrent_activation=self._recurrent_activation,
        preserve_final_state=self._return_state,
        name=self._name,
        options=options,
        options_bwd=options_bwd)

    return self._make_call_result(popnn_result, seq_len)

  @property
  def saveable(self):
    return False


class PopnnGRU(_PopnnRNN):
  # pylint:disable=line-too-long
  """XLA compatible, time-major Popnn implementation of a GRU layer.

  Below is a typical workflow:

  .. code-block:: python

    with tf.Graph().as_default():
      gru = PopnnGRU(num_units, ...)

      outputs, output_state = gru(inputs, initial_state, training=True)

  """
  # pylint:enable=line-too-long
  _rnn_mode = POPNN_GRU
  _num_gates_per_layer = POPNN_GRU_NUM_GATES

  def __init__(self,
               num_units,
               dtype=tf.float32,
               partials_dtype=tf.float32,
               seed=None,
               weights_initializer=None,
               bias_initializer=None,
               activation='tanh',
               recurrent_activation='sigmoid',
               return_state=True,
               name=None,
               reset_after=False,
               options=None,
               options_bwd=None):
    """Creates a PopnnGRU model from model spec.

    Args:
      num_units: the number of units within the GRU model.
      dtype: tf.float16 or tf.float32
      partials_dtype: the type used by Popnn to perform partial calculations.
        Either tf.float16 or tf.float32.
      seed: A Python integer. Used to create the default Glorot uniform
        initializer weights_initializer.
      weights_initializer: starting value to initialize the weights
        (default is Glorot uniform initializer).
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      activation: Activation function. Defaults to "tanh".
        Accepted values: "tanh", "relu", "softmax", "sigmoid", "hard_sigmoid".
      recurrent_activation: Recurrent activation function. Defaults to
        "sigmoid". Must generate output in the [0,1] range.
        Accepted values: "tanh", "softmax", "sigmoid", "hard_sigmoid".
      return_state: Boolean. Whether to return the last state in addition to the
        output. Default: `True`.
      name: VariableScope for the created subgraph; defaults to class name.
        This only serves the default scope if later no scope is specified when
        invoking ``__call__()``.
      reset_after:  GRU convention (whether to apply reset gate
        after or before matrix multiplication). False = "before" (default),
        True = "after".
        Leave as default (False) to match the behaviour of the standard
        TensorFlow GRU.
      options: A Python dictionary.
        Implementation or debug options for the forward LSTM cell in PopLibs.
        See the LSTM documentation in the PopLibs API reference for the full
        list of options.
      options_bwd: A Python dictionary.
        Implementation or debug options for the backward LSTM cell in PopLibs.
        See the LSTM documentation in the PopLibs API reference for the full
        list of options.
    """
    super().__init__(num_units=num_units,
                     dtype=dtype,
                     partials_dtype=partials_dtype,
                     seed=seed,
                     weights_initializer=weights_initializer,
                     bias_initializer=bias_initializer,
                     activation=activation,
                     recurrent_activation=recurrent_activation,
                     return_state=return_state,
                     name=name,
                     options=options,
                     options_bwd=options_bwd)
    self._reset_after = reset_after

  def build(self, input_shape):
    """Create variables of the PopnnGRU.

    It can be called manually before `__call__()` or automatically through
    `__call__()`. In the former case, any subsequent `__call__()` will skip
    creating variables.

    Args:
      input_shape: a TensorShape object with 3 dimensions.

    Raises:
      ValueError: if input_shape has wrong dimension or unknown 3rd dimension.
    """
    self._build(input_shape)

  def call(self, inputs, initial_state=None, training=True):
    """Runs the forward step for the GRU model.

    Args:
      inputs: 3D tensor with shape [time_len, batch_size, input_size].
      initial_state: Initial state tensor, shaped `[batch_size, num_units]`. If
        not provided, the state is initialized to zeros.
      training: Set to False to use the GRU model in inference mode.

    Returns:
      A tuple of output and output_state.

      * output: a tensor of shape [time_len, batch_size, num_units].
      * output_state: The output state of the last cell.

    Raises:
      ValueError: if initial_state is not valid.

    """

    dtype = self.dtype
    inputs = tf.convert_to_tensor(inputs, dtype=dtype)

    batch_size = tf.shape(inputs)[1]

    if initial_state is None:
      # Create a zero state.
      initial_state = self._zero_state(batch_size)

    initial_state = tf.convert_to_tensor(initial_state, dtype=dtype)

    options = json.dumps(self._options)
    options_bwd = json.dumps(self._options_bwd)

    popnn_result = gen_popnn_ops.popnn_gru_layer(
        inputs=inputs,
        num_channels=self._num_units,
        kernel=self.kernel,
        biases=self.biases,
        initial_state=initial_state,
        is_training=training,
        partials_dtype=self._partials_dtype,
        activation=self._activation,
        recurrent_activation=self._recurrent_activation,
        name=self._name,
        reset_after=self._reset_after,
        options=options,
        options_bwd=options_bwd)

    return self._make_call_result(popnn_result)

  def state_shape(self, batch_size):
    """Shape of Popnn GRU state.

    State shape is [batch_size, num_units].

    Args:
      batch_size: an int

    Returns:
      A Python array.
    """
    return [batch_size, self.num_units]

  def _zero_state(self, batch_size):
    return tf.zeros(self.state_shape(batch_size), dtype=self.dtype)

  def _canonical_bias_shape(self, unused_layer):
    """Shapes of Popnn canonical bias tensors for given layer."""
    if self._reset_after:
      return [self._num_gates_per_layer, 2, self._num_units]
    return super()._canonical_bias_shape(unused_layer)

  def _make_call_result(self, popnn_result, seq_len=None):
    """Takes the result from the popnn call and converts it to the correct
    format.
    """
    outputs, _ = popnn_result
    if self._return_state:
      h_state = self._extract_final_state(outputs, seq_len)
      return outputs, h_state
    return outputs

  @property
  def saveable(self):
    return False


class PopnnDynamicGRU(PopnnGRU):
  # pylint:disable=line-too-long
  """XLA compatible, time-major Popnn implementation of an GRU layer,
  with a sequence length input.

  Below is a typical workflow:

  .. code-block:: python

    with tf.Graph().as_default():
      gru = PopnnDynamicGRU(num_units, ...)

      outputs, output_state = gru(
        inputs, seq_len, initial_state, training=True)

  """
  # pylint:enable=line-too-long
  _rnn_mode = POPNN_DYNAMIC_GRU
  _num_gates_per_layer = POPNN_DYNAMIC_GRU_NUM_GATES

  def __init__(self,
               num_units,
               dtype=tf.float32,
               partials_dtype=tf.float32,
               seed=None,
               weights_initializer=None,
               bias_initializer=None,
               activation='tanh',
               recurrent_activation='sigmoid',
               return_state=True,
               name=None,
               reset_after=False,
               options=None,
               options_bwd=None):
    """Creates a PopnnDynamicGRU model from model spec.

    Args:
      num_units: the number of units within the RNN model.
      dtype: tf.float16 or tf.float32
      partials_dtype: the type used by Popnn to perform partial calculations.
        Either tf.float16 or tf.float32.
      seed: A Python integer. Used to create the default Glorot uniform
        initializer weights_initializer.
      weights_initializer: starting value to initialize the weight
        (default is Glorot uniform initializer).
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      activation: Activation function. Defaults to "tanh".
        Accepted values: "tanh", "relu", "softmax", "sigmoid", "hard_sigmoid".
      recurrent_activation: Recurrent activation function. Defaults to
        "sigmoid". Must generate output in the [0,1] range.
        Accepted values: "tanh", "softmax", "sigmoid", "hard_sigmoid".
      return_state: Boolean. Whether to return the last state in addition to the
        output. Default: `True`.
      name: VariableScope for the created subgraph; defaults to class name.
        This only serves the default scope if later no scope is specified when
        invoking ``__call__()``.
      reset_after:  GRU convention (whether to apply reset gate after or before
        matrix multiplication). False = "before" (default), True = "after".
        Leave as default (False) to match the behaviour of the standard
        TensorFlow GRU.
      options: A Python dictionary.
        Implementation or debug options for the forward LSTM cell in PopLibs.
        See the LSTM documentation in the PopLibs API reference for the full
        list of options.
      options_bwd: A Python dictionary.
        Implementation or debug options for the backward LSTM cell in PopLibs.
        See the LSTM documentation in the PopLibs API reference for the full
        list of options.
    """
    super().__init__(num_units=num_units,
                     dtype=dtype,
                     partials_dtype=partials_dtype,
                     seed=seed,
                     weights_initializer=weights_initializer,
                     bias_initializer=bias_initializer,
                     activation=activation,
                     recurrent_activation=recurrent_activation,
                     return_state=return_state,
                     name=name,
                     reset_after=reset_after,
                     options=options,
                     options_bwd=options_bwd)

  @property
  def saveable(self):
    return False

  #pylint: disable=arguments-differ
  def call(self,
           inputs,
           seq_len,
           initial_state=None,
           training=True,
           time_major=True):
    """Runs the forward step for the DynamicGRU model.

    Args:
      inputs: 3-D tensor with shape [batch_size, time_len, input_size].
      seq_len: 1-D tensor with the sequence length of samples in each batch.
      initial_state: Initial state tensor, shaped `[batch_size, num_units]`.
        If not provided, the state is initialized to zeros.
      training: whether this operation will be used in training or inference.
      time_major: whether the time dimension is the first demension.

    Returns:
      A tuple of output and output state.

      * output: a tensor of shape [time_len, batch_size, num_units].
      * output_state: The output state of the last cell.

    Raises:
      ValueError: if initial_state is not valid.
    """

    dtype = self.dtype

    inputs = tf.convert_to_tensor(inputs, dtype=dtype)
    if not time_major:
      inputs = tf.transpose(inputs, [1, 0, 2])

    batch_size = tf.shape(inputs)[1]

    if initial_state is None:
      # Create a zero state.
      initial_state = self._zero_state(batch_size)

    initial_state = tf.convert_to_tensor(initial_state, dtype=dtype)
    if self._reset_after:
      self.biases = tf.expand_dims(self.biases, 1)
      self.biases = tf.concat([self.biases, self.biases], axis=1)

    options = json.dumps(self._options)
    options_bwd = json.dumps(self._options_bwd)

    popnn_result = gen_popnn_ops.popnn_dynamic_gru_layer(
        inputs=inputs,
        seq_len=seq_len,
        num_channels=self._num_units,
        kernel=self.kernel,
        biases=self.biases,
        initial_state=initial_state,
        is_training=training,
        partials_dtype=self._partials_dtype,
        activation=self._activation,
        recurrent_activation=self._recurrent_activation,
        name=self._name,
        reset_after=self._reset_after,
        options=options,
        options_bwd=options_bwd)

    return self._make_call_result(popnn_result, seq_len)


class PopnnAUGRU(PopnnGRU):
  # pylint:disable=line-too-long
  """XLA compatible, time-major Popnn implementation of an AUGRU layer.

  Below is a typical workflow:

  .. code-block:: python

    with tf.Graph().as_default():
      augru = PopnnAUGRU(num_units, ...)

      outputs, output_state = augru(inputs, initial_state, training=True)

  """
  # pylint:enable=line-too-long
  _rnn_mode = POPNN_AUGRU
  _num_gates_per_layer = POPNN_AUGRU_NUM_GATES

  def __init__(self,
               num_units,
               dtype=tf.float32,
               partials_dtype=tf.float32,
               seed=None,
               weights_initializer=None,
               bias_initializer=None,
               activation='tanh',
               recurrent_activation='sigmoid',
               return_state=True,
               name=None,
               reset_after=False,
               options=None,
               options_bwd=None):
    """Creates a PopnnAUGRU model from model spec.

    Args:
      num_units: the number of units within the RNN model.
      dtype: tf.float16 or tf.float32
      partials_dtype: the type used by Popnn to perform partial calculations.
        Either tf.float16 or tf.float32.
      seed: A Python integer. Used to create the default Glorot uniform
        initializer weights_initializer.
      weights_initializer: starting value to initialize the weight
        (default is Glorot uniform initializer).
      activation: Activation function. Defaults to "tanh".
        Accepted values: "tanh", "relu", "softmax", "sigmoid", "hard_sigmoid".
      recurrent_activation: Recurrent activation function. Defaults to
        "sigmoid". Must generate output in the [0,1] range.
        Accepted values: "tanh", "softmax", "sigmoid", "hard_sigmoid".
      return_state: Boolean. Whether to return the last state in addition to the
        output. Default: `True`.
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      name: VariableScope for the created subgraph; defaults to class name.
        This only serves the default scope if later no scope is specified when
        invoking ``__call__()``.
      options: A Python dictionary.
        Implementation or debug options for the forward LSTM cell in PopLibs.
        See the LSTM documentation in the PopLibs API reference for the full
        list of options.
      options_bwd: A Python dictionary.
        Implementation or debug options for the backward LSTM cell in PopLibs.
        See the LSTM documentation in the PopLibs API reference for the full
        list of options.
    """
    super().__init__(num_units=num_units,
                     dtype=dtype,
                     partials_dtype=partials_dtype,
                     seed=seed,
                     weights_initializer=weights_initializer,
                     bias_initializer=bias_initializer,
                     activation=activation,
                     recurrent_activation=recurrent_activation,
                     return_state=return_state,
                     name=name,
                     reset_after=reset_after,
                     options=options,
                     options_bwd=options_bwd)

  #pylint: disable=arguments-differ
  def call(self,
           inputs,
           seq_len,
           attention_score,
           initial_state=None,
           training=True,
           time_major=True):
    """Runs the forward step for the AUGRU model.

    Args:
      inputs: 3-D tensor with shape [time_len, batch_size, input_size].
      seq_len: 1-D tensor with the sequence length of samples in each batch.
      attention_score: The output of attention layer, the score of samples
        in each batch, shaped `[batch_size, max_seq_len]`.
      initial_state: Initial state tensor, shaped `[batch_size, num_units]`.
        If not provided, the state is initialized to zeros.
      training: whether this operation will be used in training or inference.
      time_major: whether the time dimension is the first dimension.

    Returns:
      A tuple of output and output state.

      * output: a tensor of shape [time_len, batch_size, num_units].
      * output_state: The output state of the last cell.

    Raises:
      ValueError: if initial_state is not valid.

    """

    dtype = self.dtype
    inputs = tf.convert_to_tensor(inputs, dtype=dtype)
    if not time_major:
      inputs = tf.transpose(inputs, [1, 0, 2])
      attention_score = tf.transpose(attention_score, [1, 0])

    batch_size = tf.shape(inputs)[1]

    if initial_state is None:
      # Create a zero state.
      initial_state = self._zero_state(batch_size)

    initial_state = tf.convert_to_tensor(initial_state, dtype=dtype)
    augru_biases_r_u = tf.get_variable("bias_r_u",
                                       dtype=inputs.dtype,
                                       initializer=tf.ones_initializer(),
                                       shape=[2, self._num_units])
    augru_biases_c = tf.get_variable("bias_c",
                                     dtype=inputs.dtype,
                                     initializer=tf.zeros_initializer(),
                                     shape=[1, self._num_units])
    augru_biases = tf.concat([augru_biases_r_u, augru_biases_c], axis=0)
    if self._reset_after:
      augru_biases = tf.expand_dims(augru_biases, 1)
      augru_biases = tf.concat([augru_biases, augru_biases], axis=1)

    options = json.dumps(self._options)
    options_bwd = json.dumps(self._options_bwd)

    popnn_result = gen_popnn_ops.popnn_augru_layer(
        inputs=inputs,
        att_score=attention_score,
        seq_len=seq_len,
        num_channels=self._num_units,
        kernel=self.kernel,
        biases=augru_biases,
        initial_state=initial_state,
        is_training=training,
        partials_dtype=self._partials_dtype,
        activation=self._activation,
        recurrent_activation=self._recurrent_activation,
        name=self._name,
        reset_after=self._reset_after,
        options=options,
        options_bwd=options_bwd)

    return self._make_call_result(popnn_result, seq_len)

  @property
  def saveable(self):
    return False
