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
Dense Keras layer
~~~~~~~~~~~~~~~~~
"""

import tensorflow.compat.v2 as tf
from tensorflow import keras
from tensorflow.python.ipu.ops import math_ops as ipu_math_ops
from tensorflow.python.ipu.ops.f8_ops import f8_matmul, create_metadata, QuarterTensor, Format

from keras import backend as K
from keras.engine.input_spec import InputSpec
import numpy as np


class SerialDense(keras.layers.Layer):
  """Densely-connected NN layer where the dot operation is serialized to reduce
  the size of this operation.

  `Dense` implements the operation:
  `output = activation(dot(input, kernel) + bias)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a weights matrix
  created by the layer, and `bias` is a bias vector created by the layer
  (only applicable if `use_bias` is `True`).

  Given the `input` tensor with shape `[..., m, k]` and `kernel` tensor with
  shape `[k, n]`, the matrix multiplication can be serialized as follows:

  * Along the `m` dimension of `input`, by setting `serialization_dimension` to
    `input_columns`.
  * Along the `k` dimension of `input` and `kernel` by setting
    `serialization_dimension` to `input_rows_kernel_columns`.
  * Along `n` dimension of `kernel`, by setting `serialization_dimension` to
    `kernel_rows`.

  Example:

  .. code-block:: python

    # as first layer in a sequential model:
    model = Sequential()
    model.add(SerialDense(32, input_shape=(16,)))
    # now the model will take as input arrays of shape (*, 16)
    # and output arrays of shape (*, 32)

    # after the first layer, you don't need to specify
    # the size of the input anymore:
    model.add(SerialDense(32))

  Arguments:
    units: Positive integer, dimensionality of the output space.
    serialization_factor: An integer indicating the number of smaller matrix
      multiplies this operation is broken up into. Must divide the dimension
      along which the operation is serialized on.
    serialization_dimension: A string, must be one of `input_columns`,
      `input_rows_kernel_columns` or `kernel_rows`. Indicates the dimension
      along which the operation is serialzed on.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation").
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.

  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.

  Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.
  """

  def __init__(self,
               units,
               serialization_factor,
               serialization_dimension,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super().__init__(
        activity_regularizer=keras.regularizers.get(activity_regularizer),
        **kwargs)
    self.serialization_factor = int(serialization_factor)
    self.serialization_dimension = serialization_dimension

    self.units = int(units) if not isinstance(units, int) else units
    self.activation = keras.activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = keras.initializers.get(kernel_initializer)
    self.bias_initializer = keras.initializers.get(bias_initializer)
    self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
    self.bias_regularizer = keras.regularizers.get(bias_regularizer)
    self.kernel_constraint = keras.constraints.get(kernel_constraint)
    self.bias_constraint = keras.constraints.get(bias_constraint)

    self.supports_masking = True
    self.input_spec = keras.layers.InputSpec(min_ndim=2)

  def build(self, input_shape):
    dtype = tf.as_dtype(self.dtype or keras.backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `SerialDense` layer with non-floating '
                      'point dtype %s' % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs to `SerialDense` '
                       'should be defined. Found `None`.')
    last_dim = tf.compat.dimension_value(input_shape[-1])
    self.input_spec = keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})
    self.kernel = self.add_weight('kernel',
                                  shape=[last_dim, self.units],
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint,
                                  dtype=self.dtype,
                                  trainable=True)
    if self.use_bias:
      self.bias = self.add_weight('bias',
                                  shape=[
                                      self.units,
                                  ],
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint,
                                  dtype=self.dtype,
                                  trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
    """
    Args:
      inputs: The tensor to apply the dense weights to.

    Returns:
      The tensor resulting from applying the dense weights.
    """
    if keras.backend.is_sparse(inputs):
      raise TypeError(
          'Unable to build `SerialDense` layer with sparse inputs.')

    if self.serialization_factor < 1:
      raise ValueError(
          'serialization_factor has to be at least 1, but was {}.'.format(
              self.serialization_factor))

    inputs = tf.cast(inputs, self._compute_dtype)

    # Transform the dimension name.
    serialization_dimension = self.serialization_dimension
    if serialization_dimension == "input_columns":
      serialization_dimension = "a_columns"
    elif serialization_dimension == "input_rows_kernel_columns":
      serialization_dimension = "a_rows_b_columns"
    elif serialization_dimension == "kernel_rows":
      serialization_dimension = "b_rows"
    else:
      raise ValueError('Invalid serialization_dimension={}, expected one of: '
                       '\'input_columns\', \'input_rows_kernel_columns\', '
                       '\'kernel_rows\'.'.format(serialization_dimension))

    outputs = ipu_math_ops.serialized_matmul(inputs, self.kernel,
                                             self.serialization_factor,
                                             serialization_dimension)
    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = {
        'units':
        self.units,
        'serialization_factor':
        self.serialization_factor,
        'serialization_dimension':
        self.serialization_dimension,
        'activation':
        keras.activations.serialize(self.activation),
        'use_bias':
        self.use_bias,
        'kernel_initializer':
        keras.initializers.serialize(self.kernel_initializer),
        'bias_initializer':
        keras.initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
        keras.regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
        keras.regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
        keras.regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
        keras.constraints.serialize(self.kernel_constraint),
        'bias_constraint':
        keras.constraints.serialize(self.bias_constraint)
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Dense(keras.layers.Dense):
  """Dense layer with support for fp8 matrix multiplication.

  The layer uses fp8 when it is passed a list as its input, in which case it
  expects this input to come from a ConvertToF8 layer.

  Otherwise you should be able to pass most of the options available for the
  normal keras Dense layer.

  Note: you should not pass the output of convert_to_f8 directly to this layer,
  as it returns a QuarterTensor instead of a list that this layer expects.

  The default initializer for the kernel data is uniformly random in all
  possible uint8 values (other than the error value 0x80 which gets mapped to
  0). You can change this by passing an initializer to the constructor through
  `kernel_data_initializer`. Keep in mind that this will need to return uint8
  data, which you will most likely want to get from a call to `convert_to_f8`.

  By default the kernel metadata will be set to a scale of 0 and `F143` format.
  If you need a different kernel scale / format, you can achieve that by
  passing `kernel_scale` and `kernel_format` parameters to the constructor. The
  passed scale should be in the inclusive range [-32, 31], which multiplies the
  numeric value of the kernel by `2^kernel_scale`. The format should be
  of type :py:class:`~tensorflow.python.ipu.ops.f8_ops.Format`.

  You can also use the `get_weights` / `set_weights` methods to manipulate
  the weights.

  An example of using this layer eagerly:

  .. code-block:: python

    from tensorflow.python.ipu.ops.f8_ops import create_metadata, Format
    from keras.ipu.layers import Dense, ConvertToF8
    from tensorflow.python.ipu.ipu_strategy import IPUStrategyV1

    strategy = IPUStrategyV1()
    with strategy.scope():
      input_array = np.array([[1., 2.], [3., -1.]])
      f8_tensor = ConvertToF8()(input_array,
                                metadata=create_metadata(Format.F143))
      output = Dense(units=3)(f8_tensor)

  An example of using this layer in a Functional model:

  .. code-block:: python

    from tensorflow.python.ipu.ops.f8_ops import create_metadata, Format
    from keras.ipu.layers import Dense, ConvertToF8
    from tensorflow.python.ipu.ipu_strategy import IPUStrategyV1

    strategy = IPUStrategyV1()
    with strategy.scope():
      inputs = Input(dtype="float16", shape=[2], batch_size=2)
      outputs = ConvertToF8()(inputs, metadata=create_metadata(Format.F143))
      outputs = Dense(units=3)(outputs)
      model = keras.Model(inputs, outputs)

      input_array = np.array([[1., 2.], [3., -1.]])
      model.predict(input_array, batch_size=2)

  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.
    In case of passing an fp8 input, the input should be the output of a
    `ConvertToF8` layer.

  Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.

  Args:
    units: Positive integer, dimensionality of the output space.
    kernel_format: Format of the kernel tensor when using fp8; one of
      `Format.F143` or `Format.F152`. `Format` can be imported
      from `tensorflow.python.ipu.ops.f8_ops`.
    kernel_scale: Scale for the kernel tensor when using fp8.
    kernel_data_initializer: An initializer for the kernel data when using fp8.
  """

  def __init__(self,
               units,
               kernel_format=Format.F143,
               kernel_scale=0,
               kernel_data_initializer=None,
               **kwargs):
    self._initialised_weights = False
    self.kernel_format = kernel_format
    if not -32 <= kernel_scale <= 31:
      raise ValueError("`kernel_scale` should be in the range [-32, 31]. "
                       f"The passed scale was {kernel_scale}.")
    self.kernel_scale = kernel_scale
    self.kernel_data_initializer = kernel_data_initializer
    super().__init__(units, **kwargs)

    # This is necessary since the input spec will depend on the input
    self.input_spec = None

  def call(self, inputs):
    """Use fp8 MatMul if `inputs` is an instance of QuarterTensor.
    Otherwise behave like a normal Dense layer.
    """
    if not isinstance(inputs, list):
      if not self._initialised_weights:
        super().build(inputs.shape)
        self._initialised_weights = True
      # Use keras.layers.Dense
      return super().call(inputs)
    inputs = QuarterTensor(*inputs)
    # Perform fp8 deferred weight building
    if not self._initialised_weights:
      last_dim = tf.compat.dimension_value(inputs.shape[-1])
      data_initializer = self.kernel_data_initializer
      if not data_initializer:
        data = np.random.randint(0, 256, [last_dim, self.units])
        # Make sure that the data is not set to the error value 0x80.
        data[data == 0x80] = 0
        data_initializer = keras.initializers.Constant(data)
      self.kernel_data = self.add_weight('kernel_data',
                                         shape=[last_dim, self.units],
                                         initializer=data_initializer,
                                         dtype="uint8",
                                         trainable=True)
      metadata_initializer = keras.initializers.Constant(
          create_metadata(self.kernel_format, self.kernel_scale))
      self.kernel_metadata = self.add_weight('kernel_metadata',
                                             shape=[],
                                             dtype="uint8",
                                             trainable=True,
                                             initializer=metadata_initializer)

      if self.use_bias:
        # The bias should be float16 when multiplying fp8 values
        self.bias = self.add_weight('bias',
                                    shape=[
                                        self.units,
                                    ],
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    dtype="float16",
                                    trainable=True)
      else:
        self.bias = None
      self._initialised_weights = True
    kernel = QuarterTensor(self.kernel_data, self.kernel_metadata)
    outputs = f8_matmul(lhs=inputs, rhs=kernel)

    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)

    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs

  def build(self, input_shape):
    """Stripped down version of keras.layers.Dense.build.

    Defers weight construction to the call method so that we know if
    we're dealing with fp8 matmul or not, depending on inputs.
    """
    dtype = tf.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      f'dtype {dtype}')
    if isinstance(input_shape, list):
      return
    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
    self.built = True
