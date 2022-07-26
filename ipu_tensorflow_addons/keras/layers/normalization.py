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
Normalization Keras layers
~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import operator
from functools import reduce
import tensorflow.compat.v2 as tf
from tensorflow import keras
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from ipu_tensorflow_addons.keras.layers import ipu_layer


# We implement all three algorithms through a common generic group norm algorithm.
class GroupNormalization(ipu_layer.IPULayer):
  """Group normalization layer optimized for running on the IPU.

  This layer is used like the standard Keras BatchNormalization layer.
  However, it has beta and gamma trainable parameters, but no statistics
  gathering.

  Group normalization is described in this paper:
  https://arxiv.org/abs/1803.08494.

  Arguments:
    groups: The number of groups to use in the normalization.
    channels_axis: Integer, the axis that should be normalized
      (typically the features axis).
    center: If True, add offset of `beta` to normalized tensor.
      If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
      If False, `gamma` is not used.
    epsilon: Small float added to variance to avoid dividing by zero.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    strided_channel_grouping: Selects whether to group the channels dimension
      for group normalisation with a stride between channels. This makes the
      PopLibs implementation more efficient but is unconventional. Among other
      things this will mean that using pre-trained weights would not be possible
      if not produced with this unconventional implementation.
    trainable: Boolean, if `True` the variables will be marked as trainable.
  """
  def __init__(self,
               groups=2,
               channels_axis=-1,
               center=True,
               scale=True,
               epsilon=1e-3,
               beta_initializer=None,
               gamma_initializer=None,
               strided_channel_grouping=True,
               trainable=True,
               **kwargs):
    super().__init__(**kwargs)

    self.groups = groups
    self.channels_axis = channels_axis
    self.center = center
    self.scale = scale
    self.epsilon = epsilon

    self.beta_initializer = keras.initializers.get(beta_initializer)
    self.gamma_initializer = keras.initializers.get(gamma_initializer)

    self.strided_channel_grouping = strided_channel_grouping

    self.data_format = ""
    self.channels = 1
    self.trainable = trainable

  def build(self, input_shape):
    return self._build_impl(input_shape)

  def _build_impl(self, input_shape, parameter_shape=None):
    if input_shape is None:
      raise ValueError('Input has undefined rank.')
    if self.channels_axis > (len(input_shape) - 1):
      raise ValueError('Axis is out of bounds.')

    # Standardize the channels_axis to be positive and identify # of channels.
    if self.channels_axis < 0:
      self.channels_axis = len(input_shape) + self.channels_axis
    self.channels = input_shape[self.channels_axis]

    if self.groups > self.channels:
      raise ValueError('Invalid groups %d for %d channels.' %
                       (self.groups, self.channels))
    if self.channels % self.groups != 0:
      raise ValueError('%d channels is not commensurate with %d groups.' %
                       (self.channels, self.groups))

    # Check which format the data is in.
    if self.channels_axis == 1:
      self.data_format = "NCHW"
    elif self.channels_axis == len(input_shape) - 1:
      self.data_format = "NHWC"
    else:
      raise ValueError('Unsupported data format, group norm only supports NCHW'
                       '(channel axis 1) and NHWC (channel axis -1).')

    parameter_shape = parameter_shape if parameter_shape else [self.channels]

    if self.scale:
      self.gamma = self.add_weight("gamma",
                                   dtype=self.dtype,
                                   initializer=self.gamma_initializer,
                                   shape=parameter_shape,
                                   trainable=self.trainable)

    if self.center:
      self.beta = self.add_weight("beta",
                                  dtype=self.dtype,
                                  initializer=self.beta_initializer,
                                  shape=parameter_shape,
                                  trainable=self.trainable)

    self.built = True

  # pylint: disable=arguments-differ
  def call(self, inputs, training=None):
    """
    Args:
      inputs: The tensor to apply normalization to.

    Returns:
      The tensor resulting from applying normalization.
    """
    params_shape = [self.channels]

    # TensorFlow doesn't like constants being created in the build func.
    if not self.center:
      self.beta = tf.constant(0.0, dtype=self.dtype, shape=params_shape)

    if not self.scale:
      self.gamma = tf.constant(1.0, dtype=self.dtype, shape=params_shape)

    # Flatten beta and gamma as this operation is 2D.
    beta = tf.reshape(self.beta, [-1])
    gamma = tf.reshape(self.gamma, [-1])

    def group_norm_training():
      outputs, _, _ = gen_popnn_ops.popnn_group_norm_training(
          inputs=inputs,
          gamma=gamma,
          beta=beta,
          data_format=self.data_format,
          epsilon=self.epsilon,
          num_groups=self.groups,
          strided_channel_grouping=self.strided_channel_grouping)
      return outputs

    def group_norm_inference():
      # Calculate the moments.
      mean, inv_std_dev = gen_popnn_ops.popnn_group_norm_statistics(
          inputs=inputs,
          data_format=self.data_format,
          epsilon=self.epsilon,
          num_groups=self.groups,
          strided_channel_grouping=self.strided_channel_grouping)

      outputs = gen_popnn_ops.popnn_group_norm_inference(
          inputs=inputs,
          gamma=gamma,
          beta=beta,
          mean=mean,
          inv_std_dev=inv_std_dev,
          data_format=self.data_format,
          epsilon=self.epsilon,
          num_groups=self.groups,
          strided_channel_grouping=self.strided_channel_grouping)
      return outputs

    outputs = keras.backend.in_train_phase(group_norm_training,
                                           group_norm_inference,
                                           training=training)

    return outputs

  def get_config(self):
    return {
        "groups": self.groups,
        "channels_axis": self.channels_axis,
        "center": self.center,
        "scale": self.scale,
        "epsilon": self.epsilon,
        "beta_initializer":
        keras.initializers.serialize(self.beta_initializer),
        "gamma_initializer":
        keras.initializers.serialize(self.gamma_initializer),
        "strided_channel_grouping": self.strided_channel_grouping,
        "trainable": self.trainable,
    }


class InstanceNormalization(GroupNormalization):
  """Instance normalization layer optimized for use on the IPU.

  This layer is used like the standard Keras InstanceNormalization layer.
  However, it has beta and gamma trainable parameters, but no statistics
  gathering.

  Instance normalization is described in this paper:
  https://arxiv.org/abs/1607.08022.

  Arguments:
    channels_axis: Integer, the axis that should be normalized
      (typically the features axis).
    center: If True, add offset of `beta` to normalized tensor.
      If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
      If False, `gamma` is not used.
    epsilon: Small float added to variance to avoid dividing by zero.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
  """
  def __init__(self,
               channels_axis=-1,
               center=True,
               scale=True,
               epsilon=1e-3,
               beta_initializer=None,
               gamma_initializer=None,
               trainable=True,
               **kwargs):
    super().__init__(
        # We set this in the build function, once we know what the shape is.
        groups=0,
        channels_axis=channels_axis,
        center=center,
        scale=scale,
        epsilon=epsilon,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        trainable=trainable,
        **kwargs)

  # pylint: disable=useless-super-delegation
  def build(self, input_shape):
    # Change the groups based on the input shape.
    self.groups = input_shape[self.channels_axis]
    super().build(input_shape)

  # pylint: disable=useless-super-delegation
  def call(self, inputs, training=None):
    return super().call(inputs, training)

  def get_config(self):
    return {
        "channels_axis": self.channels_axis,
        "center": self.center,
        "scale": self.scale,
        "epsilon": self.epsilon,
        "beta_initializer":
        keras.initializers.serialize(self.beta_initializer),
        "gamma_initializer":
        keras.initializers.serialize(self.gamma_initializer),
        "trainable": self.trainable,
    }


class LayerNormalization(GroupNormalization):
  """Layer normalization layer optimized for use on the IPU.

  This layer is used like the standard Keras LayerNormalization layer.
  However, it has beta and gamma trainable parameters, but no statistics
  gathering.

  Layer normalization is described in this paper:
  https://arxiv.org/abs/1607.06450.

  Arguments:
    axis: Integer or List/Tuple. The axis that should be normalized
      (typically the features axis).
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
      If False, `gamma` is not used.
      When the next layer is linear (also e.g. `nn.relu`),
      this can be disabled since the scaling
      will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.
    trainable: Boolean, if `True` the variables will be marked as trainable.
  """
  def __init__(self,
               axis=-1,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               trainable=True,
               **kwargs):
    if isinstance(axis, (list, tuple)):
      self.axis = axis[:]
    elif isinstance(axis, int):
      self.axis = axis
    else:
      raise ValueError('Expected an int or a list/tuple of ints for the '
                       'argument \'axis\', but received instead: %s' % axis)

    channels_axis = -1
    super().__init__(groups=1,
                     channels_axis=channels_axis,
                     center=center,
                     scale=scale,
                     epsilon=epsilon,
                     beta_initializer=beta_initializer,
                     gamma_initializer=gamma_initializer,
                     trainable=trainable,
                     **kwargs)

    self._check_unsupported(beta_regularizer, "beta_regularizer")
    self._check_unsupported(gamma_regularizer, "gamma_regularizer")
    self._check_unsupported(beta_constraint, "beta_constraint")
    self._check_unsupported(gamma_constraint, "gamma_constraint")

    self.beta_regularizer = keras.regularizers.get(beta_regularizer)
    self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
    self.beta_constraint = keras.constraints.get(beta_constraint)
    self.gamma_constraint = keras.constraints.get(gamma_constraint)

  def build(self, input_shape):
    ndims = len(input_shape)
    if ndims is None:
      raise ValueError('Input shape %s has undefined rank.' % input_shape)

    # Convert axis to list and resolve negatives
    if isinstance(self.axis, int):
      self.axis = [self.axis]
    elif isinstance(self.axis, tuple):
      self.axis = list(self.axis)
    for idx, x in enumerate(self.axis):
      if x < 0:
        self.axis[idx] = ndims + x

    # Create the parameter shape before flattening the non-reduced dimensions.
    parameter_shape = [input_shape[dim] for dim in self.axis]

    self.axis = sorted(self.axis)

    # Validate axes
    for x in self.axis:
      if x < 0 or x >= ndims:
        raise ValueError('Invalid axis: %d' % x)
    if len(self.axis) != len(set(self.axis)):
      raise ValueError('Duplicate axis: {}'.format(tuple(self.axis)))

    if any([input_shape[dim] is None for dim in self.axis]):
      raise ValueError(
          "Input shape %s has unknown dimensions - all dimensions need to be "
          "fully specified." % input_shape)

    # Reshape into a 2D tensor, with the 0th dimension corresponding to the
    # non-reduced dimensions and the 1st dimension corresponding to the reduced
    # dimensions.

    # Find all the dimensions which are not reduced and find the permutation.
    self.non_reduced_dims = sorted(list(set(range(ndims)) - set(self.axis)))
    self.permutation = self.non_reduced_dims + self.axis

    # Get the number of elements which are reduced.
    self.num_reduced_elements = reduce(operator.mul,
                                       [input_shape[dim] for dim in self.axis],
                                       1)

    inputs_shape = [None, self.num_reduced_elements]
    super()._build_impl(inputs_shape, parameter_shape)

  # pylint: disable=useless-super-delegation
  def call(self, inputs, training=None):
    input_shape = inputs.shape

    if any([input_shape[dim] is None for dim in self.non_reduced_dims]):
      raise ValueError(
          "Input shape %s has unknown dimensions - all dimensions need to be "
          "fully specified." % input_shape)

    # Create the 2D shape.
    num_non_reduced_elements = reduce(
        operator.mul, [input_shape[dim] for dim in self.non_reduced_dims], 1)
    input_shape_2d = [num_non_reduced_elements, self.num_reduced_elements]

    # Permute the inputs to move the reduction and non reduction dimensions.
    permuted = tf.transpose(inputs, self.permutation)
    permuted_shape = permuted.shape
    # Reshape into 2D.
    permuted = tf.reshape(permuted, input_shape_2d)

    # Call the group norm.
    outputs = super().call(permuted, training)

    # Reshape back to the original shape.
    outputs = tf.reshape(outputs, permuted_shape)
    # Inverse the transpose.
    outputs = tf.transpose(outputs,
                           tf.math.invert_permutation(self.permutation))

    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    return outputs

  def get_config(self):
    return {
        "axis": self.axis,
        "epsilon": self.epsilon,
        "center": self.center,
        "scale": self.scale,
        "beta_initializer":
        keras.initializers.serialize(self.beta_initializer),
        "gamma_initializer":
        keras.initializers.serialize(self.gamma_initializer),
        "beta_regularizer":
        keras.regularizers.serialize(self.beta_regularizer),
        "gamma_regularizer":
        keras.regularizers.serialize(self.gamma_regularizer),
        "beta_constraint": keras.constraints.serialize(self.beta_constraint),
        "gamma_constraint": keras.constraints.serialize(self.gamma_constraint),
        "trainable": self.trainable,
    }


GroupNorm = GroupNormalization
InstanceNorm = InstanceNormalization
LayerNorm = LayerNormalization
