# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""IPU Addons base class for optimizers in TensorFlow."""
from __future__ import absolute_import, division, print_function

import functools

from keras.optimizer_v2 import optimizer_v2
from keras import backend
import tensorflow.compat.v2 as tf
from tensorflow.python.distribute import \
  distribution_strategy_context as distribute_ctx
from tensorflow import keras
from tensorflow.python.training.tracking import base as trackable


class IpuOptimizerBase(keras.optimizers.Optimizer):
  """Base class for optimizers utilising mixed precision and IPU
  features. Should not be used directly but instead you should
  instantiate one of its subclasses.
  """
  def __init__(self,
               name,
               optimizer_compute_precisions=(tf.float32,),
               outline_apply_gradients=False,
               outline_apply_gradients_kwargs=None,
               **kwargs):
    """
    Args:
      name: Name for the operations created when applying gradients.
      optimizer_compute_precisions: Tuple of TensorFlow dtypes that
        determine what precision the stages of optimizer compute are
        done in.
      outline_apply_gradients: If True, the operations to apply the
        gradients to the vars will be outlined
      outline_apply_gradients_kwargs: If using `outline_apply_gradients`,
        the kwargs used by outlining can be specified here as a
        dict.
      **kwargs: keyword arguments. Allowed to be {`clipnorm`,
        `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
        norm; `clipvalue` is clip gradients by value, `decay` is
        included for backward compatibility to allow time inverse
        decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """
    super().__init__(name, **kwargs)
    self.opt_dtypes = optimizer_compute_precisions
    self.outline_apply_gradients = outline_apply_gradients
    if outline_apply_gradients_kwargs:
      self.outline_apply_gradients_kwargs = outline_apply_gradients_kwargs
    else:
      self.outline_apply_gradients_kwargs = {}

  def _resource_apply_dense(self, grad, handle, apply_state):
    raise NotImplementedError("Must be implemented in subclasses.")

  def _resource_apply_sparse(self, grad, handle, indices, apply_state):
    raise NotImplementedError("Must be implemented in subclasses.")

  def get_slot_dtype(self, var, slot_name):  # pylint: disable=unused-argument
    """Returns the slot dtype for `var` and `slot_name`.

    Args:
      var: a `Variable` object.
      slot_name: name of the slot variable.

    Returns:
      The `dtype` of the slot variable.
    """
    return var.dtype

  def add_slot(  # pylint: disable=arguments-differ
      self,
      var,
      slot_name,
      initializer="zeros",
      shape=None,
      dtype=None):
    """Add a new slot variable for `var`.

    A slot variable is an additional variable associated with `var` to train.
    It is allocated and managed by optimizers, e.g. `Adam`.

    Args:
      var: a `Variable` object.
      slot_name: name of the slot variable.
      initializer: initializer of the slot variable
      shape: (Optional) shape of the slot variable. If not set, it will default
      to the shape of `var`.
      dtype: (Optional) data type the slot variable. If not set, it will use
      the data type of `get_slot_dtype()`. If `get_slot_dtype()` returns `None`
      then the data type of `var` is used.

    Returns:
      A slot variable.
    """
    # This function is a copy of the `Optimizer.add_slot()` function with
    # extra handling for the `dtype` to allow for mixed precision.
    if slot_name not in self._slot_names:
      self._slot_names.append(slot_name)
    var_key = optimizer_v2._var_key(var)  # pylint: disable=protected-access
    slot_dict = self._slots.setdefault(var_key, {})
    weight = slot_dict.get(slot_name, None)
    dtype = dtype or self.get_slot_dtype(var, slot_name) or var.dtype
    if weight is None:
      if isinstance(initializer, str) or callable(initializer):
        initializer = keras.initializers.get(initializer)
        if isinstance(
            initializer,
            trackable.CheckpointInitialValueCallable) or (shape is not None):
          slot_shape = shape
        else:
          slot_shape = var.shape
        initial_value = functools.partial(initializer,
                                          shape=slot_shape,
                                          dtype=dtype)
      else:
        initial_value = initializer

      with self._distribution_strategy_scope():
        strategy = distribute_ctx.get_strategy()
        if not strategy.extended.variable_created_in_scope(var):
          raise ValueError(
              "Trying to create optimizer slot variable under the scope for "
              "tf.distribute.Strategy ({}), which is different from the scope "
              "used for the original variable ({}). Make sure the slot "
              "variables are created under the same strategy scope. This may "
              "happen if you're restoring from a checkpoint outside the scope".
              format(strategy, var))

        with strategy.extended.colocate_vars_with(var):
          weight = tf.Variable(
              name="%s/%s" % (var._shared_name, slot_name),  # pylint: disable=protected-access
              dtype=dtype,
              trainable=False,
              initial_value=initial_value)
      backend.track_variable(weight)
      slot_dict[slot_name] = weight
      self._restore_slot_variable(slot_name=slot_name,
                                  variable=var,
                                  slot_variable=weight)
      self._weights.append(weight)
    return weight

  def add_slot_with_dtype(self, var, slot_name, dtype, initializer="zeros"):
    return self.add_slot(var,
                         slot_name=slot_name,
                         initializer=initializer,
                         dtype=dtype)

  def get_config(self):
    config = super(IpuOptimizerBase, self).get_config()
    config.update({
        'optimizer_compute_precisions':
        self.opt_dtypes,
        'outline_apply_gradients':
        self.outline_apply_gradients,
        'outline_apply_gradients_kwargs':
        self.outline_apply_gradients_kwargs,
    })
    return config
