# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# ==============================================================================
# Parts of this code are derived from TensorFlow.
#
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
# ==============================================================================
"""IPU Addons base class for optimizers in TensorFlow."""
from __future__ import absolute_import, division, print_function

import functools

import six
from tensorflow.python.distribute import \
  distribution_strategy_context as distribute_ctx
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend, initializers
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import _var_key
from tensorflow.python.ops import variables as tf_variables


class IpuOptimizerBase(optimizer_v2.OptimizerV2):
  """Base class for optimizers utilising mixed precision and IPU
  features. Should not be used directly but instead you should
  instantiate one of its subclasses.
  """
  def __init__(self,
               name,
               optimizer_compute_precisions=(dtypes.float32,),
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

  def add_slot_with_dtype(self, var, slot_name, dtype, initializer="zeros"):
    """Add a new slot variable for `var` with a specific dtype.

    Args:
      var: A variable to add.
      slot_name: The name of the slot.
      dtype: Dtype to create the slot in.
      initializer: Default initializer for `var`.
    """
    if slot_name not in self._slot_names:
      self._slot_names.append(slot_name)
    if dtype is None:
      dtype = var.dtype
    var_key = _var_key(var)
    slot_dict = self._slots.setdefault(var_key, {})
    weight = slot_dict.get(slot_name, None)
    if weight is None:
      if isinstance(initializer, six.string_types) or callable(initializer):
        initializer = initializers.get(initializer)
        initial_value = functools.partial(initializer,
                                          shape=var.shape,
                                          dtype=dtype)
      else:
        initial_value = initializer
      strategy = distribute_ctx.get_strategy()
      with strategy.extended.colocate_vars_with(var):
        weight = tf_variables.Variable(
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
