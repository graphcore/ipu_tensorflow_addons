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
"""SGD optimizer implementation."""

import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

from ipu_tensorflow_addons.keras.optimizers import IpuOptimizerBase


class SGDIpuOptimizer(gradient_descent.SGD, IpuOptimizerBase):
  """Optimizer that implements the gradient descent algorithm with momentum.

  This optimizer allows setting the optimizer state precisions differently to
  the var precisions. It also supports outlining the optimizer update, which
  can save memory at the expense of passing variables around by making the
  optimizer update a reusable code block.

  For `nesterov=True`, see `[`Sutskever et al., 2013
  <http://jmlr.org/proceedings/papers/v28/sutskever13.pdf>`_.
  """
  def __init__(self,
               learning_rate=0.01,
               momentum=0.0,
               nesterov=False,
               name="SGD",
               momentum_accum_dtype=None,
               **kwargs):
    """
    Args:
      learning_rate: A `Tensor` or a floating point value. or a schedule
        that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
        The learning rate.
      momentum: A `float` value or a constant `float` tensor that
        accelerates gradient descent in the relevant direction and
        dampens oscillations
      nesterov: boolean. Whether to apply Nesterov momentum.
        Defaults to `False`.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to `"SGD"`.
      momentum_accum_dtype: Dtype of the momentum accumulation optimizer state.
        If None, will set to dtypes of the corresponding vars.
      **kwargs: keyword arguments. Allowed to be {`clipnorm`,
        `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
        norm; `clipvalue` is clip gradients by value, `decay` is
        included for backward compatibility to allow time inverse
        decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """
    super().__init__(learning_rate=learning_rate,
                     momentum=momentum,
                     nesterov=nesterov,
                     name=name,
                     **kwargs)
    self.momentum_accum_dtype = momentum_accum_dtype

  def get_slot_dtype(self, var, slot_name):
    assert slot_name == 'momentum'
    return self.momentum_accum_dtype

  def _create_slots(self, var_list):
    if self._momentum:
      for var in var_list:
        self.add_slot(var,
                      'momentum',
                      dtype=self.get_slot_dtype(var, 'momentum'))

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(SGDIpuOptimizer, self)._prepare_local(var_device, var_dtype,
                                                apply_state)
    apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(
        self._get_hyper("momentum", self.opt_dtypes[0]))

  def _sgd_step(self, grad, var, apply_state):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))
    return tf.raw_ops.ResourceApplyGradientDescent(
        var=var.handle,
        alpha=coefficients["lr_t"],
        delta=grad,
        use_locking=self._use_locking)

  def _sgd_momentum_step(self, grad, var, apply_state):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))
    mom_accum = self.get_slot(var, "momentum")
    mom_accum_dtype = mom_accum.dtype
    compute_dtype = self.opt_dtypes[0]
    momentum = coefficients["momentum"]
    assignments = []

    def grad_fn(grad, var, accum):
      cast_grad = math_ops.cast(grad, compute_dtype)
      cast_accum = math_ops.cast(accum, compute_dtype)

      lr = math_ops.cast(coefficients['lr_t'], compute_dtype)

      # Update accumulator
      # velocity = momentum * velocity - learning_rate * g
      next_accum = cast_accum * momentum - lr * cast_grad

      if self.nesterov:
        # Parameter update for Nesterov momentum:
        # velocity = momentum * velocity - learning_rate * g
        next_var = var + math_ops.cast(next_accum * momentum - lr * cast_grad,
                                       var.dtype)
      else:
        # Parameter update:
        # w = w + velocity
        next_var = var + math_ops.cast(next_accum, var.dtype)

      next_accum_cast = math_ops.cast(next_accum, mom_accum_dtype)

      return next_var, next_accum_cast

    if self.outline_apply_gradients:
      grad_fn = ipu.outlined_function(grad_fn,
                                      **self.outline_apply_gradients_kwargs)

    next_var, next_mom_accum = grad_fn(grad, var, mom_accum)
    assignments.extend(
        [var.assign(next_var),
         mom_accum.assign(next_mom_accum)])
    return control_flow_ops.group(*assignments)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    if self._momentum:
      return self._sgd_momentum_step(grad, var, apply_state)
    return self._sgd_step(grad, var, apply_state)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    raise NotImplementedError(
        "_resource_apply_sparse is not implemented for the IPU Addons"
        " SGD optimizer.")

  def get_config(self):
    config = super(SGDIpuOptimizer, self).get_config()
    config.update({"momentum_accum_dtype": self.momentum_accum_dtype})
    return config
