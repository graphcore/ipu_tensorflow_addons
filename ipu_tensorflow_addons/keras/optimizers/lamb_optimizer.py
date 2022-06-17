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
"""Layer-wise Adaptive Moments (LAMB) optimizer."""

import re

import tensorflow.compat.v2 as tf
from tensorflow.python import ipu
from ipu_tensorflow_addons.keras.optimizers import IpuOptimizerBase


class LAMBIpuOptimizer(IpuOptimizerBase):
  """Optimizer that implements the Layer-wise Adaptive Moments (LAMB).
  See paper `Large Batch Optimization for Deep Learning: Training BERT
  in 76 minutes <https://arxiv.org/abs/1904.00962>`_.

  This optimizer allows setting the optimizer state precisions independently
  and differently to the var precisions. It also supports outlining the
  optimizer update, which can save memory at the expense of passing variables
  around by making the optimizer update a reusable code block.
  """
  def __init__(
      self,
      learning_rate=0.001,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      weight_decay_rate=0.0,
      exclude_from_weight_decay=None,
      exclude_from_layer_adaptation=None,
      name="LAMB",
      debiasing=True,
      m_dtype=None,
      v_dtype=None,
      weight_norm_clip=None,
      optimizer_compute_precisions=(tf.float32, tf.float32),
      **kwargs,
  ):
    """
    Args:
      learning_rate: A `Tensor` or a floating point value. or a schedule
        that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
        The learning rate.
      beta_1: A `float` value or a constant `float` tensor.
        The exponential decay rate for the 1st moment estimates.
      beta_2: A `float` value or a constant `float` tensor.
        The exponential decay rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability.
      weight_decay_rate: weight decay rate.
      exclude_from_weight_decay: List of regex patterns of
        variables excluded from weight decay. Variables whose name
        contain a substring matching the pattern will be excluded.
      exclude_from_layer_adaptation: List of regex patterns of
        variables excluded from layer adaptation. Variables whose name
        contain a substring matching the pattern will be excluded.
      name: Optional name for the operations created when applying
        gradients. Defaults to "LAMB".
      debiasing: Debias m and v to correct for initialisation.
      m_dtype: Dtype of the optimizer state m. If None, will set to
        dtypes of the vars.
      v_dtype: Dtype of the optimizer state v. If None, will set to
        dtypes of the vars.
      weight_norm_clip: Clip the weight norms by this value.
      optimizer_compute_precisions: Tuple of TF dtypes that determine
        what precision the stages of optimizer compute are done in.
        This optimizer has two stages of compute precision so the
        tuple must be of size 2.
      **kwargs: keyword arguments. Allowed to be {`clipnorm`,
        `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
        norm; `clipvalue` is clip gradients by value, `decay` is
        included for backward compatibility to allow time inverse
        decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """
    super().__init__(name, **kwargs)
    self._set_hyper("weight_decay_rate", weight_decay_rate)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("decay", self._initial_decay)
    self._set_hyper("beta_1", beta_1)
    self._set_hyper("beta_2", beta_2)
    self.epsilon = epsilon or tf.backend_config.epsilon()
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    self.weight_norm_clip = weight_norm_clip
    self.m_dtype = m_dtype
    self.v_dtype = v_dtype
    self.debiasing = debiasing

    self.opt_dtypes = optimizer_compute_precisions
    if len(self.opt_dtypes) != 2:
      raise ValueError(
          "Must provide a list of two elements for the optimizer"
          " compute precision. The final stage of the weight update"
          " can be done in a different precision to the initial stage.")

  def get_slot_dtype(self, var, slot_name):
    if slot_name == 'm':
      return self.m_dtype
    assert slot_name == 'v'
    return self.v_dtype

  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var, 'm', dtype=self.get_slot_dtype(var, 'm'))  # pylint: disable=unexpected-keyword-arg
    for var in var_list:
      self.add_slot(var, 'v', dtype=self.get_slot_dtype(var, 'v'))  # pylint: disable=unexpected-keyword-arg

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super()._prepare_local(var_device, var_dtype, apply_state)
    compute_dtype = self.opt_dtypes[0]
    local_step = tf.cast(self.iterations + 1, compute_dtype)
    beta_1_t = tf.identity(self._get_hyper("beta_1", compute_dtype))
    beta_2_t = tf.identity(self._get_hyper("beta_2", compute_dtype))
    weight_decay_rate = tf.identity(
        self._get_hyper("weight_decay_rate", compute_dtype))
    beta_1_power = tf.pow(beta_1_t, local_step)
    beta_2_power = tf.pow(beta_2_t, local_step)
    apply_state[(var_device, var_dtype)].update(
        dict(
            weight_decay_rate=weight_decay_rate,
            epsilon=tf.convert_to_tensor(self.epsilon, compute_dtype),
            beta_1_t=beta_1_t,
            beta_1_power=beta_1_power,
            one_minus_beta_1_t=1 - beta_1_t,
            beta_2_t=beta_2_t,
            beta_2_power=beta_2_power,
            one_minus_beta_2_t=1 - beta_2_t,
        ))

  def _resource_apply_dense(self, grad, handle, apply_state):
    var_device, var_dtype = handle.device, handle.dtype.base_dtype
    var_name = self._get_variable_name(handle.name)
    coefficients = (apply_state or {}).get(
        (var_device, var_dtype)) or self._fallback_apply_state(
            var_device, var_dtype)

    m = self.get_slot(handle, "m")
    v = self.get_slot(handle, "v")
    m_dtype = m.dtype
    v_dtype = v.dtype
    compute_dtype = self.opt_dtypes[0]
    apply_ratio_dtype = self.opt_dtypes[1]
    beta_1_t = coefficients['beta_1_t']
    beta_2_t = coefficients['beta_2_t']
    assignments = []

    def grad_fn(grad, var, m, v):
      cast_grad = tf.cast(grad, dtype=compute_dtype)
      cast_m = tf.cast(m, dtype=compute_dtype)
      cast_v = tf.cast(v, dtype=compute_dtype)

      # Update biased first moment estimate
      # m_t = m_t-1 * beta1 + g_t * (1 - beta1)
      next_m = (tf.multiply(cast_m, beta_1_t) +
                tf.multiply(cast_grad, 1.0 - beta_1_t))

      # Update biased second raw moment estimate
      # v_t = v_t-1 * beta2 + g_t^2 * (1 - beta2)
      next_v = (tf.multiply(cast_v, beta_2_t) +
                tf.multiply(tf.square(cast_grad), 1.0 - beta_2_t))

      next_m_cast = tf.cast(next_m, dtype=m_dtype)
      next_v_cast = tf.cast(next_v, dtype=v_dtype)

      if self.debiasing:
        # Compute bias-corrected first moment estimate
        # m_hat = m_t / (1 - beta1^t)
        m_hat = next_m / (1.0 - coefficients['beta_1_power'])
        # Compute bias-corrected second raw moment estimate
        # v_hat = v_t / (1 - beta2^t)
        v_hat = next_v / (1.0 - coefficients['beta_2_power'])
      else:
        m_hat = next_m
        v_hat = next_v

      # Compute ratio
      # m_hat / (sqrt(v_hat) + epsilon)
      update = m_hat / (tf.sqrt(v_hat) + coefficients['epsilon'])

      # Apply weight decay
      if self._do_use_weight_decay(var_name):
        update += tf.cast(var, update.dtype) * tf.cast(self.weight_decay_rate,
                                                       update.dtype)

      reshaped_update = tf.reshape(update, [-1])

      ratio = 1.0

      # Do layer wise normalisation on update
      if self._do_layer_adaptation(var_name):
        reshaped_param = tf.reshape(var, [-1])

        w_norm = tf.norm(tf.cast(reshaped_param, dtype=compute_dtype),
                         ord=2,
                         axis=-1)
        u_norm = tf.norm(reshaped_update, ord=2, axis=-1)

        # Clip norm of parameters by value
        if self.weight_norm_clip is not None:
          w_norm = tf.minimum(
              w_norm, tf.cast(self.weight_norm_clip, dtype=w_norm.dtype))

        w_norm_cast = tf.cast(w_norm, dtype=compute_dtype)
        u_norm_cast = tf.cast(u_norm, dtype=compute_dtype)

        ratio = tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(u_norm, 0), w_norm_cast / u_norm_cast,
                     tf.constant(1.0, dtype=compute_dtype,
                                 shape=w_norm.shape)),
            tf.constant(1.0, dtype=compute_dtype, shape=w_norm.shape))
        ratio = tf.reshape(ratio, shape=ratio.shape.as_list() + [1])

      # Perform parameter update
      ratio = tf.cast(coefficients['lr_t'], compute_dtype) * ratio
      ratio = tf.cast(ratio, dtype=apply_ratio_dtype)
      reshaped_update = tf.cast(reshaped_update, dtype=apply_ratio_dtype)
      update_with_lr = ratio * reshaped_update
      update_with_lr = tf.reshape(update_with_lr, shape=var.shape)
      update_with_lr = tf.cast(update_with_lr, dtype=var.dtype)

      next_var = var - update_with_lr

      return next_var, next_m_cast, next_v_cast

    if self.outline_apply_gradients:
      grad_fn = ipu.outlined_function(grad_fn,
                                      **self.outline_apply_gradients_kwargs)

    next_var, next_m, next_v = grad_fn(grad, handle, m, v)
    assignments.extend(
        [handle.assign(next_var),
         m.assign(next_m),
         v.assign(next_v)])
    return tf.group(*assignments)

  def _resource_apply_sparse(self, grad, handle, indices, apply_state):
    raise NotImplementedError(
        "_resource_apply_sparse is not implemented for the IPU Addons"
        " LAMB optimizer.")

  def get_config(self):
    config = super().get_config()
    config.update({
        "learning_rate":
        self._serialize_hyperparameter("learning_rate"),
        "weight_decay_rate":
        self._serialize_hyperparameter("weight_decay_rate"),
        "decay":
        self._serialize_hyperparameter("decay"),
        "beta_1":
        self._serialize_hyperparameter("beta_1"),
        "beta_2":
        self._serialize_hyperparameter("beta_2"),
        "epsilon":
        self.epsilon,
        'm_dtype':
        self.m_dtype,
        'v_dtype':
        self.v_dtype,
        'debiasing':
        self.debiasing,
    })
    return config

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for
    `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.exclude_from_layer_adaptation:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
