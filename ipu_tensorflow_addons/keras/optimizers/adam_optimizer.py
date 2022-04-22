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
"""Adam optimizer implementation."""

import tensorflow.compat.v2 as tf
from tensorflow.python import ipu
from tensorflow import keras
from ipu_tensorflow_addons.keras.optimizers import IpuOptimizerBase


class AdamIpuOptimizer(keras.optimizers.Adam, IpuOptimizerBase):
  """Optimizer that implements the Adam algorithm.

  Adam optimization is a stochastic gradient descent method that is based on
  adaptive estimation of first-order and second-order moments.
  According to the paper
  `Adam: A Method for Stochastic Optimization. Kingma et al.,
  2014 <http://arxiv.org/abs/1412.6980>`_, the method is "*computationally
  efficient, has little memory requirement, invariant to diagonal rescaling
  of gradients, and is well suited for problems that are large in terms of
  data/parameters*".

  For AMSGrad see `On The Convergence Of Adam And Beyond.
  Reddi et al., 5-8 <https://openreview.net/pdf?id=ryQu7f-RZ>`_

  This optimizer allows setting the optimizer state precisions independently
  and differently to the var precisions. It also supports outlining the
  optimizer update, which can save memory at the expense of passing variables
  around by making the optimizer update a reusable code block.
  """
  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               name="Adam",
               m_dtype=None,
               v_dtype=None,
               vhat_dtype=None,
               debiasing=True,
               **kwargs):
    r"""
    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta_1: A float value or a constant float tensor. The exponential decay
        rate for the 1st moment estimates.
      beta_2: A float value or a constant float tensor. The exponential decay
        rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
      amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from
        the paper "On the Convergence of Adam and beyond".
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".
      m_dtype: Dtype of the optimizer state m. If None, will set to
        dtypes of the corresponding vars.
      v_dtype: Dtype of the optimizer state v. If None, will set to
        dtypes of the corresponding vars.
      vhat_dtype: Dtype of the optimizer state vhat. If None, will set to
        dtypes of the corresponding vars.
      debiasing: Debias m and v to correct for initialisation.
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """
    super().__init__(learning_rate=learning_rate,
                     beta_1=beta_1,
                     beta_2=beta_2,
                     epsilon=epsilon,
                     amsgrad=amsgrad,
                     name=name,
                     **kwargs)
    self.m_dtype = m_dtype
    self.v_dtype = v_dtype
    self.vhat_dtype = vhat_dtype
    self.debiasing = debiasing

  def get_slot_dtype(self, var, slot_name):
    if slot_name == 'm':
      return self.m_dtype
    elif slot_name == 'v':
      return self.v_dtype
    assert slot_name == 'vhat'
    return self.vhat_dtype

  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var, 'm', dtype=self.get_slot_dtype(var, 'm'))  # pylint: disable=unexpected-keyword-arg
    for var in var_list:
      self.add_slot(var, 'v', dtype=self.get_slot_dtype(var, 'v'))  # pylint: disable=unexpected-keyword-arg
    if self.amsgrad:
      for var in var_list:
        self.add_slot(var, 'vhat', dtype=self.get_slot_dtype(var, 'vhat'))  # pylint: disable=unexpected-keyword-arg

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super()._prepare_local(var_device, var_dtype, apply_state)

    local_step = tf.cast(self.iterations + 1, self.opt_dtypes[0])
    beta_1_t = tf.identity(self._get_hyper('beta_1', self.opt_dtypes[0]))
    beta_2_t = tf.identity(self._get_hyper('beta_2', self.opt_dtypes[0]))
    beta_1_power = tf.pow(beta_1_t, local_step)
    beta_2_power = tf.pow(beta_2_t, local_step)
    apply_state[(var_device, var_dtype)].update(
        dict(epsilon=tf.convert_to_tensor(self.epsilon, self.opt_dtypes[0]),
             beta_1_t=beta_1_t,
             beta_1_power=beta_1_power,
             one_minus_beta_1_t=1 - beta_1_t,
             beta_2_t=beta_2_t,
             beta_2_power=beta_2_power,
             one_minus_beta_2_t=1 - beta_2_t))

  def _adam_step(self, grad, var, apply_state):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    m_dtype = m.dtype
    v_dtype = v.dtype

    if self.amsgrad:
      v_hat_t = self.get_slot(var, 'vhat')
      v_hat_dtype = v_hat_t.dtype

    compute_dtype = self.opt_dtypes[0]
    beta_1_t = coefficients['beta_1_t']
    beta_2_t = coefficients['beta_2_t']
    assignments = []

    def grad_fn(grad, var, m, v, v_hat_t=None):
      cast_grad = tf.cast(grad, dtype=compute_dtype)
      cast_m = tf.cast(m, dtype=compute_dtype)
      cast_v = tf.cast(v, dtype=compute_dtype)

      # Update biased first moment estimate
      # m_t = beta1 * m_t-1 + (1 - beta1) * g_t
      next_m = (tf.multiply(beta_1_t, cast_m) +
                tf.multiply(1.0 - beta_1_t, cast_grad))

      # Update biased second raw moment estimate
      # v_t = beta2 * v_t-1 + (1 - beta2) * g_t^2
      next_v = (tf.multiply(beta_2_t, cast_v) +
                tf.multiply(1.0 - beta_2_t, tf.square(cast_grad)))

      next_m_cast = tf.cast(next_m, dtype=m_dtype)
      next_v_cast = tf.cast(next_v, dtype=v_dtype)

      if self.debiasing:
        # Compute bias-corrected first moment estimate
        # m_hat = m_t / (1 - beta1^2)
        m_hat = next_m / (1.0 - coefficients['beta_1_power'])
        # Compute bias-corrected second raw moment estimate
        # v_hat = v_t / (1 - beta2^2)
        v_hat = next_v / (1.0 - coefficients['beta_2_power'])
      else:
        m_hat = next_m
        v_hat = next_v

      if self.amsgrad:
        # Get maximum of past second raw moment estimate
        # v_hat = max(v_hat_t-1, v_hat)
        v_hat_t = tf.maximum(v_hat_t, v_hat)
        next_v_hat_t_cast = tf.cast(v_hat_t, dtype=v_hat_dtype)
      else:
        v_hat_t = v_hat
        next_v_hat_t_cast = None

      # Update parameters
      # var_t = var_t-1 - lr * m_hat / (sqrt(v_hat) + epsilon)

      update = m_hat / (tf.sqrt(v_hat_t) + coefficients['epsilon'])

      update_with_lr = tf.cast(coefficients['lr_t'], compute_dtype) * update

      next_var = var - tf.cast(update_with_lr, var.dtype)

      return next_var, next_v_cast, next_m_cast, next_v_hat_t_cast

    if self.outline_apply_gradients:
      grad_fn = ipu.outlined_function(grad_fn,
                                      **self.outline_apply_gradients_kwargs)

    if self.amsgrad:
      next_var, next_v, next_m, next_v_hat_t = grad_fn(grad, var, m, v,
                                                       v_hat_t)
      assignments.append(v_hat_t.assign(next_v_hat_t))
    else:
      next_var, next_v, next_m, _ = grad_fn(grad, var, m, v)

    assignments.extend([
        var.assign(next_var),
        m.assign(next_m),
        v.assign(next_v),
    ])

    return tf.group(*assignments)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    return self._adam_step(grad, var, apply_state)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    raise NotImplementedError(
        "_resource_apply_sparse is not implemented for the IPU Addons"
        " Adam optimizer.")

  def get_config(self):
    config = super(AdamIpuOptimizer, self).get_config()
    config.update({
        'm_dtype': self.m_dtype,
        'v_dtype': self.v_dtype,
        'vhat_dtype': self.vhat_dtype,
        'debiasing': self.debiasing,
    })
    return config
