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

import tempfile
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from tensorflow import keras
from tensorflow.python.platform import googletest
from ipu_tensorflow_addons.keras.optimizers import AdamIpuOptimizer
from ipu_tensorflow_addons.keras.optimizers.test_util import OptimizerTest


def adam_update_numpy(
    param,
    g_t,
    t,
    m,
    v,
    learning_rate,
    beta_1,
    beta_2,
    epsilon,
    debiasing,
):

  # Update biased first moment estimate
  # m_t = beta1 * m_t-1 + (1 - beta1) * g_t
  m_t = beta_1 * m + (1 - beta_1) * g_t

  # Update biased second raw moment estimate
  # v_t = beta2 * v_t-1 + (1 - beta2) * g_t^2
  v_t = beta_2 * v + (1 - beta_2) * g_t * g_t

  if debiasing:
    # Compute bias-corrected first moment estimate
    # m_hat = m_t / (1 - beta1^t)
    m_t_hat = m_t / (1 - beta_1**(t + 1))
    # Compute bias-corrected second raw moment estimate
    # v_hat = v_t / (1 - beta2^t)
    v_t_hat = v_t / (1 - beta_2**(t + 1))
  else:
    m_t_hat = m_t
    v_t_hat = v_t

  # Update parameters
  # var_t = var_t-1 - lr * m_hat / (sqrt(v_hat) + epsilon)
  update = m_t_hat / (np.sqrt(v_t_hat) + epsilon)
  param_t = param - learning_rate * update

  return param_t, m_t, v_t


def adam_update_numpy_amsgrad(
    param,
    g_t,
    t,
    m,
    v,
    v_hat,
    learning_rate,
    beta_1,
    beta_2,
    epsilon,
    debiasing,
):

  # Update biased first moment estimate
  # m_t = beta1 * m_t-1 + (1 - beta1) * g_t
  m_t = beta_1 * m + (1 - beta_1) * g_t

  # Update biased second raw moment estimate
  # v_t = beta2 * v_t-1 + (1 - beta2) * g_t^2
  v_t = beta_2 * v + (1 - beta_2) * g_t * g_t

  if debiasing:
    # Compute bias-corrected first moment estimate
    # m_hat = m_t / (1 - beta1^2)
    m_t_hat = m_t / (1 - beta_1**(t + 1))
    # Compute bias-corrected second raw moment estimate
    # v_hat = v_t / (1 - beta2^2)
    v_t_hat = v_t / (1 - beta_2**(t + 1))
  else:
    m_t_hat = m_t
    v_t_hat = v_t

  # Get maximum of past second raw moment estimate
  # v_hat = max(v_hat_t-1, v_hat)
  v_t_hat = np.maximum(v_hat, v_t_hat)  # pylint: disable=assignment-from-no-return

  # Update parameters
  # var_t = var_t-1 - lr * m_hat / (sqrt(v_hat) + epsilon)
  update = m_t_hat / (np.sqrt(v_t_hat) + epsilon)
  param_t = param - learning_rate * update

  return param_t, m_t, v_t, v_t_hat


# pylint: disable=protected-access
def get_beta_accumulators(opt):
  local_step = tf.cast(opt.iterations + 1, opt.opt_dtypes[0])
  beta_1_t = tf.identity(opt._get_hyper('beta_1', opt.opt_dtypes[0]))
  beta_2_t = tf.identity(opt._get_hyper('beta_2', opt.opt_dtypes[0]))
  beta_1_power = tf.pow(beta_1_t, local_step)
  beta_2_power = tf.pow(beta_2_t, local_step)
  return beta_1_power, beta_2_power


class AdamOptimizerTest(OptimizerTest):
  @parameterized.product(
      learning_rate=[0.01, 0.001],
      beta_1=[0.9],
      beta_2=[0.999],
      epsilon=[1e-7],
      debiasing=[True, False],
      dtype=[tf.float32],
  )
  def testFunctionality(
      self,
      learning_rate,
      beta_1,
      beta_2,
      epsilon,
      debiasing,
      dtype,
  ):
    m_0_np, v_0_np, m_1_np, v_1_np = 0.0, 0.0, 0.0, 0.0
    var0_np = np.array([1.0, 1.0, 2.0], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.0, 0.1], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.0, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np, name="var0", dtype=dtype)
    var1 = tf.Variable(var1_np, name="var1", dtype=dtype)
    grads0 = tf.constant(grads0_np, dtype=dtype)
    grads1 = tf.constant(grads1_np, dtype=dtype)

    opt = AdamIpuOptimizer(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        debiasing=debiasing,
        optimizer_compute_precisions=(dtype,),
        m_dtype=dtype,
        v_dtype=dtype,
    )

    self.evaluate(tf.compat.v1.global_variables_initializer())

    for t in range(3):
      beta_1_power, beta_2_power = get_beta_accumulators(opt)
      self.assertAllCloseAccordingToType(beta_1**(t + 1),
                                         self.evaluate(beta_1_power))
      self.assertAllCloseAccordingToType(beta_2**(t + 1),
                                         self.evaluate(beta_2_power))

      opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

      var0_np, m_0_np, v_0_np = adam_update_numpy(var0_np, grads0_np, t,
                                                  m_0_np, v_0_np,
                                                  learning_rate, beta_1,
                                                  beta_2, epsilon, debiasing)

      var1_np, m_1_np, v_1_np = adam_update_numpy(var1_np, grads1_np, t,
                                                  m_1_np, v_1_np,
                                                  learning_rate, beta_1,
                                                  beta_2, epsilon, debiasing)

      m_0 = opt.get_slot(var0, "m")
      v_0 = opt.get_slot(var0, "v")
      m_1 = opt.get_slot(var1, "m")
      v_1 = opt.get_slot(var1, "v")

      self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
      self.assertAllCloseAccordingToType(m_0_np, self.evaluate(m_0))
      self.assertAllCloseAccordingToType(v_0_np, self.evaluate(v_0))
      self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))
      self.assertAllCloseAccordingToType(m_1_np, self.evaluate(m_1))
      self.assertAllCloseAccordingToType(v_1_np, self.evaluate(v_1))

  @parameterized.product(
      learning_rate=[0.01, 0.001],
      beta_1=[0.9],
      beta_2=[0.999],
      epsilon=[1e-7],
      debiasing=[True, False],
      dtype=[tf.float32],
  )
  def testFunctionalityAmsGrad(
      self,
      learning_rate,
      beta_1,
      beta_2,
      epsilon,
      debiasing,
      dtype,
  ):
    m_0_np, v_0_np, m_1_np, v_1_np = 0.0, 0.0, 0.0, 0.0
    vhat_0_np, vhat_1_np = 0.0, 0.0
    var0_np = np.array([1.0, 1.0, 2.0], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.0, 0.1], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.0, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np, name="var0", dtype=dtype)
    var1 = tf.Variable(var1_np, name="var1", dtype=dtype)
    grads0 = tf.constant(grads0_np, dtype=dtype)
    grads1 = tf.constant(grads1_np, dtype=dtype)

    opt = AdamIpuOptimizer(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        debiasing=debiasing,
        optimizer_compute_precisions=(dtype,),
        m_dtype=dtype,
        v_dtype=dtype,
        vhat_dtype=dtype,
        amsgrad=True,
    )

    self.evaluate(tf.compat.v1.global_variables_initializer())

    for t in range(3):
      beta_1_power, beta_2_power = get_beta_accumulators(opt)
      self.assertAllCloseAccordingToType(beta_1**(t + 1),
                                         self.evaluate(beta_1_power))
      self.assertAllCloseAccordingToType(beta_2**(t + 1),
                                         self.evaluate(beta_2_power))

      opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

      var0_np, m_0_np, v_0_np, vhat_0_np = adam_update_numpy_amsgrad(
          var0_np, grads0_np, t, m_0_np, v_0_np, vhat_0_np, learning_rate,
          beta_1, beta_2, epsilon, debiasing)

      var1_np, m_1_np, v_1_np, vhat_1_np = adam_update_numpy_amsgrad(
          var1_np, grads1_np, t, m_1_np, v_1_np, vhat_1_np, learning_rate,
          beta_1, beta_2, epsilon, debiasing)

      m_0 = opt.get_slot(var0, "m")
      v_0 = opt.get_slot(var0, "v")
      vhat_0 = opt.get_slot(var0, "vhat")
      m_1 = opt.get_slot(var1, "m")
      v_1 = opt.get_slot(var1, "v")
      vhat_1 = opt.get_slot(var1, "vhat")

      self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
      self.assertAllCloseAccordingToType(m_0_np, self.evaluate(m_0))
      self.assertAllCloseAccordingToType(v_0_np, self.evaluate(v_0))
      self.assertAllCloseAccordingToType(vhat_0_np, self.evaluate(vhat_0))
      self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))
      self.assertAllCloseAccordingToType(m_1_np, self.evaluate(m_1))
      self.assertAllCloseAccordingToType(v_1_np, self.evaluate(v_1))
      self.assertAllCloseAccordingToType(vhat_1_np, self.evaluate(vhat_1))

  @parameterized.product(
      learning_rate=[0.01, 0.001],
      beta_1=[0.9],
      beta_2=[0.999],
      epsilon=[1e-7],
      debiasing=[True, False],
      dtype=[tf.float32],
  )
  def testFunctionalityLrSchedule(
      self,
      learning_rate,
      beta_1,
      beta_2,
      epsilon,
      debiasing,
      dtype,
  ):
    m_0_np, v_0_np, m_1_np, v_1_np = 0.0, 0.0, 0.0, 0.0
    var0_np = np.array([1.0, 1.0, 2.0], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.0, 0.1], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.0, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np, name="var0", dtype=dtype)
    var1 = tf.Variable(var1_np, name="var1", dtype=dtype)
    grads0 = tf.constant(grads0_np, dtype=dtype)
    grads1 = tf.constant(grads1_np, dtype=dtype)

    decay = 0.5
    lr_schedule = keras.optimizers.schedules.InverseTimeDecay(learning_rate,
                                                              decay_steps=1.0,
                                                              decay_rate=decay)

    opt = AdamIpuOptimizer(
        learning_rate=lr_schedule,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        debiasing=debiasing,
        optimizer_compute_precisions=(dtype,),
        m_dtype=dtype,
        v_dtype=dtype,
    )

    self.evaluate(tf.compat.v1.global_variables_initializer())

    for t in range(3):
      beta_1_power, beta_2_power = get_beta_accumulators(opt)
      self.assertAllCloseAccordingToType(beta_1**(t + 1),
                                         self.evaluate(beta_1_power))
      self.assertAllCloseAccordingToType(beta_2**(t + 1),
                                         self.evaluate(beta_2_power))

      opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

      lr = learning_rate / (1 + decay * t)

      var0_np, m_0_np, v_0_np = adam_update_numpy(var0_np, grads0_np, t,
                                                  m_0_np, v_0_np, lr, beta_1,
                                                  beta_2, epsilon, debiasing)

      var1_np, m_1_np, v_1_np = adam_update_numpy(var1_np, grads1_np, t,
                                                  m_1_np, v_1_np, lr, beta_1,
                                                  beta_2, epsilon, debiasing)

      m_0 = opt.get_slot(var0, "m")
      v_0 = opt.get_slot(var0, "v")
      m_1 = opt.get_slot(var1, "m")
      v_1 = opt.get_slot(var1, "v")

      self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
      self.assertAllCloseAccordingToType(m_0_np, self.evaluate(m_0))
      self.assertAllCloseAccordingToType(v_0_np, self.evaluate(v_0))
      self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))
      self.assertAllCloseAccordingToType(m_1_np, self.evaluate(m_1))
      self.assertAllCloseAccordingToType(v_1_np, self.evaluate(v_1))

  @parameterized.product(
      mixed_prec_policy=["mixed_float16", "float16", "float32"],
      m_dtype=[tf.float16, tf.float32, None],
      v_dtype=[tf.float16, tf.float32, None],
      optimizer_compute_precisions=[(tf.float16,), (tf.float32,)],
      debiasing=[True, False],
  )
  def testKerasMixedPrecisionSupported(self, mixed_prec_policy, m_dtype,
                                       v_dtype, optimizer_compute_precisions,
                                       debiasing):
    def get_optimizer():
      return AdamIpuOptimizer(
          m_dtype=m_dtype,
          v_dtype=v_dtype,
          optimizer_compute_precisions=optimizer_compute_precisions,
          debiasing=debiasing,
      )

    optimizer = get_optimizer()
    model = self.train_simple_model_on_sample(optimizer, mixed_prec_policy)
    sample_var = model.layers[1].trainable_variables[0]

    expected_m_dtype = m_dtype or sample_var.dtype
    self.assertEqual(
        optimizer.get_slot(sample_var, "m").dtype, expected_m_dtype)

    expected_v_dtype = v_dtype or sample_var.dtype
    self.assertEqual(
        optimizer.get_slot(sample_var, "v").dtype, expected_v_dtype)

    with tempfile.TemporaryDirectory() as tmp_dir:
      old_weights = model.get_weights()
      model.save_weights(tmp_dir)
      model = self.get_simple_model(get_optimizer(), mixed_prec_policy)
      model.load_weights(tmp_dir)
      new_weights = model.get_weights()
      for old_weight, new_weight in zip(old_weights, new_weights):
        self.assertAllEqual(old_weight, new_weight)


if __name__ == '__main__':
  googletest.main()
