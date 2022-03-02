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
from absl.testing import parameterized
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest

from ipu_tensorflow_addons.keras.optimizers import SGDIpuOptimizer
from ipu_tensorflow_addons.keras.optimizers.test_util import OptimizerTest


def sgd_momentum_numpy(param, g_t, learning_rate, momentum, accum):
  # Update when momentum is 0:
  # w = w - learning_rate * g
  # Update when momentum is larger than 0:
  # velocity = momentum * velocity - learning_rate * g
  # w = w + velocity
  accum = accum * momentum - learning_rate * g_t
  param = param + accum
  return param, accum


def sgd_nesterov_momentum_numpy(param, g_t, learning_rate, momentum, accum):
  # Update rule when using Nesterov momentum:
  # velocity = momentum * velocity - learning_rate * g
  # w = w + momentum * velocity - learning_rate * g
  accum = accum * momentum - learning_rate * g_t
  param = param + accum * momentum - learning_rate * g_t
  return param, accum


class SGDOptimizerTest(OptimizerTest):
  @parameterized.product(
      learning_rate=[0.01, 0.001],
      momentum=[0.0, 0.9],
      nesterov=[False, True],
      dtype=[dtypes.float32],
  )
  def test_functionality(
      self,
      learning_rate,
      momentum,
      nesterov,
      dtype,
  ):
    accum_0, accum_1 = 0.0, 0.0
    var0_np = np.array([1.0, 1.0, 2.0], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.0, 0.1], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.0, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = variables.Variable(var0_np, name="var0", dtype=dtype)
    var1 = variables.Variable(var1_np, name="var1", dtype=dtype)
    grads0 = constant_op.constant(grads0_np, dtype=dtype)
    grads1 = constant_op.constant(grads1_np, dtype=dtype)

    opt = SGDIpuOptimizer(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
        optimizer_compute_precisions=(dtype,),
        momentum_accum_dtype=dtype,
    )

    self.evaluate(variables.global_variables_initializer())

    for _ in range(3):
      opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

      if nesterov:
        var0_np, accum_0 = sgd_nesterov_momentum_numpy(var0_np, grads0_np,
                                                       learning_rate, momentum,
                                                       accum_0)

        var1_np, accum_1 = sgd_nesterov_momentum_numpy(var1_np, grads1_np,
                                                       learning_rate, momentum,
                                                       accum_1)
      else:
        var0_np, accum_0 = sgd_momentum_numpy(var0_np, grads0_np,
                                              learning_rate, momentum, accum_0)

        var1_np, accum_1 = sgd_momentum_numpy(var1_np, grads1_np,
                                              learning_rate, momentum, accum_1)

      if momentum:
        slot_0 = opt.get_slot(var0, "momentum")
        slot_1 = opt.get_slot(var1, "momentum")
        self.assertAllCloseAccordingToType(accum_0, self.evaluate(slot_0))
        self.assertAllCloseAccordingToType(accum_1, self.evaluate(slot_1))

      self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
      self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @parameterized.product(
      learning_rate=[0.01, 0.001],
      momentum=[0.0, 0.9],
      nesterov=[False, True],
      dtype=[dtypes.float32],
  )
  def test_functionality_lr_schedule(
      self,
      learning_rate,
      momentum,
      nesterov,
      dtype,
  ):
    accum_0, accum_1 = 0.0, 0.0
    var0_np = np.array([1.0, 1.0, 2.0], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.0, 0.1], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.0, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = variables.Variable(var0_np, name="var0", dtype=dtype)
    var1 = variables.Variable(var1_np, name="var1", dtype=dtype)
    grads0 = constant_op.constant(grads0_np, dtype=dtype)
    grads1 = constant_op.constant(grads1_np, dtype=dtype)

    decay = 0.5
    lr_schedule = InverseTimeDecay(learning_rate,
                                   decay_steps=1.0,
                                   decay_rate=decay)

    opt = SGDIpuOptimizer(
        learning_rate=lr_schedule,
        momentum=momentum,
        nesterov=nesterov,
        optimizer_compute_precisions=(dtype,),
        momentum_accum_dtype=dtype,
    )

    self.evaluate(variables.global_variables_initializer())

    for t in range(3):
      opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

      lr = learning_rate / (1 + decay * t)

      if nesterov:
        var0_np, accum_0 = sgd_nesterov_momentum_numpy(var0_np, grads0_np, lr,
                                                       momentum, accum_0)
        var1_np, accum_1 = sgd_nesterov_momentum_numpy(var1_np, grads1_np, lr,
                                                       momentum, accum_1)
      else:
        var0_np, accum_0 = sgd_momentum_numpy(var0_np, grads0_np, lr, momentum,
                                              accum_0)
        var1_np, accum_1 = sgd_momentum_numpy(var1_np, grads1_np, lr, momentum,
                                              accum_1)

      if momentum:
        slot_0 = opt.get_slot(var0, "momentum")
        slot_1 = opt.get_slot(var1, "momentum")
        self.assertAllCloseAccordingToType(accum_0, self.evaluate(slot_0))
        self.assertAllCloseAccordingToType(accum_1, self.evaluate(slot_1))

      self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
      self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @parameterized.product(
      mixed_prec_policy=["mixed_float16", "float16", "float32"],
      mom_accum_dtype=[dtypes.float16, dtypes.float32, None],
      optimizer_compute_precisions=[(dtypes.float16,), (dtypes.float32,)],
      momentum=[0.0, 0.9],
  )
  def test_keras_mixed_precision_supported(self, mixed_prec_policy,
                                           mom_accum_dtype,
                                           optimizer_compute_precisions,
                                           momentum):
    def get_optimizer():
      return SGDIpuOptimizer(
          momentum=momentum,
          momentum_accum_dtype=mom_accum_dtype,
          optimizer_compute_precisions=optimizer_compute_precisions,
      )

    optimizer = get_optimizer()
    model = self.train_simple_model_on_sample(optimizer, mixed_prec_policy)
    sample_var = model.layers[1].trainable_variables[0]

    if momentum:
      expected_dtype = mom_accum_dtype or sample_var.dtype
      self.assertEqual(
          optimizer.get_slot(sample_var, "momentum").dtype, expected_dtype)

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
