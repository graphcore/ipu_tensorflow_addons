# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import numpy as np
from absl.testing import parameterized
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.python.framework import test_util


class OptimizerTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  def get_simple_model(self, optimizer, mixed_prec_policy):
    policy = mixed_precision.Policy(mixed_prec_policy)
    mixed_precision.set_global_policy(policy)

    inputs = keras.Input(shape=(5,), name="digits")
    dense1 = layers.Dense(5, activation="relu", name="dense_1")
    x = dense1(inputs)
    outputs = layers.Activation("softmax", dtype="float32",
                                name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model

  def train_simple_model_on_sample(self, optimizer, mixed_prec_policy):
    model = self.get_simple_model(optimizer, mixed_prec_policy)
    x_train = np.array([[1, 1, 1, 1, 1]], dtype="float32")
    y_train = np.array([1], dtype="float32")
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model
