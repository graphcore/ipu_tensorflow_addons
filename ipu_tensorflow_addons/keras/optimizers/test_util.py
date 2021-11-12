# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
from absl.testing import parameterized
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.python.framework import test_util


class OptimizerTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  def keras_mixed_precision_test_helper(self, optimizer, mixed_prec_policy):
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
    x_train = np.array([[1, 1, 1, 1, 1]], dtype="float32")
    y_train = np.array([1], dtype="float32")
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return dense1.trainable_variables[0]
