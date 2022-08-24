# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
"""Test keras_utils"""
from tensorflow.python import ipu
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from absl.testing import parameterized
import keras
from keras import layers
import numpy as np
from ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils import (
    keras_utils)


def create_strategy():
  cfg = ipu.config.IPUConfig()
  cfg.auto_select_ipus = 1
  cfg.configure_ipu_system()
  return ipu.ipu_strategy.IPUStrategyV1()


def create_model_intermediate_test_1():
  # InputLayer does not appear in pipeline_stage_assignment
  x0 = layers.Input(shape=(2048,))
  # 1st Layer
  # CarryOver []        Size: 0
  # Input     [x0]      Size: 16*2048*4
  # Output    [x1]      Size: 16*1024*4
  x1 = layers.Dense(1024)(x0)
  # 2nd Layer
  # CarryOver []        Size: 0
  # Input     [x1]      Size: 16*1024*4
  # Output    [x2]      Size: 16*1024*4
  x2 = layers.Dense(1024)(x1)
  # 3rd Layer
  # CarryOver []        Size: 0
  # Input     [x2]      Size: 16*1024*4
  # Output    [x3]      Size: 16*1024*4
  x3 = layers.Dense(1024)(x2)
  # 4th Layer
  # CarryOver []        Size: 0
  # Input     [x3]      Size: 16*1024*4
  # Output    [x4]      Size: 16*1024*4
  x4 = layers.Dense(1024)(x3)
  # 5th Layer
  # CarryOver []        Size: 0
  # Input     [x4]      Size: 16*1024*4
  # Output    [x5]      Size: 16*16*4
  x5 = layers.Dense(16)(x4)
  return keras.Model(x0, x5)


def create_model_intermediate_test_2():
  # InputLayer does not appear in pipeline_stage_assignment
  x0 = layers.Input(shape=(2048,))
  # 1st Layer
  # CarryOver []        Size: 0
  # Input     [x0]      Size: 16*2048*4
  # Output    [x1]      Size: 16*1024*4
  x1 = layers.Dense(1024)(x0)
  # 2nd Layer
  # CarryOver [x1]      Size: 16*1024*4
  # Input     [x1]      Size: 16*1024*4
  # Output    [x1,x2]   Size: 16*2048*4
  x2 = layers.Dense(1024)(x1)
  # 3rd Layer
  # CarryOver [x1]      Size: 16*1024*4
  # Input     [x1,x2]   Size: 16*2048*4
  # Output    [x1,x3]   Size: 16*2048*4
  x3 = layers.Dense(1024)(x2)  #carry x1, input x2, output x1 x3
  # 4th Layer
  # CarryOver []        Size: 0
  # Input     [x1,x3]   Size: 16*2048*4
  # Output    [x4]      Size: 16*1024*4
  x4 = x1 + x3
  # 5th Layer
  # CarryOver []        Size: 0
  # Input     [x4]      Size: 16*1024*4
  # Output    [x5]      Size: 16*16*4
  x5 = layers.Dense(16)(x4)
  return keras.Model(x0, x5)


def create_model_intermediate_test_3():
  # InputLayer does not appear in pipeline_stage_assignment
  x0 = layers.Input(shape=(128, 768))
  # 1st Layer
  # CarryOver [x0]        Size: 16*128*768*4
  # Input     [x0]        Size: 16*128*768*4
  # Output    [x0,x1]     Size: 2*16*128*768*4
  x1 = layers.MultiHeadAttention(num_heads=2, key_dim=768 // 2)(x0, x0, x0)
  # 2nd Layer
  # CarryOver []          Size: 0
  # Input     [x0,x1]     Size: 2*16*128*768*4
  # Output    [x2]        Size: 16*128*768*4
  x2 = x0 + x1
  return keras.Model(x0, x2)


def create_model_intermediate_test_4():
  # InputLayer does not appear in pipeline_stage_assignment
  x0 = layers.Input(shape=(128, 768))
  # 1st Layer
  # CarryOver []          Size: 16*128*768*4
  # Input     [x0]        Size: 16*128*768*4
  # Output    [x1]        Size: 16*128*768*4
  x1 = layers.MultiHeadAttention(num_heads=2, key_dim=768 // 2)(x0, x0, x0)
  return keras.Model(x0, x1)


MODEL_INTERMEDIATE_TEST_CASES = [{
    "testcase_name":
    "ModelNoResidual",
    "create_model":
    create_model_intermediate_test_1,
    "batch_size":
    16,
    "ans_intermediate":
    np.array([0, 0, 0, 0, 0]),
    "ans_input":
    np.array([2048, 1024, 1024, 1024, 1024]) * 16 * 4,
    "ans_output":
    np.array([1024, 1024, 1024, 1024, 16]) * 16 * 4
}, {
    "testcase_name":
    "ModelWithResidual",
    "create_model":
    create_model_intermediate_test_2,
    "batch_size":
    16,
    "ans_intermediate":
    np.array([0, 1024, 1024, 0, 0]) * 16 * 4,
    "ans_input":
    np.array([2048, 1024, 2048, 2048, 1024]) * 16 * 4,
    "ans_output":
    np.array([1024, 2048, 2048, 1024, 16]) * 16 * 4
}, {
    "testcase_name":
    "BertAttention",
    "create_model":
    create_model_intermediate_test_3,
    "batch_size":
    16,
    "ans_intermediate":
    np.array([128, 0]) * 16 * 768 * 4,
    "ans_input":
    np.array([128, 128 * 2]) * 16 * 768 * 4,
    "ans_output":
    np.array([128 * 2, 128]) * 16 * 768 * 4
}, {
    "testcase_name":
    "MultiHeadAttention",
    "create_model":
    create_model_intermediate_test_4,
    "batch_size":
    16,
    "ans_intermediate":
    np.array([0]) * 16 * 768 * 4,
    "ans_input":
    np.array([128]) * 16 * 768 * 4,
    "ans_output":
    np.array([128]) * 16 * 768 * 4
}]


class ParseModelIntermediateTest(test_util.TensorFlowTestCase,
                                 parameterized.TestCase):
  """Test keras_utils.parse_model_intermediate."""

  @parameterized.named_parameters(*MODEL_INTERMEDIATE_TEST_CASES)
  @test_util.run_v2_only
  def testAll(self, create_model, batch_size, ans_intermediate, ans_input,
              ans_output):
    strategy = create_strategy()
    with strategy.scope():
      model = create_model()
      (memory_intermediate, memory_input,
       memory_output) = keras_utils.parse_model_intermediate(
           model, batch_size)
      self.assertAllEqual(memory_intermediate, ans_intermediate)
      self.assertAllEqual(memory_input, ans_input)
      self.assertAllEqual(memory_output, ans_output)


def create_model_layer_type_test_1():
  model_input = layers.Input(shape=(2048,))
  x0 = layers.Dense(1024)(model_input)
  x1 = layers.Dense(1024, activation="relu")(x0)
  x2 = layers.Dense(1024, activation="softmax")(x1)
  x3 = layers.Dense(1024, activation="relu", kernel_initializer="constant")(x2)
  x4 = layers.Dense(1024, activation="relu", use_bias=False)(x3)
  x5 = layers.Dense(16)(x4)
  return keras.Model(model_input, x5)


def create_model_layer_type_test_2():
  model_input = layers.Input(shape=(128, 768))
  x0 = layers.MultiHeadAttention(num_heads=2,
                                 key_dim=768 // 2)(model_input, model_input,
                                                   model_input)
  x1 = layers.LayerNormalization(epsilon=1e-6)(x0)
  x2 = layers.LayerNormalization(epsilon=1e-10)(x1)
  return keras.Model(model_input, x2)


LAYER_TYPE_TEST_CASES = [
    {
        # Different argument shapes.
        "testcase_name": "InputShape",
        "create_model": create_model_layer_type_test_1,
        "layer1_id": 0,
        "layer2_id": 1,
        "ans": False
    },
    {
        # Different activation functions.
        "testcase_name": "Activation",
        "create_model": create_model_layer_type_test_1,
        "layer1_id": 1,
        "layer2_id": 2,
        "ans": False
    },
    {
        # Different weight initializer.
        # But the resulting computation should be equivalent.
        "testcase_name": "Initializer",
        "create_model": create_model_layer_type_test_1,
        "layer1_id": 1,
        "layer2_id": 3,
        "ans": True
    },
    {
        # Different useBias.
        "testcase_name": "UseBias",
        "create_model": create_model_layer_type_test_1,
        "layer1_id": 1,
        "layer2_id": 4,
        "ans": False
    },
    {
        # Different eps.
        # But the resulting computation should be mostly equivalent.
        "testcase_name": "eps",
        "create_model": create_model_layer_type_test_2,
        "layer1_id": 1,
        "layer2_id": 2,
        "ans": True
    },
    {
        # Different class.
        "testcase_name": "class",
        "create_model": create_model_layer_type_test_2,
        "layer1_id": 0,
        "layer2_id": 1,
        "ans": False
    }
]


class LayerTypeTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  """Test keras_utils.get_assignment_layer_type."""

  @parameterized.named_parameters(*LAYER_TYPE_TEST_CASES)
  @test_util.run_v2_only
  def testAll(self, create_model, layer1_id, layer2_id, ans):
    strategy = create_strategy()
    with strategy.scope():
      model = create_model()
      assignments = model.get_pipeline_stage_assignment()
      node1_type = keras_utils.get_assignment_layer_type(
          assignments[layer1_id], 16)
      node2_type = keras_utils.get_assignment_layer_type(
          assignments[layer2_id], 16)
      self.assertEqual((node1_type == node2_type), ans)


if __name__ == "__main__":
  googletest.main()
