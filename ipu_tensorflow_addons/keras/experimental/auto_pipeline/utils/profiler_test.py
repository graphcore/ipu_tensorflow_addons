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
"""Test profiler"""
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.ipu import test_utils as tu
from absl.testing import parameterized
import keras
from keras import layers
from ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils import (
    profiler)


def create_model_1():
  x0 = layers.Input(shape=(64,))
  x1 = layers.Dense(32)(x0)
  x2 = layers.Dense(16)(x1)
  return keras.Model(x0, x2)


def create_model_2():
  x0 = layers.Input(shape=(4, 16))
  x1 = layers.MultiHeadAttention(num_heads=2, key_dim=16 // 2)(x0, x0, x0)
  x2 = x0 + x1
  return keras.Model(x0, x2)


TEST_CASES = [{
    "testcase_name": "ModelWithTwoDenseLayers",
    "create_model": create_model_1,
    "batch_size": 16
}, {
    "testcase_name": "ModelWithMultiHeadAttention",
    "create_model": create_model_2,
    "batch_size": 16
}]


class ProfileSingleLayerForwardTest(test_util.TensorFlowTestCase,
                                    parameterized.TestCase):
  """Test `profiler.profile_layer_from_assignment` can generate a pva report for
  each layer in a model without runtime errors."""

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testAll(self, create_model, batch_size):
    report_helper = tu.ReportHelper()
    strategy = profiler.create_strategy(1, report_helper, True)
    with strategy.scope():
      model = create_model()
      assignments = model.get_pipeline_stage_assignment()
      for assignment in assignments:
        profiler.profile_layer_from_assignment(assignment, batch_size,
                                               strategy, report_helper)


if __name__ == "__main__":
  googletest.main()
