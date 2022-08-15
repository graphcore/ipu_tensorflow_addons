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
import pva
import keras
from keras import layers
from ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils import (
    profiler, pva_utils)


def create_model_1():
  x0 = layers.Input(shape=(4096,))
  x1 = layers.Dense(1024)(x0)
  return keras.Model(x0, x1)


# As of Poplar version: 3.0.0 (e295b81196) Poplar package: 6e1d109c88
# THIS MAY NEED CHANGING AS POPLAR VERSION CHANGES!
TEST_CASES_SINGLE_LAYER = [{
    "testcase_name":
    "Model1Dense",
    "create_model":
    create_model_1,
    "batch_size":
    16,
    "ans": [{
        # From Liveness Report - Always Live Variables - vertexCode.
        "vertexCode": 3.4 * 1024**2,
        # From Memory Report - Exchanges - programs starting dense.
        "internalExchangeCode": 360 * 1024**1,
        # From Liveness Report - Always Live Variables - vertexInstanceState.
        "vertexInstanceState": 178 * 1024**1,
        # Not displayed in Graph Analyzer. This is usually a lot smaller than
        # Liveness Report - Always Live Variables - controlCode,
        # as this value contains a lot non-layer related program.
        "controlCode": 100 * 1024**1,
        # Liveness Report - Not always live variable - find a step with maximum
        # not-always-live memory.
        # Then remove any not-always-live variable that looks like a parameter.
        "maxLive": 9.8 * 1024**2 - 0,
        # Liveness Report - Always-Live Variables - Variables with /
        # Plus value you subtracted from maxLive.
        "parameter": 16.1 * 1024**2,
        # Memory Report - Memory Excluding Gaps.
        "totalMemory": 32.8 * 1024**2,
        # Liveness Report - Step that seems to relate to layer computation -
        # Cycle Estimate - Maximum among all tiles.
        "cycleEst": 8700
    }]
}]


class PVASingleLayerTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  """Test `profiler.profile_layer_from_assignment` can generate a pva report for
  each layer in a model without runtime error."""

  def create_report_dict(self, report, name):
    return {
        "vertexCode":
        pva_utils.get_vertex_code(report, 0),
        "internalExchangeCode":
        pva_utils.get_internal_exchange_code(report, 0, name),
        "vertexInstanceState":
        pva_utils.get_vertex_instance_state(report, 0),
        "controlCode":
        pva_utils.get_control_code(report, 0, name),
        "maxLive":
        pva_utils.get_max_live_memory(report, 0),
        "parameter":
        pva_utils.get_parameter_from_always_live(report, 0, name) +
        pva_utils.get_parameter_from_not_always_alive(report, 0),
        "totalMemory":
        pva_utils.get_total_memory(report, 0),
        "cycleEst":
        pva_utils.get_layer_cycle(report, name, True)
    }

  def assert_est_close(self, estimated_dict, actual_dict):
    """Allow 10% of difference between the estimated value and the
    actual value."""
    for k in actual_dict.keys():
      estimated = estimated_dict[k]
      actual = actual_dict[k]
      d = actual / 10
      return self.assertAllInRange(estimated, actual - d, actual + d)

  @parameterized.named_parameters(*TEST_CASES_SINGLE_LAYER)
  @test_util.run_v2_only
  def testAll(self, create_model, batch_size, ans):
    report_helper = tu.ReportHelper()
    strategy = profiler.create_strategy(1, report_helper, True)
    with strategy.scope():
      model = create_model()
      assignments = model.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        pva_pop = profiler.profile_layer_from_assignment(
            assignment, batch_size, strategy, report_helper)
        report = pva.openReport(pva_pop)
        est_dict = self.create_report_dict(report, assignment.layer.name)
        self.assert_est_close(est_dict, ans[i])


if __name__ == "__main__":
  googletest.main()
