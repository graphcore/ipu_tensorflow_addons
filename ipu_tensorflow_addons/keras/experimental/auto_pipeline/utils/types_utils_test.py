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
"""Test types_utils"""
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from absl.testing import parameterized
from ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils import (
    types_utils)

# 10 byte per cycle for ipu-ipu
# 1 byte per cycle for cpu-ipu
cluster_2_ipu = {
    "clock":
    1024**3,
    "memory":
    128,
    "connection": [{
        "recvGBps": 1 if i == 0 else 10,
        "recvIdleCycle": 1000000 if i == 0 else 100000,
        "sendGBps": 1 if i == 1 else 10,
        "sendIdleCycle": 1000000 if i == 1 else 100000
    } for i in range(2)]
}

cluster_4_ipu = {
    "clock":
    1024**3,
    "memory":
    128,
    "connection": [{
        "recvGBps": 1 if i == 0 else 10,
        "recvIdleCycle": 1000000 if i == 0 else 100000,
        "sendGBps": 1 if i == 3 else 10,
        "sendIdleCycle": 1000000 if i == 3 else 100000
    } for i in range(4)]
}

model_profile_1 = {
    "layerName": [str(i) for i in range(12)],
    "layerType": [str(i) for i in range(12)],
    "layerInfo": [{
        "cycle": {
            "forward": 10000,
            "backward": 0,
        },
        "memory": {
            "shared": {},
            "exclusive": {},
            "inputs": (i + 1) * 1000,
            "outputs": (i + 1) * 1000,
            "maxTemporary": 0,
            "curCarryOver": 0,
            "curActivation": 0
        },
    } for i in range(12)]
}

TRANSFER_CYCLE_TEST_CASES = [
    {
        "testcase_name": "Cluster_2_IPU_IPU0",
        "args": [cluster_2_ipu, 0, model_profile_1, 0, 1],
        # IPU 0 send 2000 bytes: 200 + 100000 cycles
        # IPU 0 recv 1000 bytes: 1000 + 1000000 cycles
        # IPU 1 send 12000 bytes: 12000 + 1000000 cycles
        # Total = (200 + 100000) + (1000 + 1000000) + (12000 + 1000000)
        "ans": 2113200
    },
    {
        "testcase_name": "Cluster_2_IPU_IPU1",
        "args": [cluster_2_ipu, 1, model_profile_1, 6, 11],
        # IPU 1 recv 7000 bytes: 700 + 100000 cycles
        # IPU 0 recv 1000 bytes: 1000 + 1000000 cycles
        # IPU 1 send 12000 bytes: 12000 + 1000000 cycles
        # Total = (700 + 100000) + (1000 + 1000000) + (12000 + 1000000)
        "ans": 2113700
    },
    {
        "testcase_name": "Cluster_4_IPU_IPU1",
        "args": [cluster_4_ipu, 1, model_profile_1, 2, 4],
        # IPU 1 recv 3000 bytes: 300 + 100000 cycles
        # IPU 1 send 5000 bytes: 500 + 100000 cycles
        # IPU 0 recv 1000 bytes: 1000 + 1000000 cycles
        # IPU 3 send 12000 bytes: 12000 + 1000000 cycles
        # Total = (300 + 100000) +  (500 + 100000) \
        #        +(1000+ 1000000) + (12000 + 1000000)
        "ans": 2213800
    },
]


class TransferCycleTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.named_parameters(*TRANSFER_CYCLE_TEST_CASES)
  @test_util.run_v2_only
  def testAll(self, args, ans):
    self.assertEqual(types_utils.transfer_cycle(*args), ans)


AGGREGATE_INFO_TEST_CASES = [
    {
        "testcase_name": "1",
        "old_range": {
            "cycle": {
                "forward": 12345,
                "backward": 123456
            },
            "memory": {
                "shared": {
                    "code_1": 1000
                },
                "exclusive": {
                    "var_1": 10000,
                },
                "maxTemporary": 999999,
                "curActivation": 0,
                "totalMemory": 1000 + 10000 + 999999,
                "inputs": 100,
                "outputs": 100
            }
        },
        "new_layer": {
            "cycle": {
                "forward": 123,
                "backward": 1234
            },
            "memory": {
                "shared": {
                    "code_1": 1000,
                    "code_2": 1000,
                },
                "exclusive": {
                    "var_1": 10000,
                    "var_2": 10000,
                },
                "inputs": 233,
                "outputs": 233,
                "maxTemporary": 9000000,
                "curCarryOver": 999999,
                "curActivation": 0
            }
        },
        "ans_new_range": {
            "cycle": {
                "forward": 12345 + 123,
                "backward": 123456 + 1234
            },
            "memory": {
                "shared": {
                    "code_1": 1000,
                    "code_2": 1000,
                },
                "exclusive": {
                    "var_1": 20000,
                    "var_2": 10000,
                },
                "maxTemporary": 9000000 + 999999,
                "curActivation": 0,
                "totalMemory": 1000 + 1000 + 20000 + 10000 + 9000000 + 999999,
                "inputs": 100,
                "outputs": 233
            }
        }
    },
]


class AggregateInfoTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.named_parameters(*AGGREGATE_INFO_TEST_CASES)
  @test_util.run_v2_only
  def testAll(self, old_range, new_layer, ans_new_range):
    new_range = types_utils.aggregate_info_inplace(old_range, new_layer)
    self.assertDictEqual(new_range, ans_new_range)


if __name__ == "__main__":
  googletest.main()
