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
"Test auto pipelining"
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from absl.testing import parameterized
from ipu_tensorflow_addons.keras.experimental.auto_pipeline.prediction import (
    partition)

# 1 byte per cycle
cluster_2_ipu_no_transfer_idle = {
    "clock":
    1024**3,
    "memory":
    894,
    "connection": [{
        "recvGBps": 1,
        "recvIdleCycle": 0,
        "sendGBps": 1,
        "sendIdleCycle": 0
    } for i in range(2)]
}

cluster_4_ipu_no_transfer_idle = {
    "clock":
    1024**3,
    "memory":
    894,
    "connection": [{
        "recvGBps": 1,
        "recvIdleCycle": 0,
        "sendGBps": 1,
        "sendIdleCycle": 0
    } for i in range(4)]
}

profile_2_simple_even_split = {
    "layerName": [str(i) for i in range(6)],
    "layerType": [str(0) for i in range(6)],
    "layerInfo": [{
        "cycle": {
            "forward": 10000,
            "backward": 0,
        },
        "memory": {
            "shared": {},
            "exclusive": {},
            "inputs": 0,
            "outputs": 0,
            "maxTemporary": 0,
            "curCarryOver": 0,
            "curActivation": 0
        },
    } for i in range(6)]
}

profile_2_param_constraint = {
    "layerName": [str(i) for i in [0, 0, 1, 1, 1, 1]],
    "layerType": [str(i) for i in [0, 0, 1, 1, 1, 1]],
    "layerInfo": [{
        "cycle": {
            "forward": 10000,
            "backward": 0,
        },
        "memory": {
            "shared": {},
            "exclusive": {},
            "inputs": 0,
            "outputs": 0,
            "maxTemporary": 0,
            "curCarryOver": 0,
            "curActivation": 0
        },
    } for i in range(6)]
}

profile_2_memory_constraint = {
    "layerName": [str(i) for i in range(6)],
    "layerType": [str(i) for i in range(6)],
    "layerInfo": [{
        "cycle": {
            "forward": 10000,
            "backward": 0,
        },
        "memory": {
            "shared": {},
            "exclusive": {
                "vertexCode": 600 * (1024**2) if i < 2 else 0
            },
            "inputs": 0,
            "outputs": 0,
            "maxTemporary": 0,
            "curCarryOver": 0,
            "curActivation": 0
        }
    } for i in range(6)]
}

profile_2_transfer_constraint = {
    "layerName": [str(i) for i in range(6)],
    "layerType": [str(i) for i in range(6)],
    "layerInfo": [{
        "cycle": {
            "forward": 10000,
            "backward": 0
        },
        "memory": {
            "shared": {},
            "exclusive": {},
            "inputs": 0 if i == 2 else 99999999999,
            "outputs": 0 if i == 1 else 99999999999,
            "maxTemporary": 0,
            "curCarryOver": 0,
            "curActivation": 0
        }
    } for i in range(6)]
}

profile_4_simple_even_split = {
    "layerName": [str(i) for i in range(16)],
    "layerType": [str(0) for i in range(16)],
    "layerInfo": [{
        "cycle": {
            "forward": 10000,
            "backward": 0,
        },
        "memory": {
            "shared": {},
            "exclusive": {},
            "inputs": 0,
            "outputs": 0,
            "maxTemporary": 0,
            "curCarryOver": 0,
            "curActivation": 0
        }
    } for i in range(16)]
}

profile_4_param_constraint = {
    "layerName": [str(i) for i in ([0] * 2 + [1] * 7 + [2] * 3 + [3] * 4)],
    "layerType": [str(i) for i in ([0] * 2 + [1] * 7 + [2] * 3 + [3] * 4)],
    "layerInfo": [{
        "cycle": {
            "forward": 10000,
            "backward": 0,
        },
        "memory": {
            "shared": {},
            "exclusive": {},
            "inputs": 0,
            "outputs": 0,
            "maxTemporary": 0,
            "curCarryOver": 0,
            "curActivation": 0
        }
    } for i in range(16)]
}

profile_4_memory_constraint = {
    "layerName": [str(i) for i in range(16)],
    "layerType": [str(i) for i in range(16)],
    "layerInfo": [{
        "cycle": {
            "forward": 10000,
            "backward": 0,
        },
        "memory": {
            "shared": {},
            "exclusive": {
                "vertexCode": 600 * (1024**2) if i in [0, 1, 6, 7] else 0
            },
            "inputs": 0,
            "outputs": 0,
            "maxTemporary": 0,
            "curCarryOver": 0,
            "curActivation": 0
        }
    } for i in range(16)]
}

profile_4_transfer_constraint = {
    "layerName": [str(i) for i in range(16)],
    "layerType": [str(i) for i in range(16)],
    "layerInfo": [{
        "cycle": {
            "forward": 10000,
            "backward": 0,
        },
        "memory": {
            "shared": {},
            "exclusive": {},
            "inputs": 0 if i in [2, 5, 6, 8] else 99999999999,
            "outputs": 0 if i + 1 in [2, 5, 6, 8] else 99999999999,
            "maxTemporary": 0,
            "curCarryOver": 0,
            "curActivation": 0
        }
    } for i in range(16)]
}

TEST_CASES = [
    {
        # Simple even split of cycles.
        "testcase_name": "Simple Even Split 2",
        "model_info": profile_2_simple_even_split,
        "cluster_info": cluster_2_ipu_no_transfer_idle,
        "ans_partition": [[0, 3, 6]]
    },
    {
        # There is only one possible split,
        # because layers 01/2345 needs to be on the same stage.
        "testcase_name": "Shared Parameter Constraint 2",
        "model_info": profile_2_param_constraint,
        "cluster_info": cluster_2_ipu_no_transfer_idle,
        "ans_partition": [[0, 2, 6]]
    },
    {
        # There is only one possible split,
        # because layers 01 do not fit on one IPU.
        "testcase_name": "Memory Constraint 2",
        "model_info": profile_2_memory_constraint,
        "cluster_info": cluster_2_ipu_no_transfer_idle,
        "ans_partition": [[0, 1, 6]]
    },
    {
        # Only splitting between layer 1 and 2 gives a small transfer cost.
        "testcase_name": "Transfer Constraint 2",
        "model_info": profile_2_transfer_constraint,
        "cluster_info": cluster_2_ipu_no_transfer_idle,
        "ans_partition": [[0, 2, 6]]
    },
    {
        # Simple even split of cycles.
        "testcase_name": "Simple Even Split 4",
        "model_info": profile_4_simple_even_split,
        "cluster_info": cluster_4_ipu_no_transfer_idle,
        "ans_partition": [[0, 4, 8, 12, 16]]
    },
    {
        # There is only one possible split,
        # because layers 01/2345678/91011/12131415 need to be on the same stage.
        "testcase_name": "Shared Parameter Constraint 4",
        "model_info": profile_4_param_constraint,
        "cluster_info": cluster_4_ipu_no_transfer_idle,
        "ans_partition": [[0, 2, 9, 12, 16]]
    },
    {
        # Layers 01/16/67 does not fit on one IPU.
        # Split at layer 1 so that layers 01 is not on a same IPU.
        # Split at layer 7 so that layers 67 is not on a same IPU.
        # Split between 2 and 6 so that layers 1,6 is not on a same IPU.
        # There is no constraint for the last split, because the computation
        # cost is dominated by the last pipeline stage.
        "testcase_name":
        "Memory Constraint 4",
        "model_info":
        profile_4_memory_constraint,
        "cluster_info":
        cluster_4_ipu_no_transfer_idle,
        "ans_partition": [[0, 1, 2, 7, 16], [0, 1, 3, 7, 16], [0, 1, 4, 7, 16],
                          [0, 1, 5, 7, 16], [0, 1, 6, 7, 16]]
    },
    {
        # Only splitting at layers 2568 gives a small transfer cost.
        # Split at layers 258 to minimize computation cost.
        "testcase_name": "Transfer Constraint 4",
        "model_info": profile_4_transfer_constraint,
        "cluster_info": cluster_4_ipu_no_transfer_idle,
        "ans_partition": [[0, 2, 5, 8, 16]]
    }
]

config = {"memoryProportion": 0.85, "usePoplarEstimation": True}


class PartitionTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testAll(self, model_info, cluster_info, ans_partition):
    pipe_part = partition.get_auto_pipeline_partition_by_cycle_and_transfer(
        model_info, cluster_info, config)
    self.assertTrue(pipe_part in ans_partition)


if __name__ == "__main__":
  googletest.main()
