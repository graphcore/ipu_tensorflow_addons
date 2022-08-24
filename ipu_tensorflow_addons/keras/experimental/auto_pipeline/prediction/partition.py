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
"Automatically partition a model in inference mode"
import logging
from tensorflow.python.ipu import test_utils as tu
import numpy as np
import pva
from ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils import (
    profiler, pva_utils, types_utils, keras_utils)
try:
  # Attempt to import tqdm.
  from tqdm import tqdm
except ImportError:
  logging.warning(
      ("tqdm not installed."
       "No progress bar will be shown during per-layer compilation."))
  tqdm = lambda x: x


def parse_layer_profile(layer, pva_pop, config):
  """Read a single-layer PVA profile and return an analyzed layer profile.

  Args:
    layer: The Keras Layer to be analyzed.
    pva_pop: The path to the single-layer PVA profile.
    config: A `ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils
    .types.AutoPipelineConfiguration` dictionary. The configuration for
    automatic pipelining.

  Returns:
    A `ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils.types
    .LayerInfo` dictionary.

  """
  compile_only = config["usePoplarEstimation"]
  report = pva.openReport(pva_pop)
  if not pva_utils.check_report(report, layer.name):
    return types_utils.get_empty_layer_info()
  return {
      "cycle": {
          "forward": pva_utils.get_layer_cycle(report, layer.name,
                                               compile_only),
          "backward": 0
      },
      "memory": {
          "shared": {
              f"vertexCode_{id(layer)}":
              pva_utils.get_vertex_code(report, 0),
              f"internalExchangeCode_{id(layer)}":
              pva_utils.get_internal_exchange_code(report, 0, layer.name)
          },
          "exclusive": {
              "vertexInstanceState":
              pva_utils.get_vertex_instance_state(report, 0)
              # Remove random seed and supervisor vertex state,
              # which is usually 48 KB.
              - 48 * 1024,
              "controlCode":
              pva_utils.get_control_code(report, 0, layer.name),
              "parameter":
              pva_utils.get_parameter_from_always_live(report, 0, layer.name) +
              pva_utils.get_parameter_from_not_always_alive(report, 0)
          },
          "inputs": 0,
          "outputs": 0,
          "maxTemporary": pva_utils.get_max_live_memory(report, 0),
          "curCarryOver": 0,
          "curActivation": 0
      }
  }


def create_model_profile(create_model, batch_size, config):
  """Profile each layer in the model and return a profile of the model.

  Args:
    create_model: A function to create the model.
    batch_size: Batch size to be used with the model.

  Return:
    A `ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils.types
    .RangeInfo` dictionary.
  """
  report_helper = tu.ReportHelper()
  compile_only = config["usePoplarEstimation"]
  strategy = profiler.create_strategy(1, report_helper, compile_only)
  with strategy.scope():
    model = create_model()
    assignments = model.get_pipeline_stage_assignment()

    (layer_carry_over, layer_input,
     layer_output) = keras_utils.parse_model_intermediate(model, batch_size)

    # Find type of all layers.
    layer_types = [
        keras_utils.get_assignment_layer_type(assignment, batch_size)
        for assignment in assignments
    ]

    # Layer Type -> Layer Profile
    # This dictionary works as a cache for layer profiles.
    layer_info_cache = {}

    # The profile of each layer in the model.
    layer_info = []

    for i, assignment in enumerate(tqdm(assignments)):
      if layer_types[i] in layer_info_cache:
        # A layer of the same type has been compiled.
        # Use the cached
        original_result = layer_info_cache[layer_types[i]]
      else:
        # The first time compiling the layer.
        # Profile the layer and save the LayerInfo in cache.
        profile_pop = profiler.profile_layer_from_assignment(
            assignment, batch_size, strategy, report_helper)
        original_result = parse_layer_profile(assignment.layer, profile_pop,
                                              config)
        layer_info_cache[layer_types[i]] = original_result

      # Make a copy of the original profile and update some location-specific
      # properties.
      copy_result = types_utils.copy_layer_info(original_result)
      copy_result["memory"]["curCarryOver"] = int(layer_carry_over[i])
      copy_result["memory"]["inputs"] = int(layer_input[i])
      copy_result["memory"]["outputs"] = int(layer_output[i])
      layer_info.append(copy_result)
  return {
      "layerName": [assignment.layer.name for assignment in assignments],
      "layerType": layer_types,
      "layerInfo": layer_info
  }


INF = int(1e16)


def get_model_cost_matrix(model_profile, mem_limit):
  """Get the cost of running each range as a pipeline stage in the model.

  `cost[i,j]` equals `INF` if running `[i,j]` layers on a IPU requires more
  memory than `mem_limit`, or if some variables in these layers are used outside
  this IPU.

  `cost[i,j]` equals to the sum of estimated cycle count from `[i,j]` layers.

  Args:
    model_profile: An `ipu_tensorflow_addons.keras.experimental.auto_pipeline
    .utils.types.ModelInfo` dictionary. The profile of the model to be
    pipelined.
    mem_limit: The memory capacity of an IPU in MB.

  Return:
    A `np.ndarray` with `np.long` dtype and `(n_layer, n_layer)` shape.
  """
  n_layer = len(model_profile["layerInfo"])
  # For each layer in the model, total number of invocations in the model.
  model_layer_total = {}
  for layername in model_profile["layerName"]:
    model_layer_total.setdefault(layername, 0)
    model_layer_total[layername] += 1

  # The cost array to be returned.
  range_cost = np.full((n_layer, n_layer), INF, dtype=np.long)

  # Set of layers in the range.
  range_layers = set()

  for start in range(0, n_layer):
    range_layers.clear()
    # For all layers in the pipeline stage, number of layer invocations outside
    # the pipeline stage.
    cnt = 0

    # Estimated profile of running `[start,end]` layers as a pipeline stage.
    range_info = types_utils.get_empty_range_info()

    for end in range(start, n_layer):
      # Update `range_layers` and `cnt` with the `end`-th layer.
      layername = model_profile["layerName"][end]
      if layername not in range_layers:
        range_layers.add(layername)
        cnt += model_layer_total[layername]
      cnt -= 1

      # Update `range_info` with the `end`-th layer.
      now_info = model_profile["layerInfo"][end]
      range_info = types_utils.aggregate_info_inplace(range_info, now_info)

      if range_info["memory"]["totalMemory"] > mem_limit:
        # If running `[start, end]` triggers an OOM error, running
        # `[start, end+...]` layers will also trigger an OOM error.
        # So break early and use the default INF value.
        break

      if cnt == 0:
        # Current range is valid for a pipeline stage.
        # Set the cost to be the sum of cycle counts from layers.
        range_cost[start, end] = range_info["cycle"]["forward"]

  return range_cost


def get_transfer_cost_matrix(model_profile, cluster_info):
  """Get the estimated data transfer cycle count of running each range as a
  pipeline stage in the model.

  Args:
    model_profile: An `ipu_tensorflow_addons.keras.experimental.auto_pipeline
    .utils.types.ModelInfo` dictionary. The profile of the model to be
    pipelined.
    cluster_info: An `ipu_tensorflow_addons.keras.experimental.auto_pipeline
    .utils.types.IPUClusterInfo` dictionary. The description of the IPU Cluster.

  Return:
    A `np.ndarray` with `np.long` dtype and `(n_ipu, n_layer, n_layer)` shape.
  """
  n_ipu = len(cluster_info["connection"])
  n_layer = len(model_profile["layerInfo"])
  transfer_cost = np.full((n_ipu, n_layer, n_layer), INF, dtype=np.long)
  for ipu_id in range(n_ipu):
    for range_start in range(n_layer):
      for range_end in range(range_start, n_layer):
        cycle = types_utils.transfer_cycle(cluster_info, ipu_id, model_profile,
                                           range_start, range_end)
        transfer_cost[ipu_id, range_start, range_end] = cycle
  return transfer_cost


def get_auto_pipeline_partition_by_cycle_and_transfer(model_profile,
                                                      cluster_info, config):
  """Find a pipeline stage partition that minimizes overall computation and data
  transferring cycle counts.

  Based on https://arxiv.org/abs/1806.03377

  Args:
    model_profile: An `ipu_tensorflow_addons.keras.experimental.auto_pipeline
    .utils.types.ModelInfo` dictionary. The profile of the model to be
    pipelined.
    cluster_info: An `ipu_tensorflow_addons.keras.experimental.auto_pipeline
    .utils.types.IPUClusterInfo` dictionary. The description of the IPU Cluster.
    config: A `ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils
    .types.AutoPipelineConfiguration` dictionary. The configuration for
    automatic pipelining.

  Returns:
    A list of `n_ipu+1` integers. The `i`-th element in the list indicates the
    first layer of the `i`-th pipeline stage.
  """
  mem_limit = int(cluster_info["memory"] * config["memoryProportion"] *
                  1024**2)

  # See README for a more detailed explanation
  n_ipu = len(cluster_info["connection"])
  n_layer = len(model_profile["layerInfo"])
  range_cost = get_model_cost_matrix(model_profile, mem_limit)
  transfer_cost = get_transfer_cost_matrix(model_profile, cluster_info)

  # All unique value from transfer_cost.
  transfer_cost_values = np.unique(transfer_cost)
  n_transfer = transfer_cost_values.shape[0]

  # dynamic programming array
  partition_cost = np.full((n_ipu, n_layer, n_transfer), INF, dtype=np.long)

  # From which state the state is updated from.
  partition_from = np.full((n_ipu, n_layer, n_transfer), -1, dtype=np.long)

  # Partition cost on the first IPU.
  partition_cost[0] = range_cost[0, :, np.newaxis]
  partition_from[0] = -1

  # Set `INF` if the first pipeline stage uses more transfer cycle than the third
  # dimension of the DP state.
  mask_above_transfer_cost = np.greater(transfer_cost[0, 0, :, np.newaxis],
                                        transfer_cost_values[np.newaxis, :])
  partition_cost[0][mask_above_transfer_cost] = INF

  for now_ipu in range(1, n_ipu):
    for now_end in range(1, n_layer):
      # We are now finding `partition_cost[now_ipu, now_end, :]`

      # Candidate value `partition_cost[now_ipu, now_end, :]`
      # Shape: (now_end, n_transfer)
      dp_temp = np.maximum(partition_cost[now_ipu - 1, 0:now_end, :],
                           range_cost[1:now_end + 1, now_end, np.newaxis])

      # Set `INF` if the latest pipeline stage uses more transfer cycle than the
      # third dimension of the DP state.
      mask_above_transfer_cost = np.greater(
          transfer_cost[now_ipu, 1:now_end + 1, now_end, np.newaxis],
          transfer_cost_values[np.newaxis, :])
      dp_temp[mask_above_transfer_cost] = INF

      # Choose the optimal option from the candidate values.
      best_from = np.argmin(dp_temp, axis=0)
      best_cost = dp_temp[best_from, np.arange(n_transfer)]
      partition_cost[now_ipu, now_end] = best_cost
      partition_from[now_ipu, now_end] = best_from

  # Now consider all max_transfer_cycles, find one minimizes overall cost
  final_cost_by_transfer_cycle = np.copy(transfer_cost_values)
  final_cost_by_transfer_cycle += partition_cost[n_ipu - 1, n_layer - 1, :]
  best_transfer_cycle_id = np.argmin(final_cost_by_transfer_cycle)

  # Recover the optimal partition.
  ans_partition = [0 for i in range(n_ipu + 1)]
  cur_layer = n_layer - 1
  for now_ipu in range(n_ipu, 0, -1):
    ans_partition[now_ipu] = cur_layer + 1
    cur_layer = int(partition_from[now_ipu - 1, cur_layer,
                                   best_transfer_cycle_id])
  return ans_partition
