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
"""Automatically find the combination of IPU Configuration, number of IPUs and
batch size for the optimal performance."""

import copy
import pprint
from ipu_tensorflow_addons.keras.experimental.auto_pipeline.prediction import (
    partition, partition_evaluate)

INF = int(1e16)


def get_pipeline_cost(pipeline_profile, batch_size):
  """Get the cycle cost of predicting a batch from an analyzed profile.

  Args:
    pipeline_profile: A list of `ipu_tensorflow_addons.keras.experimental
    .auto_pipeline.utils.types.RangeInfo`, which contains an analyzed profile
    of a pipelined model.
    batch_size: Batch size to be used with the model.

  Return:
    An integer. The average cycle cost to predict a batch.
  """

  n_ipu = len(pipeline_profile)
  max_compute_cycle = max(profile["cycle"]["total"]
                          for profile in pipeline_profile)
  return int(max_compute_cycle * n_ipu / batch_size)


# Disable Lint for bare except
# Because errors from strategy.run / model.predict do not have a specific class.
# pylint: disable=bare-except
def search_for_max_throughput(binary_search_configs,
                              grid_search_configs,
                              create_args,
                              create_model,
                              create_dataset=None):
  """Perform an efficient search to find an optimal set of configurations to
  achieve a maximum throughput for a model.

  Args:
    binary_search_configs: A dictionary from each configuration name to a list
    of candidate values for this configuration. Candidate values toward the end
    of the list should achieve a higher throughput but more likely to fail
    compilation. Searching in this dictionary is done using binary search.
    grid_search_configs: A dictionary from each configuration name to a list
    of candidate values for this configuraiton. There is no constraint on the
    order of the candidate value list.
    create_args: A function that accepts configurations defined in
    `binary_search_configs` and `grid_search_configs` and returns a dictionary
    of arguments to the autopipe.
    create_model: A function to create the model.
    create_dataset: A function to create the dataset or `None`.
      If `None`, `search_for_max_throughput` will estimate the throughput from
      cycle estimates, which is faster but less accurate.
      If function, `search_for_max_throughput` will estimate the throughput from
      an actual model execution, which is slower but more accurate.
      `create_dataset` should be a function that accepts two numbers: number of
      batches in the dataset and batchsize. The function should return a
      `tf.data.Dataset`.

  Example:
    In this example, we use binary search to find the maximum batch size, and
    use grid search to find an appropriate number of IPUs to run the model.

    .. code-block:: python
      # Configuration for AutoPipe.
      config = {"memoryProportion": 0.85, "usePoplarEstimation": False}

      # Dict of configurations for binary search.
      binary_search_options = {
        "batch_size": list(range(1,128))
      }

      # Dict of configurations for grid search.
      grid_search_options = {
        "n_ipu": [1,2,4],
        "mem_portion": [0.3,0.6,1]
      }

      # A function to generate configurations for AutoPipe and Profiler from
      # the two dictionaries of configurations.
      def create_args(batch_size, n_ipu, mem_portion):
        # Set `availableMemoryProportion` in ipu_config.
        mem_str = str(mem_portion)
        ipu_config = ipu.config.IPUConfig()
        ipu_config.matmuls.poplar_options["availableMemoryProportion"] = mem_str

        return {
          # Use the default IPUConfig, with no extra configurations.
          "ipuConfig": ipu_config,

          # Batch size to use this time.
          "batchSize": batch_size,

          # AutoPipe configuration.
          "autoPipeConfig": config,

          # Description of the IPU system. Here `cluster_infos[n_ipu]` is the
          # `ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils.types
          # .IPUClusterInfo` dictionary for a `n_ipu` system.
          "clusterInfo": cluster_infos[n_ipu]
      }

      # Run the search.
      max_throughput.search_for_max_throughput(binary_search_options,
                                         grid_search_options, create_args,
                                         create_model, create_dataset)
  """
  binary_search_config_list = list(binary_search_configs.items())
  grid_search_config_list = list(grid_search_configs.items())

  cur_config = {}
  best_model_cost = INF
  best_config_dict = {}
  best_partition = []

  def do_grid_search(cur_config_id):
    """Use grid search to find an optimal value for the `cur_config_id`-th
    configuration in `grid_search_configs`.

    Args:
      cur_config_id: The index of the configuration to search.

    Returns:
      The best per-batch throughput achieved, with fixed `[0,cur_config_id)`
      configuration in `grid_search_config`.
    """
    if cur_config_id >= len(grid_search_config_list):
      # All grid_search_configs are chosen
      return do_binary_search(0)

    [config_name, config_values] = grid_search_config_list[cur_config_id]

    min_pipe_cost = INF
    for v in config_values:
      cur_config[config_name] = v
      now_through_put = do_grid_search(cur_config_id + 1)
      min_pipe_cost = min(min_pipe_cost, now_through_put)

    return min_pipe_cost

  def do_binary_search(cur_config_id):
    """Use binary search to find an optimal value for the `cur_config_id`-th
    configuration in `binary_search_configs`.

    Args:
      cur_config_id: The index of the configuration to search.

    Returns:
      The best per-batch throughput achieved, with fixed `grid_search_configs`
      and `[0,cur_config_id)` configuration in `binary_search_configs`.
    """
    if cur_config_id >= len(binary_search_config_list):
      return do_profile()

    [config_name, config_values] = binary_search_config_list[cur_config_id]

    min_pipe_cost = INF
    l, r = 0, len(config_values)

    while l + 1 != r:
      mid = (l + r) // 2
      cur_config[config_name] = config_values[mid]
      now_through_put = do_binary_search(cur_config_id + 1)
      if now_through_put < INF:
        l = mid
        min_pipe_cost = now_through_put
      else:
        r = mid

    return min_pipe_cost

  def do_profile():
    """Perform a profile with the given configuration.

    Returns:
      Throughput for this model with the configuration.
    """
    nonlocal cur_config

    # Generate arguments from the config dictionary.
    args_dict = create_args(**cur_config)
    if args_dict is None:
      return INF

    ipu_cfg = args_dict["ipuConfig"]
    batch_size = args_dict["batchSize"]
    autopipe_config = args_dict["autoPipeConfig"]
    cluster_info = args_dict["clusterInfo"]

    print("=====Profiling the model=====")
    pprint.pprint(cur_config)

    if len(cluster_info["connection"]) == 1:
      # Use a dummy pipeline partition.
      pipe = [0, 0]
    else:
      # Profile each layer in the model.
      try:
        layer_profile = partition.create_model_profile(create_model,
                                                       batch_size,
                                                       autopipe_config,
                                                       ipu_cfg)
      except:
        print("Layer compilation failed.")
        return INF

      # Use AutoPipe to find a pipeline stage partition.
      try:
        pipe = partition.get_auto_pipeline_partition_by_cycle_and_transfer(
            layer_profile, cluster_info, autopipe_config)
      except:
        print("Pipeline Partition Failed.")
        return INF

      print("Pipeline stage assignment found.")
      print(pipe)

    # Find cycle per batch
    try:
      pipe_profile = partition_evaluate.profile_pipelined_model(
          create_model, create_dataset, batch_size, pipe, ipu_cfg)
    except:
      print("Pipeline compilation failed.")
      return INF

    # Compare current configuraiton with best configuration found
    cycle_per_batch = get_pipeline_cost(pipe_profile, batch_size)
    nonlocal best_model_cost, best_config_dict, best_partition
    if cycle_per_batch < best_model_cost:
      best_model_cost = cycle_per_batch
      best_config_dict = copy.deepcopy(cur_config)
      best_partition = pipe

    # Print info
    print("Profiling Done!")
    pprint.pprint(cur_config)
    print("Cost=", cycle_per_batch)

    return cycle_per_batch

  # Start search
  do_grid_search(0)

  # Print final result
  print()
  print("Searching complete")
  print("Best Config:")
  pprint.pprint(best_config_dict)
  print("Best Partition:")
  pprint.pprint(best_partition)
  print("Best Cost")
  print(best_model_cost)
  return best_config_dict, best_partition
