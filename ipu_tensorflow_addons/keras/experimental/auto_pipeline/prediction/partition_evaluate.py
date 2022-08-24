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
"""Helper function to test a generated partition"""
import functools
import shutil
from tensorflow.python.ipu import test_utils as tu
import pva
from ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils import (
    profiler, pva_utils, types_utils)


def estimate_pipelined_model_profile(model_profile, partition):
  """Estimate the profile of the pipelined model from profiles of layers in the
  model.

  Args:
    model_profile: An `ipu_tensorflow_addons.keras.experimental.auto_pipeline
    .utils.types.ModelInfo` dictionary, which contains a profile for every layer
    in the model.
    cluster_info: An `ipu_tensorflow_addons.keras.experimental.auto_pipeline
    .utils.types.IPUClusterInfo` dictionary. The description of the IPU Cluster.
    partition: A list of `n_ipu+1` integers. The `i`-th element in the list
    indicates the first layer of the `i`-th pipeline stage.

  Returns:
    A list of `ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils
    .types.RangeInfo` dictionary. The estimated running cost of each pipeline
    stage.
  """
  range_infos = [
      functools.reduce(
          types_utils.aggregate_info_inplace,
          model_profile["layerInfo"][partition[i]:partition[i + 1]],
          types_utils.get_empty_range_info())
      for i in range(len(partition) - 1)
  ]
  return range_infos


def parse_pipelined_model_profile(profile_pop):
  """Read a PVA profile from a pipelined model and return an analyzed profile
  for each pipeline stage.

  Args:
    pva_pop: The path to the PVA profile of the pipelined model.

  Return:
    A list of `ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils
    .types.RangeInfo` dictionary. The actual running cost of each pipeline
    stage.
  """
  report = pva.openReport(profile_pop)
  n_ipu = len(report.compilation.ipus)
  pipeline_cycle = pva_utils.get_pipeline_stage_cycles(report)
  pipeline_profiles = [
      {
          "cycle": {
              "forward": int(pipeline_cycle[i]),
              "backward": 0
          },
          "memory": {
              "shared": {
                  "vertexCode":
                  pva_utils.get_vertex_code(report, i),
                  "internalExchangeCode":
                  pva_utils.get_internal_exchange_code(report, i)
              },
              "exclusive": {
                  "vertexInstanceState":
                  pva_utils.get_vertex_instance_state(report, i),
                  "controlCode":
                  pva_utils.get_control_code(report, i),
                  "parameter":
                  pva_utils.get_parameter_from_always_live(report, i) +
                  pva_utils.get_parameter_from_not_always_alive(report, i)
              },
              # Ignore inputs and outputs because they are not compared against
              # the estimated profile.
              "inputs": 0,
              "outputs": 0,
              "curActivation": 0,
              "maxTemporary": pva_utils.get_max_live_memory(report, i),
              "totalMemory": pva_utils.get_total_memory(report, i)
          }
      } for i in range(n_ipu)
  ]
  return pipeline_profiles


def profile_pipelined_model(create_model,
                            create_data,
                            partition,
                            save_profile_path=None):
  """Run and profile a model with minimal data to evaluate pipeline performance.

  Args:
    create_model: A function to create the model.
    create_data: A function which takes an integer argument `x` and returns
    a dataset with `x` batches.
    partition: A list of `n_ipu+1` integers. The `i`-th element in the list
    indicates the first layer of the `i`-th pipeline stage.
    save_profile_path: The path to save a copy of the profile.
      If `None`, the profile will be deleted after analyzing the profile.

  Return:
    A list of `ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils
    .types.RangeInfo` dictionary. The actual running cost of each pipeline
    stage.
  """
  report_helper = tu.ReportHelper()
  n_ipu = len(partition) - 1
  strategy = profiler.create_strategy(n_ipu, report_helper, compile_only=False)
  with strategy.scope():
    model = create_model()
    assignments = model.get_pipeline_stage_assignment()
    for i in range(n_ipu):
      for j in range(partition[i], partition[i + 1]):
        assignments[j].pipeline_stage = i

    model.set_pipeline_stage_assignment(assignments)

    # Run the model with `2*n_ipu` batches.
    # At `n_ipu`-th stage, the model is fully ramped up.
    data = create_data(n_ipu * 2)
    model.compile(stages_per_execution=n_ipu * 2)
    model.predict(data)

  pva_pop = report_helper.find_reports()[-1]

  if save_profile_path:
    shutil.copy(pva_pop, save_profile_path)

  return parse_pipelined_model_profile(pva_pop)
