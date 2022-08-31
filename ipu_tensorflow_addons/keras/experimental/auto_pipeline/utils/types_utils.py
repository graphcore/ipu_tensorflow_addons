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
"""Helper function for types defined for auto pipelining"""
import copy


def get_empty_layer_info():
  """Create an empty analyzed layer profile with all necessary keys.

  Returns:
    An `ipu_tensorflow_addons.keras.experimental.auto_pipeline.util.types
    .LayerInfo` dictionary.
  """
  return {
      "cycle": {
          "forward": 0,
          "backward": 0,
          "total": 0,
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
  }


def get_empty_range_info():
  """Create an empty range profile with all necessary keys.

  Returns:
    An `ipu_tensorflow_addons.keras.experimental.auto_pipeline.util.types
    .RangeInfo` dictionary.
  """
  return {
      "cycle": {
          "forward": 0,
          "backward": 0,
          "total": 0,
      },
      "memory": {
          # The "inputs" and "outputs" properties are intentionally excluded,
          # to make it simpler in `aggregate_info_inplace`.
          "shared": {},
          "exclusive": {},
          "maxTemporary": 0,
          "curActivation": 0,
          "totalMemory": 0
      }
  }


def copy_layer_info(info):
  """Create a copy of a `LayerInfo` dictionary. The cloned dictionary can be
  modified without changing the original copy.

  Args:
    info: An `ipu_tensorflow_addons.keras.experimental.auto_pipeline.util.types
    .LayerInfo` dictionary.

  Returns:
    A cloned `ipu_tensorflow_addons.keras.experimental.auto_pipeline.util.types
    .LayerInfo` dictionary.
  """
  return copy.deepcopy(info)


def transfer_cycle(cluster_info, ipu_id, model_profile, range_start,
                   range_end):
  """Estimated cycle count for the `ipu_id`-th IPU to transfer data, if
  `[range_start,range_end]` layers are on this IPU. This includes both inter-ipu
  and ipu-host exchange cycles.

  Args:
    cluster_info: Description of the given IPU.
    ipu_id: The index of the IPU in the cluster.
    model_profile: An analyzed model profile.
    range_start: Starting layer of the pipeline stage.
    range_end: Last layer of the pipeline stage.
  """

  def recv_cycle(ipu_id, layer_start):
    """Cycle count for the `ipu_id`-th IPU to receive data from the previous
    device in the pipeline, if `layer_start` is the first layer in the pipeline
    stage.
    """
    recv_byte_per_clock = (
        (cluster_info["connection"][ipu_id]["recvGBps"] * 1024**3) /
        cluster_info["clock"])
    return int(cluster_info["connection"][ipu_id]["recvIdleCycle"] +
               model_profile["layerInfo"][layer_start]["memory"]["inputs"] /
               recv_byte_per_clock)

  def send_cycle(ipu_id, layer_end):
    """Cycle count for the `ipu_id`-th IPU to send data to the next device
    in the pipeline, if `layer_end` is the last layer in the pipeline stage.
    """
    recv_byte_per_clock = (
        (cluster_info["connection"][ipu_id]["sendGBps"] * 1024**3) /
        cluster_info["clock"])
    return int(cluster_info["connection"][ipu_id]["sendIdleCycle"] +
               model_profile["layerInfo"][layer_end]["memory"]["outputs"] /
               recv_byte_per_clock)

  n_layer = len(model_profile["layerInfo"])
  n_ipu = len(cluster_info["connection"])
  cycle = 0
  # Currently cpu-ipu StreamCopy does not seem to overlap with ipu-ipu Exchange.
  # After the computation of a pipeline stage, all IPUs exchange their pipeline
  # output. Then the last IPU sends the model output to CPU. Finally, the first
  # CPU receive the model input for next round from CPU.
  #
  # 0 [Global Exchange] [StreamCopy---------------] [StreamCopy-]
  #   [               ] [Begin            |Mid|End] [Begin|M|End]
  # 1 [Global Exchange-----------] [StreamCopy----] [StreamCopy-]
  #   [                          ] [Begin|M|End---] [Begin|M|End]
  #
  # However note the last two stages uses constant cycle with a same
  # `cluster_info`. So the last two stage will not affect pipeline partition.
  if ipu_id != 0:
    cycle += recv_cycle(ipu_id, range_start)
  if ipu_id != n_ipu - 1:
    cycle += send_cycle(ipu_id, range_end)
  cycle += recv_cycle(0, 0)
  cycle += send_cycle(n_ipu - 1, n_layer - 1)
  return cycle


def aggregate_info_inplace(range_info, layer_info):
  """Update the range profile `range_info` to include a new layer.

  Arg:
    range_info: An `ipu_tensorflow_addons.keras.experimental.auto_pipeline.util
    .types.RangeInfo` dictionary to be updated.
    layer_info: An `ipu_tensorflow_addons.keras.experimental.auto_pipeline.util
    .types.LayerInfo` dictionary to be included in the range.

  Return:
    The updated `range_info`.
  """
  range_cycle = range_info["cycle"]
  layer_cycle = layer_info["cycle"]
  range_memory = range_info["memory"]
  layer_memory = layer_info["memory"]

  # Update CycleInfo.
  range_cycle["forward"] += layer_cycle["forward"]
  range_cycle["backward"] += layer_cycle["backward"]

  # Update Input and Output Size.
  range_memory.setdefault("inputs", layer_memory["inputs"])
  range_memory["outputs"] = layer_memory["outputs"]

  # Update maxTemporary.
  # range_memory["curActivation"] is updated after range_memory["maxTemporary"],
  # because layer_memory["maxTemporary"] includes layer_memory["curActivation"].
  new_max_temporary = (layer_memory["maxTemporary"] +
                       layer_memory["curCarryOver"] +
                       range_memory["curActivation"])
  if new_max_temporary > range_memory["maxTemporary"]:
    range_memory["totalMemory"] -= range_memory["maxTemporary"]
    range_memory["totalMemory"] += new_max_temporary
    range_memory["maxTemporary"] = new_max_temporary

  # Update curActivation.
  range_memory["curActivation"] += layer_memory["curActivation"]

  # Shared variables.
  for name, size in layer_memory["shared"].items():
    if name not in range_memory["shared"]:
      #Include the variable if it is not in the shared items
      range_memory["shared"][name] = size
      range_memory["totalMemory"] += size

  # Exclusive variables.
  for name, size in layer_memory["exclusive"].items():
    range_memory["exclusive"].setdefault(name, 0)
    range_memory["exclusive"][name] += size
    range_memory["totalMemory"] += size
  return range_info


def compare_pipelined_model_profile(estimated_profile, actual_profile):
  """Compare an estimated pipeline profile with an actual pipeline profile, and
  pretty-print the difference.

  Args:
    estimated_profile: A list of `ipu_tensorflow_addons.keras.experimental
    .auto_pipeline.util.types.RangeInfo`, which is the estimated profile of a
    model running under a pipeline stage assignment.
    actual_profile: A list of `ipu_tensorflow_addons.keras.experimental
    .auto_pipeline.util.types.RangeInfo`, which is the actual profile of a model
    running under a pipeline stage assignment.
  """

  def compare_value(act, est, div, sigfig=2):
    """Compare estimated value with actual value.

    Args:
      act: Value from the actual profile.
      est: Value from the estimated profile.
      div: The denominator of act and div.
      sigfig: Significant figure to keep.
    """
    return {
        "Act": round(act / div, sigfig),
        "Est": round(est / div, sigfig),
        "diffAbs": round((est - act) / div, sigfig),
        "diffPercent": round((est - act) / act * 100, 2),
    }

  def compare_pipeline_stage(estimated_stage_profile, act_stage_profile):
    """Compare """
    category = {
        "cycle": {
            "total": {
                "Act": act_stage_profile["cycle"]["total"],
                "Est": estimated_stage_profile["cycle"]["total"]
            },
            "forward": {
                "Act": act_stage_profile["cycle"]["forward"],
                "Est": estimated_stage_profile["cycle"]["forward"]
            },
            "backward": {
                "Act": act_stage_profile["cycle"]["backward"],
                "Est": estimated_stage_profile["cycle"]["backward"]
            }
        },
        "memory": {
            "total": {
                "Act": act_stage_profile["memory"]["totalMemory"],
                "Est": estimated_stage_profile["memory"]["totalMemory"],
            },
            "notAlwaysAlive": {
                "Act": act_stage_profile["memory"]["maxTemporary"],
                "Est": estimated_stage_profile["memory"]["maxTemporary"],
            }
        }
    }

    for [kind, info] in [["Est", estimated_stage_profile],
                         ["Act", act_stage_profile]]:
      for var_name, var_size in (list(info["memory"]["shared"].items()) +
                                 list(info["memory"]["exclusive"].items())):
        var_category = var_name.split("_")[0]
        category["memory"].setdefault(var_category, {"Act": 0, "Est": 0})
        category["memory"][var_category][kind] += var_size

    return {
        "cycle": {
            k: compare_value(v["Act"], v["Est"], 1000, 0)
            for k, v in category["cycle"].items() if v["Act"] != 0
        },
        "memory": {
            k: compare_value(v["Act"], v["Est"], 1024**2, 2)
            for k, v in category["memory"].items() if v["Act"] != 0
        },
    }

  compare_result = [
      compare_pipeline_stage(est_stage, act_stage)
      for est_stage, act_stage in zip(estimated_profile, actual_profile)
  ]

  for i, ipu_diff_info in enumerate(compare_result):
    print(f"IPU {i}")
    for category, category_diff_info in ipu_diff_info.items():
      print(category)
      for name, diff in category_diff_info.items():
        print(f"{name:>20} |",
              f"Act: {diff['Act']:>10}",
              f"Est: {diff['Est']:>10}",
              f"Diff: {diff['diffAbs']:>10}",
              f"Percent: {diff['diffPercent']:>7}%",
              sep="  ")
