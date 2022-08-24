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
"""Helper function to get layer level / pipeline step level information from
a PVA report"""
import pva

PType = pva.Program.Type


def check_report(report, name):
  """Check if a report contains a layer with the provided `name` by iterating
  all the program names of each step in the profile.

  Arg:
    report: A PVA report.
    name: The name of the layer.

  Return:
    Boolean
  """
  return any(name in step.program.name
             for step in report.compilation.livenessProgramSteps)


def get_vertex_code(report, ipu_id=0):
  """Returns the memory consumption of vertex code on the `ipu_id`-th IPU.

  Arg:
    report: A PVA report.
    ipu_id: An integer indicating which IPU to analyze from the report.

  Return:
    An integer. Memory consumption for vertex code on the specified IPU.

  """
  return sum(tile.memory.category.vertexCode.total
             for tile in report.compilation.ipus[ipu_id].tiles)


def get_internal_exchange_code(report, ipu_id=0, name=None):
  """Returns the memory consumption of internal exchange code on the `ipu_id`-th
  IPU.

  If `name` is provided, the report should be a single-layer profile and
  non-layer related internal exchange code is excluded.

  If `name` is not provided, the function returns memory consumption of all
  internal exchange program on the selected IPU.

  Arg:
    report: A PVA report.
    ipu_id: An integer indicating which IPU to analyze from the report.
    name: The name of the layer if the report is a profile of a single layer.

  Return:
    An integer. Memory consumption for internal exchange code on a given IPU.
  """

  if ipu_id == 0 and name is not None:
    return sum(
        sum(program.codeBytesByTile) for program in report.compilation.programs
        if program.type == PType.DoExchange and name in program.name)
  return sum(tile.memory.category.internalExchangeCode.total
             for tile in report.compilation.ipus[ipu_id].tiles)


def get_vertex_instance_state(report, ipu_id=0):
  """Returns the memory consumption of vertex instance state on the `ipu_id`-th
  IPU.

  Arg:
    report: A PVA report.
    ipu_id: An integer indicating which IPU to analyze from the report.

  Return:
    An integer. Memory consumption for vertex instance state on given IPU.
  """

  return sum(tile.memory.category.vertexInstanceState.total
             for tile in report.compilation.ipus[ipu_id].tiles)


def get_control_code(report, ipu_id=0, name=None):
  """Returns the memory consumption of internal exchange code on the `ipu_id`-th
  IPU.

  If `name` is provided, the report should be a single-layer profile and only
  control program with the given layer name will be included.

  If `name` is not provided, the function returns memory consumption of all
  internal exchange program on the selected IPU.

  Arg:
    report: A PVA report.
    ipu_id: An integer indicating which IPU to analyze from the report.
    name: The name of the layer if the report is a profile of a single layer.

  Return:
    An integer. Memory consumption for control code on given IPU and with given
  """

  if name is None:
    return sum(tile.memory.category.controlCode.total
               for tile in report.compilation.ipus[ipu_id].tiles)
  return sum(
      sum(program.controlCodeByTile) for program in report.compilation.programs
      if name in program.name or program.type == PType.Sans
      or program.type == PType.SyncAns)


def get_max_live_memory(report, ipu_id=0):
  """Returns the memory consumption of max not always alive memory on the
  `ipu_id`-th IPU.

  Arg:
    report: A PVA report.
    ipu_id: An integer indicating which IPU to analyze from the report.

  Return:
    A tuple of two integers
      - The memory consumption of parameters found in not-always-alive memory.
      - The maximum size of temporary memory used to complete the operation.
  """
  max_mem_step = max(report.compilation.livenessProgramSteps,
                     key=lambda x: x.notAlwaysLiveMemoryForIpu(ipu_id).bytes)
  max_mem_step_bytes = max_mem_step.notAlwaysLiveMemoryForIpu(ipu_id).bytes
  param_bytes = get_parameter_from_not_always_alive(report, ipu_id)
  return max_mem_step_bytes - param_bytes


def get_parameter_from_not_always_alive(report, ipu_id=0):
  """Return the memory consumption of parameters in not-always-alive category,
  on the `ipu_id`-th IPU.

  Workaround for PVA containing parameters in not always alive memory.

  Currently in some profiles with large input tensors or large parameter
  tensors, PVA fails to recognize parameters as always alive variables, because
  these parameter variables are missing in a few StreamCopy steps.

  In these liveness report, the not always alive memory of first few program
  steps looks like the graph below.
  |
  |__        ________
  |        __
  |      __
  |  ____
  |_____________________

  To workaround this, we look at the first program step, where the computation
  has not started. We then take all variables in this program step as always
  alive variables.

  Arg:
    report: A PVA report.
    ipu_id: An integer indicating which IPU to analyze from the report.

  Return:
    An integer. Memory consumption for parameters in not-always-alive category
    on a given IPU.
  """
  step = report.compilation.livenessProgramSteps[0]
  return step.notAlwaysLiveMemoryForIpu(ipu_id).bytes


def get_parameter_from_always_live(report, ipu_id=0, name=None):
  """Return the memory consumption of parameters in always alive category,
  on the `ipu_id`-th IPU.

  If `name` is provided, the report should be a single-layer profile and only
  parameters related to the given layer name will be included.

  If `name` is not provided, the function returns memory consumption of all
  parameters on the selected IPU.

  Arg:
    report: A PVA report.
    ipu_id: An integer indicating which IPU to analyze from the report.
    name: The name of the layer if the report is a profile of a single layer.

  Return:
    An integer. Memory consumption for parameters in always alive category on a
    given IPU.
  """
  if ipu_id == 0 and name is not None:
    # "/" in v.name is used not exclude vertexCode, vertexInstanceState in
    # always alive memory
    return sum(v.size for v in report.compilation.alwaysLiveVariables
               if "/" in v.name and v.size > 256)

  # Workaround for PVA is too slow when traversing always alive variables.
  # When analyzing a profile for a whole model, we find the size of parameter
  # memory by subtracting other types of memory from always alive memory.
  return sum(tile.memory.alwaysLiveBytes -
             tile.memory.category.vertexCode.total -
             tile.memory.category.internalExchangeCode.total -
             tile.memory.category.vertexInstanceState.total -
             tile.memory.category.controlCode.total
             for tile in report.compilation.ipus[ipu_id].tiles)


def get_total_memory(report, ipu_id=0):
  """Returns the total memory consumption on the `ipu_id`-th IPU.

  Arg:
    report: A PVA report.
    ipu_id: An integer indicating which IPU to analyze from the report.

  Return:
    An integer. Total memory usage on the specified IPU.
  """
  return sum(tile.memory.total.excludingGaps
             for tile in report.compilation.ipus[ipu_id].tiles)


def get_program_step_cycle(step):
  """Get estimated cycle for a program step.

  Arg:
    step: A PVA compilation program step.

  Return:
    An integer. Number of estimated cycle to complete the step."""
  if step.program.type == PType.DoExchange:
    return max(step.program.estimatedCyclesByTile)
  if step.program.type == PType.OnTileExecute:
    return max(step.program.computeset.estimatedCyclesByTile)
  return 0


def get_exec_step_cycle(step, ipu_id=0):
  """Get actual cycle for a program step.

  Arg:
    step: A PVA execution program step.

  Return:
    An integer. Number of cycle to complete the step in execution.
  """
  if step.program.type in [PType.DoExchange, PType.OnTileExecute]:
    return step.ipus[ipu_id].cycles
  return 0


def get_layer_cycle(report, name, compile_only):
  """Find the number of cycles to complete a layer

  Arg:
    report: A single-layer profile.
    name: The name of the layer if the report is a profile of a single layer.
    use_execution_profile: If `True`, this function will return a cycle number
    from an actual layer execution. If `False`, this function will return an
    estimation number from cycle estimation for each program step.

  Return:
    An integer. Estimated/Actual number of cycles required to complete the
      computation of the given layer.
  """
  if compile_only:
    return sum(
        get_program_step_cycle(step)
        for step in report.compilation.livenessProgramSteps
        if name in step.program.name)
  return sum(
      get_exec_step_cycle(step, 0) for step in report.execution.runs[0].steps
      if name in step.program.name)


def get_pipeline_stage_cycles(report, compute_cycle_only=True):
  """Find the cycle count of a pipeline stage on all IPUs, when the pipeline is
  fully ramped up.

  For the execution trace of running a pipelined model for prediction,
  ___________
  |_|_|_|_|_|
    |_|_|_|_|
      |_|_|_|
        |_|_|
           ^ <- Only cycle counts from this pipeline stage are used.

  Arg:
    report: A pipelined model profile. The model must predict at least
    `2*n_ipu` batches.
    compute_cycle_only: If `True`, the pipeline stage cycle will include cycle
    counts of layer computation only, excluding cycle counts of CPU-IPU
    `StreamCopy`, IPU-IPU `GlobalExchange`, and IPU `ExternalSync` steps.
    If `False`, this function will include every cycle during the pipeline
    stage, including CPU-IPU `StreamCopy`, IPU-IPU `GlobalExchange`, and IPU
    `ExternalSync` steps.

  Return:
    A list of `n_ipu` integers. Number of cycles required to complete a pipeline
    stage.
  """
  n_ipu = report.compilation.target.numIPUs

  # All global changes that transfer more than 1024 Bytes of data
  global_exchanges = [
      step for step in report.execution.runs[0].steps
      if step.program.type == PType.GlobalExchange
      if sum(ipus.dataIn + ipus.dataOut for ipus in step.ipus) > 1024
  ]

  # Start of the fully ramped pipeline stage
  pipeline_stage_start = global_exchanges[n_ipu - 1]

  # End of the fully ramped pipeline stage
  pipeline_stage_end = global_exchanges[n_ipu]

  def in_pipeline_stage(step):
    return all(ipu_cycles.activeCycles.cyclesFrom.max >=
               pipeline_stage_start.ipus[ipu_id].activeCycles.cyclesTo.min
               and ipu_cycles.activeCycles.cyclesTo.min <=
               pipeline_stage_end.ipus[ipu_id].activeCycles.cyclesFrom.max
               for ipu_id, ipu_cycles in enumerate(step.ipus))

  # Pipeline stage cycles including data transferring cycles idling cycles
  pipeline_cycles = [
      pipeline_stage_end.ipus[i].activeCycles.cyclesTo.max -
      pipeline_stage_start.ipus[i].activeCycles.cyclesTo.max
      for i in range(n_ipu)
  ]

  # If we want data transferring steps in the cycle counts
  if not compute_cycle_only:
    return pipeline_cycles

  # External syncs during this pipeline stage
  external_syncs = [
      step for step in report.execution.runs[0].steps
      if step.program.type == PType.Sync and step.program.name == "External"
      and in_pipeline_stage(step)
  ]

  # Stream copy during this pipeline stage, including Model Input from CPU to
  # the first IPU and model ouput from last IPU to CPU
  stream_copys = [
      step for step in report.execution.runs[0].steps if step.program.type in
      [PType.StreamCopyBegin, PType.StreamCopyMid, PType.StreamCopyEnd]
      and in_pipeline_stage(step)
  ]

  # Remove `GlobalExchange`, `ExternalSync`, `StreamCopy` cycle counts
  # to find the computation cycle counts of each IPU in this pipeline stage.
  for step in [pipeline_stage_end] + external_syncs + stream_copys:
    for ipu_id, ipu_cycles in enumerate(step.ipus):
      cycle = ipu_cycles.activeCycles.cyclesTo.max
      cycle -= ipu_cycles.activeCycles.cyclesFrom.max
      pipeline_cycles[ipu_id] -= cycle

  return pipeline_cycles
