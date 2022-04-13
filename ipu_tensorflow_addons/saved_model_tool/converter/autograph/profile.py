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
from collections import defaultdict
import glob
import logging
import warnings
from statistics import mean
from datetime import datetime

import pva
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.tfv1graph import TFv1Graph, Node
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.utils import frame_info

CYCLE_DEFAULT = 500
CYCLE_UNASSIGNED = -1

OTHER_PROGRAM_NAME = {
    'controlId',
    'hostExchangePacketHeader',
    'globalExchangePacketHeader',
    'copyDescriptor',
    'instrumentationResults',
    'hostExchangeCode',
    'globalExchangeCode',
    'internalExchangeCode',
    'vertexCode',
    'vertexFieldData',
    'vectorListDescriptor',
    'vertexInstanceState',
    'controlCode',
    'stack',
}


class MemInfo():
  def __init__(self, alwayslive=0, nonalwayslive=0):
    self.alwayslive = alwayslive
    self.nonalwayslive = nonalwayslive

  def __repr__(self):
    return (f"MemInfo(alwayslive={self.alwayslive},"
            f" nonalwayslive={self.nonalwayslive})")

  def __str__(self):
    return (f"MemInfo(alwayslive={self.alwayslive},"
            f" nonalwayslive={self.nonalwayslive})")


class CycleInfo():
  def __init__(self, cycles=CYCLE_UNASSIGNED):
    self.cycles = cycles

  def __repr__(self):
    return f"CycleInfo(cycles={self.cycles})"

  def __str__(self):
    return f"CycleInfo(cycles={self.cycles})"


class OPTypeCycleEstimateKey():
  """The custom key for searching nodes have the same inputs, outputs and op_type"""
  def __init__(self, node: Node):
    self.op_type = node.op_type
    self.inputs_size = [tuple(inp.shape) for inp in node.inputs]
    self.outputs_size = [tuple(out.shape) for out in node.outputs]

  def __repr__(self):
    return f"{self.op_type}, {self.inputs_size}, {self.outputs_size}"

  def __str__(self):
    return f"{self.op_type}, {self.inputs_size}, {self.outputs_size}"

  def __eq__(self, other):
    return (other and self.op_type == other.op_type
            and self.inputs_size == other.inputs_size
            and self.outputs_size == other.outputs_size)

  def __hash__(self):
    return hash(f"{self.op_type}, {self.inputs_size}, {self.outputs_size}")


class TrieDictionary():
  def __init__(self, sep="/"):
    self.sep = sep
    self.root = {}

  def add(self, name_scope):
    if name_scope:
      words = [w for w in name_scope.split(self.sep) if w]
      n = len(words)
      cur = self.root

      for idx, word in enumerate(words):
        if word not in cur:
          cur[word] = {} if idx != n - 1 else True
        if cur[word] is True and idx != n - 1:
          cur[word] = {}
        cur = cur[word]

  def top_match(self, name_scope_unmatched):
    if name_scope_unmatched and self.root:
      found_words_list = []
      words = [w for w in name_scope_unmatched.split(self.sep) if w]
      cur = self.root
      for word in words:
        if cur is True:
          break
        if word in cur:
          found_words_list.append(word)
          cur = cur[word]
        else:
          return self.sep.join(found_words_list) if found_words_list else False

      return self.sep.join(found_words_list)

    return False

  def isempty(self):
    return len(self.root) == 0


class ProfileAnalyzer():
  def __init__(self, file_path: str, namescope_sep="/", logger=None):
    self.file_path = file_path
    self.logger = logger
    self.report = None
    self.ipu_model_type = None
    self.report_version = None
    self.stable_format = None
    self.num_ipus = None
    self.ipu_oom_list = []

    self.MembytesPerTile = None
    self.MembytesPerIPU = None
    self.totalMemory = None
    self.clockFrequency = None
    self.tilesPerIpu = None
    self.numTiles = None
    self.replicas = None
    self.ipusPerReplica = None
    self.tilesPerReplica = None
    self.memoryPerReplica = None

    self.mem_info_per_tile = []
    self.ipu_other_ops = {}
    self.max_mem_non_alwayslive_var = defaultdict(int)
    self.meminfo_dict = {}
    self.tile_balance = []
    self.activate_tile_balance = []
    self.cycle_info_dict = {}
    self.cycle_info_estimate_dict_with_op_type = defaultdict(int)
    self.namescope_sep = namescope_sep
    self.namescope_trie = TrieDictionary(sep=namescope_sep)

  def is_parsed(self):
    return not self.report is None

  @property
  def pop_profile_path(self):
    pop_path = glob.glob(f"{self.file_path}/**/*.pop", recursive=True)
    if not pop_path:
      raise FileNotFoundError(
          "Could not find profile.pop in directory '{self.file_path}'.")
    return pop_path[0]

  def parse(self):
    self.report = self.open()
    self.ipu_model_type = self._ipu_model_type()
    self.report_version, self.stable_format = self._report_version()
    self.num_ipus = self.report.compilation.target.numIPUs
    self.MembytesPerTile = self.report.compilation.target.bytesPerTile
    self.MembytesPerIPU = self.report.compilation.target.bytesPerIPU
    self.totalMemory = self.report.compilation.target.totalMemory
    self.clockFrequency = self.report.compilation.target.clockFrequency
    self.tilesPerIpu = self.report.compilation.target.tilesPerIpu
    self.numTiles = self.report.compilation.target.numTiles

    self.replicas = self.report.compilation.target.numReplicas
    self.ipusPerReplica = self.report.compilation.target.ipusPerReplica
    self.tilesPerReplica = self.report.compilation.target.tilesPerReplica
    self.memoryPerReplica = self.report.compilation.target.memoryPerReplica

  def log(self,
          message,
          level="info",
          raise_and_warn=True,
          error_type=ValueError):

    if self.logger:
      self.logger.log(getattr(logging, level.upper()),
                      f"{frame_info()} - {message}")
    else:
      print((f"[AutoGraph-profile-analysor]-[{datetime.now()}]-"
             f"[{level.lower()}]-[{frame_info()}]: {message}"))

    if raise_and_warn:
      if level == "error":
        raise error_type(message)

      if level == "warn":
        warnings.warn(message)

  def _ipu_model_type(self):
    return {
        pva.Target.Type.Ipu: "Ipu",
        pva.Target.Type.IpuModel: "IpuModel",
        pva.Target.Type.Cpu: "Cpu",
        pva.Target.Type.Unknown: "Unknown",
    }[self.report.compilation.target.type]

  def _report_version(self):
    return ".".join([
        str(self.report.version.major),
        str(self.report.version.minor),
        str(self.report.version.point)
    ]), not self.report.version.isUnstableFormat

  def open(self):
    self.log(self.pop_profile_path)
    return pva.openReport(self.pop_profile_path)

  def _build_trie_from_tfgraph(self, graph):
    for node_name in graph.nodes:
      self.namescope_trie.add(node_name)

  def _update_ipu_oom_list(self):
    if self.ipu_oom_list:
      return
    ipu_oom_list = [0] * self.num_ipus
    for ipuid, ipu in enumerate(self.report.compilation.ipus):
      for tile in ipu.tiles:
        if (tile.memory.overflowed.excludingGaps
            or tile.memory.overflowed.includingGaps):
          ipu_oom_list[ipuid] += tile.memory.overflowed.excludingGaps
    self.ipu_oom_list = ipu_oom_list.copy()

  def which_ipu_oom(self):
    self._update_ipu_oom_list()
    return [idx for idx, oom_flag in enumerate(self.ipu_oom_list) if oom_flag]

  def check_if_oom(self):
    return not len(self.which_ipu_oom()) == 0

  def mem_state_per_tile(self):
    if self.mem_info_per_tile:
      return
    for ipu in self.report.compilation.ipus:
      ipu_mem_per_tile = []
      for tile in ipu.tiles:
        ipu_mem_per_tile.append(tile.memory.total.excludingGaps)
      self.mem_info_per_tile.append(ipu_mem_per_tile)

  def mem_used_per_ipu(self):
    self.mem_state_per_tile()
    return [sum(ipu_tiles) for ipu_tiles in self.mem_info_per_tile]

  def mem_free_per_ipu(self):
    self.mem_state_per_tile()
    return [
        self.MembytesPerIPU - sum(ipu_tiles)
        for ipu_tiles in self.mem_info_per_tile
    ]

  def mem_overflow_per_ipu(self):
    self._update_ipu_oom_list()
    mem_overflow_list = self.mem_free_per_ipu()
    for idx, oom_flag in enumerate(self.ipu_oom_list):
      if oom_flag:
        mem_overflow_list[idx] = -oom_flag
    return mem_overflow_list

  def activate_tile_balance_per_ipu(self):
    if self.activate_tile_balance:
      return
    ipu_tile_balance = [[] for _ in range(self.num_ipus)]

    for s in self.report.execution.steps:
      for idx, per_ipu in enumerate(s.ipus):
        if not (s.program.type == pva.Program.Type.Sync or per_ipu.activeCycles
                .cyclesFrom.max == per_ipu.activeCycles.cyclesTo.max):
          ipu_tile_balance[idx].append((per_ipu.activeTileBalance))

    self.activate_tile_balance = [
        sum(i) / len(i) if i else 0 for i in ipu_tile_balance
    ]

  def tile_balance_per_ipu(self):
    if self.tile_balance:
      return

    ipu_tile_balance = [[] for _ in range(self.num_ipus)]

    for s in self.report.execution.steps:
      for idx, per_ipu in enumerate(s.ipus):
        if not (s.program.type == pva.Program.Type.Sync or per_ipu.activeCycles
                .cyclesFrom.max == per_ipu.activeCycles.cyclesTo.max):
          ipu_tile_balance[idx].append((per_ipu.tileBalance))

    self.tile_balance = [sum(i) / len(i) if i else 0 for i in ipu_tile_balance]

  def update_other_ops_alwayslive_mem_info(self):
    for alv in self.report.compilation.alwaysLiveVariables:
      if alv.name in OTHER_PROGRAM_NAME:
        self.ipu_other_ops[alv.name] = MemInfo(alv.size, 0)

  def micro_sec_per_run(self):
    return [
        r.microseconds.end - r.microseconds.start
        for r in self.report.execution.runs
    ]

  def estimate_throughput(self, batch_size):
    micro_sec = self.micro_sec_per_run()[-1]
    return batch_size / micro_sec

  def _get_node_name_with_cycle_dict(self, graph: TFv1Graph):
    return {node_name: CycleInfo() for node_name in graph.nodes}

  def _filter_out_node(self,
                       step,
                       program_name,
                       exclude_program_name=None,
                       exclude_program_type=None):
    if exclude_program_name is None:
      exclude_program_name = {"/get-tuple-element", "copy", "__seed"}
    if exclude_program_type is None:
      exclude_program_type = {
          pva.Program.Type.Sync,
          pva.Program.Type.GlobalExchange,
      }

    return (step.program.type not in exclude_program_type and program_name
            and not any(program_name.lower().startswith(n)
                        for n in exclude_program_name))

  def _get_unduplicate_cycles_dict(self):
    groupby_prog_name = defaultdict(list)
    for step in self.report.execution.runs[0].steps:
      prog_name = ("_".join(step.program.name.split("_")[1:]) if
                   step.program.name.startswith("cs") else step.program.name)
      if self._filter_out_node(step, prog_name):
        groupby_prog_name[prog_name].append(sum([u.cycles for u in step.ipus]))
    return {name: mean(cycles) for name, cycles in groupby_prog_name.items()}

  def _update_estimate_cycleinfo_from(self, graph):
    unique_prog_cycle_dict = self._get_unduplicate_cycles_dict()

    for name, cycle in unique_prog_cycle_dict.items():
      namescope = self.namescope_trie.top_match(name)
      if namescope and namescope in self.cycle_info_dict:
        if self.cycle_info_dict[namescope].cycles == CYCLE_UNASSIGNED:
          self.cycle_info_dict[namescope].cycles = cycle
          self.cycle_info_estimate_dict_with_op_type[OPTypeCycleEstimateKey(
              graph.nodes[namescope])] = cycle
        else:
          self.cycle_info_dict[namescope].cycles += cycle
          self.cycle_info_estimate_dict_with_op_type[OPTypeCycleEstimateKey(
              graph.nodes[namescope])] += cycle

    for name, cycle in self.cycle_info_dict.items():
      if cycle.cycles == CYCLE_UNASSIGNED:
        cycle.cycles = self.cycle_info_estimate_dict_with_op_type.get(
            OPTypeCycleEstimateKey(graph.nodes[name]), CYCLE_DEFAULT)

  def ops_cycle_info(self, graph: TFv1Graph):
    self.cycle_info_dict = self._get_node_name_with_cycle_dict(graph)
    if self.namescope_trie.isempty():
      self._build_trie_from_tfgraph(graph)
    self._update_estimate_cycleinfo_from(graph)

  def _get_node_name_with_mem_dict(self, graph: TFv1Graph):
    return {node_name: MemInfo() for node_name in graph.nodes}

  def _update_alwaysliveness_meminfo(self):
    for alv in self.report.compilation.alwaysLiveVariables:
      alv_meminfo_name = self.namescope_trie.top_match(alv.name)
      if alv_meminfo_name and alv_meminfo_name in self.meminfo_dict:
        self.meminfo_dict[alv_meminfo_name].alwayslive += alv.size

  def _update_non_alwaysliveness_meminfo(self):
    for step in self.report.compilation.livenessProgramSteps:
      non_always_var_step_dict = defaultdict(int)
      for v in step.notAlwaysLiveMemory.variables:
        if not v.name.startswith("broadcastProgramId"):
          meminfo_name = self.namescope_trie.top_match(v.name)
          if meminfo_name:
            non_always_var_step_dict[meminfo_name] += v.size

      for meminfo_name in non_always_var_step_dict:
        if meminfo_name in self.meminfo_dict:
          self.meminfo_dict[meminfo_name].nonalwayslive = max(
              self.meminfo_dict[meminfo_name].nonalwayslive,
              non_always_var_step_dict[meminfo_name])

  def ops_memory_info(self, graph: TFv1Graph):
    self.meminfo_dict = self._get_node_name_with_mem_dict(graph)
    if self.namescope_trie.isempty():
      self._build_trie_from_tfgraph(graph)
    self._update_alwaysliveness_meminfo()
    self.log("Extract always liveness mem ops done", level="debug")

  def extract_info_to_tfv1graph(self, graph: TFv1Graph):
    """Extract information from reprot `profile.pop`
    Args:
      report_file_path (str): the path of `profile.pop` file
    """
    self.log("Parse the profile", level="debug")
    self.parse()
    self.log("Extract oom IPU info", level="debug")
    self.which_ipu_oom()
    self.log("Extract mem per tile", level="debug")
    self.mem_state_per_tile()
    self.log("Extract activate tile balance", level="debug")
    self.activate_tile_balance_per_ipu()
    self.log("Extract tile balance", level="debug")
    self.tile_balance_per_ipu()
    self.log("Extract op_mem", level="debug")
    self.ops_memory_info(graph)
    self.log("Extract other ops mem", level="debug")
    self.update_other_ops_alwayslive_mem_info()
    self.log("Extract other ops cycle info", level="debug")
    self.ops_cycle_info(graph)
    self.log("Done", level="debug")
