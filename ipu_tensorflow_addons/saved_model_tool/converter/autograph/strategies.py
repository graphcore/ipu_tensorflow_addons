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
"""
Pipelining Strategies
"""
import re
from typing import List
from copy import deepcopy
from collections import defaultdict
from statistics import mean, stdev
from itertools import accumulate

from ipu_tensorflow_addons.saved_model_tool.converter.autograph import TFv1Graph
from ipu_tensorflow_addons.saved_model_tool.converter.autograph import RunConfig
from ipu_tensorflow_addons.saved_model_tool.converter.autograph import ProfileAnalyzer


def variances(former: List[int], latter: List[int], num_group: int):
  former_mean = (sum(former) + sum(latter)) / num_group
  return (sum(former) - former_mean)**2 - len(former)


def find_opt(arr, num_group):
  """Find the optimal solution with greedy search to splitting pipeline stages.

  Args:
    arr (list): the array of memory or time cost for every operations.
    num_group (int): number group that you want to divide.

  Returns:
    the nested list: the optimal solution.
  """

  # Strategy
  #
  # `Strategy` finds the best splitting point for a large model
  # that has the best performance like throughput.
  # It can manipulate the `Graph` such as adding the pipeline stage to `Node`.
  # `Strategy` can split the large model into service pipeline stages with the cycle balance,
  # which means that every pipeline stage has a similar cycle or time cost.
  # This can make use of the IPU resource. if there is out of memory error after GreedyStrategy,
  # the Strategy will modify the pipeline stage info to fix out of memory error
  # according to the memory status of each IPU regardless of the cycle balance.
  #
  # Problems
  #
  # the operator can be considered as running in sequential order (topological order)
  # since the BSP sync strategy is used in the execution. So, the balance pipeline stage time cost problem can be described as:
  # Given ordered array A, find the best splitting points to divide A
  # into M groups with minimum the standard deviation of those M Groups.
  #
  # Greedy Search Strategy
  #
  # The strategy that we use is maximizing the score for every step.
  # the score is calculated by the distance between the mean of total means
  # and the mean of left of splitting point in the array.
  # the formular is desribe below:
  # $(\sum_{i=0}^{SplittingPointIndex}{A_i} - mean(A))^2 - SplittingPointIndex$

  res_set = []

  def divide_and_conquer_helper(arr, num_group, res):
    if not arr or not num_group:
      return

    if num_group == 1:
      res.append(arr.copy())
      res_set.append(deepcopy(res))
      res.pop()
      return

    scores_for_all = [(i, variances(arr[:i], arr[i:], num_group))
                      for i in range(1,
                                     len(arr) - num_group + 2)]
    minimum_score = min([score for _, score in scores_for_all])
    scores_for_min = [(sep, score) for sep, score in scores_for_all
                      if score == minimum_score]

    for sep, _ in scores_for_min:
      res.append(arr[:sep])
      divide_and_conquer_helper(arr[sep:], num_group - 1, res)
      res.pop()

  divide_and_conquer_helper(arr, num_group, [])
  return res_set


class StrategyBase():
  def __init__(self, config: RunConfig):
    self.ipu_config = config


class ManualPipelineStrategy(StrategyBase):
  def _add_stage_info_to_node(self, node, pipeline_stage):
    node.add_attr(pipeline_stage=pipeline_stage)

  def _compile_regexes(self, regex_lists):
    if len(regex_lists) != self.ipu_config.num_pipeline_stages:
      raise ValueError(f"The length of `regex_lists` ({len(regex_lists)}) "
                       "should be equal to the number of pipeline stages "
                       f"({self.ipu_config.num_pipeline_stages}).")
    compiled_regex_lists = [
        list() for _ in range(self.ipu_config.num_pipeline_stages)
    ]
    for stage, reg_list in enumerate(regex_lists):
      for reg in reg_list:
        compiled_regex_lists[stage].append(re.compile(reg))

    return compiled_regex_lists

  def _match_node(self, node_name, regex):
    return regex.match(node_name)

  def _match_nodes_for_pipline_stage(self, node_name, regex_for_stage):
    for r in regex_for_stage:
      if self._match_node(node_name, r):
        return True
    return False

  def _matches_nodes_and_assign_pipline_info(self, graph: TFv1Graph,
                                             compiled_regex_lists):
    for node_name, node in graph.nodes.items():
      for stage, cr in enumerate(compiled_regex_lists):
        if self._match_nodes_for_pipline_stage(node_name, cr):
          self._add_stage_info_to_node(node, stage)

  def chop(self, graph: TFv1Graph, regex_for_stages):
    compiled_regex_for_stages = self._compile_regexes(regex_for_stages)
    return self._matches_nodes_and_assign_pipline_info(
        graph, compiled_regex_for_stages)


class GreedySolveStrategy(StrategyBase):
  def __init__(self, ipu_config):
    super().__init__(ipu_config)
    self.tried_table = defaultdict(list)  # score: solutions
    self.prev_split_pos = []

  def bucket_idx(self, bucket_range_list, ord_idx):
    for idx, b in enumerate(bucket_range_list):
      if ord_idx < b:
        return idx
    return len(bucket_range_list)

  def get_ticket_for_ops(self, solutions):
    return [*accumulate(solutions, lambda x, y: x + y)]

  def add_stage_info(self, graph: TFv1Graph, bucket_list: List):
    for idx, node in enumerate(graph.topology_sort()):
      node.add_attr(pipeline_stage=self.bucket_idx(bucket_list, idx))

  def first_try(self, graph: TFv1Graph):
    """The first attempt of pipelining the model.

    Pipelining the model with average numbers of ops.

    Args:
        graph (TFv1Graph): The graph to be pipelined.
    """
    pipline_stage_num = self.ipu_config.num_pipeline_stages
    quanters = graph.size_of_nodes() // pipline_stage_num + 1
    self.prev_split_pos = [quanters] * pipline_stage_num
    bucket_range_list = self.get_ticket_for_ops(self.prev_split_pos)
    self.add_stage_info(graph, bucket_range_list)

  def greedy_search_solutions(self,
                              graph: TFv1Graph,
                              pa: ProfileAnalyzer,
                              priority="memory"):
    if priority not in {"memory", "cycle"}:
      raise ValueError('The priority must be "memory" or "cycle", '
                       f'not "{priority}".')

    if priority == "memory":
      ops_mem_list = [
          pa.meminfo_dict[node.name].alwayslive +
          pa.meminfo_dict[node.name].nonalwayslive
          for node in graph.topology_sort()
      ]
      solutions, *_ = find_opt(ops_mem_list,
                               self.ipu_config.num_pipeline_stages)
    else:
      ops_cycle_list = [
          pa.cycle_info_dict[node.name].cycles
          for node in graph.topology_sort()
      ]
      solutions, *_ = find_opt(ops_cycle_list,
                               self.ipu_config.num_pipeline_stages)

    self.prev_split_pos = [len(s) for s in solutions]
    bucket_range_list = self.get_ticket_for_ops(self.prev_split_pos)
    self.add_stage_info(graph, bucket_range_list)

  def criterion(self, pa):
    mu = mean(pa.mem_used_per_ipu())
    scale = 10**(len(str(mu)) - 2)
    return stdev([i / scale for i in pa.mem_used_per_ipu()])

  def update_tried_table(self, pa):
    score = self.criterion(pa)
    self.tried_table[score].append(self.prev_split_pos)

  def log_best_solution(self, graph, file_name):
    best_tickets = self.tried_table[min(self.tried_table.keys())]
    bucket_range_list = self.get_ticket_for_ops(best_tickets)
    self.add_stage_info(graph, bucket_range_list)
    graph.save_pipeline_config(file_name)

  def review_prev_plan(self, graph):
    bucket_range_list = self.get_ticket_for_ops(self.prev_split_pos)
    self.add_stage_info(graph, bucket_range_list)

  def translate_to_grouped_mem_list(self, split_pos_list, mem_list):
    acc_split_pos_list = self.get_ticket_for_ops(split_pos_list)
    res = [[] for _ in range(self.ipu_config.num_pipeline_stages)]
    for idx, mem in enumerate(mem_list):
      res[self.bucket_idx(acc_split_pos_list, idx)].append(mem)
    return res

  def translate_from_grouped_list(self, grouped_list):
    return [len(i) for i in grouped_list]

  def move_op_left_replace(self, pos_list, number_of_ops, ipuId_from,
                           ipuId_to):
    number_of_iteration = number_of_ops if len(
        pos_list[ipuId_from]) - 1 > number_of_ops else len(
            pos_list[ipuId_from]) - 1
    tmpvar = pos_list[ipuId_from][:number_of_iteration]
    pos_list[ipuId_from] = pos_list[ipuId_from][number_of_iteration:]
    pos_list[ipuId_to] += tmpvar

  def move_op_right_replace(self, pos_list, number_of_ops, ipuId_from,
                            ipuId_to):
    number_of_iteration = number_of_ops if len(
        pos_list[ipuId_from]) - 1 > number_of_ops else len(
            pos_list[ipuId_from]) - 1
    for _ in range(number_of_iteration):
      tmpvar = pos_list[ipuId_from].pop()
      pos_list[ipuId_to].insert(0, tmpvar)

  def move_op_and_replace(self, pos_list, number_of_ops, ipuId_from, ipuId_to):
    if ipuId_from == ipuId_to or number_of_ops == 0:
      return
    if abs(ipuId_from - ipuId_to) > 1:
      raise ValueError(f"`ipuId_from` ({ipuId_from}) must be "
                       f"less than `ipuId_to` ({ipuId_to}).")
    if ipuId_from > ipuId_to:
      self.move_op_left_replace(pos_list, number_of_ops, ipuId_from, ipuId_to)
    else:
      self.move_op_right_replace(pos_list, number_of_ops, ipuId_from, ipuId_to)

  def update_mem_free_per_ipu(self, mem_free_per_ipu, sums, idx_from, idx_to):
    mem_free_per_ipu[idx_to] -= sums
    mem_free_per_ipu[idx_from] += sums

  def move_ops_to_left(self, move_after, ipuId, mem_free_per_ipu):
    for idx, sums in enumerate(
        accumulate(move_after[ipuId], lambda x, y: x + y)):
      if len(move_after[ipuId]) > 1 and sums > abs(
          mem_free_per_ipu[ipuId] - mem_free_per_ipu[ipuId - 1]) // 2:
        sum_actual = max(move_after[ipuId][idx], sums - move_after[ipuId][idx])
        self.move_op_and_replace(move_after, max(1, idx), ipuId, ipuId - 1)
        # mem_free_per_ipu[ipuId - 1] -= sums
        self.update_mem_free_per_ipu(mem_free_per_ipu, sum_actual, ipuId,
                                     ipuId - 1)
        return

    sums = list(accumulate(move_after[ipuId], lambda x, y: x + y))[-1]
    self.move_op_and_replace(move_after,
                             len(move_after[ipuId]) - 1, ipuId, ipuId - 1)
    self.update_mem_free_per_ipu(mem_free_per_ipu,
                                 sums - move_after[ipuId][-1], ipuId,
                                 ipuId - 1)

  def move_ops_to_right(self, move_after, ipuId, mem_free_per_ipu):
    for idx, sums in enumerate(
        accumulate(move_after[ipuId][::-1], lambda x, y: x + y)):
      if len(move_after[ipuId]) > 1 and sums > abs(
          mem_free_per_ipu[ipuId] - mem_free_per_ipu[ipuId + 1]) // 2:
        sums_actual = max(move_after[ipuId][-idx - 1],
                          sums - move_after[ipuId][-idx - 1])
        self.move_op_and_replace(move_after, max(1, idx), ipuId, ipuId + 1)
        self.update_mem_free_per_ipu(mem_free_per_ipu, sums_actual, ipuId,
                                     ipuId + 1)
        return

    sums = list(accumulate(move_after[ipuId], lambda x, y: x + y))[-1]
    self.move_op_and_replace(move_after,
                             len(move_after[ipuId]) - 1, ipuId, ipuId + 1)
    self.update_mem_free_per_ipu(mem_free_per_ipu, sums - move_after[ipuId][0],
                                 ipuId, ipuId + 1)

  def tune_if_OOM(self, graph: TFv1Graph, pa: ProfileAnalyzer):
    mem_free_per_ipu = pa.mem_used_per_ipu()
    which_ipu_oom = pa.which_ipu_oom()
    move_after = self.translate_to_grouped_mem_list(
        self.prev_split_pos.copy(), [
            pa.meminfo_dict[node.name].alwayslive +
            pa.meminfo_dict[node.name].nonalwayslive
            for node in graph.topology_sort()
        ])
    for ipuId in which_ipu_oom:
      if ipuId + 1 >= len(mem_free_per_ipu) and ipuId - 1 not in which_ipu_oom:
        self.move_ops_to_left(move_after, ipuId, mem_free_per_ipu)
      elif ipuId - 1 <= 0 and ipuId + 1 not in which_ipu_oom:
        self.move_ops_to_right(move_after, ipuId, mem_free_per_ipu)
      elif ipuId - 1 not in which_ipu_oom or ipuId + 1 not in which_ipu_oom:
        if mem_free_per_ipu[max(0, ipuId - 1)] > mem_free_per_ipu[ipuId + 1]:
          self.move_ops_to_left(move_after, ipuId, mem_free_per_ipu)
        else:
          self.move_ops_to_right(move_after, ipuId, mem_free_per_ipu)

    split_ops_list = self.translate_from_grouped_list(move_after)
    self.prev_split_pos = split_ops_list.copy()
    bucket_range_list = self.get_ticket_for_ops(split_ops_list)
    self.add_stage_info(graph, bucket_range_list)
