# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
#
# This file has been modified by Graphcore Ltd.
# ==============================================================================
"""
Add Auto pipeline Converter.
"""
import glob
import math
import os
from multiprocessing import get_context
from collections import OrderedDict
import logging
from datetime import datetime
import warnings

from tensorflow.python import ipu
from ipu_tensorflow_addons.saved_model_tool.converter import Converter
from ipu_tensorflow_addons.saved_model_tool.converter import utils

from ipu_tensorflow_addons.saved_model_tool.converter.autograph import RunConfig, TFv1Experiment, TFv1Graph
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.strategies import GreedySolveStrategy
from ipu_tensorflow_addons.saved_model_tool.converter.utils import deserialize_graph_def
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.utils import frame_info

VERTEX_SIZE = 160000000  # the estimation for vertices in profiling memory 160MB

AUTOPIPELINECFG_KWDS = OrderedDict({
    "converter": None,
    "fine_tune_iter": 5,
    "ipu_model": True,
    "max_ipu_quantity": 64,
    "min_ipu_quantity": 1,
    "priority": "cycle",
    "profiling_root_dir": "profiling",
    "solution_dir": "solution",
})


def get_latest_pipeconf(path):
  list_of_files = glob.glob(f"{path}{os.path.sep}*")
  latest_file = max(list_of_files, key=os.path.getctime)
  return latest_file


class AutoPipelineImpl():
  def __init__(self,
               batch_size,
               matmul_amp,
               matmul_partial_type,
               conv_amp,
               conv_partial_type,
               fine_tune_iter,
               ipu_model,
               max_ipu_quantity,
               min_ipu_quantity,
               priority,
               profiling_root_dir,
               solution_dir,
               logger=None):
    # pylint:disable=line-too-long
    """The implementation class of AutoPipeline Converter.
    """

    self.batch_size = batch_size
    self.matmul_amp = matmul_amp
    self.matmul_partial_type = matmul_partial_type
    self.conv_amp = conv_amp
    self.conv_partial_type = conv_partial_type

    self.code_prefix = 'auto_pipeline_solver'

    self.fine_tune_iter = fine_tune_iter
    self.ipu_model = ipu_model
    self.max_ipu_quantity = max_ipu_quantity
    self.min_ipu_quantity = min_ipu_quantity
    self.priority = priority
    self.profiling_root_dir = profiling_root_dir
    self.solution_dir = solution_dir

    self.logger = logger

  def log(self,
          message,
          level="info",
          raise_and_warn=True,
          error_type=ValueError):

    if self.logger:
      self.logger.log(getattr(logging, level.upper()),
                      f"{frame_info()} - {message}")
    else:
      print((f"[AutoPipeline]-[{datetime.now()}]-"
             f"[{level.lower()}]-[{frame_info()}]: {message}"))

    if raise_and_warn:
      if level == "error":
        raise error_type(message)

      if level == "warn":
        warnings.warn(message)

  def _first_try_step(self, graph, run_config, strategy):
    profiling_path = os.path.join(self.profiling_root_dir,
                                  (f"{self.code_prefix}"
                                   f"-{run_config.num_required_ipus}-"
                                   f"b{self.batch_size}-{self.priority}"))
    exper = TFv1Experiment(run_config=run_config,
                           name=f"{self.code_prefix}-first-try",
                           profiling_path=profiling_path)

    strategy.first_try(graph)
    exper.run(graph)
    self.log("Analysing the profile.")
    pa = exper.get_profile_analysis(graph)
    total_free_mem_for_model = sum(pa.mem_free_per_ipu())
    if total_free_mem_for_model < VERTEX_SIZE:
      return pa, False
    strategy.update_tried_table(pa)
    return pa, True

  def _greedy_search_solutions_step(self, graph, run_config, strategy,
                                    profile_analysor):
    profiling_path = os.path.join(self.profiling_root_dir,
                                  (f"{self.code_prefix}"
                                   f"-{run_config.num_required_ipus}-"
                                   f"b{self.batch_size}-{self.priority}"))
    exper = TFv1Experiment(run_config=run_config,
                           name=f"{self.code_prefix}-greedy-search-solutions",
                           profiling_path=profiling_path)

    os.makedirs(self.solution_dir, exist_ok=True)
    strategy.greedy_search_solutions(graph,
                                     profile_analysor,
                                     priority=self.priority)
    exper.run(graph)
    self.log("Analysing the profile.")
    pa = exper.get_profile_analysis(graph)
    total_free_mem_for_model = sum(pa.mem_free_per_ipu())
    if total_free_mem_for_model < VERTEX_SIZE:
      return pa, False
    graph.save_pipeline_config(
        os.path.join(self.solution_dir, "greedy_search_solutions.pipeconf"))
    strategy.update_tried_table(pa)
    print(strategy.tried_table)

    return pa, True  # return the profile analysor and not OOM flag

  def _tune_according_to_mem(self, graph, run_config, strategy,
                             profile_analysor):

    os.makedirs(self.solution_dir, exist_ok=True)
    strategy.tune_if_OOM(graph, profile_analysor)
    profiling_path = os.path.join(self.profiling_root_dir,
                                  (f"{self.code_prefix}"
                                   f"-{run_config.num_required_ipus}-"
                                   f"b{self.batch_size}-{self.priority}-tune"))
    for i in range(self.fine_tune_iter):
      exper = TFv1Experiment(
          run_config=run_config,
          name="TFv1-test-greedy-search-solutions-tune-with-mem",
          profiling_path=profiling_path)

      exper.run(graph)
      pa = exper.get_profile_analysis(graph)

      total_free_mem_for_model = sum(pa.mem_free_per_ipu())
      if total_free_mem_for_model < VERTEX_SIZE:
        return False

      if not pa.check_if_oom():
        graph.save_pipeline_config(
            os.path.join(self.solution_dir,
                         f"greedy_search_solutions_tune_{i}.pipeconf"))
        self.log("(Tune) Dumps pipeline config to file.")
        print(strategy.tried_table)
        return True

      self.log("(Tune) Need to fine tune the graph.")
      strategy.tune_if_OOM(graph, profile_analysor)
    return False

  def _apply(self, graph_def, signature):
    """the execution code for auto pipeline converter.
    Args:
      graph_def: the input frozen Tensorflow graph
      signature_def: the input frozen Tensorflow meta graph
    Returns:
      graph_def: the frozen Tensorflow graph after conversion.
      signature_def: the frozen Tensorflow meta graph after conversion.
    """

    self.tfgraph = deserialize_graph_def(graph_def)

    for num_ipus in map(
        lambda x: 2**x,
        range(max(math.floor(math.log2(self.min_ipu_quantity)), 1),
              math.floor(math.log2(self.max_ipu_quantity)) + 1)):
      g = TFv1Graph(
          self.tfgraph,
          signature_def=signature)  # need to new graph to search the plan

      self.log(f"(IPU-Search) Try {num_ipus} IPUs on IPU model.")
      cfg = RunConfig(
          num_ipus,
          batch_size=self.batch_size,
          matmul_amp=self.matmul_amp,
          matmul_partial_type=self.matmul_partial_type,
          conv_amp=self.conv_amp,
          conv_partial_type=self.conv_partial_type,
          ipu_model=self.ipu_model,
      )
      strate = GreedySolveStrategy(cfg)
      pa, can_search_with_greedy_strategy = self._first_try_step(
          g, cfg, strate)
      if can_search_with_greedy_strategy:
        pa, can_fit_with_current_ipus = self._greedy_search_solutions_step(
            g, cfg, strate, pa)
        if not can_fit_with_current_ipus:
          can_fit_with_current_ipus = self._tune_according_to_mem(
              g, cfg, strate, pa)

          if can_fit_with_current_ipus:
            self.log(f"The {num_ipus} IPUs are used. Done.")
            return num_ipus

          self.log("(IPU-OOM) Can not find the optimal solution.")

        self.log(f"The {num_ipus} IPUs are used. Done.")
        return num_ipus

      self.log("(IPU-OOM) Need to increase the number of IPUs.")

    raise ValueError("(IPU-END) Can not find the optimal solution.")

  def apply(self, graph_def, signature):
    return self._apply(graph_def, signature)


def auto_pipeline_helper(graph_def, signature, batch_size, matmul_amp,
                         matmul_partial_type, conv_amp, conv_partial_type,
                         fine_tune_iter, ipu_model, max_ipu_quantity,
                         min_ipu_quantity, priority, profiling_root_dir,
                         solution_dir):
  converter = AutoPipelineImpl(batch_size, matmul_amp, matmul_partial_type,
                               conv_amp, conv_partial_type, fine_tune_iter,
                               ipu_model, max_ipu_quantity, min_ipu_quantity,
                               priority, profiling_root_dir, solution_dir)
  num_ipus = converter.apply(graph_def, signature)
  solution_path = get_latest_pipeconf(converter.solution_dir)
  return num_ipus, solution_path


def process_pool_initializer(env):
  if 'TF_POPLAR_FLAGS' in env:
    env.pop("TF_POPLAR_FLAGS")
  os.environ.update(env)


class AutoPipeline(Converter):
  # pylint:disable=line-too-long
  """
    The converter can search the number of IPUs
    and spilt the graph with a cycle balanced greedy search algorithm
    to obtain better performance and throughput.
    After that, wrap it with embedded runtime compilation.
  """
  def __init__(self, param):
    self.param = param
    self.batch_size = param.batch_size
    self.matmul_amp = param.matmul_amp
    self.matmul_partial_type = param.matmul_partial_type
    self.conv_amp = param.conv_amp
    self.conv_partial_type = param.conv_partial_type
    self._batch_per_step = param.batch_per_step
    self._excluded_nodes = param.excluded_nodes
    self._remove_excluded_nodes = param.remove_excluded_nodes
    self._embedded_runtime_save_config = param.embedded_runtime_save_config
    self._int64_to_int32_conversion = param.int64_to_int32_conversion
    self._merge_subgraphs = param.merge_subgraphs
    self._validate_params(param)

    self.cfg = ipu.config.IPUConfig()

    if param.embedded_runtime_save_config:
      (self._embedded_runtime_exec_cachedir, self._poplar_exec_filepath,
       self._runtime_api_timeout_us,
       self._batch_per_step) = utils.extract_emb_setting_from_param(param)

    if param.pipeline_cfg:
      self._validate_pipeline_cfg_and_setdefaults(param)

  def _validate_pipeline_cfg_and_setdefaults(self, param):
    (_, self.fine_tune_iter, self.ipu_model, self.max_ipu_quantity,
     self.min_ipu_quantity, self.priority, self.profiling_root_dir,
     self.solution_dir) = utils.validate_pipeline_cfg(param,
                                                      AUTOPIPELINECFG_KWDS,
                                                      "auto")

  def _validate_params(self, param):
    if param.pipeline_cfg and not param.embedded_runtime_save_config:
      raise ValueError(
          "The `embedded_runtime_save_config` argument must be specified for pipelined models."
      )
    utils.validate_embedded_runtime_save_config(param)

  def _should_do_auto_pipeline(self):
    return utils.should_do_pipeline_wrapper(self.param.pipeline_cfg, "auto",
                                            self._embedded_runtime_save_config,
                                            self._batch_per_step,
                                            self._merge_subgraphs,
                                            self._int64_to_int32_conversion)

  def _do_auto_pipline(self, graph_def, signature):
    with get_context("spawn").Pool(processes=1,
                                   initializer=process_pool_initializer,
                                   initargs=(os.environ.copy(),)) as p:
      num_ipus, solution_path = p.apply(
          auto_pipeline_helper,
          args=(graph_def, signature, self.batch_size, self.matmul_amp,
                self.matmul_partial_type, self.conv_amp,
                self.conv_partial_type, self.fine_tune_iter, self.ipu_model,
                self.max_ipu_quantity, self.min_ipu_quantity, self.priority,
                self.profiling_root_dir, self.solution_dir))
    return num_ipus, solution_path

  def _apply(self, graph_def, signature):
    num_ipus, solution_path = self._do_auto_pipline(graph_def, signature)

    self.param.num_ipus = num_ipus
    tfGraph = deserialize_graph_def(graph_def)

    auto_graph = TFv1Graph(tfGraph, signature)
    auto_graph.read_pipeline_config(solution_path)

    return utils.pipeline_embedded_runtime_wrapper(
        self.param, self.cfg, auto_graph, signature, self.batch_size,
        self._batch_per_step, self._poplar_exec_filepath,
        self._runtime_api_timeout_us)

  def apply(self, graph_def, signature_def):
    if not self._should_do_auto_pipeline():
      return graph_def, signature_def

    return self._apply(graph_def, signature_def)
