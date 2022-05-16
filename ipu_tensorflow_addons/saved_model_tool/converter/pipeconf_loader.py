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
Add PipelineConfLoader Converter.
"""
import os
from multiprocessing import get_context
from collections import OrderedDict

from tensorflow.python import ipu
from ipu_tensorflow_addons.saved_model_tool.converter import Converter
from ipu_tensorflow_addons.saved_model_tool.converter import utils

from ipu_tensorflow_addons.saved_model_tool.converter.autograph import RunConfig, TFv1Experiment, TFv1Graph
from ipu_tensorflow_addons.saved_model_tool.converter.utils import deserialize_graph_def

LOADPIPELINECFG_KWDS = OrderedDict({
    "converter": "load",
    "ipu_model": True,
    "profiling_root_dir": "profiling",
    "solution_path": "solution/greedy_search.pipeconfig",
    "profiling_enable": False
})


class PipelineConfProfiler():
  def __init__(self, batch_size, matmul_amp, num_ipus, matmul_partial_type,
               conv_amp, conv_partial_type, ipu_model, profiling_root_dir,
               solution_path, profiling_enable):
    self.batch_size = batch_size
    self.matmul_amp = matmul_amp
    self.num_ipus = num_ipus
    self.matmul_partial_type = matmul_partial_type
    self.conv_amp = conv_amp
    self.conv_partial_type = conv_partial_type

    self.code_prefix = 'pipeline_loader'

    self.ipu_model = ipu_model
    self.profiling_root_dir = profiling_root_dir
    self.solution_path = solution_path
    self.profiling_enable = profiling_enable

  def _run_with_pipeline_config(self, graph, run_config):
    profiling_path = os.path.join(self.profiling_root_dir,
                                  (f"{self.code_prefix}"
                                   f"-{run_config.num_required_ipus}-"
                                   f"b{self.batch_size}"))
    exper = TFv1Experiment(run_config=run_config,
                           name=f"{self.code_prefix}-from-pipeline-config",
                           profiling_path=profiling_path)

    graph.read_pipeline_config(self.solution_path)
    exper.run(graph)

  def _apply(self, graph_def, signature):
    """The execution code for the converter pipeline.
    Args:
      graph_def: The input frozen TensorFlow graph.
      signature_def: The input frozen TensorFlow metagraph.
    Returns:
      graph_def: The frozen TensorFlow graph after conversion.
      signature_def: The frozen TensorFlow metagraph after conversion.
    """

    self.tfgraph = deserialize_graph_def(graph_def)
    cfg = RunConfig(
        self.num_ipus,
        batch_size=self.batch_size,
        matmul_amp=self.matmul_amp,
        matmul_partial_type=self.matmul_partial_type,
        conv_amp=self.conv_amp,
        conv_partial_type=self.conv_partial_type,
        ipu_model=self.ipu_model,
    )
    g = TFv1Graph(self.tfgraph, signature_def=signature)
    self._run_with_pipeline_config(g, cfg)

  def apply(self, graph_def, signature):
    return self._apply(graph_def, signature)


def pipeline_profile_taker_helper(graph_def, signature_def, batch_size,
                                  matmul_amp, num_ipus, matmul_partial_type,
                                  conv_amp, conv_partial_type, ipu_model,
                                  profiling_root_dir, solution_path,
                                  profiling_enable):
  PipelineConfProfiler(batch_size, matmul_amp, num_ipus, matmul_partial_type,
                       conv_amp, conv_partial_type, ipu_model,
                       profiling_root_dir, solution_path,
                       profiling_enable).apply(graph_def, signature_def)


def process_pool_initializer(env):
  if 'TF_POPLAR_FLAGS' in env:
    env.pop("TF_POPLAR_FLAGS")
  os.environ.update(env)


class PipelineConfLoader(Converter):
  # pylint:disable=line-too-long
  """Loading pipline configuration from file."""
  def __init__(self, param):
    self.param = param
    self.batch_size = param.batch_size
    self.matmul_amp = param.matmul_amp
    self.num_ipus = param.num_ipus
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
    (_, self.ipu_model, self.profiling_root_dir, self.solution_path,
     self.profiling_enable) = utils.validate_pipeline_cfg(
         param, LOADPIPELINECFG_KWDS, "loader")

  def _validate_params(self, param):
    if param.pipeline_cfg:
      if not param.embedded_runtime_save_config:
        raise ValueError(
            "The `embedded_runtime_save_config` argument must be specified for pipelined models."
        )
      if "converter" not in param.pipeline_cfg:
        raise ValueError(
            '`pipeline_cfg` must contain a value for "converter".')
      if param.pipeline_cfg[
          "converter"] == "load" and "solution_path" not in param.pipeline_cfg:
        raise ValueError(
            '`pipeline_cfg` must contain a value for "converter".')
    utils.validate_embedded_runtime_save_config(param)

  def _should_do_pipeconf_load(self):
    return utils.should_do_pipeline_wrapper(self.param.pipeline_cfg, "load",
                                            self._embedded_runtime_save_config,
                                            self._batch_per_step,
                                            self._merge_subgraphs,
                                            self._int64_to_int32_conversion)

  def _take_profile(self, graph_def, signature):
    with get_context("spawn").Pool(processes=1,
                                   initializer=process_pool_initializer,
                                   initargs=(os.environ.copy(),)) as p:
      p.apply(pipeline_profile_taker_helper,
              args=(graph_def, signature, self.batch_size, self.matmul_amp,
                    self.num_ipus, self.matmul_partial_type, self.conv_amp,
                    self.conv_partial_type, self.ipu_model,
                    self.profiling_root_dir, self.solution_path,
                    self.profiling_enable))

  def _apply(self, graph_def, signature):
    if self.param.pipeline_cfg.get("profiling_enable", False):
      self._take_profile(graph_def, signature)

    tfGraph = deserialize_graph_def(graph_def)

    auto_graph = TFv1Graph(tfGraph, signature)
    auto_graph.read_pipeline_config(self.param.pipeline_cfg["solution_path"])

    return utils.pipeline_embedded_runtime_wrapper(
        self.param, self.cfg, auto_graph, signature, self.batch_size,
        self._batch_per_step, self._poplar_exec_filepath,
        self._runtime_api_timeout_us)

  def apply(self, graph_def, signature_def):
    if not self._should_do_pipeconf_load():
      return graph_def, signature_def

    return self._apply(graph_def, signature_def)
