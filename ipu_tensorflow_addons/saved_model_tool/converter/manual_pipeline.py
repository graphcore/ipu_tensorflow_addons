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
# pylint: disable=line-too-long
"""
Add Manual pipeline Converter cutting a model with user-specified regular expressions.
"""
# pylint: enable=line-too-long
import os
from multiprocessing import get_context
from collections import OrderedDict

from tensorflow.python import ipu
from ipu_tensorflow_addons.saved_model_tool.converter import Converter
from ipu_tensorflow_addons.saved_model_tool.converter import utils

from ipu_tensorflow_addons.saved_model_tool.converter.autograph import RunConfig, TFv1Experiment, TFv1Graph
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.strategies import ManualPipelineStrategy
from ipu_tensorflow_addons.saved_model_tool.converter.utils import deserialize_graph_def

MANUALPIPELINECFG_KWDS = OrderedDict({
    "converter": "manual",
    "ipu_model": True,
    "profiling_root_dir": "profiling",
    "solution_dir": "solution",
    "manual_pipeline_config": [],
    "device_info": [],
    "profiling_enable": True,
})


class ManualPipelineImpl():
  def __init__(self, batch_size, matmul_amp, num_ipus, matmul_partial_type,
               conv_amp, conv_partial_type, ipu_model, profiling_root_dir,
               solution_dir, manual_pipeline_config, device_info,
               profiling_enable):
    # pylint:disable=line-too-long
    """The implementation class for ManualPipeline Converter"""

    self.batch_size = batch_size
    self.matmul_amp = matmul_amp
    self.num_ipus = num_ipus
    self.matmul_partial_type = matmul_partial_type
    self.conv_amp = conv_amp
    self.conv_partial_type = conv_partial_type
    self.code_prefix = 'manual_pipeline'

    self.ipu_model = ipu_model
    self.profiling_root_dir = profiling_root_dir
    self.solution_dir = solution_dir
    self.manual_pipeline_config = manual_pipeline_config
    self.device_info = device_info
    self.profiling_enable = profiling_enable

  def _run_with_pipeline_config(self, graph, run_config, strategy):
    profiling_path = os.path.join(self.profiling_root_dir,
                                  (f"{self.code_prefix}"
                                   f"-{run_config.num_required_ipus}-"
                                   f"b{self.batch_size}"))
    exper = TFv1Experiment(run_config=run_config,
                           name=f"{self.code_prefix}-from-pipeline-config",
                           profiling_path=profiling_path)

    os.makedirs(self.solution_dir, exist_ok=True)
    strategy.chop(graph, self.manual_pipeline_config)
    if self.device_info:
      graph.set_pipeline_device_info(self.device_info)
    pipeline_config_file = os.path.join(self.solution_dir, "manual.pipeconfig")
    graph.save_pipeline_config(pipeline_config_file)
    if self.profiling_enable:
      exper.run(graph)
    return pipeline_config_file

  def _apply(self, graph_def, signature):
    """The execution code for the converter pipeline.
    Args:
      graph_def: The input frozen TensorFlow graph.
      signature_def: The input frozen TensorFlow meta graph.
    Returns:
      graph_def: The frozen TensorFlow graph after conversion.
      signature_def: The frozen TensorFlow meta graph after conversion.
    """

    self.tfgraph = deserialize_graph_def(graph_def)
    cfg = RunConfig(
        self.num_ipus,
        num_pipeline_stages=len(self.manual_pipeline_config),
        batch_size=self.batch_size,
        matmul_amp=self.matmul_amp,
        matmul_partial_type=self.matmul_partial_type,
        conv_amp=self.conv_amp,
        conv_partial_type=self.conv_partial_type,
        ipu_model=self.ipu_model,
    )
    g = TFv1Graph(self.tfgraph, signature_def=signature)

    pipeline_config_file = self._run_with_pipeline_config(
        g, cfg, ManualPipelineStrategy(cfg))
    return pipeline_config_file

  def apply(self, graph_def, signature):
    return self._apply(graph_def, signature)


def manual_pipeline_helper(graph_def, signature_def, batch_size, matmul_amp,
                           num_ipus, matmul_partial_type, conv_amp,
                           conv_partial_type, ipu_model, profiling_root_dir,
                           solution_dir, manual_pipeline_config, device_info,
                           profiling_enable):
  pipeline_config_file = ManualPipelineImpl(
      batch_size, matmul_amp, num_ipus, matmul_partial_type, conv_amp,
      conv_partial_type, ipu_model, profiling_root_dir, solution_dir,
      manual_pipeline_config, device_info,
      profiling_enable).apply(graph_def, signature_def)
  return pipeline_config_file


def process_pool_initializer(env):
  if 'TF_POPLAR_FLAGS' in env:
    env.pop("TF_POPLAR_FLAGS")
  os.environ.update(env)


class ManualPipeline(Converter):
  # pylint:disable=line-too-long
  """
  The Manual Pipeline Converter splits the graph into pipeline stages
  based on a user-specified pipeline configuration.
  """
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
    (_, self.ipu_model, self.profiling_root_dir, self.solution_dir,
     self.manual_pipeline_config, self.device_info,
     self.profiling_enable) = utils.validate_pipeline_cfg(
         param, MANUALPIPELINECFG_KWDS, "manual")
    self._validate_manual_pipeline_config()

  def _validate_manual_pipeline_config(self):
    if not isinstance(self.manual_pipeline_config, list):
      raise TypeError(f"`manual_pipeline_config` must be a list, "
                      f"not {type(self.manual_pipeline_config)}.")

    for reg_list in self.manual_pipeline_config:
      if not isinstance(reg_list, list):
        raise TypeError(
            f"`manual_pipeline_config` must only contain lists of strings, "
            f"not {type(reg_list)}.")
      for str_type in reg_list:
        if not isinstance(str_type, str):
          raise TypeError(
              f"`manual_pipeline_config` must only contain lists of strings, "
              f"not lists of {type(str_type)}.")

  def _validate_params(self, param):
    if param.pipeline_cfg:
      if not param.embedded_runtime_save_config:
        raise ValueError(
            "The `embedded_runtime_save_config` argument must be specified for pipelined models."
        )
      if "converter" not in param.pipeline_cfg:
        raise ValueError(
            "The `converter` argument is required in `pipeline_cfg`.")
      if param.pipeline_cfg["converter"] == "manual":
        if "manual_pipeline_config" not in param.pipeline_cfg:
          raise ValueError(
              "The `manual_pipeline_config` argument is required in `pipeline_cfg`."
          )
        if "device_info" not in param.pipeline_cfg:
          raise ValueError(
              "The `device_info` argument is required in `pipeline_cfg`.")
    utils.validate_embedded_runtime_save_config(param)

  def _should_do_manual_pipeline(self):
    return utils.should_do_pipeline_wrapper(self.param.pipeline_cfg, "manual",
                                            self._embedded_runtime_save_config,
                                            self._batch_per_step,
                                            self._merge_subgraphs,
                                            self._int64_to_int32_conversion)

  def _do_manual_pipline(self, graph_def, signature):
    with get_context("spawn").Pool(processes=1,
                                   initializer=process_pool_initializer,
                                   initargs=(os.environ.copy(),)) as p:
      return p.apply(manual_pipeline_helper,
                     args=(graph_def, signature, self.batch_size,
                           self.matmul_amp, self.num_ipus,
                           self.matmul_partial_type, self.conv_amp,
                           self.conv_partial_type, self.ipu_model,
                           self.profiling_root_dir, self.solution_dir,
                           self.manual_pipeline_config, self.device_info,
                           self.profiling_enable))

  def _apply(self, graph_def, signature):
    pipeline_config_file = self._do_manual_pipline(graph_def, signature)

    tfGraph = deserialize_graph_def(graph_def)

    auto_graph = TFv1Graph(tfGraph, signature)
    auto_graph.read_pipeline_config(pipeline_config_file)

    return utils.pipeline_embedded_runtime_wrapper(
        self.param, self.cfg, auto_graph, signature, self.batch_size,
        self._batch_per_step, self._poplar_exec_filepath,
        self._runtime_api_timeout_us)

  def apply(self, graph_def, signature_def):
    if not self._should_do_manual_pipeline():
      return graph_def, signature_def

    return self._apply(graph_def, signature_def)
