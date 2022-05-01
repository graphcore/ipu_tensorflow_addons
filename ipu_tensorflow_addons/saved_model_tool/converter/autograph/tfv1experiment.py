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

import os
import tempfile
from datetime import datetime
import json
import logging
import warnings
from typing import Callable, List
from multiprocessing import cpu_count

import numpy as np
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import ipu
from tensorflow.python.ops import variables
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import ops
from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.ipu import ipu_compiler, pipelining_ops

from ipu_tensorflow_addons.saved_model_tool.converter.utils import tf_type_to_numpy
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.utils import str_type_to_tf_type
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.utils import configure_ipu
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.tfv1graph import Graph
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.profile import ProfileAnalyzer
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.options import RunConfig
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.utils import frame_info

MAX_NUM_COMPILATION_THREAD = 90


class Experiment():
  def __init__(self,
               run_config: RunConfig = None,
               purpose="pipeline",
               profiling_path: str = '',
               training=False):
    # pylint: disable=line-too-long
    """Simulate ML graph with IPU mode and extract the information from report.

    Args:
      num_compilation_thread (int, optional): the number of process. Defaults to 1. 0 means decided by IPU resources.
    """
    if purpose.lower() not in ("shard", "pipeline"):
      raise ValueError(
          'Unsupported purpose, value should be "shard" or "pipeline".')
    self.purpose = purpose.lower()
    self.profiling_path = self._profile_path(profiling_path)
    self.profile_analysis = ProfileAnalyzer(self.profiling_path)
    self.training = training
    self.run_config = run_config

  def _profile_path(self, path: str = ''):
    prefix = f"""profile-{datetime.now().strftime("%H-%M-%S")}-"""
    os.makedirs(path, exist_ok=True)
    profile_path = tempfile.mkdtemp(prefix=prefix, dir=path)

    return profile_path

  def initialize(self):
    """Initialize the environment of device.
    Args:
      run_config (IPUConfig): the IPU configuration.
    """
    raise NotImplementedError()

  def disengage(self):
    raise NotImplementedError()

  def shard(self, graph: Graph):
    """Split graph into several pieces and make them pipeline or sharding model.
    Args:
      graph (Graph): graph with node attribute "virtualID" and "pipeline_stage".

    Return:
      List of tensorflow pipeline stage function or sharded model
    """
    raise NotImplementedError()

  def run(self, graph: Graph):
    """run the Simulator"""
    raise NotImplementedError()


class TFv1Experiment(Experiment):
  """The experiment is the trial for running the model with IPU-target or with IPU model.

    For example:
    >>> g = ag.TFv1Graph(graph)
    >>> cfg = ag.RunConfig(num_ipus, batch_size=batch_size,
                       ipu_model=ipu_model)
    >>> exper.run(g)
  """
  def __init__(self,
               purpose: str = "pipeline",
               run_config: RunConfig = None,
               profiling: bool = True,
               profiling_path: str = '',
               name: str = "TFv1Experiment",
               epochs: int = 1,
               compiled_model_stored_path: str = "compiled_models_exec",
               num_compilation_thread: int = 0,
               fake_data_random_range=10,
               logger: logging.Logger = None):
    super().__init__(purpose=purpose,
                     run_config=run_config,
                     profiling_path=profiling_path)
    self.name = name
    self.random_range = fake_data_random_range
    self.epochs = epochs
    self.ipu_config = None
    self.logger = logger

    if not self.run_config.ipu_model:
      self.compiled_model_stored_path = tempfile.mkdtemp(
          prefix=f"{compiled_model_stored_path}_")
    else:
      self.compiled_model_stored_path = compiled_model_stored_path

    self.num_compilation_thread = (
        num_compilation_thread if num_compilation_thread else min(
            cpu_count(), MAX_NUM_COMPILATION_THREAD))
    self.env_backup = dict(os.environ)
    self.profiling = profiling

  def _get_ipu_config(self):
    self.ipu_config = self.run_config.get_ipu_config()

  def configure_ipu_system(self):
    return configure_ipu(self.ipu_config)

  def log(self,
          message,
          level="info",
          raise_and_warn=True,
          error_type=ValueError):

    if self.logger:
      self.logger.log(getattr(logging, level.upper()),
                      f"{frame_info()} - {message}")
    else:
      print((f"[AutoGraph-tfv1-experiment]-[{datetime.now()}]-"
             f"[{level.lower()}]-[{frame_info()}]: {message}"))

    if raise_and_warn:
      if level == "error":
        raise error_type(message)

      if level == "warn":
        warnings.warn(message)

  def initialize(self):
    self.log("Initializing enviornment variables", level="debug")
    if "TF_POPLAR_FLAGS" in os.environ:
      os.environ.pop("TF_POPLAR_FLAGS")
    os.environ["TF_POPLAR_FLAGS"] = ('--max_compilation_threads='
                                     f'{self.num_compilation_thread} '
                                     '--show_progress_bar=true')
    if self.run_config.ipu_model:
      os.environ["TF_POPLAR_FLAGS"] += " --use_ipu_model"
    else:
      os.environ["TF_POPLAR_FLAGS"] += (' --executable_cache_path='
                                        f'{self.compiled_model_stored_path}')

    if self.profiling:
      opts = {
          "autoReport.directory": self.profiling_path,
          "autoReport.outputExecutionProfile": "true",
          "debug.allowOutOfMemory": "true",
          "autoReport.outputSerializedGraph": "false",
          "debug.outputAllSymbols": "true",
          "autoReport.all": "true",
          "autoReport.outputDebugInfo": "true",
      }
      os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(opts)

    self.log("Configure IPU system.", level="debug")
    self._get_ipu_config()
    self.configure_ipu_system()

  def disengage(self):
    self.log("Reset default Tensorflow Graph.", level="debug")
    ops.reset_default_graph()
    self.log("Reset IPU configuration.", level="debug")
    ipu.config.reset_ipu_configuration()
    self.log("Restore the environment variables.", level="debug")
    if self.profiling:
      os.environ.pop('POPLAR_ENGINE_OPTIONS')
    if self.env_backup.get("TF_POPLAR_FLAGS"):
      os.environ['TF_POPLAR_FLAGS'] = self.env_backup["TF_POPLAR_FLAGS"]
    else:
      os.environ.pop('TF_POPLAR_FLAGS')

  def _check_ipu_model(self):
    return ipu.utils.running_on_ipu_model()

  def _pipelined_model_with_node_info(self, graph: Graph):
    computational_stages = []
    device_mapping = []
    computational_stages, device_mapping = graph.pipelined(
        self.run_config.num_required_ipus)
    if self.run_config.num_pipeline_stages != len(computational_stages):
      self.log(message=("the length of computational_stages "
                        "should be the same as `num_pipeline_stages`"),
               level="error")

    return computational_stages, device_mapping

  def _sharded_model_with_node_info(self, graph: Graph):
    raise NotImplementedError()  # return TF model

  def shard(
      self,
      graph: Graph,
  ):
    if self.purpose == 'autoshard':
      return self._sharded_model_with_node_info(graph)
    return self._pipelined_model_with_node_info(graph)

  def _get_model_input(self, graph: Graph):
    out = []
    for inp in graph.inputs:
      shape = inp.shape
      if shape[0] is None:
        shape = tuple(shape[1:])
      out.append(
          (inp.name, shape, tf_type_to_numpy(str_type_to_tf_type(inp.dtype))))
    return out

  def _prepare_random_data(self, graph: Graph):
    inputs_shape_and_dtype = self._get_model_input(graph)
    dataset = Dataset.from_tensors(
        tuple(
            np.random.randint(self.random_range, size=shape).astype(dtype)
            for _, shape, dtype in inputs_shape_and_dtype))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=self.run_config.batch_size,
                            drop_remainder=True)
    dataset = dataset.shuffle(self.run_config.batch_size * 100)
    dataset = dataset.cache()

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    return infeed_queue

  def _tf_pipeline_ops(self, infeed_queue,
                       computational_stages: List[Callable],
                       device_mapping: List):

    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
    pipeline_op = pipelining_ops.pipeline(
        computational_stages=computational_stages,
        device_mapping=device_mapping,
        gradient_accumulation_count=self.run_config.
        gradient_accumulation_count,
        repeat_count=self.run_config.batch_per_step,
        inputs=[],
        infeed_queue=infeed_queue,
        outfeed_queue=outfeed_queue,
        name=f'{self.name}_pipeline_op')

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(lambda: pipeline_op, inputs=[])

    return r, outfeed_queue

  def _tf_session_config(self):
    sess_cfg = config_pb2.ConfigProto()
    sess_cfg.graph_options.rewrite_options.memory_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF)
    return sess_cfg

  def _exec(self, infeed_queue, *model_after_args):
    if self.purpose == "shard":
      model_after, *_ = model_after_args
      return self._exec_shard(model_after)
    if self.purpose == "pipeline":
      computational_stages, device_mapping = model_after_args
      return self._exec_pipeline(infeed_queue, computational_stages,
                                 device_mapping)
    raise ValueError("Unknown model args.")

  def _exec_shard(self, model_after):
    raise NotImplementedError()

  def _init_global_var(self, sess):
    try:
      sess.run(variables.global_variables_initializer())
    except errors.OpError as e:
      if (self.run_config.ipu_model
          and "Serialisation of executables is only supported for IPU targets"
          in e.message):
        self.log(
            "Failed to save executable as "
            "running against IPU model, "
            "please set ipu_model=False in RunConfig class",
            level="error",
            raise_and_warn=False)
      raise e

  def _exec_pipeline(self, infeed_queue, computational_stages: List[Callable],
                     device_mapping: List):

    pipeline_ops, outfeed_queue = self._tf_pipeline_ops(
        infeed_queue, computational_stages, device_mapping)

    with session.Session(config=self._tf_session_config()) as sess:
      self._init_global_var(sess)
      sess.run(infeed_queue.initializer)
      for _ in range(self.epochs):
        _ = sess.run([pipeline_ops])
        _ = sess.run(outfeed_queue.dequeue())

  def _run(self, graph: Graph):
    model_after = self.shard(graph)
    ds = self._prepare_random_data(graph)
    self._exec(ds, *model_after)

  def run(self, graph: Graph):
    """this is the runner function for experiment

    Args:
        graph (Graph): the model

    Raises:
        e: [description]
    """
    self.initialize()
    if not self._check_ipu_model():
      warnings.warn("XLA is not configured to run on the IPU model.",
                    RuntimeWarning)
    try:
      self._run(graph)
    except Exception as e:
      self.log(str(e), level="error", raise_and_warn=False)
      raise e
    finally:
      self.disengage()

  def get_profile_analysis(self, graph: Graph):
    self.profile_analysis.extract_info_to_tfv1graph(graph)
    return self.profile_analysis
