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
Add infeeds/outfeeds and loop.repeat for graph
"""
from tensorflow.python.ipu import ipu_infeed_queue, ipu_outfeed_queue, loops
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python import ipu
from ipu_tensorflow_addons.saved_model_tool.converter import Converter
from ipu_tensorflow_addons.saved_model_tool.converter import utils


class LoopRepeatWrapper(Converter):
  # pylint:disable=line-too-long
  """Add infeed/outfeed queue and `loop.repeat` for non-pipeline IPU model.

  Fields:
      - **batch_size** - The batch size to use for the embedded application runtime compilation.
      - **num_ipus** - The number of IPUs to use for the embedded application runtime compilation.
      - **batch_per_step** - The repeat count for the `loop.repeat` or `repeat_count` of `pipelining_ops.pipeline`.
                             If 0, it will not turn off the loop repeat IPU wrapper.
                             If embedded application runtime is enabled and batch_per_step is 0, it will be changed to 1.
      - **embedded_runtime_save_config** - A dictionary of embedded application runtime compilation configuration values.
          It must contain the following items:
            'embedded_runtime_exec_cachedir': The cache output directory of the embedded application runtime.
            'runtime_api_timeout_us': The timeout for the embedded application.

          For example:
          embedded_runtime_save_config = {
            "embedded_runtime_exec_cachedir": "cache_dir",
            "runtime_api_timeout_us": 5000,
          }

          For more information, see [embedded_application_runtime](https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/tensorflow/embedded_application_runtime.html)
  """
  def __init__(self, param):
    self.batch_size = param.batch_size
    self._batch_per_step = param.batch_per_step
    self._num_ipus = param.num_ipus
    self._excluded_nodes = param.excluded_nodes
    self._remove_excluded_nodes = param.remove_excluded_nodes
    self._embedded_runtime_save_config = param.embedded_runtime_save_config
    self._merge_subgraphs = param.merge_subgraphs
    self._int64_to_int32_conversion = param.int64_to_int32_conversion
    self._validate_params(param)

    self.cfg = ipu.config.IPUConfig()
    utils.extract_ipu_config_from_param(param, self.cfg)

    if param.embedded_runtime_save_config:
      (self._embedded_runtime_exec_cachedir, self._poplar_exec_filepath,
       self._runtime_api_timeout_us,
       self._batch_per_step) = utils.extract_emb_setting_from_param(param)

  def _validate_params(self, param):
    utils.validate_embedded_runtime_save_config(param)

  def _should_do_loop_repeat_ipu_wrapper(self):
    return utils.should_do_embedded_runtime(self._embedded_runtime_save_config,
                                            self._batch_per_step,
                                            self._merge_subgraphs,
                                            self._int64_to_int32_conversion)

  def _apply(self, graph_def, signature):
    with ops.Graph().as_default() as tfGraph:
      importer.import_graph_def(graph_def, name="")

    with ops.Graph().as_default():
      inputs_shape_and_dtype = utils.input_placeholder_name_shape_dtype(
          tfGraph, signature)
      outputs_shape_and_dtype = utils.output_placeholder_name_shape_dtype(
          tfGraph, signature)

      dataset = utils.prepare_dataset(inputs_shape_and_dtype,
                                      batch_size=self.batch_size)

      input_names = [inp_name for inp_name, _, _ in inputs_shape_and_dtype]
      output_names = [outp_name for outp_name, _, _ in outputs_shape_and_dtype]
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def ipu_functions_wrapper(*input_tensors):
        input_map = dict(zip(input_names, input_tensors))
        output = importer.import_graph_def(graph_def,
                                           name="",
                                           input_map=input_map,
                                           return_elements=output_names)

        outfeed = outfeed_queue.enqueue(output)
        return outfeed

      def compile_func():
        r = loops.repeat(self._batch_per_step, ipu_functions_wrapper, [],
                         infeed_queue)
        return r

    new_graph_def, signature = utils.embedded_application_runtime_save(
        signature, inputs_shape_and_dtype, outputs_shape_and_dtype,
        self._poplar_exec_filepath, self._runtime_api_timeout_us, compile_func,
        self.batch_size)

    return new_graph_def, signature

  def apply(self, graph_def, signature_def):
    if not self._should_do_loop_repeat_ipu_wrapper():
      return graph_def, signature_def
    self.cfg.configure_ipu_system()
    return self._apply(graph_def, signature_def)
