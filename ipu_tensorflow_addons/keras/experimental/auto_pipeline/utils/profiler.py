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
"""Helper functions for profiling a single layer"""
import tensorflow as tf
from tensorflow.python import ipu
from ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils import (
    keras_utils)


def create_strategy(n_ipu, report_helper, compile_only=True):
  """Configures IPUs and creates an IPU Strategy.

  Arg:
    n_ipu: Number of IPUs to configure.
    report_helper: A Report helper
      (`tensorflow.python.ipu.test_utils.ReportHelper`).
    compile_only: If `True`, models under this strategy will only be compiled to
      provide a compilation profile.

  Return:
    An `IPUStrategy`.
  """
  cfg = ipu.config.IPUConfig()
  cfg.auto_select_ipus = n_ipu
  cfg.device_connection.version = "ipu2"

  if compile_only:
    cfg.device_connection.type = ipu.config.DeviceConnectionType.PRE_COMPILE
    cfg.device_connection.enable_remote_buffers = True
  else:
    cfg.device_connection.type = ipu.config.DeviceConnectionType.ALWAYS

  report_helper.set_autoreport_options(cfg, output_execution_profile=True)
  cfg.compilation_poplar_options["debug.allowOutOfMemory"] = "true"
  cfg.compilation_poplar_options["profiler.includeCycleEstimates"] = "true"
  cfg.configure_ipu_system()

  return ipu.ipu_strategy.IPUStrategyV1()


def profile_layer_from_assignment(assignment, batch_size, strategy,
                                  report_helper):
  """Profile a `layer.call` on IPU by creating a `tf.function`.

  Args:
    assignment: A `PipelineStageAssignment` object, which includes the layer and
    call arguments of this layer invocation.
    batch_size: Batch size to be used with the model.
    strategy: `IPUStrategy` the layer will run under.
    report_helper: A report helper used to find the PVA report.

  Return:
    A string. Directory to layer profile.
  """
  # tensorShape identity -> random generated tensor of same shape and dtype
  input_dict = {}
  layer = assignment.layer
  node_index = getattr(assignment, "node_index", 0)
  node = node = layer.inbound_nodes[node_index]

  def create_input_dict(x):
    shape = keras_utils.prepend_batchsize(x.shape, batch_size)
    input_dict[id(x)] = tf.zeros(shape, x.dtype)

  @tf.function
  def layer_profile_function(input_dict):
    """The `tf.function` wrapping the given layer. If this layer invocation uses
    a tensor multiple times, then this wrapping function also supplies the same
    tensor for these arguments. This is achieved by using a dictionary, from
    identity of the tensorShape object to generated tensor value.

    Args:
      input_dict: input_dict generated outside the function. So tensor argument
      for the layer invocation will be copied from CPU to IPU.
    """
    # Replace tensorShape object in args with tensors from input_dict
    [call_args, call_kwargs] = keras_utils.tf_nested_map_structure_tensor(
        lambda x: input_dict[id(x)], [node.call_args, node.call_kwargs])

    # Call layer with generated data
    return layer(*call_args, **call_kwargs)

  # Create a tensor with random value, for every tensorShape in args
  keras_utils.tf_nested_map_structure_tensor(
      create_input_dict, [node.call_args, node.call_kwargs])

  # Compile/Run the wrapping function on IPU
  strategy.run(layer_profile_function, args=[input_dict])
  # TODO(T67704): Auto Pipeline - Parallel compilation of layers
  # Currently we use the latest profile from the report helper, which works in
  # sequential compilation. But this may not return the correct profile
  # with parallel compilation.
  pva_pop = report_helper.find_reports()[-1]

  return pva_pop
