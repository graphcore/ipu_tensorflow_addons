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
"""Helper functions for analyzing a `keras.Model` instance."""
import functools
import logging
import json
import tensorflow as tf
import keras
import numpy as np
from ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils import types


def prepend_batchsize(shape, batch_size):
  """Prepend `batch_size` to tensor `shape`, if the tensor `shape` does not
  include `batch_size` already.

  Args:
    shape: Shape of the tensor.
    batch_size: Batch size to be used with the model.

  Return:
    Shape of the tensor prepended with `batch_size`.
  """
  if shape[0] is None:
    return (batch_size,) + shape[1:]
  return shape


def tf_tensor_size(x, batch_size):
  """Returns the memory occupied by the tensor.

  Args:
    x: A `tf.TensorShape` or `tf.Tensor` object.
    batch_size: Batch size to be used with the model.

  Return:
    An integer batch size.
  """
  return functools.reduce(lambda x, y: x * y,
                          prepend_batchsize(x.shape, batch_size), x.dtype.size)


def tf_nested_map_structure_tensor(tensor_func, struct):
  """Creates a new structure by applying `tensor_func` to every keras dummy
  tensor in the struct.

  Args:
    tensor_func: Function to apply to a dummy Keras tensor.
    struct: The structure to be mapped.

  Return:
    The object returned by mapping every Keras Tensor.
  """

  # Wrap tensor_func for tf.nest.map_structure
  def all_func(x):
    if isinstance(x, keras.engine.keras_tensor.KerasTensor):
      return tensor_func(x)
    return x

  return tf.nest.map_structure(all_func, struct)


def parse_model_intermediate(model, batch_size):
  """For each layer invocation of the model, finds the following:
    - Memory occupied by intermediate tensors, used by later layers.
  invocations.
    - Memory occupied by input tensors (including intermediate tensors).
    - Memory occupied by output tensors (including intermediate tensors).

  This is used to complete estimation for not-always-alive memory and estimation
  for data transfering cycles.

  Args:
    model: A `keras.Model` created under an `IPUStrategy`.
    batch_size: Integer batch size to be used for the model.

  Return:
    A three-element tuple of `np.ndarray`.

  Example:
    The below code snippet shows how the three arrays are calculated for a model
    with a residual connection.
    .. code-block:: python
      def create_model():
        # InputLayer does not appear in pipeline_stage_assignment
        x0 = layers.Input(shape=(2048,))
        # 1st Layer
        # CarryOver []        Size: 0
        # Input     [x0]      Size: 16*2048*4
        # Output    [x1]      Size: 16*1024*4
        x1 = layers.Dense(1024)(x0)
        # 2nd Layer
        # CarryOver [x1]      Size: 16*1024*4
        # Input     [x1]      Size: 16*1024*4
        # Output    [x1,x2]   Size: 16*2048*4
        x2 = layers.Dense(1024)(x1)
        # 3rd Layer
        # CarryOver [x1]      Size: 16*1024*4
        # Input     [x1,x2]   Size: 16*2048*4
        # Output    [x1,x3]   Size: 16*2048*4
        x3 = layers.Dense(1024)(x2)  #carry x1, input x2, output x1 x3
        # 4th Layer
        # CarryOver []        Size: 0
        # Input     [x1,x3]   Size: 16*2048*4
        # Output    [x4]      Size: 16*1024*4
        x4 = x1 + x3
        # 5th Layer
        # CarryOver []        Size: 0
        # Input     [x4]      Size: 16*1024*4
        # Output    [x5]      Size: 16*16*4
        x5 = layers.Dense(16)(x4)
        return keras.Model(x0, x5)

      >>> parse_model_intermediate(model, 16)
      [array([    0, 65536, 65536,     0,     0]),
       array([131072,  65536, 131072, 131072,  65536]),
       array([ 65536, 131072, 131072,  65536,   1024])]
  """

  assignments = model.get_pipeline_stage_assignment()
  n_layer = len(assignments)
  # id -> [created by which layer, last used by which layer, size]
  tensors = {}

  def new_tensor(x, loc):
    """Initialize a record in the `tensors` dictionary, for a tensor `x`
    outputted by the `loc`-th layer.

    Arg:
      x: The tensor to be added to the `tensors` dictionary.
      loc: A integer indicating which layer outputs `x`.
    """
    tensors[id(x)] = [loc, loc, tf_tensor_size(x, batch_size)]

  def use_tensor(x, loc):
    """Update the record of tensor `x` in the `tensors` dictionary, for the
    `loc`-th layer using tensor `x` as its argument.

    Arg:
      x: The tensor to be updated.
      loc: A integer indicating which layer uses `x`.
    """

    if id(x) in tensors:
      tensors[id(x)][1] = loc
    else:
      logging.warning(("Tensor used as argument to layer %s but the tensor is"
                       "not an output from previous layers"), loc)
      logging.warning(x)

  # Initialize model.input tensor
  tf_nested_map_structure_tensor(lambda x: new_tensor(x, -1), [model.input])
  for i, assignment in enumerate(assignments):
    node_index = getattr(assignment, "node_index", 0)
    layer = assignment.layer
    node = layer.inbound_nodes[node_index]
    # layer argument
    #pylint: disable=cell-var-from-loop
    tf_nested_map_structure_tensor(lambda x: use_tensor(x, i),
                                   [node.call_args, node.call_kwargs])
    # layer output
    tf_nested_map_structure_tensor(lambda x: new_tensor(x, i),
                                   [node.output_tensors])

  memory_intermediate = np.zeros((n_layer,), dtype=np.long)
  for [created_by, last_use, size] in tensors.values():
    # if range (createdby, lastuse) is non empty
    # i.e. some layer is carrying this tensor
    if last_use - created_by > 1:
      memory_intermediate[created_by + 1] += size
      memory_intermediate[last_use] -= size
  memory_intermediate = np.cumsum(memory_intermediate)
  memory_input = np.copy(memory_intermediate)
  memory_output = np.copy(memory_intermediate)

  # Set of non-carryover input tensor for each layer.
  layer_inputs = [set() for i, _ in enumerate(assignments)]

  def add_input(x, loc):
    """Update the `loc`-th layer's input size, for the layer using tensor `x` as
    an argument.

    Arg:
      x: The tensor used by the `loc`-th layer.
      loc: A integer indicating which layer uses `x`.
    """
    if id(x) in tensors:
      [created_by, last_use, size] = tensors[id(x)]
      # First, check if layer `loc` is already carrying the tensor `x`.
      if not created_by < loc < last_use:
        # Second, check if tensor `x` is already included in the input.
        if id(x) not in layer_inputs[loc]:
          memory_input[loc] += size
          layer_inputs[loc].add(id(x))

  def add_output(x, loc):
    """Update the `loc`-th layer's output size, for the layer outputting tensor
    `x`.

    Arg:
      x: The tensor generated by the `loc`-th layer.
      loc: A integer indicating which layer outputs `x`.
    """
    memory_output[loc] += tf_tensor_size(x, batch_size)

  for i, assignment in enumerate(assignments):
    node_index = getattr(assignment, "node_index", 0)
    layer = assignment.layer
    node = layer.inbound_nodes[node_index]
    #pylint: disable=cell-var-from-loop
    tf_nested_map_structure_tensor(lambda x: add_input(x, i),
                                   [node.call_args, node.call_kwargs])
    tf_nested_map_structure_tensor(lambda x: add_output(x, i),
                                   [node.output_tensors])

  return memory_intermediate, memory_input, memory_output


def tf_nested_map_structure_all_tensor(tensor_func, struct):
  """Creates a new structure by applying `tensor_func` to every `tf.Tensor` or
  `tf.TensorShape` in the struct.

  The difference between this function and `tf_nested_map_structure_tensor` is
  that `tf_nested_map_structure_tensor` maps keras dummy tensors only.

  Args:
    tensor_func: Function to apply to a tensor.
    struct: The structure to be mapped.

  Return:
    The object returned by mapping every tensor.
  """

  # Wrap tensor_func for tf.nest.map_structure
  def all_func(x):
    if hasattr(x, "shape") and hasattr(x, "dtype"):
      return tensor_func(x)
    return x

  return tf.nest.map_structure(all_func, struct)


def get_assignment_layer_type(assignment, batch_size):
  """Create a string for identifying an invocation of a layer.

  This function should return an equal string for two layer invocations only
  when the computation of the two layers is equal, but possibly with different
  parameters or arguments.

  Args:
    assignment: A `PipelineStageAssignment` object, which includes the layer and
    call arguments of this layer invocation.
    batch_size: Batch size to be used with the model.

  Return:
    A string identifying the invocation.
  """
  layer = assignment.layer
  node_index = getattr(assignment, "node_index", 0)
  node = layer.inbound_nodes[node_index]

  layer_prop = {
      # Layer class name
      "class": [layer.__class__.__module__, layer.__class__.__name__],
      # Layer argument shape and dtype
      "args":
      tf_nested_map_structure_all_tensor(
          lambda x: str([prepend_batchsize(x.shape, batch_size), x.dtype]),
          [node.call_args, node.call_kwargs]),
      # weight shape and dtype
      "weight": [str([x.shape, x.dtype]) for x in layer.get_weights()],
      # Config
      "config": {
          k: str(v)
          for k, v in layer.get_config().items()
          # These keys do not affect computation of a layer.
          if not any(s in k for s in ["trainable", "initializer", "name"])
          # A float value is usually a constant value.
          and not isinstance(v, float)
      }
  }
  return json.dumps(layer_prop, sort_keys=True)
