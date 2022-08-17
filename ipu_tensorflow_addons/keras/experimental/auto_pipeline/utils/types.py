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
"""Type definition for types used in auto pipelining"""
from typing import Dict, List
from typing_extensions import TypedDict


# Disable name check because the class definition should be in JSON style
# pylint: disable=invalid-name
class CycleInfo(TypedDict):
  """Information about the execution time of a layer."""
  forward: int
  """Number of cycles to complete prediction for a batch, excluding IO cycles.
  """
  backward: int
  """Number of cycles to complete training for a batch, excluding IO cycles."""


class MemoryBaseInfo(TypedDict):
  """Common memory breakdown for a layer or a pipeline step run on an IPU."""

  shared: Dict[str, int]
  '''This dictionary keeps information of variables that could be shared between
  layers.

  Name of a variable -> Size of the variable. The name of a variable is in the
  form of "category_number". The name of a variables identifies the variables.

  Example:
    Suppose there are two `Dense` layers with the same kernel shape, bias shape,
    and activation function. Then the second layer should reuse the vertex code
    of the first layer.

    Assume the size of vertex code for both layers is 1024 Bytes. The `shared`
    property of their profiles should both include `{"vertexCode_1":1024}`. If
    the these two layers are scheduled on the same IPU, "vertexCode_1" should
    only be counted once for memory estimation.

    Suppose there is another `Dense` layer with a different kernel shape and
    bias shape. And the size of vertex Code for this new layer is 2048 Bytes.
    The `shared` property for this layer should have `{"vertexCode_2":2048}`.
    "vertexCode_2" is always counted independent of "vertexCode_1".
  '''

  exclusive: Dict[str, int]
  '''This dictionary keeps information of variables that are never shared
  between layers.

  Category of a variable -> Size of the variable. The string key is only for
  identifying the category of the variable.

  Example:
    Suppose there are two `Dense` layers with the same kernel shape, bias shape,
    and activation function. But these two layers take different parameter
    values. In such case, these two layers should not share the parameter
    memory.

    Suppose parameter takes 4096 Bytes for each layer. Then the `exclusive`
    dictionary should include `{"param":4096}` for both layers. If a pipeline
    step includes only these two `Dense` layer, the pipeline step should have
    a parameter size of 8192 bytes.
  '''

  inputs: int
  """Size of input tensors for a layer or a range of layer, including
  carry-over tensors for later layers.
  """

  outputs: int
  """Size of output tensors for a layer or a range of layer, including
  carry-over tensors for later layers."""

  curActivation: int
  """For a training estimation/profile, this is equivalent to the
  size of activation tensor just before computing the backward pass.

  For any prediction profile, this should always be zero.
  """


class MemoryLayerInfo(MemoryBaseInfo):
  """Memory breakdown for a layer run on an IPU."""
  maxTemporary: int
  """Maximum temporary memory used when running this layer alone.

  This is equivalent to the maximum not-always-alive memory from the single
  layer profile. And this does not include carry-over tensors or activation
  tensors from previous layers.
  """

  curCarryOver: int
  """For a single-layer prediction profile, this is equivalent to the size of
  carry-over tensors, which are produced by earlier layers in the model but
  required by later layers.

  For any training profile, this should always be zero.
  """


class MemoryRangeInfo(MemoryBaseInfo):
  """Memory breakdown for a pipeline step run on an IPU."""
  maxTemporary: int
  """Maximum temporary memory used.

  This is the maximum temporary memory across all program steps. Each program
  step should include the memory of carry-over tensors from earlier layers in
  the model, or activation tensors from layers within the range.
  """

  totalMemory: int
  """Sum of memory from all categories."""


class LayerInfo(TypedDict):
  """Analyzed profile for a layer from per-layer compilation."""
  cycle: CycleInfo
  """Cycle information for a layer."""
  memory: MemoryLayerInfo
  """Memory information for a layer."""


class RangeInfo(TypedDict):
  """Analyzed profile for a range of layers from many single-layer profile or
  actual pipelined model execution."""
  cycle: CycleInfo
  """Cycle information for a range of layer."""
  memory: MemoryRangeInfo
  """Categories of memory for a range of layer."""


class ModelInfo(TypedDict):
  """Analyzed profile for a model from per-layer compilation."""
  layerName: List[str]
  """Name of layer.

  Used to identity which layers need to be in the same pipeline step, for the
  variable constraint.
  """
  layerType: List[str]
  """Type of layer.
  """
  layerInfo: List[LayerInfo]
  """Analyzed profile for """


class ConnectionInfo(TypedDict):
  """Description of the IPU connection."""
  sendGBps: int
  """Data transfer speed to the previous device in the pipeline.

  For the first IPU, this is the speed to receive data from the CPU.
  """
  sendIdleCycle: int
  """Number of idling cycles for the IPU before receiving data from the previous
  device.
  """
  recvGBps: int
  """Data transfer speed to the next device in the pipeline.

  For the last IPU, this is the speed to send data to the CPU."""
  recvIdleCycle: int
  """Number of idling cycles for the IPU before sending data to the next device.
  """


class IPUClusterInfo(TypedDict):
  """Description of the IPU cluster."""
  clock: int
  """The clock frequency of an IPU in Hz."""
  memory: int
  """The memory capacity of an IPU in MB."""
  connection: List[ConnectionInfo]
  """Data transferring speed between devices in the pipeline."""


class AutoPipelineConfiguration(TypedDict):
  """Configuration for automatic pipelining."""
  memoryProportion: float
  """Proportion of memory to use on IPUs.

  The auto pipelining algorithm finds a pipeline stage assignment, such that
  each pipeline step is estimated to fit in `memoryProportion` * IPU memory.

  This is used to increase the chance of finding a valid pipeline stage
  assignment. If we under-estimate the memory consumption of a pipeline step,
  the under-estimated part of memory may still fit in the remaining memory.

  If a model triggers an OOM (out of memory) error with the suggested pipeline
  stage assignment, a lower value should be tried. With a lower value, the auto
  pipelining algorithm may find a new pipeline stage assignment that better
  balances memory across IPUs at the cost of more execution cycles.
  """
  usePoplarEstimation: bool
  """If `True`, `auto_pipeline` will use cycle estimation from poplar for
  the running cost of a layer.

  If `False`, `auto_pipeline` will estimate running cost of a layer from the
  execution profile. This is slower than poplar's cycle estimation but it is
  likely to be more accurate.
  """
