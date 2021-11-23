# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
Recomputation Keras layers
~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ipu.ops import pipelining_ops


class RecomputationCheckpoint(Layer):
  """
  Layer for checkpointing values in a computational pipeline stage.
  When recomputation is enabled, these values will not be recomputed and they
  will be stored in memory instead.

  This layer can reduce memory liveness peaks when using recomputation if
  there are too many activations which need to be recomputed before the
  backpropagation operations can be executed.

  This layer should be used with the
  `RecomputationMode.RecomputeAndBackpropagateInterleaved` pipelining
  recomputation mode.

  Note that this layer has no effect when used with the
  `RecomputationMode.RecomputeThenBackpropagate` pipelining
  recomputation mode.
  """
  def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
    """
    Checkpoint the input tensors.

    Args:
      inputs: A tensor or a structure of tensors which should be checkpointed.

    Returns:
      A tensor or a structure of tensors which matches shape and type of
      `inputs`.
    """
    return pipelining_ops.recomputation_checkpoint(inputs, name=self.name)

  def get_config(self):
    return {}
