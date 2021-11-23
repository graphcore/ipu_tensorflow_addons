# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
Assume Equal Across Replicas Keras layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.ipu.ops import cross_replica_ops
from tensorflow.python.keras.engine.base_layer import Layer


class AssumeEqualAcrossReplicas(Layer):
  """
  Layer for marking values as equal across replicas to try and prevent divergent
  control flow compilation errors.

  Divergent control flow describes the situation where program flow differs
  among replicas. This happens when the value of a conditional is not the same
  across all replicas. This is a problem if the conditional body requires a
  cross-replica sync, as only some replicas will reach it. If this happens,
  the execution will hang as the operation waits for all replicas to sync.

  To warn the user about this, Poplar checks for divergent control flow during
  compilation. However since the values of tensors are unknown at compilation
  time it can't be certain whether a tensor will lead to divergent control
  flow or not. `assume_equal_across_replicas` can be used to mark tensors
  which are equal across all replicas and in doing so prevents them causing
  divergency errors, if used in a conditional.

  Args:
    inplace: A bool for controlling whether or not the given tensor(s) is copied
      or operated on inplace. This is needed when using
      `AssumeEqualAcrossReplicas` with tensor slices.
  """
  def __init__(self, inplace=False, **kwargs):
    super().__init__(**kwargs)
    self.inplace = inplace

  def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
    return cross_replica_ops.assume_equal_across_replicas(inputs, self.inplace)

  def get_config(self):
    return {
        'inplace': self.inplace,
    }
