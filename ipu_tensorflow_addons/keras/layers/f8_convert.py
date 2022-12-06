# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import keras
from tensorflow.python.ipu.ops.f8_ops import convert_to_f8, convert_from_f8, QuarterTensor
import tensorflow as tf


class ConvertToF8(keras.layers.Layer):
  """A wrapper layer around convert_to_f8.

  This layer expects 2 inputs: (floating point) data and metadata, and returns
  the output of convert_to_f8 wrapped in a list instead of a QuarterTensor."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def call(self, data, metadata, **kwargs):  # pylint: disable=unused-argument, arguments-differ
    """Args:
      data: A floating point tensor.
      metadata: Output of :py:func:`~tensorflow.python.ipu.ops.f8_ops.create_metadata`."""
    output = convert_to_f8(data, metadata)
    return [output.data, output.metadata]


class ConvertFromF8(keras.layers.Layer):
  """Layer to convert from fp8 to another floating point datatype."""

  def __init__(self, dtype=tf.half, **kwargs):
    """Args:
      dtype: The dtype to convert to. Anything other than tf.half will incur
        an extra cast to tf.half first."""
    super().__init__(dtype=dtype, **kwargs)

  def call(self, inputs, **kwargs):  # pylint: disable=unused-argument, arguments-differ
    """Args:
      inputs: Output of a layer that returns an f8 tensor.
        More specifically, inputs should have the form [data, metadata]."""
    inputs = QuarterTensor(*inputs)
    return convert_from_f8(inputs, dtype=self.dtype)
