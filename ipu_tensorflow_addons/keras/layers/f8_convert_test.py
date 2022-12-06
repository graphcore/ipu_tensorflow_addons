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

from tensorflow.python.ipu.ops.f8_ops \
  import create_metadata, convert_to_f8, convert_from_f8, Format
import tensorflow as tf
from tensorflow.python.platform import googletest
from absl.testing import parameterized
import numpy as np

from ipu_tensorflow_addons.keras.layers import ConvertToF8, ConvertFromF8


class F8ConvertTest(tf.test.TestCase, parameterized.TestCase):
  """Test functionality of ipu_tensorflow_addons.keras.layers.f8_convert ops.
  """

  data = np.array([[1., 2.], [3., -1.], [-3., -5.]])
  """Sample data to be used in tests.
  """

  metadata = create_metadata(Format.F143, 0)
  """Sample metadata to be used in tests.
  """

  def test_to_f8(self):
    """Test that ConvertToF8 returns the correct result.
    """
    layer = ConvertToF8()
    output = layer(F8ConvertTest.data, F8ConvertTest.metadata)
    expected = convert_to_f8(F8ConvertTest.data, F8ConvertTest.metadata)
    self.assertAllClose(output[0], expected.data)
    self.assertAllClose(output[1], expected.metadata)

  def test_from_f8(self):
    """Test that ConvertFromF8 returns the correct result.
    """
    inputs = convert_to_f8(F8ConvertTest.data, F8ConvertTest.metadata)
    layer = ConvertFromF8()
    output = layer([inputs.data, inputs.metadata])
    expected = convert_from_f8(inputs)
    self.assertAllClose(output, expected)

  def test_from_f8_with_dtype(self):
    """Test that ConvertFromF8 returns the correct result with different dtype.
    """
    inputs = convert_to_f8(F8ConvertTest.data, F8ConvertTest.metadata)
    layer = ConvertFromF8(dtype=tf.float32)
    output = layer([inputs.data, inputs.metadata])
    expected = convert_from_f8(inputs, dtype=tf.float32)
    self.assertAllClose(output, expected)


if __name__ == '__main__':
  googletest.main()
