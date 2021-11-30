# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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
"""
A collection of addons for IPU TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

try:
  # Test if IPU TensorFlow is available.
  import tensorflow.python.ipu as ipu
  # Remove the reference we just created from this namespace.
  del ipu
except ModuleNotFoundError:
  raise ImportError(
      "Failed to import IPU TensorFlow. Make sure you have IPU TensorFlow "
      "installed, and not a different TensorFlow release.")
