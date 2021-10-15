# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
A collection of addons for IPU Tensorflow
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
