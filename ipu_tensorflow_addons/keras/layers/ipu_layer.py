# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
Base IPU Keras layer
~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.keras.engine.base_layer import Layer


class IPULayer(Layer):
  def _check_unsupported(self, arg, arg_name, method="__init__"):
    if arg:
      raise NotImplementedError(
          "ipu.keras.%s does not support %s"
          " argument %s. It is included for API consistency"
          "with keras.%s." %
          (self.__class__.__name__, method, arg_name, self.__class__.__name__))
