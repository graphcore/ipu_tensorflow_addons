# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

"""
TensorFlow layers made for IPU TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from ipu_tensorflow_addons.layers.rnn_ops import PopnnLSTM
from ipu_tensorflow_addons.layers.rnn_ops import PopnnDynamicLSTM
from ipu_tensorflow_addons.layers.rnn_ops import PopnnGRU
from ipu_tensorflow_addons.layers.rnn_ops import PopnnDynamicGRU
from ipu_tensorflow_addons.layers.rnn_ops import PopnnAUGRU
