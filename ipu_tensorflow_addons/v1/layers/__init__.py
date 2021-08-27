# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from ipu_tensorflow_addons.v1.layers.rnn_ops import PopnnLSTM
from ipu_tensorflow_addons.v1.layers.rnn_ops import PopnnDynamicLSTM
from ipu_tensorflow_addons.v1.layers.rnn_ops import PopnnGRU
from ipu_tensorflow_addons.v1.layers.rnn_ops import PopnnDynamicGRU
from ipu_tensorflow_addons.v1.layers.rnn_ops import PopnnAUGRU

# Registers gradient functions for the rnn layers.
from ipu_tensorflow_addons.v1.layers import rnn_ops_grad

