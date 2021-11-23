# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
Keras layers made for IPU TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from ipu_tensorflow_addons.keras.layers.assume_equal_across_replicas import AssumeEqualAcrossReplicas
from ipu_tensorflow_addons.keras.layers.ctc import CTCInferenceLayer, CTCLoss, CTCPredictionsLayer
from ipu_tensorflow_addons.keras.layers.dense import SerialDense
from ipu_tensorflow_addons.keras.layers.dropout import Dropout
from ipu_tensorflow_addons.keras.layers.effective_transformer import EffectiveTransformer
from ipu_tensorflow_addons.keras.layers.embedding_lookup import Embedding
from ipu_tensorflow_addons.keras.layers.normalization import GroupNorm, InstanceNorm, LayerNorm
from ipu_tensorflow_addons.keras.layers.normalization import GroupNormalization, InstanceNormalization, LayerNormalization
from ipu_tensorflow_addons.keras.layers.recomputation import RecomputationCheckpoint
from ipu_tensorflow_addons.keras.layers.rnn import PopnnLSTM, LSTM
from ipu_tensorflow_addons.keras.layers.rnn import PopnnGRU, GRU
