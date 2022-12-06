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
Keras layers made for IPU TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from ipu_tensorflow_addons.keras.layers.assume_equal_across_replicas import AssumeEqualAcrossReplicas
from ipu_tensorflow_addons.keras.layers.ctc import CTCInferenceLayer, CTCLoss, CTCPredictionsLayer
from ipu_tensorflow_addons.keras.layers.dense import SerialDense, Dense
from ipu_tensorflow_addons.keras.layers.f8_convert import ConvertToF8, ConvertFromF8
from ipu_tensorflow_addons.keras.layers.dropout import Dropout
from ipu_tensorflow_addons.keras.layers.effective_transformer import EffectiveTransformer
from ipu_tensorflow_addons.keras.layers.embedding_lookup import Embedding
from ipu_tensorflow_addons.keras.layers.normalization import GroupNorm, InstanceNorm, LayerNorm
from ipu_tensorflow_addons.keras.layers.normalization import GroupNormalization, InstanceNormalization, LayerNormalization
from ipu_tensorflow_addons.keras.layers.recomputation import RecomputationCheckpoint
from ipu_tensorflow_addons.keras.layers.rnn import PopnnLSTM, LSTM
from ipu_tensorflow_addons.keras.layers.rnn import PopnnGRU, GRU
