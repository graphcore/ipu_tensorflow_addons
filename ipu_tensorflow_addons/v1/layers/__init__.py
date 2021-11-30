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
TensorFlow layers made for IPU TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from ipu_tensorflow_addons.v1.layers.rnn_ops import PopnnLSTM
from ipu_tensorflow_addons.v1.layers.rnn_ops import PopnnDynamicLSTM
from ipu_tensorflow_addons.v1.layers.rnn_ops import PopnnGRU
from ipu_tensorflow_addons.v1.layers.rnn_ops import PopnnDynamicGRU
from ipu_tensorflow_addons.v1.layers.rnn_ops import PopnnAUGRU
