# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
from ipu_tensorflow_addons.saved_model_tool.converter import IPUPlacement
from ipu_tensorflow_addons.saved_model_tool.converter import PrecisionConversion
from ipu_tensorflow_addons.saved_model_tool.converter import IPUCompilerWrapper
from ipu_tensorflow_addons.saved_model_tool.converter import ManualSharding
from ipu_tensorflow_addons.saved_model_tool.converter import Int64Conversion
from ipu_tensorflow_addons.saved_model_tool.converter import LoopRepeatWrapper


class ConverterPipeline():
  # pylint: disable=unused-argument
  def __init__(self, param, signatrue_key):
    self._converters = list()
    self._converters.append(IPUPlacement(param))
    self._converters.append(Int64Conversion(param))
    self._converters.append(PrecisionConversion(param))
    self._converters.append(ManualSharding(param))
    self._converters.append(IPUCompilerWrapper(param))
    self._converters.append(LoopRepeatWrapper(param))

  def ApplyConverters(self, graph_def, signature_def):
    for converter in self._converters:
      graph_def, signature_def = converter.apply(graph_def, signature_def)

    return graph_def, signature_def
