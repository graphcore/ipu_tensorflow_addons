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

from ipu_tensorflow_addons.saved_model_tool.converter.converter import Converter
from ipu_tensorflow_addons.saved_model_tool.converter.ipu_compiler_wrapper import IPUCompilerWrapper
from ipu_tensorflow_addons.saved_model_tool.converter.ipu_placement import IPUPlacement
from ipu_tensorflow_addons.saved_model_tool.converter.precision_conversion import PrecisionConversion
from ipu_tensorflow_addons.saved_model_tool.converter.manual_shard import ManualSharding
from ipu_tensorflow_addons.saved_model_tool.converter.int64_to_int32_conversion import Int64Conversion
from ipu_tensorflow_addons.saved_model_tool.converter.loop_repeat_wrapper import LoopRepeatWrapper
from ipu_tensorflow_addons.saved_model_tool.converter.pipeconf_loader import PipelineConfLoader
from ipu_tensorflow_addons.saved_model_tool.converter.gelu_replacement import GeluReplacement
from ipu_tensorflow_addons.saved_model_tool.converter.converter_pipeline import ConverterPipeline
