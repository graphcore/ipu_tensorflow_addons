# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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

import json
from ipu_tensorflow_addons.saved_model_tool.converter.autograph.utils import next_power_of_two, get_ipu_config


class RunConfig(object):
  def __init__(self,
               num_required_ipus: int = 1,
               num_pipeline_stages=None,
               gradient_accumulation_count=None,
               batch_size=1,
               batch_per_step=1,
               ipu_model=True,
               matmul_partial_type="float",
               conv_partial_type="float",
               matmul_amp=0.6,
               conv_amp=0.6):

    self.num_required_ipus = next_power_of_two(num_required_ipus)
    self.num_pipeline_stages = (num_pipeline_stages if num_pipeline_stages else
                                self.num_required_ipus)
    self.gradient_accumulation_count = (gradient_accumulation_count
                                        if gradient_accumulation_count else
                                        self.num_pipeline_stages)
    self.batch_per_step = (batch_per_step
                           if batch_per_step > self.num_pipeline_stages else
                           self.num_pipeline_stages)
    self.batch_size = batch_size
    self.ipu_model = ipu_model
    self.matmul_partial_type = matmul_partial_type
    self.matmul_amp = matmul_amp
    self.conv_amp = conv_amp
    self.conv_partial_type = conv_partial_type

  def __str__(self):
    start = "autograph RunConfig".center(50, "-") + '\n'
    end = '\n' + "-".center(50, "-") + '\n'
    template = start + "\n".join(
        [f"{k}: {v}" for k, v in self.__dict__.items()]) + end
    return template

  def to_json(self, filename):
    with open(filename, "w") as f:
      json.dump({"RunConfig": self.__dict__}, fp=f, indent=2)

  @classmethod
  def from_json(cls, filename):
    with open(filename) as f:
      config_json = json.load(f)
    return cls(**config_json["RunConfig"])

  def __getstate__(self):
    return self.__dict__

  def __setstate__(self, state):
    self.__dict__ = state.copy()

  def get_ipu_config(self):
    return get_ipu_config(self.num_required_ipus,
                          matmul_amp=self.matmul_amp,
                          matmul_partial_type=self.matmul_partial_type,
                          conv_amp=self.conv_amp,
                          conv_partial_type=self.conv_partial_type)
