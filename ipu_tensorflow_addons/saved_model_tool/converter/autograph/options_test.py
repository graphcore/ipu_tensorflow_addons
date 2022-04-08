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
import os
import shutil
import tempfile
import unittest
import pickle

from ipu_tensorflow_addons.saved_model_tool.converter.autograph.options import RunConfig


class RunConfigTestCase(unittest.TestCase):
  def setUp(self):
    self.params = {
        "num_required_ipus": 1,
        "num_pipeline_stages": 2,
        "gradient_accumulation_count": 10,
        "batch_size": 1,
        "batch_per_step": 2,
        "ipu_model": True,
        "matmul_partial_type": "half",
        "matmul_amp": 0.6,
        "conv_amp": "half",
        "conv_partial_type": 0.6
    }
    self.config = RunConfig(**self.params)
    self.temp_dir = tempfile.mkdtemp()

  def test_to_json(self):
    cfg_path = os.path.join(self.temp_dir, "test.json")
    self.config.to_json(cfg_path)

    with open(cfg_path) as f:
      cfg_dict = json.load(f)
    self.assertDictEqual(cfg_dict["RunConfig"], self.params)

  def test_from_json(self):
    cfg_path = os.path.join(self.temp_dir, "cfg.json")
    self.config.to_json(cfg_path)

    cfg = RunConfig.from_json(cfg_path)

    self.assertEqual(str(cfg), str(self.config))

  def test_pickle(self):
    pickled_cfg = pickle.dumps(self.config)
    unpickled_cfg = pickle.loads(pickled_cfg)
    self.assertEqual(str(unpickled_cfg), str(self.config))

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
  unittest.main()
