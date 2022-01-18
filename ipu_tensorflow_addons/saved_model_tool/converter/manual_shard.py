# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
"""Shard the large model to multiple IPUs"""

import re

from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.ipu.sharding_utils import set_ipu_shard
from ipu_tensorflow_addons.saved_model_tool.converter import Converter


class ManualSharding(Converter):
  # pylint:disable=line-too-long
  """Parameters that are used for IPU conversion.

  Fields:
     - **num_ipus** - the number of IPUs that you would like to shard.
     - **manual_sharding** - specify regular expressions to control which nodes will be sharded.
              For example,
              the following configuration will put the nodes whose names start with "MatMul" to IPU0,
              and whose names end with "AddV2$" to IPU1:
                manual_sharding = [
                    ["^MatMul"],
                    ["AddV2$"]
                ]
  """
  def __init__(self, param):
    self._num_ipus = param.num_ipus
    self._manual_sharding = param.manual_sharding
    self._validate_shard_config()

  def apply(self, graph_def, signature_def):
    if not self._manual_sharding:  # is None or False, maps to no sharding or auto sharding
      return graph_def, signature_def
    print('Do manual sharding...')
    graph_def = self._do_manual_sharding(graph_def, signature_def)
    return graph_def, signature_def

  def _validate_shard_config(self):
    if not self._manual_sharding:
      return
    if not isinstance(self._manual_sharding, list):
      raise TypeError(
          f"manual_sharding must be a list, not {type(self._manual_sharding)}."
      )

    for reg_list in self._manual_sharding:
      if not isinstance(reg_list, list):
        raise TypeError(
            f"manual_sharding must only contain lists of strings, not {type(reg_list)}."
        )
      for str_type in reg_list:
        if not isinstance(str_type, str):
          raise TypeError(
              f"manual_sharding must only contain lists of strings, not lists of {type(str_type)}."
          )

    if self._num_ipus != len(self._manual_sharding):
      raise ValueError(
          f"The length of manual_sharding ({len(self._manual_sharding)})"
          f" should be equal to num_ipus ({self._num_ipus}).")

  def _manual_sharding_for_inference(self, output_tensors):
    output_ops = [output_ts.op for output_ts in output_tensors]

    ipu_ops = list(
        filter(lambda o: 'IPU' in o.device, output_ops[0].graph.get_operations(
        )))  # list all the IPU operations

    if not ipu_ops:
      raise ValueError("No ops placed on IPU device to shard.")

    # Note: one op may be sharded multiple times
    sharded_ops = set()
    for shard_idx, patterns in enumerate(self._manual_sharding):
      for pattern in patterns:
        for op in ipu_ops:
          if re.search(pattern, op.name):
            set_ipu_shard(op, shard_idx)
            sharded_ops.add(op)

    unsharded_ops = set(ipu_ops) - sharded_ops
    for op in unsharded_ops:
      set_ipu_shard(op, 0)

  def _do_manual_sharding(self, frozen_graph_def, signature_def):
    with ops.Graph().as_default():
      importer.import_graph_def(frozen_graph_def, name="")
      outputs = list(signature_def.outputs.values())
      imported_graph = ops.get_default_graph()
      output_tensors = [
          imported_graph.get_tensor_by_name(output.name) for output in outputs
      ]
      self._manual_sharding_for_inference(output_tensors)

      sharding_graph = ops.get_default_graph()
      sharding_graph_def = sharding_graph.as_graph_def()

      return sharding_graph_def
