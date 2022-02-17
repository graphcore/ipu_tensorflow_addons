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
import copy
from itertools import chain

from tensorflow import disable_v2_behavior
from tensorflow.python.client import session
from tensorflow.python.platform import test
from tensorflow.python.framework import dtypes, importer, ops, test_util
from tensorflow.python.ops import (array_ops, math_ops, random_ops, sparse_ops,
                                   standard_ops, variable_scope)
from ipu_tensorflow_addons.saved_model_tool.converter import \
    PrecisionConversion
from ipu_tensorflow_addons.saved_model_tool.ipu_convert import \
    IpuConversionParams
from ipu_tensorflow_addons.saved_model_tool.saved_model_test_utils import \
    ModelForTest, declare_signature

disable_v2_behavior()


class PrecisionTestSavedModel(ModelForTest):
  @declare_signature(input_name_keys=["x"], output_name_keys=["y"])
  def create(self):
    x = array_ops.placeholder(dtypes.float32, [8, 8], name="x")
    label = array_ops.expand_dims(array_ops.constant([0, 1, 2, 3, 4, 6, 7, 9]),
                                  1)
    index = array_ops.expand_dims(math_ops.range(0, 8), 1)
    concated = array_ops.concat([index, label], 1)
    onehot_labels = sparse_ops.sparse_to_dense(concated, [8, 8], 1.0, 0.0)
    w0 = variable_scope.get_variable("w0", shape=[8, 8], dtype=dtypes.float32)
    d = random_ops.random_normal([8, 8],
                                 mean=0.5,
                                 stddev=1,
                                 dtype=dtypes.float16)
    x = math_ops.matmul(w0, x)
    x = math_ops.cast(math_ops.add(x, onehot_labels), dtype=dtypes.float16)
    x = math_ops.cast(math_ops.matmul(d, x), dtype=dtypes.float16)
    w1 = variable_scope.get_variable("w1", shape=[8, 8], dtype=dtypes.float16)
    x = math_ops.cast(math_ops.matmul(w1, x), dtype=dtypes.float32)
    x = array_ops.gather(x, [1, 2, 3, 4])
    y = standard_ops.reduce_sum(x)
    return y


class PrecisionConversionTestCase(test_util.TensorFlowTestCase):
  def setUp(self):
    self.model = PrecisionTestSavedModel(freeze=True)
    self._graph_def, self._signature_def = (self.model.graph_def,
                                            self.model.signature_def)

  def test_fp32_to_fp16(self):
    graph_def = copy.deepcopy(self._graph_def)
    signature_def = copy.deepcopy(self._signature_def)

    params = IpuConversionParams(precision_mode='FP16')
    graph_def, _ = PrecisionConversion(params).apply(graph_def, signature_def)

    with session.Session(graph=ops.Graph()):
      importer.import_graph_def(self._graph_def, name="")
      tensor_list = {
          tensor
          for op in ops.get_default_graph().get_operations()
          for tensor in op.values()
      }
      input_and_output_tensor_names = [
          i.name for i in chain(self._signature_def.inputs.values(),
                                self._signature_def.outputs.values())
      ]
      FP32_tensor_name = [
          tensor.name for tensor in tensor_list
          if tensor.dtype == dtypes.float32
          and tensor.name not in input_and_output_tensor_names
      ]

    with session.Session(graph=ops.Graph()):
      importer.import_graph_def(graph_def, name="")
      FP16_tensor_dict = {
          tensor.name: tensor
          for op in ops.get_default_graph().get_operations()
          for tensor in op.values()
      }

    flag = all((FP16_tensor_dict[name].dtype == dtypes.float16
                for name in FP32_tensor_name))
    self.assertGreater(len(FP16_tensor_dict), 0)
    self.assertGreater(len(FP32_tensor_name), 0)
    self.assertIsNotNone(graph_def)
    self.assertGreater(len(graph_def.node), 0)
    self.assertTrue(flag)

  def test_precision_conversion_excluded_nodes(self):
    graph_def = copy.deepcopy(self._graph_def)
    signature_def = copy.deepcopy(self._signature_def)

    precision_conversion_excluded_nodes = ['MatMul', '^Sum']
    params = IpuConversionParams(
        precision_mode='FP16',
        precision_conversion_excluded_nodes=precision_conversion_excluded_nodes
    )
    graph_def, _ = PrecisionConversion(params).apply(graph_def, signature_def)

    with ops.Graph().as_default():
      importer.import_graph_def(graph_def, name="")

    # After conversion, the node type in precision_conversion_excluded_nodes should be float
    # and the other nodes should be half-float
    node_dict = {node.name: node for node in graph_def.node}
    self.assertEqual(node_dict['MatMul'].attr['T'].type, dtypes.float32)
    self.assertEqual(node_dict['MatMul_1'].attr['T'].type, dtypes.float16)
    self.assertEqual(node_dict['Sum'].attr['T'].type, dtypes.float32)
    self.assertEqual(node_dict['Sum'].attr['Tidx'].type, dtypes.int32)
    self.assertEqual(node_dict['GatherV2'].attr['Tparams'].type,
                     dtypes.float16)
    self.assertEqual(node_dict['SparseToDense'].attr['Tindices'].type,
                     dtypes.int32)
    self.assertTrue('MatMul/CastInsertion_0' in node_dict)
    self.assertTrue('Sum/CastInsertion_0' in node_dict)


if __name__ == '__main__':
  test.main()
