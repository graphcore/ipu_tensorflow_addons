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
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2

import numpy as np

FLOAT_TYPE_LIST = [
    types_pb2.DT_HALF,
    types_pb2.DT_FLOAT,
]

ATTR_TYPE_LIST = [
    'dtype',
    'T',
    'Tparams',
    'DstT',
    'SrcT',
]

INPUT_NODES_TYPE_LIST = [
    'T',
    'Tparams',
    'dtype',
    'DstT',
]

NODES_TYPE_LIST = [
    'T',
    'Tparams',
    'dtype',
    'SrcT',
]


def str_to_dtype(type_name):
  STR_2_DTYPE = {
      'FP32': types_pb2.DT_FLOAT,
      'FP16': types_pb2.DT_HALF,
      'FP64': types_pb2.DT_DOUBLE
  }
  return STR_2_DTYPE[type_name]


def tf_type_to_dtype(tf_type):
  TFTYPE_2_DTYPE = {
      dtypes.float32: types_pb2.DT_FLOAT,
      dtypes.int32: types_pb2.DT_INT32,
      dtypes.float16: types_pb2.DT_HALF,
      dtypes.int64: types_pb2.DT_INT64,
      dtypes.bool: types_pb2.DT_BOOL
  }
  return TFTYPE_2_DTYPE[tf_type]


def deserialize_graph_def(graph_def):
  with session.Session(graph=ops.Graph()) as sess:
    importer.import_graph_def(graph_def, name="")
    return sess.graph


def tf_type_to_numpy(dtype):
  DTYPE_2_NP = {
      dtypes.float32: np.float32,
      dtypes.int32: np.int32,
      dtypes.float16: np.float16,
      dtypes.int64: np.int64,
      dtypes.bool: np.bool
  }
  return DTYPE_2_NP[dtype]


def np_type_to_tf_type(dtype):
  DTYPE_2_TF = {
      np.float32: dtypes.float32,
      np.int32: dtypes.int32,
      np.float16: dtypes.float16,
      np.int64: dtypes.int64,
      np.bool: dtypes.bool
  }
  return DTYPE_2_TF[dtype]


def add_ipu_scope(node, deviceinfo):
  node.device = deviceinfo
  node.attr['_XlaCompile'].CopyFrom(attr_value_pb2.AttrValue(b=True))
  node.attr['_XlaScope'].CopyFrom(
      attr_value_pb2.AttrValue(s='jit_scope_ipu_0'.encode()))
  node.attr['_XlaSeparateCompiledGradients'].CopyFrom(
      attr_value_pb2.AttrValue(b=False))


def input_name_to_node_name(name):
  return name if ":" not in name else name.split(":")[0]


def input_name_to_tensor_name(name):
  return name if ":" in name else name + ":0"


def input_name_to_placeholder_name(name):
  return name.replace(":", "_")


def tensor_name_to_placehoder_name(name):
  if ":0" in name:
    return name.split(":0")[0]
  elif ":" in name:
    return name.replace(":", "_")
  else:
    raise ValueError(f"The input name ({name}) is not a tensor name.")


def tensor_name_to_node_name(name):
  return name.split(":")[0]


def split_graph_by_device_placement(origin_graph_def):
  cpu_graph_def = graph_pb2.GraphDef()
  cpu_graph_def.versions.CopyFrom(origin_graph_def.versions)
  ipu_graph_def = graph_pb2.GraphDef()
  ipu_graph_def.versions.CopyFrom(origin_graph_def.versions)

  for node in origin_graph_def.node:
    if node.device == "/device:IPU:0":
      new_node = ipu_graph_def.node.add()
      new_node.device = "/device:IPU:0"
    else:
      new_node = cpu_graph_def.node.add()
      new_node.device = "/device:CPU:0"
    new_node.op = node.op
    new_node.name = node.name
    new_node.input.extend(node.input)
    for attr in list(node.attr.keys()):
      new_node.attr[attr].CopyFrom(node.attr[attr])
  return ipu_graph_def, cpu_graph_def


def get_edge_tensor(cpu_graph_def, ipu_graph_def):
  edge_tensor_name = list()
  cpu_nodes = {node.name: node for node in cpu_graph_def.node}
  ipu_nodes = {node.name: node for node in ipu_graph_def.node}

  # Assert the input of cpu_nodes do not appear in ipu_nodes.
  for node in cpu_nodes:
    for input_name in cpu_nodes[node].input:
      node_name = input_name_to_node_name(input_name)
      if node_name in ipu_nodes:
        raise ValueError(
            f"Please place {node_name} on CPU, a complete CPU head is required."
        )

  for ipu_node in ipu_nodes:
    for input_tensor in ipu_nodes[ipu_node].input:
      input_name = input_name_to_node_name(input_tensor)
      if input_name in cpu_nodes:
        edge_tensor_name.append(input_name_to_tensor_name(input_tensor))

  edge_tensor_name = list(set(edge_tensor_name))
  with session.Session(graph=ops.Graph()) as sess:
    importer.import_graph_def(cpu_graph_def, name="")
    edge_tensors = {
        v: sess.graph.get_tensor_by_name(v)
        for v in edge_tensor_name
    }

  return edge_tensors
