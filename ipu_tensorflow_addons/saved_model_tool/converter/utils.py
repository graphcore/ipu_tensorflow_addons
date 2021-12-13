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
import tensorflow.compat.v1 as tf
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
      tf.float32: types_pb2.DT_FLOAT,
      tf.int32: types_pb2.DT_INT32,
      tf.float16: types_pb2.DT_HALF,
      tf.int64: types_pb2.DT_INT64,
      tf.bool: types_pb2.DT_BOOL
  }
  return TFTYPE_2_DTYPE[tf_type]


def deserialize_graph_def(graph_def):
  with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name="")
    return sess.graph


def tf_type_to_numpy(dtype):
  DTYPE_2_NP = {
      tf.float32: np.float32,
      tf.int32: np.int32,
      tf.float16: np.float16,
      tf.int64: np.int64,
      tf.bool: np.bool
  }
  return DTYPE_2_NP[dtype]


def np_type_to_tf_type(dtype):
  DTYPE_2_TF = {
      np.float32: tf.float32,
      np.int32: tf.int32,
      np.float16: tf.float16,
      np.int64: tf.int64,
      np.bool: tf.bool
  }
  return DTYPE_2_TF[dtype]
