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
from tensorflow import as_dtype
from tensorflow.python.platform import gfile
from tensorflow.python import ops
from tensorflow.python.client import session
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import loader


def convert_graph_def_to_graph(graph_def):
  with ops.Graph().as_default() as graph:
    import_graph_def(graph_def, name="")
  return graph


def load_tf_graph(model_path, with_meta_graph=True, tag=tag_constants.SERVING):
  if with_meta_graph:
    with session.Session(graph=ops.Graph()) as sess:
      meta_graph = loader.load(sess, [tag] if tag else [], model_path)
      graph = ops.get_default_graph()
      return graph, meta_graph
  else:
    with gfile.GFile(model_path, "rb") as f:
      graph_def = ops.GraphDef()
      graph_def.ParseFromString(f.read())
    graph = convert_graph_def_to_graph(graph_def)
    return graph, None


def import_from_graph(tf_graph: ops.Graph):
  with ops.Graph().as_default() as copied_graph:
    graph_def = tf_graph.as_graph_def(add_shapes=True)
    import_graph_def(graph_def, name="")
  return copied_graph


def analyze_pb_inputs_outputs(graph):
  operations = graph.get_operations()
  outputs_set = set(operations)
  inputs = []
  for op in operations:
    if not op.inputs and op.type != 'Const':
      inputs.append(op)
    else:
      for input_tensor in op.inputs:
        if input_tensor.op in outputs_set:
          outputs_set.remove(input_tensor.op)
  outputs = list(outputs_set)

  inputs.sort(key=lambda x: x.name)
  outputs.sort(key=lambda x: x.name)

  return inputs, outputs


def tf_type_to_str_type(tf_type):
  return repr(tf_type).split(".")[-1]


def str_type_to_tf_type(str_type):
  return as_dtype(str_type)
