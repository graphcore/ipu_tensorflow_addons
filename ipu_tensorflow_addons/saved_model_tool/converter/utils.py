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
import os
from uuid import uuid4
from collections import OrderedDict

import numpy as np
from tensorflow import as_dtype
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2, rewriter_config_pb2
from tensorflow.python import ipu, ops
from tensorflow.python.client import session
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import importer
from tensorflow.core.framework import attr_value_pb2, graph_pb2
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.ipu import ipu_infeed_queue, ipu_outfeed_queue

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

DEFAULT_BATCH_PER_STEP = 1
DEFAULT_EMBEDDED_RUNTIME_CACHEDIR = "poplar_exec"
DEFAULT_RUNTIME_API_TIMEOUT_US = 5000


def str_to_dtype(type_name):
  STR_2_DTYPE = {
      'FP32': types_pb2.DT_FLOAT,
      'FP16': types_pb2.DT_HALF,
      'FP64': types_pb2.DT_DOUBLE
  }
  return STR_2_DTYPE[type_name]


def tf_type_to_dtype(tf_type):
  return tf_type.as_datatype_enum


def deserialize_graph_def(graph_def):
  with session.Session(graph=ops.Graph()) as sess:
    importer.import_graph_def(graph_def, name="")
    return sess.graph


def tf_type_to_numpy(tf_dtype):
  return tf_dtype.as_numpy_dtype


def np_type_to_tf_type(dtype):
  return as_dtype(dtype)


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
  if ":" in name:
    return name.replace(":", "_")

  raise ValueError(f"The input name ({name}) is not a tensor name.")


def node_name_from_tensor_name(tensor_name):
  if ":" in tensor_name:
    tensor_name = tensor_name.split(":")[0]
  if "^" in tensor_name:
    tensor_name = tensor_name.replace('^', '')
  return tensor_name


def tensor_name_from_node_name(node_name, index=0):
  return f"{node_name}:{index}"


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


def get_edge_tensor(cpu_graph_def, ipu_graph_def, is_remove_excluded_nodes):
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
    for idx, input_tensor in enumerate(ipu_nodes[ipu_node].input):
      input_name = input_name_to_node_name(input_tensor)
      if input_name in cpu_nodes:
        edge_tensor_name.append(input_name_to_tensor_name(input_tensor))
        if is_remove_excluded_nodes:
          placehoder_name = input_name_to_placeholder_name(input_tensor)
          ipu_nodes[ipu_node].input[idx] = placehoder_name + ":0"

  edge_tensor_name = list(set(edge_tensor_name))
  with session.Session(graph=ops.Graph()) as sess:
    importer.import_graph_def(cpu_graph_def, name="")
    edge_tensors = {
        v: sess.graph.get_tensor_by_name(v)
        for v in edge_tensor_name
    }

  return edge_tensors


def get_tensor_shape(tensor):
  return tensor.get_shape().as_list()


def casted_input_from_signature(inputs_shape_and_dtype,
                                signature,
                                batch_size=0):
  inputs_placeholders = []
  input_names_2_type_from_sig = {
      i.name: i.dtype
      for i in signature.inputs.values()
  }
  if batch_size:
    inputs_shape_and_dtype_ = [(name, tuple([
        batch_size,
    ] + shape[1:]), dtype) for name, shape, dtype in inputs_shape_and_dtype]
  else:
    inputs_shape_and_dtype_ = inputs_shape_and_dtype
  for name, shape, dtype in inputs_shape_and_dtype_:
    if dtype != input_names_2_type_from_sig[name]:
      inputs_placeholders.append(
          math_ops.cast(
              array_ops.placeholder(
                  as_dtype(input_names_2_type_from_sig[name]),
                  shape=shape,
                  name=node_name_from_tensor_name(name)), dtype))
    else:
      inputs_placeholders.append(
          array_ops.placeholder(as_dtype(input_names_2_type_from_sig[name]),
                                shape=shape,
                                name=node_name_from_tensor_name(name)))
  return inputs_placeholders


def casted_output_from_signature(output_tensors, outputs_shape_and_dtype,
                                 signature):
  output_placeholders = []
  output_names_2_type_from_sig = {
      i.name: i.dtype
      for i in signature.outputs.values()
  }

  for out_tensor, (name, _, dtype) in zip(output_tensors,
                                          outputs_shape_and_dtype):
    if dtype != output_names_2_type_from_sig[name]:
      output_placeholders.append(
          math_ops.cast(out_tensor,
                        as_dtype(output_names_2_type_from_sig[name]),
                        name=node_name_from_tensor_name(name)))
    else:
      output_placeholders.append(
          array_ops.identity(out_tensor,
                             name=node_name_from_tensor_name(name)))

  return output_placeholders


def input_placeholder_name_shape_dtype(graph, signature):
  input_ph = [
      graph.get_tensor_by_name(i.name) for i in signature.inputs.values()
  ]
  return sorted([(iph.name, get_tensor_shape(iph), iph.dtype)
                 for iph in input_ph])


def output_placeholder_name_shape_dtype(graph, signature):
  output_ph = [
      graph.get_tensor_by_name(i.name) for i in signature.outputs.values()
  ]
  return sorted([(oph.name, get_tensor_shape(oph), oph.dtype)
                 for oph in output_ph])


def prepare_dataset(inputs_shape_and_dtype, batch_size=0):
  if not batch_size:
    inputs_shape_and_dtype_for_dataset = inputs_shape_and_dtype.copy()
  else:
    inputs_shape_and_dtype_for_dataset = [
        (name, shape[1:], dtype)
        for name, shape, dtype in inputs_shape_and_dtype
    ]
  dataset = Dataset.from_tensors(
      tuple(
          np.random.randint(10, size=shape).astype(tf_type_to_numpy(dtype))
          for _, shape, dtype in inputs_shape_and_dtype_for_dataset))
  dataset = dataset.repeat()
  if batch_size:
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
  return dataset


def _graph_def_reconstruction(signature,
                              inputs_shape_and_dtype,
                              outputs_shape_and_dtype,
                              poplar_exec_filepath,
                              runtime_api_timeout_us,
                              batch_size=1):
  # pylint:disable=line-too-long
  """Keep the same shape and dtype as the original signature definition.

  For example:
    input tensor dtype from signature is input_1: tf.int64
    output tensor dtype from signature is tf.float32
    and graph imported from graph_def is like:

    input_1(tf.int32) --> ApplicationCall --> output(tf.float16)

    after reconstruction:

    input_1(tf.int64) --> Cast(<int64 - int32>) --> ApplicationCall --> Cast(<float16 - float32>) -> output(tf.float32)

  Args:
      signature : model signature
      inputs_shape_and_dtype (List of Tuple): the input shape and dtype of the input tensor, like [ (name, shape, dtype <tensorflow dtype> ), ... ]
      outputs_shape_and_dtype (List of Tuple): the output shape and dtype of the output tensor, like [ (name, shape, dtype <tensorflow dtype> ), ... ]

  Returns:
      graph_def: a reconstructed embedded application runtime graph_def proto that is consistant with the signature.
      signature: unmodified signature.
  """
  with ops.Graph().as_default() as graph:
    ctx = ipu.embedded_runtime.embedded_runtime_start(
        poplar_exec_filepath, [],
        f"application-{uuid4()}",
        timeout=runtime_api_timeout_us)

    # If input_dtype_from_graph == input_dtype_from_signature,
    # then there is no need to add `Cast` after input placeholders.
    input_placeholder = casted_input_from_signature(inputs_shape_and_dtype,
                                                    signature,
                                                    batch_size=batch_size)

    result = ipu.embedded_runtime.embedded_runtime_call(input_placeholder, ctx)

    # If output_dtype_from_graph == output_dtype_from_signature,
    # then there is no need to add `Cast` before the output of `ApplicationCall` op.
    casted_output_from_signature(result, outputs_shape_and_dtype, signature)

  return graph.as_graph_def(), signature


def embedded_application_runtime_save(signature,
                                      inputs_shape_and_dtype,
                                      outputs_shape_and_dtype,
                                      poplar_exec_filepath,
                                      runtime_api_timeout_us,
                                      compile_func,
                                      batch_size=1):

  sess_cfg = config_pb2.ConfigProto()
  sess_cfg.graph_options.rewrite_options.memory_optimization = (
      rewriter_config_pb2.RewriterConfig.OFF)

  with session.Session(config=sess_cfg) as sess:
    compile_op = (
        ipu.ops.application_compile_op.experimental_application_compile_op(
            compile_func, output_path=poplar_exec_filepath))
    sess.run(compile_op)

  return _graph_def_reconstruction(signature, inputs_shape_and_dtype,
                                   outputs_shape_and_dtype,
                                   poplar_exec_filepath,
                                   runtime_api_timeout_us, batch_size)


def use_ipu_wrapper(signature, compile_func):
  sess_cfg = config_pb2.ConfigProto()
  sess_cfg.graph_options.rewrite_options.memory_optimization = (
      rewriter_config_pb2.RewriterConfig.OFF)

  with session.Session(config=sess_cfg) as sess:
    with ops.device("/device:IPU:0"):
      compile_op = ipu.ipu_compiler.compile(compile_func, [])
    sess.run(compile_op)

    graph_def = sess.graph_def

  return graph_def, signature


def extract_emb_setting_from_param(param):
  embedded_runtime_exec_cachedir = (param.embedded_runtime_save_config.get(
      "embedded_runtime_exec_cachedir", DEFAULT_EMBEDDED_RUNTIME_CACHEDIR))
  poplar_exec_filepath = os.path.join(f"{embedded_runtime_exec_cachedir}",
                                      "application.poplar_exec")
  runtime_api_timeout_us = param.embedded_runtime_save_config.get(
      "runtime_api_timeout_us", DEFAULT_RUNTIME_API_TIMEOUT_US)
  batch_per_step = (param.batch_per_step
                    if param.batch_per_step else DEFAULT_BATCH_PER_STEP)
  os.makedirs(embedded_runtime_exec_cachedir, exist_ok=True)
  return (embedded_runtime_exec_cachedir, poplar_exec_filepath,
          runtime_api_timeout_us, batch_per_step)


def extract_ipu_config_from_param(param, cfg):
  cfg.auto_select_ipus = param.num_ipus
  cfg.matmuls.poplar_options.update({
      "availableMemoryProportion":
      str(param.matmul_amp),
      "partialsType":
      param.matmul_partial_type,
  })
  cfg.convolutions.poplar_options.update({
      "availableMemoryProportion":
      str(param.conv_amp),
      "partialsType":
      param.conv_partial_type,
  })


def validate_embedded_runtime_save_config(param):
  if param.embedded_runtime_save_config:
    if param.merge_subgraphs:
      raise ValueError(
          ("The `merge_subgraphs` must be false "
           "with `merge_subgraphs=False` "
           "during the embedded application runtime compilation."))
    if not param.int64_to_int32_conversion:
      raise ValueError(
          ("The `int64_to_int32_conversion` must be True "
           "with `int64_to_int32_conversion=True` "
           "during the embedded application runtime compilation."))
    help_message = (
        "`embedded_runtime_save_config` must be a dictionary "
        "containing the following items { "
        "'embedded_runtime_exec_cachedir': "
        "The cache output directory of the embedded application runtime, "
        "'runtime_api_timeout_us': "
        "The timeout for the embedded application }.")
    if not isinstance(param.embedded_runtime_save_config, dict):
      raise TypeError(
          f"Invalid type ({type(param.embedded_runtime_save_config).__name__}) "
          f"for `embedded_runtime_save_config`. {help_message}")
    for expected_key in [
        "embedded_runtime_exec_cachedir", "runtime_api_timeout_us"
    ]:
      if not expected_key in param.embedded_runtime_save_config:
        raise ValueError(f"Key '{expected_key}' missing from "
                         f"`embedded_runtime_save_config`. {help_message}")


def should_do_embedded_runtime(embedded_runtime_save_config, batch_per_step,
                               merge_subgraphs, int64_to_int32_conversion):
  if not embedded_runtime_save_config:
    return False
  if merge_subgraphs:
    return False
  if not batch_per_step:
    return False
  if not int64_to_int32_conversion:
    return False
  return True


def validate_pipeline_cfg(param, kwargs: OrderedDict, behavior: str):
  if "converter" not in param.pipeline_cfg:
    raise ValueError("`pipeline_cfg` must contain `behavior` keywords.")

  if param.pipeline_cfg["converter"].lower() == behavior:
    for key in param.pipeline_cfg:
      if key not in kwargs:
        raise ValueError(
            f"Unkown keyword {key} in `pipeline_cfg` in {behavior} converter.")

  return tuple(param.pipeline_cfg.get(kw, kwargs[kw]) for kw in kwargs)


def pipeline_embedded_runtime_wrapper(param, ipu_cfg, autograph_tfv1graph,
                                      signature, batch_size, batch_per_step,
                                      poplar_exec_filepath,
                                      runtime_api_timeout_us):
  extract_ipu_config_from_param(param, ipu_cfg)
  ipu_cfg.configure_ipu_system()

  computational_stages, device_mapping = autograph_tfv1graph.pipelined()

  with ops.Graph().as_default():
    inputs_shape_and_dtype = input_placeholder_name_shape_dtype(
        autograph_tfv1graph.graph, signature)
    outputs_shape_and_dtype = output_placeholder_name_shape_dtype(
        autograph_tfv1graph.graph, signature)

    dataset = prepare_dataset(inputs_shape_and_dtype, batch_size=batch_size)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    gradient_accumulation_count = batch_per_step * len(computational_stages)

    def compile_func():
      pipeline_op = ipu.pipelining_ops.pipeline(
          computational_stages=computational_stages,
          device_mapping=device_mapping,
          gradient_accumulation_count=gradient_accumulation_count,
          repeat_count=1,
          inputs=[],
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          name='pipeline_op')
      return pipeline_op

  return embedded_application_runtime_save(signature, inputs_shape_and_dtype,
                                           outputs_shape_and_dtype,
                                           poplar_exec_filepath,
                                           runtime_api_timeout_us,
                                           compile_func, batch_size)


def should_do_pipeline_wrapper(pipeline_cfg, behavior,
                               embedded_runtime_save_config, batch_per_step,
                               merge_subgraphs, int64_to_int32_conversion):
  if not pipeline_cfg:
    return False
  if pipeline_cfg["converter"].lower() != behavior:
    return False

  return should_do_embedded_runtime(embedded_runtime_save_config,
                                    batch_per_step, merge_subgraphs,
                                    int64_to_int32_conversion)
