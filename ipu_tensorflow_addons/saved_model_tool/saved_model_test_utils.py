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

import tempfile
import shutil
from functools import wraps
from abc import ABCMeta, abstractmethod

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util, ops
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import signature_constants
from tensorflow.saved_model import builder, tag_constants
from tensorflow.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.platform import test


def _analyze_pb_inputs_outputs(graph: ops.Graph):
  """Analyse the inputs and outputs of the tensorflow graph.

  Args:
      graph (ops.Graph): tensorflow graph.

  Returns:
      List[Tensor]: the input tensors list.
      List[Tensor]: the output tensors list.
  """
  operators = graph.get_operations()
  outputs_set = set(operators)
  inputs = []
  for op in operators:
    if not op.inputs and op.type == 'Placeholder':  # take Placeholder as input
      inputs.append(op)
    else:
      for input_tensor in op.inputs:
        if input_tensor.op in outputs_set:
          outputs_set.remove(input_tensor.op)
  outputs = list(outputs_set)

  inputs.sort(key=lambda x: x.name)
  outputs.sort(key=lambda x: x.name)

  return [graph.get_tensor_by_name(f"{inp.name}:0") for inp in inputs], outputs


def declare_signature(input_name_keys=None, output_name_keys=None):
  # pylint:disable=line-too-long
  """The function wrapper for declaring the name key for the signature proto.

  For example:

  ```python
  class MultiInputsModelForTest(ModelForTest)
    @declare_signature(input_name_keys=sorted(['input1', 'input2']), output_name_keys=sorted(['output1', 'output2']))
    def create(self):
      # <input defination>
      a = tf.placeholder(tf.int32, name="input1")
      b = tf.placeholder(tf.int32, name="input2")
      self.register_inputs(a, b)
      outputs = model_defination(here)
      return outputs
  ```

  Args:
      input_name_keys (List[str], optional):
          the input name keys for the signature,
          must be in alphabetical order.
          if None, the name keys will be the same as tensor names.
          Defaults to None.
      output_name_keys (List[str], optional):
          the input name keys for the signature,
          must be in alphabetical order.
          if None, the name keys will be the same as tensor names.
          Defaults to None.
  """
  def declare_signature_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
      self.input_name_keys = input_name_keys
      self.output_name_keys = output_name_keys
      return func(self, *args, **kwargs)

    return wrapper

  return declare_signature_decorator


class ModelForTest(metaclass=ABCMeta):
  # pylint:disable=line-too-long
  """
  This a base model class for defining models for tests.

  Example usage:

  ```python
  class MultiInputsModelForTest(ModelForTest)
    def create(self):
      <model defination here>
  ```

  if you turn off the `auto_infer_inputs_and_outputs`
  the example will be:

  ```python
  class MultiInputsModelForTest(ModelForTest)
    def create(self):
      <input defination>
      a = tf.placeholder(tf.int32)
      b = tf.placeholder(tf.int32)
      self.register_inputs(a, b)
      outputs = <model defination here>
      return outputs
  ```

  >>> model = MultiInputsModelForTest()
  >>> model.signature_def
  >>> model.graph_def

  Args:
    tag: the server tag for saving the model.
    signature_def_key: the signature_def_key for saving the model.
    freeze (boolean) : freeze the model,
                    this means using const folder on the graph_def.
    auto_infer_inputs_and_outputs (boolean) :
                    infer inputs and outputs from user defined graph.
    save(boolean) : save the model pb to temporary files.
    input_name_keys (List[string]):
        the name key for the input dictionary in signature_def,
        defaults to None. If None, the name key will be same as the tensor name.
    output_name_keys (List[string]):
        the name key for the output dictionary in signature_def,
        defaults to None. If None, the name key will be same as the tensor name.

  """
  def __init__(self,
               tag=None,
               signature_def_key=None,
               freeze=False,
               auto_infer_inputs_and_outputs=True,
               input_name_keys=None,
               output_name_keys=None,
               save=False):
    self._graph = ops.Graph()
    self._graph_def = graph_pb2.GraphDef()
    self._signature_def = None
    self._inputs = []
    self._outputs = []
    self.input_name_keys = input_name_keys
    self.output_name_keys = output_name_keys
    self._tag = tag if tag else tag_constants.SERVING
    self._signature_def_key = (
        signature_def_key if signature_def_key else
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    self._create_graph()
    if freeze:
      self._freeze()
    if auto_infer_inputs_and_outputs:
      self._infer_inputs_and_outputs()
    self._meta_graph_def = meta_graph_pb2.MetaGraphDef()
    self._meta_graph()
    self.model_path = None
    if save:
      self.save()

  @property
  def graph_def(self):
    return self._graph_def

  @property
  def signature_def(self):
    return self._signature_def

  @property
  def meta_graph_def(self):
    return self._meta_graph_def

  @abstractmethod
  def create(self):
    raise NotImplementedError("Need to implement this method.")

  def regist_inputs(self, *args, **kwargs):
    self._inputs.extend(args if args else [] + list(kwargs.values()))
    self._inputs.sort()

  def regist_outputs(self, *args, **kwargs):
    self._outputs.extend(args if args else [] + list(kwargs.values()))
    self._outputs.sort()

  def _infer_inputs_and_outputs(self):
    self._inputs, _ = _analyze_pb_inputs_outputs(self._graph)

  def _create_graph(self):
    with ops.Graph().as_default():
      outputs = self.create()
      if outputs is not None:
        self._outputs.extend(
            [outputs] if isinstance(outputs, ops.Tensor) else outputs)
      else:
        raise ValueError("No outputs here.")
      self._graph = ops.get_default_graph()
      self._graph_def = self._graph.as_graph_def()

  def _freeze(self):
    with session.Session(graph=self._graph) as sess_freeze:
      variables.global_variables_initializer().run()
      self._graph_def = graph_util.convert_variables_to_constants(
          sess_freeze, sess_freeze.graph_def,
          [o.name.split(":")[0] for o in self._outputs])

  def _signature(self):
    input_name_key_dict = {
        inp.name.split(":")[0]: inp
        for inp in self._inputs
    } if not self.input_name_keys else dict(
        zip(self.input_name_keys, self._inputs))
    output_name_key_dict = {
        out.name.split(":")[0]: out
        for out in self._outputs
    } if not self.output_name_keys else dict(
        zip(self.output_name_keys, self._outputs))
    self._signature_def = predict_signature_def(inputs=input_name_key_dict,
                                                outputs=output_name_key_dict)

  def _meta_graph(self):
    self._signature()
    self._meta_graph_def.meta_info_def.tags.append(self._tag)
    self._meta_graph_def.signature_def[self._signature_def_key].CopyFrom(
        self._signature_def)

  def save(self):
    if self.model_path:
      shutil.rmtree(self.model_path)
    self.model_path = tempfile.mkdtemp(dir=test.get_temp_dir())
    save_builder = builder.SavedModelBuilder(self.model_path)
    with ops.Graph().as_default():
      import_graph_def(self._graph_def, name="")
      with session.Session() as sess_save:
        save_builder.add_meta_graph_and_variables(
            sess=sess_save,
            tags=[self._tag],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                self._signature_def
            })
        save_builder.save()
