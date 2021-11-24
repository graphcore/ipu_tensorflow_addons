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
"""Exposes the Python wrapper conversion to be able to run on ipu."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import platform

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python import ipu

from ipu_tensorflow_addons.saved_model_tool.converter import ConverterPipeline

if platform.system() == "Windows":
  raise RuntimeError("Windows platform is not supported")


class IpuConversionParams(object):
  # pylint:disable=line-too-long
  """Parameters that are used for IPU conversion.

  Fields:
    - **excluded_nodes** - list of node names to prevent the converter from touching.
    - **num_ipus** - number ipus to run the model.
    - **ipu_placement** - do ipu placement or not.
  """

  # pylint:enable=line-too-long
  def __init__(self, excluded_nodes=None, num_ipus=1, ipu_placement=True):
    self.excluded_nodes = excluded_nodes
    self.num_ipus = num_ipus
    self.ipu_placement = ipu_placement
    self.pb_md5sum = None

  def load_from_json_file(self, config_file):
    if not os.access(config_file, os.R_OK):
      raise ValueError("config_file {} is not readable.".format(config_file))

    with open(config_file, 'r') as fp:
      config = json.load(fp)

    if "num_ipus" in config and isinstance(config["num_ipus"], (int)):
      self.num_ipus = config["num_ipus"]
    if "no_ipu_placement" in config and isinstance(config["no_ipu_placement"],
                                                   bool):
      self.ipu_placement = not bool(config["no_ipu_placement"])
    if "excluded_nodes" in config and isinstance(config["excluded_nodes"],
                                                 list):
      self.excluded_nodes = config["excluded_nodes"]

  def save_to_json_file(self, directory):
    if not os.path.isdir(directory) or not os.access(directory, os.W_OK):
      raise ValueError(
          "dir {} is not a directory or writeable.".format(directory))

    config = {}
    config["num_ipus"] = self.num_ipus
    config["no_ipu_placement"] = True if self.ipu_placement is False else False
    config["md5sum"] = self.pb_md5sum
    config["excluded_nodes"] = self.excluded_nodes

    config_file_path = os.path.join(directory, 'conversion_params.json')
    with open(config_file_path, 'w') as fp:
      json.dump(config, fp, indent=4)


def _check_conversion_params(conversion_params):
  """Validate the provided IpuConversionParams.

  Args:
    conversion_params: a IpuConversionParams instance.

  Raises:
    TypeError: if any of the parameters are of unexpected type.
    ValueError: if any of the parameters are of unexpected value.
  """
  if not isinstance(conversion_params.num_ipus,
                    int) or conversion_params.num_ipus <= 0:
    raise ValueError("num_ipus should be an integer, and greater than 0")

  if conversion_params.excluded_nodes is not None:
    if not isinstance(conversion_params.excluded_nodes, list):
      raise ValueError("excluded_nodes should be a list")

  if not isinstance(conversion_params.ipu_placement, bool):
    raise ValueError("ipu_placement should be True or False")


class IpuGraphConverter(object):
  # pylint:disable=line-too-long
  """A converter for IPU transformation for TF 1.x GraphDef/SavedModels.
  """
  def __init__(self,
               input_saved_model_dir=None,
               input_saved_model_tags=None,
               input_saved_model_signature_key=None,
               input_meta_graph_def=None,
               input_sess=None,
               conversion_params=None,
               output_saved_model_dir=None):
    """Initializes the converter.

    Args:
      input_saved_model_dir: the directory to load the SavedModel which contains
        the input graph to transforms. Used only when input_graph_def is None.
      input_saved_model_tags: list of tags to load the SavedModel.
      input_saved_model_signature_key: the key of the signature to optimize the
        graph for.
      input_meta_graph_def: a MetaGraphDef object containing a model to be transformed.
        If set to None, the graph will be read from the SavedModel loaded from
        input_saved_model_dir.
      input_sess: Session bound to input_meta_graph_def.
      conversion_params: a IpuConversionParams instance.

    Raises:
      ValueError: if the combination of the parameters is invalid.
    """

    if input_meta_graph_def and input_saved_model_dir:
      raise ValueError(
          "Can only specify one of input_meta_graph_def and input_saved_model_dir"
      )
    if not input_meta_graph_def and not input_saved_model_dir:
      raise ValueError("Must specify one of input_meta_graph_def and "
                       "input_saved_model_dir")
    if input_meta_graph_def and not input_sess:
      raise ValueError(
          "Must specify input_sess when specify input_meta_graph_def")

    self._input_meta_graph_def = input_meta_graph_def
    self._input_sess = input_sess
    self._input_saved_model_dir = input_saved_model_dir
    self._input_saved_model_tags = (input_saved_model_tags
                                    or [tag_constants.SERVING])
    self._input_saved_model_signature_key = (
        input_saved_model_signature_key
        or signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)

    _check_conversion_params(conversion_params)
    self._conversion_params = conversion_params

    self._converted = False
    self._converted_graph_def = None
    self._grappler_meta_graph_def = None
    self._output_saved_model_dir = output_saved_model_dir

  @staticmethod
  def _names_and_keys_lists(tensor_info):
    """ Get the keys and corresponding tensor names as a pair of lists.
        Ordering is determined by the natural order of the keys.
    """
    keys_list = []
    names_list = []
    for key in sorted(tensor_info.keys()):
      keys_list.append(key)
      names_list.append(tensor_info[key].name.split(":")[0])

    return names_list, keys_list

  def _store_input_names_and_keys(self, signature_def):
    self._input_names, self._input_keys = self._names_and_keys_lists(
        signature_def.inputs)

  def _store_output_names_and_keys(self, signature_def):
    self._output_names, self._output_keys = self._names_and_keys_lists(
        signature_def.outputs)

  def convert(self):
    if self._converted:
      raise RuntimeError("The model can't be converted multiple times.")

    new_graph_def, input_signature_def = self.call_converter()

    self._grappler_meta_graph_def = meta_graph_pb2.MetaGraphDef()
    self._grappler_meta_graph_def.graph_def.CopyFrom(new_graph_def)
    self._grappler_meta_graph_def.signature_def[
        self._input_saved_model_signature_key].CopyFrom(input_signature_def)
    self._converted_graph_def = new_graph_def
    self._converted = True

    return self._converted_graph_def

  def call_converter(self):
    """Run the ipu conversion.

    Returns:
      The converted GraphDef for TF 1.x.
    """
    graph = ops.Graph()
    with session.Session(graph=graph) as sess:
      input_meta_graph_def = loader.load(sess, self._input_saved_model_tags,
                                         self._input_saved_model_dir)

      #get signature_def of graph
      input_signature_def = input_meta_graph_def.signature_def[
          self._input_saved_model_signature_key]

      # Get input and outputs from all SignatureDef.
      self._store_input_names_and_keys(input_signature_def)
      self._store_output_names_and_keys(input_signature_def)

      # Freeze the variables in the SavedModel graph and copy the frozen
      # graph over.
      new_graph_def = graph_util.convert_variables_to_constants(
          sess, sess.graph.as_graph_def(add_shapes=True),
          list(self._output_names))
      # call converter pipline
      pipeline = ConverterPipeline(self._conversion_params,
                                   self._input_saved_model_signature_key)
      new_graph_def, input_signature_def = pipeline.ApplyConverters(
          new_graph_def, input_signature_def)
    return new_graph_def, input_signature_def

  def save(self, output_saved_model_dir):
    """Save the converted graph as a SavedModel.

    Args:
      output_saved_model_dir: construct a SavedModel using the converted
        GraphDef and save it to the specified directory. This option only works
        when the input graph is loaded from a SavedModel, i.e. when
        input_saved_model_dir is specified and input_graph_def is None in
        __init__().

    Raises:
      ValueError: if the input to the converter is a GraphDef instead of a
      SavedModel.
    """
    assert self._converted

    # Write the transformed graphdef as SavedModel.
    saved_model_builder = builder.SavedModelBuilder(output_saved_model_dir)
    with ops.Graph().as_default():
      importer.import_graph_def(self._converted_graph_def, name="")
      # We don't use any specific converter here.
      with session.Session() as sess:
        saved_model_builder.add_meta_graph_and_variables(
            sess,
            self._input_saved_model_tags,
            signature_def_map=self._grappler_meta_graph_def.signature_def)
    # Ignore other meta graphs from the input SavedModel.
    saved_model_builder.save()

    # Save conversion_params info to output_saved_model_dir
    resp = os.popen('md5sum ' + output_saved_model_dir +
                    "/saved_model.pb").readlines()
    self._conversion_params.pb_md5sum = resp[0].split(" ")[0]
    self._conversion_params.save_to_json_file(output_saved_model_dir)


def create_inference_graph(input_saved_model_dir=None,
                           input_saved_model_tags=None,
                           input_saved_model_signature_key=None,
                           output_saved_model_dir=None,
                           excluded_nodes=None,
                           num_ipus=1,
                           ipu_placement=True,
                           config_file=None):
  """Python wrapper for the IPU transformation.

  Args:
    input_saved_model_dir: the directory to load the SavedModel which contains
      the input graph to transforms. Used only when input_graph_def is None.
    input_saved_model_tags: list of tags to load the SavedModel.
    input_saved_model_signature_key: the key of the signature to optimize the
      graph for.
    output_saved_model_dir: if not None, construct a SavedModel using the
      returned GraphDef and save it to the specified directory. This option only
      works when the input graph is loaded from a SavedModel, i.e. when
      input_saved_model_dir is specified and input_graph_def is None.
    excluded_nodes: list of node names to prevent the converter from touching.
    num_ipus: number ipus to run the model.
    ipu_placement: do ipu placement or not.
    config_file: config file path.

  Returns:
    A GraphDef transformed from the SavedModel graph def
    loaded from input_saved_model_dir

  Raises:
    ValueError: if the combination of the parameters is invalid.
  """
  conversion_params = IpuConversionParams(excluded_nodes=excluded_nodes,
                                          num_ipus=num_ipus,
                                          ipu_placement=ipu_placement)

  if config_file:
    conversion_params.load_from_json_file(config_file)

  ipu_converter = IpuGraphConverter(
      input_saved_model_dir=input_saved_model_dir,
      input_saved_model_tags=input_saved_model_tags,
      input_saved_model_signature_key=input_saved_model_signature_key,
      conversion_params=conversion_params,
      output_saved_model_dir=output_saved_model_dir)

  converted_graph_def = ipu_converter.convert()
  if output_saved_model_dir:
    ipu_converter.save(output_saved_model_dir)
  return converted_graph_def
