# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from ipu_tensorflow_addons.saved_model_tool.converter import IPUPlacement


class ConverterPipeline(object):
  # pylint: disable=unused-argument
  def __init__(self, param, signatrue_key):
    self._converters = list()
    self._converters.append(IPUPlacement(param))

  def ApplyConverters(self, graph_def, signature_def):
    for converter in self._converters:
      graph_def, signature_def = converter.apply(graph_def, signature_def)

    return graph_def, signature_def
