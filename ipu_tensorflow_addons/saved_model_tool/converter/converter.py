# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
Abstract base class and pipeline manager for all passes
"""
from abc import ABCMeta, abstractmethod


class Converter(metaclass=ABCMeta):
  @abstractmethod
  def apply(self, graph_def, signature_def):
    raise NotImplementedError("Abstract method")
