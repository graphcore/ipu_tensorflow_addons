# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
"""
A utility to check if the correct versions of TensorFlow and Keras have been
installed.
"""

import importlib
from ipu_tensorflow_addons.util import _dependency_versions


def _get_module_spec(module_name):
  return importlib.util.find_spec(module_name)


def _get_module_version(module_name):
  return importlib.import_module(module_name).__version__


def _not_ipu_optimised_package_error_msg(package_name):
  return (f"Failed to import IPU {package_name}. Make sure you have IPU "
          f"{package_name} installed, and not a different {package_name} "
          "release.")


def _wrong_package_version_error_msg(package_name, actual_version,
                                     expected_version):
  return (f"Incompatible version of {package_name} was found. Version "
          f"{expected_version} was expected, but version {actual_version} was "
          "found.")


def _check_package_exists(package_name, package_spec):
  if package_spec is None:
    raise ImportError(_not_ipu_optimised_package_error_msg(package_name))


def _check_package_version(package_name, actual_version, expected_version):
  if actual_version != expected_version:
    raise ImportError(
        _wrong_package_version_error_msg(package_name, actual_version,
                                         expected_version))


def _check_ipu_tensorflow_exists():
  spec = _get_module_spec("tensorflow.python.ipu")
  _check_package_exists("TensorFlow", spec)


def _check_ipu_tensorflow_version():
  actual_version = _get_module_version("tensorflow")
  expected_version = _dependency_versions.TENSORFLOW_VERSION
  _check_package_version("Tensorflow", actual_version, expected_version)


def check_dependencies():
  _check_ipu_tensorflow_exists()
  _check_ipu_tensorflow_version()
