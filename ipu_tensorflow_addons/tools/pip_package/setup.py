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
import sys
import re
from setuptools import setup


def find_packages(pattern=r".*"):
  packages = []
  for root, _, files in os.walk(".", followlinks=True):
    for name in files:
      if name == "__init__.py":
        dir_path = os.path.relpath(root)
        package_name = ".".join(dir_path.split(os.sep))
        if re.fullmatch(pattern, package_name) is not None:
          packages.append(package_name)
  return packages


if "TF_VERSION" not in os.environ:
  sys.exit(
      "ERROR: TF_VERSION is not set. Please set TF_VERSION to the TensorFlow "
      "version the wheel is being built.")
tf_version = os.environ["TF_VERSION"]

# pylint: disable=line-too-long
CONSOLE_SCRIPTS = [
    'ipu_saved_model_cli = ipu_tensorflow_addons.saved_model_tool.saved_model_cli:main',
]
# pylint: enable=line-too-long

setup(
    name='ipu_tensorflow_addons',
    description='A collection of addons for IPU TensorFlow',
    version=tf_version,
    python_requires='>=3.6',
    url='https://www.graphcore.ai/',
    author='Graphcore Ltd.',
    # Contained modules and scripts.
    packages=find_packages(r"ipu_tensorflow_addons\.?.*"),
    license='Apache 2.0',
    install_requires=[f"tensorflow=={tf_version}"],
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    },
)
