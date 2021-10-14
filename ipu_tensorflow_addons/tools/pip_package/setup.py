# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
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


setup(
    name='ipu_tensorflow_addons',
    version='0.0.0',
    python_requires='>=3.6',
    url='https://www.graphcore.ai/',
    author='Graphcore Ltd.',
    # Contained modules and scripts.
    packages=find_packages(r"ipu_tensorflow_addons\.?.*"),
    license='Apache 2.0',
)
