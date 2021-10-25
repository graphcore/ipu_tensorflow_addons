# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

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

setup(
    name='ipu_tensorflow_addons',
    description='A collection of addons for IPU TensorFlow',
    version=tf_version,
    python_requires='>=3.6',
    url='https://www.graphcore.ai/',
    author='Graphcore Ltd.',
    # Contained modules and scripts.
    packages=find_packages(r"ipu_tensorflow_addons\.?.*"),
    license='MIT',
    install_requires=[f"tensorflow=={tf_version}"],
)
