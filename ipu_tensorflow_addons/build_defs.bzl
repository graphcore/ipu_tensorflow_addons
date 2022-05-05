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

"""Utilities for defining dependency versions."""

VersionInfo = provider()

def _dependency_version_impl(ctx):
    version = ctx.build_setting_value
    version_components = version.split(".")

    if not all([len(version_components) == 3, all([i.isdigit() for i in version_components])]):
        fail("{} is not a valid version for '{}'. It should be in the semver format. Did you forget to confgure?".format(version, ctx.label.name))

    return VersionInfo(version = version)

dependency_version = rule(
    implementation = _dependency_version_impl,
    build_setting = config.string(flag = True),
)

def _py_dependency_versions_impl(ctx):
    tensorflow_version = ctx.attr._tensorflow_version[VersionInfo].version
    keras_version = ctx.attr._keras_version[VersionInfo].version

    out = ctx.actions.declare_file("{}.py".format(ctx.label.name))

    content = [
        'TENSORFLOW_VERSION = "{}"'.format(tensorflow_version),
        'KERAS_VERSION = "{}"'.format(keras_version),
    ]
    content = "\n".join(content) + "\n"

    ctx.actions.write(output = out, content = content)

    return [DefaultInfo(files = depset([out]))]

py_dependency_versions = rule(
    implementation = _py_dependency_versions_impl,
    attrs = {
        "_tensorflow_version": attr.label(default = "//ipu_tensorflow_addons:tensorflow_version"),
        "_keras_version": attr.label(default = "//ipu_tensorflow_addons:keras_version"),
    },
)
