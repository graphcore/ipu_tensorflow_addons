package(default_visibility = ["//ipu_tensorflow_addons/tools/pip_package:__pkg__"])

load(":build_defs.bzl", "dependency_version", "py_dependency_versions")

# This rule will create a file `_dependency_versions.py` which contains
# information about the version of TensorFlow.
py_dependency_versions(name = "_dependency_versions")

# The rule below define the versions of TensorFlow (defaults to "None"). This
# should be explicitly defined from the command line with the build setting
# `--//ipu_tensorflow_addons:tensorflow_version`. This is automatically done
# during configuration. This build setting can be seen in
# `.addons_configure.bazelrc`.
dependency_version(
    name = "tensorflow_version",
    build_setting_default = "None",
)

py_library(
    name = "ipu_tensorflow_addons",
    deps = [
        "//ipu_tensorflow_addons/layers",
        "//ipu_tensorflow_addons/optimizers",
        "//ipu_tensorflow_addons/saved_model_tool",
    ],
)
