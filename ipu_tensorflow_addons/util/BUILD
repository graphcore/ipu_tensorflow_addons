package(default_visibility = ["//ipu_tensorflow_addons:__pkg__"])

load("//ipu_tensorflow_addons/util:build_defs.bzl", "dependency_version", "py_dependency_versions")

# This rule will create a file `_dependency_versions.py` which contains
# information about the versions of TensorFlow and Keras.
py_dependency_versions(name = "_dependency_versions")

# The two rules below define the versions of TensorFlow and Keras (defaults to
# "None"). These should be explicitly defined from the command line with the
# build settings `--//ipu_tensorflow_addons/util:tensorflow_version` and
# `--//ipu_tensorflow_addons/util:keras_version`. This is automatically done
# during configuration. These build settings can be seen in
# `.addons_configure.bazelrc`.
dependency_version(
    name = "tensorflow_version",
    build_setting_default = "None",
)

dependency_version(
    name = "keras_version",
    build_setting_default = "None",
)

py_library(
    name = "util",
    srcs = [
        "__init__.py",
        "_dependency_checker.py",
    ],
    deps = ["//ipu_tensorflow_addons/util:_dependency_versions"],
)
