package(default_visibility = ["//ipu_tensorflow_addons/tools/pip_package:__pkg__"])

py_library(
    name = "ipu_tensorflow_addons",
    deps = [
        "//ipu_tensorflow_addons/layers",
        "//ipu_tensorflow_addons/optimizers",
        "//ipu_tensorflow_addons/saved_model_tool",
    ],
)

test_suite(
    name = "all_tests",
    tests = [
        "//ipu_tensorflow_addons/layers:all_tests",
        "//ipu_tensorflow_addons/optimizers:all_tests",
        "//ipu_tensorflow_addons/saved_model_tool:all_tests",
    ],
)
