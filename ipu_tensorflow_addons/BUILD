package(default_visibility = ["//visibility:private"])

py_library(
    name = "ipu_tensorflow_addons",
    deps = [
        "//ipu_tensorflow_addons/v1/layers",
        "//ipu_tensorflow_addons/v1/optimizers",
        "//ipu_tensorflow_addons/keras/layers",
        "//ipu_tensorflow_addons/keras/optimizers",
    ],
)

test_suite(
    name = "all_tests",
    tests = [
        "//ipu_tensorflow_addons/v1/layers:all_tests",
        "//ipu_tensorflow_addons/v1/optimizers:all_tests",
        "//ipu_tensorflow_addons/keras/layers:all_tests",
        "//ipu_tensorflow_addons/keras/optimizers:all_tests",
    ],
)
