package(default_visibility = ["//ipu_tensorflow_addons:__pkg__"])

py_test(
    name = "example_ipu_specific_keras_layers",
    size = "large",
    srcs = ["example_ipu_specific_keras_layers.py"],
    args = [
        "--epochs=1",
        "--steps-per-epoch=16",
    ],
    deps = ["//ipu_tensorflow_addons"],
)

test_suite(
    name = "all_tests",
)
