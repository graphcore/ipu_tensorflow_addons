package(default_visibility = ["//ipu_tensorflow_addons:__pkg__"])

py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "rnn_ops.py",
    ],
)

py_test(
    name = "gru_size_test",
    size = "small",
    srcs = ["gru_size_test.py"],
    deps = [
        ":layers",
        "//ipu_tensorflow_addons/test_utils",
    ],
)

py_test(
    name = "gru_training_test",
    size = "large",
    srcs = ["gru_training_test.py"],
    deps = [
        ":layers",
        "//ipu_tensorflow_addons/test_utils",
    ],
)

py_test(
    name = "lstm_size_test",
    size = "small",
    srcs = ["lstm_size_test.py"],
    deps = [
        ":layers",
        "//ipu_tensorflow_addons/test_utils",
    ],
)

py_test(
    name = "lstm_training_test",
    size = "large",
    srcs = ["lstm_training_test.py"],
    deps = [
        ":layers",
        "//ipu_tensorflow_addons/test_utils",
    ],
)

test_suite(
    name = "all_tests",
    tests = [
        "gru_size_test",
        "gru_training_test",
        "lstm_size_test",
        "lstm_training_test",
    ],
)
