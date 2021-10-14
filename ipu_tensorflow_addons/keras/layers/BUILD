package(default_visibility = ["//ipu_tensorflow_addons:__pkg__"])

py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "effective_transformer.py",
    ],
)

py_test(
    name = "keras_effective_transformer_test",
    size = "large",
    srcs = ["keras_effective_transformer_test.py"],
    shard_count = 12,
    deps = [":layers"],
)

test_suite(
    name = "all_tests",
    tests = [
        "keras_effective_transformer_test",
    ],
)
