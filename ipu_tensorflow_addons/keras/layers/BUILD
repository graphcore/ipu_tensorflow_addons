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
    shard_count = 12,
    srcs = ["keras_effective_transformer_test.py"],
    deps = [":layers"],
)

test_suite(
    name = "all_tests",
    tests = [
        "keras_effective_transformer_test",
    ],
)
