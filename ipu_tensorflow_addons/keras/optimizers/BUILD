package(default_visibility = ["//ipu_tensorflow_addons:__pkg__"])

py_library(
    name = "optimizers",
    srcs = [
        "__init__.py",
        "adam_optimizer.py",
        "ipu_optimizer_base.py",
    ],
)

py_test(
    name = "keras_adam_optimizer_test",
    size = "medium",
    srcs = [
        "keras_adam_optimizer_test.py",
        "test_util.py",
    ],
    shard_count = 4,
    deps = [":optimizers"],
)

test_suite(
    name = "all_tests",
    tests = ["keras_adam_optimizer_test"],
)
