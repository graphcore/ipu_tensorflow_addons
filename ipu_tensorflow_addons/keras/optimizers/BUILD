package(default_visibility = ["//ipu_tensorflow_addons:__pkg__"])

py_library(
    name = "optimizers",
    srcs = [
        "__init__.py",
        "adam_optimizer.py",
        "ipu_optimizer_base.py",
        "lamb_optimizer.py",
        "sgd_optimizer.py",
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

py_test(
    name = "keras_lamb_optimizer_test",
    size = "medium",
    srcs = [
        "keras_lamb_optimizer_test.py",
        "test_util.py",
    ],
    shard_count = 5,
    deps = [":optimizers"],
)

py_test(
    name = "keras_sgd_optimizer_test",
    size = "medium",
    srcs = [
        "keras_sgd_optimizer_test.py",
        "test_util.py",
    ],
    shard_count = 3,
    deps = [":optimizers"],
)
