package(default_visibility = ["//ipu_tensorflow_addons:__pkg__"])

py_library(
    name = "auto_pipeline",
    srcs = [
        "prediction/partition.py",
        "prediction/partition_evaluate.py",
        "prediction/partition_test.py",
        "utils/keras_utils.py",
        "utils/profiler.py",
        "utils/pva_utils.py",
        "utils/types.py",
        "utils/types_utils.py",
    ],
)

py_test(
    name = "partition_test",
    size = "medium",
    srcs = ["prediction/partition_test.py"],
    deps = [
        ":auto_pipeline",
    ],
)

py_test(
    name = "keras_utils_test",
    size = "medium",
    srcs = ["utils/keras_utils_test.py"],
    deps = [
        ":auto_pipeline",
    ],
)

py_test(
    name = "profiler_test",
    size = "medium",
    srcs = ["utils/profiler_test.py"],
    deps = [
        ":auto_pipeline",
    ],
)

py_test(
    name = "pva_utils_test",
    size = "medium",
    srcs = ["utils/pva_utils_test.py"],
    deps = [
        ":auto_pipeline",
    ],
)
