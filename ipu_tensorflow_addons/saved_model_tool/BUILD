package(default_visibility = ["//ipu_tensorflow_addons:__pkg__"])

py_library(
    name = "saved_model_tool",
    srcs = [
        "__init__.py",
        "converter/__init__.py",
        "converter/converter.py",
        "converter/converter_pipeline.py",
        "converter/gelu_replacement.py",
        "converter/int64_to_int32_conversion.py",
        "converter/ipu_compiler_wrapper.py",
        "converter/ipu_placement.py",
        "converter/manual_shard.py",
        "converter/precision_conversion.py",
        "converter/utils.py",
        "ipu_convert.py",
        "saved_model_cli.py",
    ],
)

py_test(
    name = "ipu_placement_test",
    srcs = [
        "converter/ipu_placement_test.py",
        "saved_model_test_utils.py",
    ],
    deps = [
        ":saved_model_tool",
    ],
)

py_test(
    name = "precision_conversion_test",
    srcs = [
        "converter/precision_conversion_test.py",
        "saved_model_test_utils.py",
    ],
    deps = [
        ":saved_model_tool",
    ],
)

py_test(
    name = "manual_shard_test",
    srcs = [
        "converter/manual_shard_test.py",
        "saved_model_test_utils.py",
    ],
    deps = [
        ":saved_model_tool",
    ],
)

py_test(
    name = "int64_to_int32_conversion_test",
    srcs = [
        "converter/int64_to_int32_conversion_test.py",
        "saved_model_test_utils.py",
    ],
    deps = [
        ":saved_model_tool",
    ],
)

py_test(
    name = "ipu_compiler_wrapper_test",
    srcs = [
        "converter/ipu_compiler_wrapper_test.py",
        "saved_model_test_utils.py",
    ],
    deps = [
        ":saved_model_tool",
    ],
)

py_test(
    name = "gelu_replacement_test",
    srcs = [
        "converter/gelu_replacement_test.py",
        "saved_model_test_utils.py",
    ],
    deps = [
        ":saved_model_tool",
    ],
)

py_test(
    name = "saved_model_cli_test",
    srcs = [
        "saved_model_cli_test.py",
        "saved_model_test_utils.py",
    ],
    deps = [
        ":saved_model_tool",
    ],
)

test_suite(
    name = "all_tests",
)
