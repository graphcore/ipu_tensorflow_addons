package(default_visibility = ["//ipu_tensorflow_addons:__pkg__"])

filegroup(
    name = "profile_test_data",
    srcs = glob(["testdata/**"]),
)

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
        "converter/loop_repeat_wrapper.py",
        "converter/manual_shard.py",
        "converter/precision_conversion.py",
        "converter/utils.py",
        "ipu_convert.py",
        "saved_model_cli.py",
    ] + glob(
        ["converter/autograph/**/*.py"],
        exclude = ["converter/autograph/**/*_test.py"],
    ),
)

py_test(
    name = "profile_test",
    srcs = [
        "converter/autograph/profile_test.py",
        "saved_model_test_utils.py",
    ],
    data = [
        ":profile_test_data",
    ],
    deps = [
        ":saved_model_tool",
    ],
)

py_test(
    name = "strategies_test",
    srcs = [
        "converter/autograph/profile_test.py",
        "converter/autograph/strategies_test.py",
        "saved_model_test_utils.py",
    ],
    data = [
        ":profile_test_data",
    ],
    deps = [
        ":saved_model_tool",
    ],
)

py_test(
    name = "tfv1graph_test",
    srcs = [
        "converter/autograph/tfv1graph_test.py",
        "saved_model_test_utils.py",
    ],
    deps = [
        ":saved_model_tool",
    ],
)

py_test(
    name = "options_test",
    srcs = [
        "converter/autograph/options_test.py",
    ],
    deps = [
        ":saved_model_tool",
    ],
)

py_test(
    name = "tfv1experiment_test",
    srcs = [
        "converter/autograph/tfv1experiment_test.py",
        "saved_model_test_utils.py",
    ],
    deps = [
        ":saved_model_tool",
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
    name = "loop_repeat_wrapper_test",
    srcs = [
        "converter/loop_repeat_wrapper_test.py",
        "saved_model_test_utils.py",
    ],
    tags = ["hw_poplar_test_2_ipus"],
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
    tags = ["hw_poplar_test_1_ipus"],
    deps = [
        ":saved_model_tool",
    ],
)

py_test(
    name = "utils_test",
    srcs = [
        "converter/utils_test.py",
    ],
    deps = [
        ":saved_model_tool",
    ],
)
