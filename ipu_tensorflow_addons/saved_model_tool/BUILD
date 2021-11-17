package(default_visibility = ["//ipu_tensorflow_addons:__pkg__"])

py_library(
    name = "saved_model_tool",
    srcs = [
        "__init__.py",
        "converter/__init__.py",
        "converter/converter.py",
        "converter/converter_pipeline.py",
        "converter/ipu_placement.py",
        "ipu_convert.py",
    ],
)

py_test(
    name = "ipu_placement_test",
    srcs = ["converter/ipu_placement_test.py"],
    deps = [
        ":saved_model_tool",
    ],
)

test_suite(
    name = "all_tests",
)
