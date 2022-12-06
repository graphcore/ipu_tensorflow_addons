package(default_visibility = ["//ipu_tensorflow_addons:__pkg__"])

py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "assume_equal_across_replicas.py",
        "ctc.py",
        "dense.py",
        "dropout.py",
        "effective_transformer.py",
        "embedding_lookup.py",
        "f8_convert.py",
        "ipu_layer.py",
        "normalization.py",
        "recomputation.py",
        "rnn.py",
    ],
)

py_test(
    name = "assume_equal_across_replicas_test",
    size = "large",
    srcs = ["assume_equal_across_replicas_test.py"],
    tags = ["hw_poplar_test_2_ipus"],
    deps = [
        ":layers",
    ],
)

py_test(
    name = "ctc_test",
    size = "large",
    srcs = ["ctc_test.py"],
    deps = [":layers"],
)

py_test(
    name = "dense_test",
    size = "large",
    srcs = ["dense_test.py"],
    shard_count = 12,
    deps = [":layers"],
)

py_test(
    name = "dropout_test",
    size = "medium",
    srcs = ["dropout_test.py"],
    deps = [":layers"],
)

py_test(
    name = "effective_transformer_test",
    size = "large",
    srcs = ["effective_transformer_test.py"],
    shard_count = 12,
    deps = [":layers"],
)

py_test(
    name = "embedding_lookup_test",
    size = "large",
    srcs = ["embedding_lookup_test.py"],
    deps = [":layers"],
)

py_test(
    name = "f8_convert_test",
    size = "small",
    srcs = ["f8_convert_test.py"],
    deps = [":layers"],
)

py_test(
    name = "normalization_test",
    size = "large",
    srcs = ["normalization_test.py"],
    shard_count = 12,
    deps = [":layers"],
)

py_test(
    name = "recomputation_test",
    size = "large",
    srcs = ["recomputation_test.py"],
    shard_count = 2,
    tags = ["isolated_poplar_tests"],
    deps = [
        ":layers",
    ],
)

py_test(
    name = "rnn_test",
    size = "large",
    srcs = ["rnn_test.py"],
    shard_count = 5,
    deps = [":layers"],
)

py_test(
    name = "rnn_hw_test",
    size = "large",
    srcs = ["rnn_hw_test.py"],
    tags = ["hw_poplar_test_1_ipus"],
    deps = [
        ":layers",
    ],
)
