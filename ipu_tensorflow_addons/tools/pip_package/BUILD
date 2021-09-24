package(default_visibility = ["//visibility:private"])

sh_binary(
    name = "build_pip_package",
    data = [
        "//ipu_tensorflow_addons",
        "setup.py",
    ],
    srcs = ["build_pip_package.sh"],
)
