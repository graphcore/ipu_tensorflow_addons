package(default_visibility = ["//visibility:private"])

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "setup.py",
        "//ipu_tensorflow_addons",
    ],
)
