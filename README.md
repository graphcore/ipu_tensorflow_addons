# IPU TensorFlow Addons

This document explains how to build a wheel file, and how to run tests.

## Prerequisites
Bazel 0.26.1 is required for building wheel files and running tests.

For running the tests, you will need a wheel file for a compatible version of IPU TensorFlow, and a
matching Poplar installation (from the same Poplar SDK).

## TL;DR
```
lang=sh
# Prerequisites:
# Make sure you have the prerequisites listed in the above section.
# Navigate to your ipu_tensorflow_addons repository.

# Build a wheel:
bazel run //ipu_tensorflow_addons/tools/pip_package:build_pip_package <absolute-path-to-output-directory> <tf-version-number>

# Configure:
export TF_POPLAR_BASE=<absolute-path-to-poplar-install>
bash ./configure <path-to-tf-wheel>

# Run all non-HW tests:
bazel test //ipu_tensorflow_addons:all_tests --test_env=TF_POPLAR_FLAGS="--max_infeed_threads=8 --use_ipu_model --max_compilation_threads=1 --ipu_model_tiles=8" --test_env=TF_POPLAR_VLOG_LEVEL=1 --test_size_filters=small,medium,large --test_timeout="240,360,900,3600" --test_output=errors

# Run all HW tests:
bazel test //ipu_tensorflow_addons:all_tests --test_tag_filters=hw_poplar_test --test_env=TF_POPLAR_FLAGS="--max_infeed_threads=8 --max_compilation_threads=1" --test_env=TF_POPLAR_VLOG_LEVEL=1 --test_size_filters=small,medium,large --test_timeout="1200,1200,1200,1200" --test_output=errors

```

## Building a wheel
The bazel script target `//ipu_tensorflow_addons/tools/pip_package:build_pip_package` is used to
build wheel files. It links a subset of the python source files into a build directory,
and uses setuptools to build a wheel file from them.

The script requires positional arguments. For more information pass `--help` using any of the
methods below.

The script can be run completely through bazel using the following command:

```
lang=sh
bazel run //ipu_tensorflow_addons/tools/pip_package:build_pip_package -- <args>
```

Using `bazel run` means the script will be run from within the `bazel-bin` directory, so it's
recommended that you specify an absolute path for the `output_directory` argument. Alternatively,
you can build the script using bazel, then run it manually with the following commands:

```
lang=sh
bazel build //ipu_tensorflow_addons/tools/pip_package:build_pip_package
bash bazel-bin/ipu_tensorflow_addons/tools/pip_package/build_pip_package <args>
```

## Configuring
If you want to run tests, you will first need to run the `configure` script found in the root of the
repository. This creates a `.bazelrc.user` which specifies bazel configuration options required to
run the tests. One thing this does is set up the test environment so that TensorFlow and Poplar are
available.

Before running the script you need to set `TF_POPLAR_BASE` to the absolute path to your Poplar
installation. The script also requires positional arguments. Call `bash configure --help` for more
information.

```
lang=sh
export TF_POPLAR_BASE=<absolute-path-to-poplar-install>
bash ./configure <path-to-tf-wheel>
```

After calling the configure script, you can add customisations to the `.bazelrc.user` file.

## Running the tests
The following command can be used to run all of the tests. Unless you know what they do, it is
recommended that you pass all of the arguments shown in the command when running tests.

```
lang=sh
bazel test --test_env=TF_POPLAR_FLAGS="--max_infeed_threads=8 --use_ipu_model --max_compilation_threads=1 --ipu_model_tiles=8" --test_env=TF_POPLAR_VLOG_LEVEL=1 --test_size_filters=small,medium,large --test_timeout="240,360,900,3600" --test_output=errors //ipu_tensorflow_addons:all_tests
```

You can specify a different bazel test target to run a more specific set of tests.

```
lang=sh
bazel test --test_env=TF_POPLAR_FLAGS="--max_infeed_threads=8 --use_ipu_model --max_compilation_threads=1 --ipu_model_tiles=8" --test_env=TF_POPLAR_VLOG_LEVEL=1 --test_size_filters=small,medium,large --test_timeout="240,360,900,3600" --test_output=errors //ipu_tensorflow_addons/layers:gru_size_test
```

You can also run a single test by specifying `--test_arg`.

```
lang=sh
bazel test --test_env=TF_POPLAR_FLAGS="--max_infeed_threads=8 --use_ipu_model --max_compilation_threads=1 --ipu_model_tiles=8" --test_env=TF_POPLAR_VLOG_LEVEL=1 --test_size_filters=small,medium,large --test_timeout="240,360,900,3600" --test_output=errors //ipu_tensorflow_addons/layers:gru_size_test --test_arg IPUEffectiveTransformerLayerTest.testMismatchedSequenceLen
```

### Running HW tests
To run hardware tests, you need to pass a different set of arguments to bazel test.

```
lang=sh
bazel test --test_env=TF_POPLAR_FLAGS="--max_infeed_threads=8 --max_compilation_threads=1" --test_env=TF_POPLAR_VLOG_LEVEL=1 --test_size_filters=small,medium,large --test_timeout="1200,1200,1200,1200" --test_output=errors --test_tag_filters=hw_poplar_test --jobs N --test_env="TF_IPU_COUNT=N" //ipu_tensorflow_addons:all_tests
```
The `N` in `--jobs N` and `--test_env="TF_IPU_COUNT=N"` specifies the number of IPUs in the system
that a single test can use.

`--test_tag_filters=hw_poplar_test` tells bazel to only run tests tagged as hw tests.

Note that `--use_ipu_model` and `--ipu_model_tiles` have been omitted from `--test_env=TF_POPLAR_FLAGS`.
