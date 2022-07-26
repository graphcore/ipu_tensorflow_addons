#!/usr/bin/env bash
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e # Stop on error
DIR=$(realpath $(dirname $0))

print_usage_and_exit() {
  cat <<EOM
Usage: $0 output-directory [--help|-h]

  output-directory
      The directory to place the wheel file in. The current working directory is used by default.
  -h, --help
      Print this help message.
EOM
  exit 1
}

# Check for -h or --help
if echo $@ | grep -w -- "-h\|--help" > /dev/null
  then
    print_usage_and_exit
fi

# Check argument count
if [ $# -ne 1 ]
  then
    echo "ERROR: Incorrect number of arguments"
    print_usage_and_exit
fi

OUTPUT_DIRECTORY=$(realpath $1)
if [ ! -d ${OUTPUT_DIRECTORY} ]; then
  echo "ERROR: Specified output directory could not be found: ${OUTPUT_DIRECTORY}"
  exit 1
fi
shift

cleanup_temp_dir() {
  cd "${DIR}"
  if [ -d "${TMP_DIR}" ]
    then
      rm -rf "${TMP_DIR}"
  fi
}

# Create temporary directory to build package in.
TMP_DIR="$(mktemp -d -t tmp.XXXXXXXXXX)"
trap 'cleanup_temp_dir' EXIT

# Link runfiles into temp directory.
# Runfiles is where the built //ipu_tensorflow_addons target is located.
if [ -d "${DIR}/runfiles/" ]
  then
    ln -s "${DIR}/runfiles/ai_graphcore/ipu_tensorflow_addons/" "${TMP_DIR}/"
elif [ -d "${DIR}/build_pip_package.runfiles/" ]
  then
    ln -s "${DIR}/build_pip_package.runfiles/ai_graphcore/ipu_tensorflow_addons/" "${TMP_DIR}/"
else
  echo "Unable to locate runfiles. Try rebuilding the bazel target."
  exit 1
fi

cp ipu_tensorflow_addons/tools/pip_package/setup.py ${TMP_DIR}
cp ${TMP_DIR}/ipu_tensorflow_addons/util/_dependency_versions.py ${TMP_DIR}

# Build wheel and copy it to specified output directory.
cd "${TMP_DIR}"
"${PYTHON_BIN_PATH:-python3}" setup.py bdist_wheel --dist-dir "${TMP_DIR}/" >/dev/null
WHEEL_NAME=$(basename ipu_tensorflow_addons-*.whl)
if [ ! -f ${WHEEL_NAME} ]; then
  echo "ERROR: Failed to locate wheel file: ${WHEEL_NAME}"
  exit 1
fi
cp -f "${WHEEL_NAME}" "${OUTPUT_DIRECTORY}/"
echo "Wheel created in ${OUTPUT_DIRECTORY}/${WHEEL_NAME}"
