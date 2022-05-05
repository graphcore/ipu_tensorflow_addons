# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ADDONS_DIR = Path(os.path.realpath(__file__)).parent

RED = "\033[0;31m"
GREEN = "\033[0;32m"
GRAY = "\033[0;90m"
NO_COLOR = "\033[0m"

red = lambda x: f"{RED}{x}{NO_COLOR}"
green = lambda x: f"{GREEN}{x}{NO_COLOR}"
gray = lambda x: f"{GRAY}{x}{NO_COLOR}"


def error(message):
  print(f"{red('ERROR:')} {message}")


def info(message):
  print(f"{green('INFO:')} {message}")


def exit_with_error(message, exit_code=1):
  error(message)
  sys.exit(exit_code)


def run_cmd(cmd, env=None, verbose=True):
  info(f"Running command: {cmd}")
  try:
    result = subprocess.run(shlex.split(cmd),
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=60,
                            check=True)
  except subprocess.CalledProcessError as e:
    exit_with_error(f"Running command failed (return code {e.returncode}):"
                    f"\n {red(e.stderr.decode().rstrip())}")

  stdout = result.stdout.decode().rstrip()
  if verbose:
    print(gray(stdout))

  return stdout


def run_cmds_and_get_env(cmds):
  # Suppress any output from the commands.
  cmds = [f"{cmd} 1>/dev/null" for cmd in cmds]
  # Join commands.
  cmd = " && ".join(cmds)
  # Capture the env variables exported from the command.
  cmd = f"bash -c '{cmd} && env'"

  stdout = run_cmd(cmd, verbose=False)

  env = stdout.split('\n')
  env = [entry.partition("=")[0::2] for entry in env]
  env = {k.strip(): v.strip() for k, v in env}

  return env


def get_tf_whl_path(args):
  return Path(args.tf_whl_path).absolute()


def get_keras_whl_path(args):
  return Path(args.keras_whl_path).absolute()


def get_venv_path(args):
  return Path(args.venv_path).absolute()


def get_python_bin(args):
  return (get_venv_path(args) / "bin/python3").absolute()


def get_python_lib(args):
  return (get_venv_path(args) / "lib/python3.6/site-packages").absolute()


def get_pip_bin(args):
  return (get_venv_path(args) / "bin/pip3").absolute()


def get_venv_activate_script(args):
  return (get_venv_path(args) / "bin/activate").absolute()


def get_disk_cache_path(args):
  return Path(args.disk_cache).absolute()


def get_tf_poplar_env_key():
  return "TF_POPLAR_BASE" if os.environ.get(
      "TF_POPLAR_BASE") is not None else "TF_POPLAR_SANDBOX"


def get_tf_poplar_path():
  tf_poplar_env_key = get_tf_poplar_env_key()
  return Path(os.environ.get(tf_poplar_env_key)).absolute()


def get_poplar_activate_script():
  tf_poplar_env_key = get_tf_poplar_env_key()
  tf_poplar_path = get_tf_poplar_path()
  paths = {
      "TF_POPLAR_BASE": tf_poplar_path / "enable.sh",
      "TF_POPLAR_SANDBOX": tf_poplar_path.parent / "activate.sh",
  }
  return paths[tf_poplar_env_key].absolute()


def get_package_version(args, package):
  """Get the version of a Python package."""
  stdout = run_cmd(f"{get_pip_bin(args)} show {package}", verbose=False)
  info_ = stdout.split('\n')
  for line in info_:
    k, _, v = line.partition(":")
    if k == "Version":
      version = v.strip()
      semver = version.split(".")
      if not (len(semver) == 3 and all((i.isdigit() for i in semver))):
        return exit_with_error(
            f"Couldn't infer '{package}' version - '{version}' is not a valid "
            "semantic version.")
      return version

  return exit_with_error(f"Couldn't infer '{package}' version.")


def get_tf_version(args):
  return get_package_version(args, "tensorflow")


def get_keras_version(args):
  return get_package_version(args, "keras")


def create_venv(args):
  # Remove old venv if it exists and `--rm-venv` is set.
  if get_venv_path(args).exists() and args.rm_venv:
    info(f"Removing old Python virtual environment from "
         f"'{get_venv_path(args)}'.")
    shutil.rmtree(get_venv_path(args))

  # If a virtual environment already exists and it wasn't removed above, then
  # reuse it.
  if get_venv_path(args).exists() and not args.rm_venv:
    info(f"Re-using old Python virtual environment from "
         f"'{get_venv_path(args)}'.")
  else:  # Otherwise, create a new virtual environment.
    # Create venv.
    run_cmd(f"python3 -m venv {get_venv_path(args)} --without-pip",
            verbose=False)

    # Install pip.
    is_python_3_6 = str(sys.version).startswith("3.6.")
    pip_url = ("https://bootstrap.pypa.io/pip/3.6/get-pip.py" if is_python_3_6
               else "https://bootstrap.pypa.io/pip/get-pip.py")
    info(f"Installing `pip` from '{pip_url}'.")
    with urllib.request.urlopen(pip_url) as response:
      with tempfile.NamedTemporaryFile("wb") as f:
        f.write(response.read())
        run_cmd(f"{get_python_bin(args)} {f.name}")

  # Install requirements.
  info(f"Installing TensorFlow wheel from '{get_tf_whl_path(args)}'.")
  run_cmd(f"{get_pip_bin(args)} install --force-reinstall "
          f"{get_tf_whl_path(args)}")
  info(f"Installing Keras wheel from '{get_keras_whl_path(args)}'.")
  run_cmd(f"{get_pip_bin(args)} install --force-reinstall "
          f"{get_keras_whl_path(args)}")


def create_action_env(args, env):
  # pylint: disable=unused-argument
  new_env = {}
  for entry in args.env:
    k, _, v = entry.partition("=")
    new_env[k] = v
  return new_env


def create_test_env(args, env):
  # pylint: disable=unused-argument
  new_env = {}
  new_env["PATH"] = env["PATH"]
  new_env["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH"]
  new_env["PYTHONPATH"] = env["PYTHONPATH"]
  return new_env


def dump_env(env, name):
  info(f"Using {name} environment:")
  for k, v in env.items():
    print(gray(f"'{k}': '{v}'"))


def write_bazelrc(args, action_env, test_env):
  bazel_rc = []
  rc = lambda x: bazel_rc.append(x)  # pylint: disable=unnecessary-lambda

  # yapf: disable
  for k, v in action_env.items():
    rc(f"build --action_env={k}={v}")
  for k, v in test_env.items():
    rc(f"build --test_env={k}={v}")
  rc(f"build --python_path={get_python_bin(args)}")
  rc(f"build --//ipu_tensorflow_addons:tensorflow_version={get_tf_version(args)}")  # pylint: disable=line-too-long
  rc(f"build --//ipu_tensorflow_addons:keras_version={get_keras_version(args)}")
  if args.disk_cache:
    rc(f"build --disk_cache={get_disk_cache_path(args)}")
  # yapf: enable

  info("Writing configuration to `.addons_configure.bazelrc`.")
  with open(ADDONS_DIR / ".addons_configure.bazelrc", "w",
            encoding="utf-8") as f:
    f.writelines([line + "\n" for line in bazel_rc])


def argument_parser():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--tf-whl-path',
      default=None,
      required=True,
      help="The path to the IPU TensorFlow .whl file you want to build the "
      "Addons for.")
  parser.add_argument(
      '--keras-whl-path',
      default=None,
      required=True,
      help="The path to the IPU Keras .whl file you want to build the Addons "
      "for.")
  parser.add_argument(
      '--venv-path',
      default=ADDONS_DIR / "venv",
      help="Choose the location of the Python virtual environment. "
      "If a virtual environment already exists at the specified path, it will "
      "be reused unless `--rm-venv` is set.")
  parser.add_argument(
      '--rm-venv',
      action="store_true",
      default=False,
      help="Delete the Python virtual environment specified by `--venv-path` if"
      " it exists.")
  parser.add_argument(
      '--env',
      action='append',
      default=[],
      help="Specify an environment variable to forward to Bazel. The variables "
      "will be set during builds and testing. Multiple uses are accumulated.")
  parser.add_argument(
      '-c',
      '--disk-cache',
      default=None,
      help="Enable and specify the directory for Bazel's disk cache.")
  return parser


def check_args(args):
  if not os.path.exists(args.tf_whl_path):
    exit_with_error(f"Specified TensorFlow wheel path does not exist: "
                    f"{args.tf_whl_path}.")
  if not os.path.exists(args.keras_whl_path):
    exit_with_error(f"Specified Keras wheel path does not exist: "
                    f"{args.keras_whl_path}.")
  for entry in args.env:
    if entry.find("=") < 1:
      exit_with_error(
          f"Wrong format for argument `--env {entry}`. The correct "
          "format is `VARIABLE_NAME=variable_value`.")


def check_env():
  path_or_none = lambda x: x if x is None else Path(x)
  tf_poplar_base = path_or_none(os.environ.get("TF_POPLAR_BASE"))
  tf_poplar_sandbox = path_or_none(os.environ.get("TF_POPLAR_SANDBOX"))

  def check_popc_path(var_name, popc_path):
    if not popc_path.exists():
      exit_with_error(f"{var_name} is set to an invalid path: '{popc_path}' "
                      "does not exist.")

  if not (tf_poplar_base is None) ^ (tf_poplar_sandbox is None):
    exit_with_error(
        f"Exactly one of `TF_POPLAR_BASE='{tf_poplar_base}'` and "
        f"`TF_POPLAR_SANDBOX='{tf_poplar_sandbox}'` should be set.")
  elif tf_poplar_base:
    check_popc_path("TF_POPLAR_BASE", tf_poplar_base / "bin/popc")
  elif tf_poplar_sandbox:
    check_popc_path("TF_POPLAR_SANDBOX", tf_poplar_sandbox / "poplar/bin/popc")


def main(args):
  info("Configuring IPU TensorFlow Addons!")

  check_args(args)
  check_env()

  create_venv(args)

  poplar_activate_script = get_poplar_activate_script()
  venv_activate_script = get_venv_activate_script(args)

  env = run_cmds_and_get_env(
      [f"source {poplar_activate_script}", f"source {venv_activate_script}"])
  action_env = create_action_env(args, env)
  test_env = create_test_env(args, env)
  dump_env(action_env, "action")
  dump_env(test_env, "test")

  write_bazelrc(args, action_env, test_env)

  info("Configure script completed successfully!")


if __name__ == '__main__':
  main(argument_parser().parse_args())
