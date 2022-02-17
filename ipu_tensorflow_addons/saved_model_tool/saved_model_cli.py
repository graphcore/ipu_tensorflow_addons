# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#
# This file has been modified by Graphcore Ltd.
# ==============================================================================
"""
Command-line interface to convert and execute a graph in a SavedModel on IPU.
"""

import ast
import sys
import textwrap
import argparse
import json
from tensorflow.python.tools import saved_model_cli as cli
from tensorflow.python.ipu.config import IPUConfig
from ipu_tensorflow_addons.saved_model_tool import ipu_convert


def _init_ipu(args):
  print('Initializing IPU System ...')
  cfg = IPUConfig()
  cfg.auto_select_ipus = args.num_ipus
  cfg.matmuls.poplar_options = {"availableMemoryProportion": args.matmul_amp}
  cfg.convolutions.poplar_options = {
      "availableMemoryProportion": args.conv_amp
  }
  cfg.matmuls.poplar_options.update({"partialsType": args.matmul_partial_type})
  cfg.convolutions.poplar_options.update(
      {"partialsType": args.conv_partial_type})
  cfg.configure_ipu_system()


def run(args):
  """Function triggered by run command.

  Args:
    args: A namespace parsed from command line.

  Raises:
    AttributeError: An error when neither --inputs nor --input_exprs is passed
    to run command.
  """
  if args.init_ipu:
    _init_ipu(args)

  cli.run(args)


def convert_with_ipu(args):
  """Function triggered by 'convert ipu' command.

  Args:
    args: A namespace parsed from command line.
  """
  ipu_convert.create_inference_graph(
      batch_size=args.batch_size,
      input_saved_model_dir=args.dir,
      input_saved_model_tags=args.tag_set.split(','),
      input_saved_model_signature_key=None,
      output_saved_model_dir=args.output_dir,
      excluded_nodes=args.excluded_nodes,
      num_ipus=args.num_ipus,
      batch_per_step=args.batch_per_step,
      matmul_amp=args.matmul_amp,
      conv_amp=args.conv_amp,
      matmul_partial_type=args.matmul_partial_type,
      conv_partial_type=args.conv_partial_type,
      ipu_placement=not bool(args.no_ipu_placement),
      int64_to_int32_conversion=bool(args.int64_to_int32_conversion),
      remove_excluded_nodes=bool(args.remove_excluded_nodes),
      precision_conversion_excluded_nodes=(
          args.precision_conversion_excluded_nodes),
      manual_sharding=args.manual_sharding,
      config_file=args.config_file,
      precision_mode=args.precision_mode,
      embedded_runtime_save_config=args.embedded_runtime_save_config)


def list_of_lists_str(strs):
  return ast.literal_eval(strs)


def json_dict(strs):
  return json.loads(strs)


def create_parser():
  """Creates a parser that parse the command line arguments.

  Returns:
    A namespace parsed from command line arguments.
  """
  parser = cli.create_parser()

  # Modify the run arguments to add IPU args and point it to our run wrapper.
  parser_run = parser._subparsers._group_actions[0].choices['run']  # pylint: disable=W0212
  parser_run.add_argument(
      '--init_ipu',
      action='store_true',
      default=None,
      help='If specified, ipu.utils.configure_ipu_system() will be called. '
      'This option should be only used if the worker is a IPU job.')
  parser_run.add_argument('--num_ipus',
                          type=int,
                          default=1,
                          help="Number of ipus to utilize of inference.")
  parser_run.add_argument('--matmul_amp',
                          type=str,
                          default='0.6',
                          help='AvailableMemoryProportion for matmul.')
  parser_run.add_argument('--conv_amp',
                          type=str,
                          default='0.6',
                          help='AvailableMemoryProportion for convolution.')
  parser_run.add_argument('--matmul_partial_type',
                          type=str,
                          default='float',
                          help='Partial type for matmul.')
  parser_run.add_argument('--conv_partial_type',
                          type=str,
                          default='float',
                          help='Partial type for convolution.')
  parser_run.set_defaults(func=run)

  # Add an `ipu` choice to the "conversion methods" subparsers
  parser_convert_with_ipu = argparse.ArgumentParser(
      'ipu',
      description="Convert the SavedModel with IPU integration.",
      formatter_class=argparse.RawTextHelpFormatter)

  parser_convert_with_ipu.add_argument(
      '--excluded_nodes',
      type=str,
      nargs='+',
      default=[],
      help='Ops that do not need to be placed on the IPU.')
  parser_convert_with_ipu.add_argument('--num_ipus',
                                       type=int,
                                       default=1,
                                       help='Number of IPUs.')
  parser_convert_with_ipu.add_argument(
      '--matmul_amp',
      type=float,
      default='0.6',
      help='AvailableMemoryProportion for matmul.')
  parser_convert_with_ipu.add_argument(
      '--conv_amp',
      type=float,
      default='0.6',
      help='AvailableMemoryProportion for convolution.')
  parser_convert_with_ipu.add_argument('--matmul_partial_type',
                                       type=str,
                                       default='float',
                                       help='Partial type for matmul.')
  parser_convert_with_ipu.add_argument('--conv_partial_type',
                                       type=str,
                                       default='float',
                                       help='Partial type for convolution.')
  parser_convert_with_ipu.add_argument('--batch_size',
                                       type=int,
                                       default=1,
                                       help='Batch size.')
  parser_convert_with_ipu.add_argument(
      '--batch_per_step',
      type=int,
      default=0,
      help=("Repeat count for `loop.repeat` or `pipelining_op`. "
            "If 0, it will not turn off the loop repeat IPU wrapper. "
            "If embedded application runtime is enabled and "
            "Batch_per_step is 0, it will be changed to 1."))
  parser_convert_with_ipu.add_argument('--precision_mode',
                                       type=str,
                                       default=None,
                                       action='store',
                                       help='The precision of output model.')
  parser_convert_with_ipu.add_argument(
      '--no_ipu_placement',
      action='store_true',
      help='If set, will not do IPU placement.')
  parser_convert_with_ipu.add_argument(
      '--int64_to_int32_conversion',
      action="store_true",
      default=False,
      help='Convert nodes with int64 type to int32 type.')
  parser_convert_with_ipu.add_argument(
      '--precision_conversion_excluded_nodes',
      type=str,
      nargs='+',
      default=[],
      help='Ops that will not have their precision changed by --precision_mode.'
  )
  parser_convert_with_ipu.add_argument(
      '--remove_excluded_nodes',
      action='store_true',
      default=False,
      help='Remove nodes in excluded_nodes from graph.')
  parser_convert_with_ipu.add_argument(
      '--merge_subgraphs',
      action='store_true',
      default=False,
      help='Merge multiple IPU subgraphs into one with `ipu compile` API.')

  parser_convert_with_ipu.add_argument(
      '--manual_sharding',
      action='store',
      type=list_of_lists_str,
      default=[],
      help=(
          "A list containing a list of regular expression strings for each IPU."
          " Nodes who's names match the expressions for a given IPU"
          " will be sharded on that IPU."
          " Nodes which match no expressions will be placed on IPU0."))
  parser_convert_with_ipu.add_argument(
      '--embedded_runtime_save_config',
      type=json_dict,
      default=None,
      help=('The configuration of embedded application runtime compilation, '
            'a JSON like string e.g.\n') + textwrap.dedent("""\
            {
                "embedded_runtime_exec_cachedir": "/path/to/exec",
                "runtime_api_timeout_us": 5000
            }"""))

  parser_convert_with_ipu.add_argument('--config_file',
                                       type=str,
                                       default=None,
                                       action="store",
                                       help='Config file path (JSON format).')
  parser_convert_with_ipu.set_defaults(func=convert_with_ipu)

  parser_convert = parser._subparsers._group_actions[0].choices['convert']  # pylint: disable=W0212
  parser_convert._subparsers._group_actions[0].choices[  # pylint: disable=W0212
      'ipu'] = parser_convert_with_ipu

  return parser


def main():
  parser = create_parser()
  args = parser.parse_args()
  if not hasattr(args, 'func'):
    parser.error('too few arguments')
  args.func(args)


if __name__ == '__main__':
  sys.exit(main())
