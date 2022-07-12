IPU SavedModel CLI
~~~~~~~~~~~~~~~~~~

The IPU TensorFlow Addons includes a preview of the SavedModel command line
interface (CLI) tool for IPUs called ``ipu_saved_model_cli``.

.. note:: This tool is still in development and subject to change without notice. Not all functions will have been fully tested.

This section documents the IPU-specific functions of the SavedModel CLI for the :ref:`run <run-command>` and :ref:`convert <convert-command>` subcommands.

For more information about the tool, see the `TensorFlow SavedModel CLI documentation
<https://www.tensorflow.org/guide/saved_model#details_of_the_savedmodel_command_line_interface>`__.

.. _run-command:

Run subcommand
...............

.. code-block:: none

  ipu_saved_model_cli run [-h]
                          --dir DIR
                          --tag_set TAG_SET
                          --signature_def SIGNATURE_DEF_KEY
                          [--inputs INPUTS]
                          [--input_exprs INPUT_EXPRS]
                          [--input_examples INPUT_EXAMPLES]
                          [--outdir OUTDIR]
                          [--overwrite]
                          [--tf_debug]
                          [--worker WORKER]
                          [--init_ipu]
                          [--num_ipus NUM_IPUS]
                          [--matmul_amp MATMUL_AMP]
                          [--conv_amp CONV_AMP]
                          [--matmul_partial_type MATMUL_PARTIAL_TYPE]
                          [--conv_partial_type CONV_PARTIAL_TYPE]

The ``run`` subcommand supports the following IPU-specific command line options:

:samp:`--conv_amp {float}`
  The "available memory proportion": the proportion of memory to use for temporary values, intermediate sums and so on for convolutions. It must be a value between 0.0 and 1.0.
  If you want to change this value, you will need to specify it to the ``run`` command, even if you have already specified it for the ``convert`` command unless you are using the  embedded application runtime (see :ref:`convert-command`).

  See :py:attr:`convolutions.poplar_options <tensorflow.python.ipu.config.IPUConfig.convolutions.poplar_options>` for more information.

  The technical note :doc:`available-memory:index` provides more details and some practical examples of using ``availableMemoryProportion``.

  Default: 0.6.

:samp:`--conv_partial_type {string}`
  The type to use for intermediate values when doing a convolution. This can be "float" (the default) or "half".
  If you want to change this type, you will need to specify it to the ``run`` command, even if you have already specified it for the ``convert`` command unless you are using the  embedded application runtime (see :ref:`convert-command`).

  See the :ref:`Memory and Performance Optimisation on the IPU <memory-performance-optimisation:partials type>` technical notes for some practical tips on selecting the type for partial results.

  Default: "float".

:samp:`--init_ipu`
  If specified, then the SavedModel will call :py:func:`~tensorflow.python.ipu.config.configure_ipu_system` when it starts execution. This option should be only used if the worker is an IPU job.

:samp:`--matmul_amp {float}`
  The proportion of memory to use for temporary values, intermediate sums and so on for matrix multiplications. Must be a value between 0.0 and 1.0.

  See :py:attr:`matmuls.poplar_options <tensorflow.python.ipu.config.IPUConfig.matmuls.poplar_options>` for more information.

  The technical note :doc:`available-memory:index` provides more details and some practical examples of using ``availableMemoryProportion``.

  Default: 0.6.

:samp:`--matmul_partial_type {string}`
  The type to use for intermediate values when doing a matrix multiply. This can be "float" (the default) or "half".

  See the :ref:`Memory and Performance Optimisation on the IPU <memory-performance-optimisation:partials type>` technical notes for some practical tips on selecting the type for partial results.

  Default: "float".

:samp:`--num_ipus {integer}`
  The number of IPUs that the SavedModel will use for inference. In most cases this must be a power of 2. The command line tool does not check this, but you may get an error from the application if any necessary constraints are not met.

  Default: 1.

.. _convert-command:

Convert subcommand
...................

Convert the SavedModel with IPU integration.

.. code-block:: none

  ipu_saved_model_cli convert ipu [-h]
                                  [--excluded_nodes EXCLUDED_NODES [EXCLUDED_NODES ...]]
                                  [--num_ipus NUM_IPUS]
                                  [--matmul_amp MATMUL_AMP]
                                  [--conv_amp CONV_AMP]
                                  [--matmul_partial_type MATMUL_PARTIAL_TYPE]
                                  [--conv_partial_type CONV_PARTIAL_TYPE]
                                  [--batch_size BATCH_SIZE]
                                  [--batch_per_step BATCH_PER_STEP]
                                  [--precision_mode PRECISION_MODE]
                                  [--gelu_replacement GELU_REPLACEMENT]
                                  [--no_ipu_placement]
                                  [--int64_to_int32_conversion]
                                  [--precision_conversion_excluded_nodes PRECISION_CONVERSION_EXCLUDED_NODES [PRECISION_CONVERSION_EXCLUDED_NODES ...]]
                                  [--remove_excluded_nodes]
                                  [--manual_sharding MANUAL_SHARDING]
                                  [--embedded_runtime_save_config EMBEDDED_RUNTIME_SAVE_CONFIG]
                                  [--pipeline_cfg PIPELINE_CFG]
                                  [--config_file CONFIG_FILE]

This has the following command line options:

:samp:`--batch_per_step {integer}`
  Repeat count for :py:func:`~tensorflow.python.ipu.loops.repeat` or :py:mod:`~tensorflow.python.ipu.pipelining_ops`. If 0, it will not turn off the loop repeat IPU wrapper. If the :ref:`ipu embedded application runtime` is enabled and batches per step is 0, then it will be changed to 1.

  Default: 0

:samp:`--batch_size {integer}`
  The :term:`micro batch size <glossary:Micro batch size>` to be used by the model.

  Default: 1

:samp:`--config_file {path}`
  Path to a JSON-format configuration file that defines all the options to the command.
  :ref:`example-configuration-file`.


:samp:`--conv_amp {float}`
  The "available memory proportion": the proportion of memory to use for temporary values, intermediate sums and so on for convolutions. Must be a value between 0.0 and 1.0.

  See :py:attr:`IPUConfig.convolutions.poplar_options <tensorflow.python.ipu.config.IPUConfig.convolutions.poplar_options>` for more information.

  The technical note :doc:`available-memory:index` provides more details and some practical examples of using ``availableMemoryProportion``.

  Default: 0.6.

:samp:`--conv_partial_type {string}`
  The type to use for intermediate values when doing a convolution. This can be "float" (the default) or "half".

  See the :ref:`Memory and Performance Optimisation on the IPU <memory-performance-optimisation:partials type>` technical notes for some practical tips on selecting the type for partial results.

  Default: "float".

:samp:`--embedded_runtime_save_config {json}`
  A JSON string defining the configuration for embedded application runtime compilation. For example:

  .. code-block:: json

    {
      "embedded_runtime_exec_cachedir": "/path/to/exec",
      "runtime_api_timeout_us": 5000
    }

  where:

  * ``embedded_runtime_exec_cachedir`` sets the directory where the compiled embedded-application runtime file is created, and
  * ``runtime_api_timeout_us`` sets the limit (in microseconds) on the time the IPU will wait for data. See :ref:`timeout` for more information.

:samp:`--excluded_nodes {string1, string2, string3, ...}`
  A list of nodes that will not be be placed on the IPU.

  Default: none.

:samp:`--gelu_replacement {string}`
  The nodes that define the GELU activation function will be replaced with the IPU-optimised GELU op (:py:func:`tensorflow.python.ipu.nn_ops.gelu`), which will reduce the amount of memory required and improve the throughput.

  This is a JSON-format string. For example:

  .. code-block:: json

    {
      "gelu_replacement": {
        "nodes": [                    // Nodes in GELU function (regular expressions)
          "intermediate/dense/Sqrt$",
          "intermediate/dense/truediv$",
          "intermediate/dense/Erf$",
          "intermediate/dense/add$",
          "intermediate/dense/mul$",
          "intermediate/dense/mul_1$",
          "intermediate/dense/Sqrt/x$",
          "intermediate/dense/add/x$",
          "intermediate/dense/mul/x$"
        ],
        "node_as_gelu_input": [       // The names of GELU function inputs (regex)
          "encoder/layer_[0-9]*/intermediate/dense/BiasAdd"
        ],
        "node_use_gelu_output": [     // The names of GELU function outputs (regex)
          "encoder/layer_[0-9]*/output/dense/MatMul"
        ]
      }
    }

:samp:`--int64_to_int32_conversion`
  Convert ops with int64 type to int32 type.

  The IPU does not support int64. Ops placed on the IPU that have int64 inputs/outputs will be modified to use int32 instead. Prior to sending data to the IPU, any int64 values will be cast to int32 values.

:samp:`--manual_sharding {regex-for-ipu0, regex-for-ipu1, regex-for-ipu2, ...}`
  A list of regular expression strings, one for each IPU. Nodes whose names match an expressions will be placed on that IPU. Nodes which do not match an expression will be placed on IPU 0.

  An error will be raised if the number of regular expressions is not equal to the number of IPUs?

  Default: none.

:samp:`--matmul_amp {float}`
  The proportion of memory to use for temporary values, intermediate sums and so on for matrix multiplications. Must be a value between 0.0 and 1.0.
  See :py:attr:`IPUConfig.matmuls.poplar_options <tensorflow.python.ipu.config.IPUConfig.matmuls.poplar_options>` for more information.

  The technical note :doc:`available-memory:index` provides more details and some practical examples of using ``availableMemoryProportion``.

  Default: 0.6.

:samp:`--matmul_partial_type {string}`
  The type to use for intermediate values when doing a matrix multiply. This can be "float" (the default) or "half".

  See the :ref:`Memory and Performance Optimisation on the IPU <memory-performance-optimisation:partials type>` technical notes for some practical tips on selecting the type for partial results.

  Default: "float".

:samp:`--merge_subgraphs`
  Merge multiple IPU subgraphs into one with the IPU :py:func:`~tensorflow.python.ipu.ipu_compiler.compile` function.

:samp:`--no_ipu_placement`
  If set, no nodes will be placed on IPUs.

:samp:`--num_ipus {integer}`
  The number of IPUs that the SavedModel will use for inference.
  Default: 1.

:samp:`--pipeline_cfg {string}`
  A JSON-format string that defines the configuration of the pipeline. See :ref:`pipeline-configuration` for more information.

:samp:`--precision_conversion_excluded_nodes {string1, string2, string3, ...}`
  A list of nodes that will not have their precision changed by the ``--precision_mode`` option.

  Default: none.

:samp:`--precision_mode {string}`
  The precision of the output model. This can be either "FP16" or "FP32".

  Default: FP32

:samp:`--remove_excluded_nodes`
  Remove the nodes listed in ``--excluded_nodes`` from the graph.


.. _pipeline-configuration:

Pipeline configuration
.......................

The pipeline configuration specifies how to distribute the nodes in the model across the IPUs. It has three options:

- ``auto``: This automatically splits the model into several pipeline stages to optimise performance, searching for the minimum number of IPUs that model should use.
- ``manual``: Specifies how to split the model into several pipeline stages with user-specified regular expressions.
- ``load``: Load the pipeline configuration from a file.

In general, you would use ``auto`` first, and then use ``load`` mode to adjust the configuration if you are not satisfied with the results.

The pipeline configuration is specified using the following options:

- `converter`: specifies the mode of the pipeline converter. This must be one of ``auto``, ``manual`` or  ``load``.
  Each of these has a set of configuration options, described below.

  - ``auto``

    - `fine_tune_iter` (optional): The maximum number of iterations for fine tuning.
    - `ipu_model`: Run on the IPU Model (default: true). If false, the pipelined model will be run on IPUs.
    - `profiling_root_dir` (optional): The directory where the SavedModel tool will write the profiling file (default ``./pipeline-profiling``).
    - `priority` (optional): The priority of the balancing strategy. Must be "cycle" (default) or "memory".

      - "cycle": balance the compute cycles for each pipeline stage
      - "memory": balance the memory use for each pipeline stage

    - `max_ipu_quantity` (optional): The maximum number of IPUs that can be used (default 64).
    - `min_ipu_quantity` (optional): The minimum number of IPUS that can be used (default 2).
    - `solution_dir` (optional): The directory where the SavedModel will write the configuration file describing the pipeline it has created.

  - ``load``

    - `ipu_model`: Run on the IPU Model (default: true).
    - `profiling_root_dir` (optional): The directory where the SavedModel tool will write the profiling file (default ``./pipeline-profiling``).
    - `solution_path` (required): The file containing the pipeline configuration to be read by the SavedModel tool.
    - `profiling_enable` (optional): Enable profiling.

  - ``manual``

    - `ipu_model`: Run on the IPU Model (default: true).
    - `profiling_root_dir` (optional): The directory where the SavedModel tool will write the profiling file (default ``./pipeline-profiling``).
    - `manual_pipeline_config` (required): A list of regular expressions, one for each IPU, to match nodes to the IPUs in the pipeline. Nodes whose names match an expressions will be placed on the corresponding IPU. Nodes which do not match an expression will be placed on IPU 0.
    - `device_info` (required): the mapping of pipeline stages to IPU targets (see the :ref:`description of device mapping <device-mapping-description>` for more information).
    - `solution_dir` (optional): The directory where the SavedModel will write the configuration file describing the pipeline.
    - `profiling_enable` (optional): Enable profiling.

Pipeline development
....................

For the auto pipeline optimization option, the SavedModel CLI tool will:

1. Run the model
2. Generate and analyse the profiling information
3. Find an optimal solution to split the model and save the result to the solution file

For step 2, the tool needs to get cycle information from the profile. However, it is possible that the model will raise an out of memory error. The tool avoids this by running the model on the IPU Model, which will does not generate out of memory errors.

The auto option has the following limitations:

- The design does not consider other data types like ``tf.resource``.
- It does not support INT64 data type.
- The SavedModel input needs to be frozen.
- The minimum number of IPU search spaces is 2, which means searching from IPU number >= 2 since the model can be fit in a single IPU and does not need the pipeline methodology.
- The input tensor shape needs to be fixed, excluding batch size.
- It cannot handle ``control_dependency`` nodes.
- The first dimension of input tensors must be ``micro_batch_size``.

Pipeline solution file
.......................

The pipeline solution file is a JSON definition of how pipeline stages are mapped to IPUs. This is generated by the ``auto`` and ``manual`` options, and read by the ``load`` option.

.. code-block:: json

  {
    "device_maping": [0, 1],
    "pipeline_mapping": {
      <node name>: <pipeline stage id>,
      <node name>: <pipeline stage id>
    }
  }

Where:

.. _device-mapping-description:

 * ``device_mapping`` is a list of length equal to the number of computational stages. Each element in the list specifies the ID of the IPU that the corresponding pipeline stage will be placed on.
 * ``pipeline_mapping`` specifies which nodes will be mapped to each pipeline stage.

.. _example-configuration-file:

Example configuration file
...........................

.. code-block:: json
  :caption: configuration.json
  :name: configuration-json
  :linenos:

    {
      "batch_size": 1,
      "num_ipus": 1,
      "batch_per_step": 1,
      "matmul_amp": 0.6,
      "matmul_partial_type": "half",
      "conv_amp": 0.6,
      "conv_partial_type": "half",
      "no_ipu_placement": false,
      "excluded_nodes": [
        "^strided_slice_1$",
        "NotEqual",
        "Assert"
      ],
      "remove_excluded_nodes": true,
      "merge_subgraphs": true,
      "precision_mode": "FP16",
      "precision_conversion_excluded_nodes": [
        "^add$"
      ],
      "int64_to_int32_conversion": true,
      "gelu_replacement": {
        "nodes": [
          "intermediate/dense/Sqrt$",
          "intermediate/dense/truediv$",
          "intermediate/dense/Erf$",
          "intermediate/dense/add$",
          "intermediate/dense/mul$",
          "intermediate/dense/mul_1$",
          "intermediate/dense/Sqrt/x$",
          "intermediate/dense/add/x$",
          "intermediate/dense/mul/x$"
        ],
        "node_as_gelu_input": [
          "encoder/layer_[0-9]*/intermediate/dense/BiasAdd"
        ],
        "node_use_gelu_output": [
          "encoder/layer_[0-9]*/output/dense/MatMul"
        ]
      },
      "manual_sharding": [
        [
          "^sharding0"
        ],
        [
          "^sharding1"
        ]
      ],
      "pipeline_cfg": {
        // auto pipeline configuration.
        "converter": "auto",
        "fine_tune_iter": 5,
        "ipu_model": true,
        "max_ipu_quantity": 64,
        "min_ipu_quantity": 2,
        "priority": "cycle",
        "profiling_root_dir": "/path/to/profiling_root_dir",
        "solution_dir": "/path/to/solution_dir",
        // pipeline configuration loader configuration.
        "converter": "load",
        "ipu_model": true,
        "profiling_root_dir": "profiling",
        "solution_path": "solution/greedy_search.pipeconfig",
        "profiling_enable": false,
        // manual pipeline configuration.
        "converter": "manual",
        "ipu_model": true,
        "profiling_root_dir": "profiling",
        "solution_dir": "solution",
        "manual_pipeline_config": [
          [
            "input_3",
            "^middle/unit_0",
            "^middle/unit_1",
            "^middle/unit_2/",
            "^middle/unit_3",
            "^middle/unit_4"
          ],
          [
            "^middle/unit_5",
            "input_1",
            "input_2",
            "^right/unit_0",
            "^right/unit_1",
            "^right/unit_2",
            "^left/unit_0"
          ],
          [
            "^left/unit_1",
            "^left/unit_2",
            "^left/unit_3",
            "^left/unit_4",
            "concat",
            "^res/unit_0/"
          ],
          [
            "^res/unit_1",
            "^res/unit_2",
            "^res/down/",
            "^res/add"
          ]
        ],
        "device_info": [
          0,
          1,
          1,
          0
        ],
        "profiling_enable": true
      },
      "embedded_runtime_save_config": {
        "runtime_api_timeout_us": 5000,
        "embedded_runtime_exec_cachedir": "bert_poplar_exec"
      }
    }
