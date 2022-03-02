IPU TensorFlow Addons Python API
--------------------------------

TensorFlow layers
^^^^^^^^^^^^^^^^^

.. automodule:: ipu_tensorflow_addons.layers
  :members:
  :special-members: __init__
  :imported-members:

TensorFlow optimizers
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: ipu_tensorflow_addons.optimizers
  :members:
  :special-members: __init__
  :imported-members:

IPU TensorFlow saved model tool CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
  :module: ipu_tensorflow_addons.saved_model_tool.saved_model_cli
  :func: create_parser
  :prog: ipu_saved_model_tool

  show : @replace
    Usage examples:

    To show all tag-sets in a SavedModel:

    .. code:: bash

      $ipu_saved_model_tool show --dir /tmp/saved_model

    To show all available SignatureDef keys in a MetaGraphDef specified by its `tag-set`:

    .. code:: bash

      $ipu_saved_model_tool show --dir /tmp/saved_model --tag_set serve


    For a MetaGraphDef with multiple tags in the `tag-set`, all tags must be passed in, separated by `,` :

    .. code:: bash

      $ipu_saved_model_tool show --dir /tmp/saved_model --tag_set serve,gpu


    To show all inputs and outputs TensorInfo for a specific SignatureDef specified by the SignatureDef key in a MetaGraph.

    .. code:: bash

      $ipu_saved_model_tool show --dir /tmp/saved_model --tag_set serve --signature_def serving_default


    To show all available information in the SavedModel:

    .. code:: bash

      $ipu_saved_model_tool show --dir /tmp/saved_model --all


  run : @replace
    Usage example:

    To run input tensors from files through a MetaGraphDef and save the output tensors to files:

    .. code:: bash

      $ipu_saved_model_tool show --dir /tmp/saved_model --tag_set serve \
        --signature_def serving_default \
        --inputs input1_key=/tmp/124.npz[x],input2_key=/tmp/123.npy \
        --input_exprs 'input3_key=np.ones(2)' \
        --input_examples 'input4_key=[{"id":[26],"weights":[0.5, 0.5]}]' \
        --outdir=/out


    For more information about input file format, please see:


    `https://www.tensorflow.org/guide/saved_model_cli`


  scan : @replace
    Usage example:

    To scan for blacklisted ops in SavedModel:

    .. code:: bash

      $ipu_saved_model_tool scan --dir /tmp/saved_model



    To scan a specific MetaGraph, pass in `--tag_set`


  convert : @replace
    Usage example:

    To convert the SavedModel to one that have TensorRT ops:

    .. code:: bash

      $ipu_saved_model_tool convert \
        --dir /tmp/saved_model \
        --tag_set serve \
        --output_dir /tmp/saved_model_trt \
        tensorrt


