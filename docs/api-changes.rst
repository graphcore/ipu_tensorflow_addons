IPU TensorFlow Addons API changes
---------------------------------

Release 2.5
~~~~~~~~~~~

The following changes have been made to the IPU TensorFlow Addons API in the Poplar SDK version 2.5.
This may require you to change your code.

Non-breaking changes
____________________

RNN available_memory_proportion_fwd/available_memory_proportion_bwd deprecated
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Thgite ``available_memory_proportion_fwd`` and ``available_memory_proportion_bwd`` arguments have been deprecated and will be removed from the following layers in a future release:

  - ipu_tensorflow_addons.v1.layers.rnn_ops.PopnnLSTM
  - ipu_tensorflow_addons.v1.layers.rnn_ops.PopnnDynamicLSTM
  - ipu_tensorflow_addons.v1.layers.rnn_ops.PopnnGRU
  - ipu_tensorflow_addons.v1.layers.rnn_ops.PopnnDynamicGRU
  - ipu_tensorflow_addons.v1.layers.rnn_ops.PopnnAUGRU

Thgitese values are now set using the ``'availableMemoryProportion'`` key of the ``options`` and ``options_bwd`` arguments correspondingly.

Release 2.4
~~~~~~~~~~~

First IPU TensorFlow Addons release.
