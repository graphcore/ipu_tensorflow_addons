Optimizers
~~~~~~~~~~

The `ipu_tensorflow_addons.keras.optimizers` namespace contains IPU-specific
implementations of Keras optimizers from TensorFlow Addons and upstream Keras:

* `AdamIpuOptimizer <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.optimizers.AdamIpuOptimizer>`_

* `LAMBIpuOptimizer <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.optimizers.LAMBIpuOptimizer>`_

* `SGDIpuOptimizer <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.optimizers.SGDIpuOptimizer>`_

They are functionally the same but have a number of additional features,
which can be used via the optimizer's keyword arguments.

The precision of any optimizer states within the optimizer can be set
independently of each other and the model parameters. This is particularly
useful when training in mixed precision.

The optimizer update can be outlined, making the optimizer update block
code reusable, which can reduce memory at the expense of passing variables
around.
