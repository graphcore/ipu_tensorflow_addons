Keras layers
~~~~~~~~~~~~

The `ipu_tensorflow_addons.keras.layers` namespace contains both IPU-specific implementations
of standard Keras layers and IPU-specific layers that do not have upstream equivalents.

IPU implementations of standard Keras layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    Swapping standard Keras layers for their IPU-specific counterparts
    will improve your model's performance when using IPUs.

* `GRU <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.layers.PopnnGRU>`_

* `LSTM <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.layers.PopnnLSTM>`_

* `GroupNorm <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.layers.GroupNorm>`_

* `InstanceNorm <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.layers.InstanceNorm>`_

* `LayerNorm <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.layers.LayerNorm>`_

* `Dropout <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.layers.Dropout>`_

* `Embedding <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.layers.Embedding>`_

* `SerialDense <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.layers.SerialDense>`_

Layers without upstream equivalents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The layers below handle IPU specific behaviour:

* `AssumeEqualAcrossReplicas <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.layers.AssumeEqualAcrossReplicas>`_

* `RecomputationCheckpoint <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.layers.RecomputationCheckpoint>`_

The layers below are designed specifically for the IPU, but are not part of upstream Keras:

* `CTCInferenceLayer <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.layers.CTCInferenceLayer>`_

* `CTCPredictionsLayer <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.layers.CTCPredictionsLayer>`_

* `CTCLoss <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.layers.CTCLoss>`_

* `EffectiveTransformer <https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/api.html#ipu_tensorflow_addons.keras.layers.EffectiveTransformer>`_

Code example
^^^^^^^^^^^^

The example below demonstrates the use of the `ipu_tensorflow_addons.keras.layers` namespace in an IPU application.

.. literalinclude:: example_ipu_specific_keras_layers.py
  :language: python
  :linenos:
  :emphasize-lines: 11, 34, 35
