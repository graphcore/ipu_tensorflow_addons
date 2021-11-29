Keras layers
~~~~~~~~~~~~

The `ipu_tensorflow_addons.keras.layers` namespace contains Keras layers optimised for running on
IPUs. These layers can be used the same way as standard Keras layers.

Some IPU-specific versions of standard Keras layers are included. Swapping out standard keras
layers for their IPU-specific counterparts will improve your model's performance when using IPUs.

.. literalinclude:: example_ipu_specific_keras_layers.py
  :language: python
  :linenos:
  :emphasize-lines: 11, 34, 35
