# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
Embedding Keras layer
~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.ipu.ops import embedding_ops
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils

from ipu_tensorflow_addons.keras.layers import ipu_layer


class Embedding(ipu_layer.IPULayer):
  """
  This is designed to be a replacement for the typical use cases of the
  Keras Embedding layer.

  Args:
    input_dim: int > 0. Size of the vocabulary,
      i.e. maximum integer index + 1.
    output_dim: int >= 0. Dimension of the dense embedding.
    embeddings_initializer: Initializer for the `embeddings` matrix.
    serialization_factor: If greater than 1, the embedding lookup will be
        broken up into `serialization_factor` smaller lookups, serialized
        along the 0th dimension. This option should not be used unless
        the parameters of this layer is used by another layer. If this is
        the case, then serialization can reduce the maximum memory at the
        cost of extra computation.

  Input shape:
    2D tensor with shape: `(batch_size, input_length)`.

  Output shape:
    3D tensor with shape: `(batch_size, input_length, output_dim)`.

  """

  # pylint: disable=useless-super-delegation
  def __init__(self,
               input_dim,
               output_dim,
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               input_length=None,
               serialization_factor=1,
               **kwargs):

    kwargs['autocast'] = False
    super(Embedding, self).__init__(**kwargs)

    self.input_dim = input_dim
    self.output_dim = output_dim
    self.embeddings_initializer = initializers.get(embeddings_initializer)
    self.serialization_factor = serialization_factor

    self._check_unsupported(embeddings_regularizer, 'embeddings_regularizer')
    self._check_unsupported(activity_regularizer, 'activity_regularizer')
    self._check_unsupported(embeddings_constraint, 'embeddings_constraint')
    self._check_unsupported(mask_zero, 'mask_zero')
    self._check_unsupported(input_length, 'input_length')

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if len(input_shape) != 2:
      raise ValueError(
          "The input shape should be a tensor of shape [batch, input_length]")

    self.embeddings = self.add_weight(shape=(self.input_dim, self.output_dim),
                                      initializer=self.embeddings_initializer,
                                      name='embeddings')
    self.built = True

  # pylint: disable=arguments-differ
  def call(self, inputs, training=None):
    """
    Perform an embedding lookup.

    Args:
        inputs: An integer tensor of indices into the embedding variable.

    Returns:
        The entries of the embedding tensor corresponding to the ids tensor
        indices.
    """
    del training
    return embedding_ops.embedding_lookup(
        self.embeddings,
        ids=inputs,
        name=self.name,
        serialization_factor=self.serialization_factor)

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape + (self.output_dim,)

  def get_config(self):
    return {
        'input_dim':
        self.input_dim,
        'output_dim':
        self.output_dim,
        'embeddings_initializer':
        initializers.serialize(self.embeddings_initializer),
        'serialization_factor':
        self.serialization_factor,
    }
