# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Tests for IPU Embedding layer."""

import numpy as np
from tensorflow.python import keras
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn
from tensorflow.python.platform import test

from ipu_tensorflow_addons.keras import layers

dataType = np.float32


def kerasIPUEmbeddingLookup(params, ids, name=None, serialization_factor=1):
  input_dim = params.shape[0]
  output_dim = params.shape[1]
  layer = layers.Embedding(
      input_dim=input_dim,
      output_dim=output_dim,
      embeddings_initializer=keras.initializers.constant(params),
      name=name,
      serialization_factor=serialization_factor)
  layer.build(input_shape=ids.shape)

  @def_function.function
  def impl(ids):
    return layer(inputs=ids)

  return impl(ids)


class IPUEmbeddingLookupTest(test.TestCase):
  def testEmbeddingLookup(self):
    ids = constant_op.constant([[1, 2, 3]])
    paras = np.array([[10], [20], [80], [40]])
    emb_lookup_tf = nn.embedding_lookup(paras, ids)
    emb_lookup_ipu = kerasIPUEmbeddingLookup(paras, ids, name="emb_test_0")
    self.assertAllClose(emb_lookup_tf, emb_lookup_ipu)

  def testEmbeddingLookupBatchSize2(self):
    ids = constant_op.constant([[1, 2, 3], [3, 4, 5]])
    paras = np.array([[10], [20], [80], [40], [50], [60]])
    emb_lookup_tf = nn.embedding_lookup(paras, ids)
    emb_lookup_ipu = kerasIPUEmbeddingLookup(paras, ids, name="emb_test_1")
    self.assertAllClose(emb_lookup_tf, emb_lookup_ipu)

  # Based on ipu/tests/embedding_lookup_test.py
  def testEmbeddingLookupBigGather(self):
    ids = np.arange(0, 8, dtype=np.int32).reshape([1, 8])
    paras = np.arange(2400000, dtype=dataType).reshape([12000, 200])
    result_ipu = kerasIPUEmbeddingLookup(paras, ids, name="emb_test_2")
    result_np = np.take(paras, ids, axis=0)
    self.assertAllClose(result_ipu, result_np)
    self.assertEqual(result_ipu.shape, (1, 8, 200))

  def testEmbeddingBadInputShape(self):
    ids = np.arange(0, 16, dtype=np.int32)
    paras = np.arange(25600, dtype=dataType).reshape([32, 200, 4])
    with self.assertRaisesRegexp(ValueError, r'The input shape should be a'):
      kerasIPUEmbeddingLookup(paras, ids, name="emb_test_4")

  def testEmbeddingLookupSerialization(self):
    ids = constant_op.constant([[1, 2, 3]])
    paras = np.array([[10], [20], [80], [40]])
    emb_lookup_tf = nn.embedding_lookup(paras, ids)
    emb_lookup_ipu = kerasIPUEmbeddingLookup(paras,
                                             ids,
                                             name="emb_test_5",
                                             serialization_factor=4)
    self.assertAllClose(emb_lookup_tf, emb_lookup_ipu)

  @test_util.run_v2_only
  def testGetConfig(self):
    layer = layers.Embedding(input_dim=32, output_dim=200)
    config = layer.get_config()
    layer2 = layers.Embedding.from_config(config)
    self.assertEqual(config, layer2.get_config())


if __name__ == '__main__':
  test.main()
