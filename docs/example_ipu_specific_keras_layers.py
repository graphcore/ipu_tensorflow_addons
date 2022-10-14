import argparse
import tensorflow as tf
from tensorflow.python import ipu
from tensorflow import keras

from ipu_tensorflow_addons.keras import layers as ipu_layers

max_features = 20000


# Define the dataset
def get_dataset():
  (x_train,
   y_train), (_, _) = keras.datasets.imdb.load_data(num_words=max_features)

  x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=80)

  ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  ds = ds.repeat()
  ds = ds.map(lambda x, y: (x, tf.cast(y, tf.int32)))
  ds = ds.batch(32, drop_remainder=True)
  return ds


# Define the model
def get_model():
  input_layer = keras.layers.Input(shape=(80), dtype=tf.int32, batch_size=32)

  with keras.ipu.PipelineStage(0):
    x = ipu_layers.Embedding(max_features, 64)(input_layer)
    x = ipu_layers.LSTM(64, dropout=0.2)(x)

  with keras.ipu.PipelineStage(1):
    a = keras.layers.Dense(8, activation='relu')(x)

  with keras.ipu.PipelineStage(2):
    b = keras.layers.Dense(8, activation='relu')(x)

  with keras.ipu.PipelineStage(3):
    x = keras.layers.Concatenate()([a, b])
    x = keras.layers.Dense(1, activation='sigmoid')(x)

  return keras.Model(input_layer, x)


#
# Main code
#

# Parse command line args
parser = argparse.ArgumentParser("Config Parser", add_help=False)
parser.add_argument('--steps-per-epoch',
                    type=int,
                    default=768,
                    help="Number of steps in each epoch.")
parser.add_argument('--epochs',
                    type=int,
                    default=3,
                    help="Number of epochs to run.")
args = parser.parse_args()

# Configure IPUs
cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 2
cfg.configure_ipu_system()

# Set up IPU strategy
strategy = ipu.ipu_strategy.IPUStrategyV1()
with strategy.scope():

  model = get_model()
  model.set_pipelining_options(gradient_accumulation_steps_per_replica=8,
                               device_mapping=[0, 1, 1, 0])
  model.compile(loss='binary_crossentropy',
                optimizer=keras.optimizers.Adam(0.005),
                steps_per_execution=16)

  model.fit(get_dataset(),
            steps_per_epoch=args.steps_per_epoch,
            epochs=args.epochs)
