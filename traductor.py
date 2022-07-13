import numpy as np
import math
import re
import pandas as pd
import bs4 as BeautifulSoup

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

cols = ["sentiment", "id", "date", "query", "user", "text"]
train_data = pd.read_csv("train.csv",
                        header=None,
                        names=cols,
                        engine="python",
                        encoding="Latin1")

test_data = pd.read_csv("test.csv",
                        header=None,
                        names=cols,
                        engine="python",
                        encoding="Latin1")

data = train_data

# clean
data.drop(["id", "date", "query", "user"],
          axis=1,
          inplace=True)

# formatear strings
def clean_tweet(tweeter):
  tweet = BeautifulSoup(tweet, "LxmL").get_text()
  # arroba
  tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
  # https / url
  tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
  # solo letras
  tweet = re.sub(r"[^a-zA-Z.!?')", ' ',tweet)
  # espacios en blanco
  tweet = re.sub(r" +", ' ', tweet)
  return tweet


data_clean = [clean_tweet(tweet) for tweet in data.text]

data_labels = data.sentiment.values
data_labels[data_labels == 4] = 1

# Tokenizar
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    data_clean, target_vocab_size=2**16)

data_inputs = [tokenizer.encode(sentence) for sentence in data_clean]

# Padding
MAX_LEN = max(len(sentence) for sentence in data_inputs)
data_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    data_inputs, value=0, padding="post", maxlen=MAX_LEN)

# Sets de entrenamiento y prueba
test_idx = np.random.randint(0, 800000, 8000)
test_idx = np.concatenate((test_idx, test_id+800000))

test_inputs = data_inputs[test_idx]
test_labels = data_labels[test_idx]

train_inputs = np.delete(data_inputs, test_idx, axis=0)
train_labels = np.delete(data_labels, test_idx)

# Red neuronal


class DCNN(tf.keras.Model):

  def __init__(self,
              vocab_size,
              emb_dim=128,
              nb_filters=50,
              FFN_units=512,
              nb_classes=2,
              dropout_rate=0.1,
              training=False,
              name="dcnn"):
      super(DCNN, self).__init__(name=name)

      self.embedding = layers.Embedding(vocab_size, emb_dim)

      self.bigram = layers.Conv1D(filters=nb_filters,
                                  kernel_size=2,
                                  padding="valid",
                                  activation="relu")

      self.trigram = layers.Conv1D(filters=nb_filters,
                                  kernel_size=3,
                                  padding="valid",
                                  activation="relu")

      self.fourgram = layers.Conv1D(filters=nb_filters,
                                    kernel_size=4,
                                    padding="valid",
                                    activation="relu")

      self.pool = layers.GlobalMaxPool1D()

      self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
      self.dropout = layers.Dropout(rate=dropout_rate)

      if nb_classes == 2:
        self.last_dense = layers.Dense(units=1,
                                      activation="sigmoid")
      else:
        self.last_dense = layers.Dense(units=nb_classes,
                                      activation="softmax")

  def call(self, inputs, training):
    x = self.embedding(inputs)
    x_1 = self.bigram(x)
    x_1 = self.pool(x_1)
    x_2 = self.trigram(x)
    x_2 = self.pool(x_2)
    x_3 = self.fourgram(x)
    x_4 = self.pool(x_3)

    merged = tf.concat([x_1,x_2,x_3], axis=-1)
    merged = self.dense_1(merged)
    merged = self.dropout(merged, training)
    merged = self.last_dense(merged)

    return output

VOCAB_SIZE = tokenizer.vocab_size
EMB_DIM = 200
NB_FILTERS = 100
FFN_UNITS = 256
NB_CLASSES = len(set(train_labels))

DROPOUT_RATE = 0.2

BATCH_SIZE = 32
NB_EPOCHS = 5

Dcnn = DCNN(vocab_size=VOCAB_SIZE,
            emb_dim=EMB_DIM,
            nb_filters=NB_FILTERS,
            FFN_units=FFN_UNITS,
            nb_classes=NB_CLASSES,
            dropout_rate=DROPOUT_RATE)



