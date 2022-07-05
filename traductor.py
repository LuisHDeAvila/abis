import numpy as np
import math
import re
import pandas as pd
import bs4 import BeautifulSoup

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
          inplace = True )

def clean_tweet(tweeter):
  tweet = BeautifulSoup(tweet, "LxmL").get_text()
  tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
  tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
  tweet = re.sub(r" +", ' ', tweet)
  return tweet


data_clean = [clean_tweet(tweet) for tweet in data.text]

data_labels = data.sentiment.values
data_labels[data_labels == 4] = 1

# Tokenizar
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(data_clean, target_vocab_size=2**16)

data_inputs = [tokenizer.encode(sentence) for sentence in data_clean]

# Padding
MAX_LEN = max(len(sentence) for sentence in data_inputs)
data_inputs = tf.keras.preprocessing.sequence.pad_sequences(data_inputs, value=0, padding="post", maxlen=MAX_LEN)

# Sets de entrenamiento y prueba
test_idx = np.random.randint(0, 800000, 8000)
test_idx = np.ooncatenate((test_idx, test_id+800000))

test_inputs = data_inputs[test_idx]
test_labels = data_labels[test_idx]

train_inputs = np.delete(data_inputsm test_idx, axis=0)
train_labels = np.delete(data_labels, test_idx)

# Red neuronal
class DCNN(tf.keras.Model):
  def __init__(self, vocab_size, )


