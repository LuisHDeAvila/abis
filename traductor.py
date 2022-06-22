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

tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(data_clean, target_vocab_size=2**16)

data_inputs = [tokenizer.encode(sentence) for sentence in data_clean]


