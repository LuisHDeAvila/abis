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
          inplace)
