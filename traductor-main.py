import numpy as np 
import math 
import re
import time 

import tensorflow as tf
from tensoflow.keras import layers
import tensorflow_datasets as tfds

# cargando archivos
with open("europarl-v7.es-en.en",
mode='r',
encoding='utf-8') as f:
europarl_en = f.read()

with open("europarl-v7.es-en.es",
mode='r',
encoding='utf-8') as f:
europarl_en = f.read()

