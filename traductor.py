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

# Limpiar datos
corpus_en = europarl_en
corpus_en = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpus_en)
# remover $$$
corpus_en = re.sub(r".\$\$\$", '', corpus_en)
# remover espacios
corpus_en = re.sub(r"   +", " ", corpus_en)
corpus_en = corpus_en.split('\n')

corpus_es = europarl_es
corpus_es = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpus_es)
# remover $$$
corpus_es = re.sub(r".\$\$\$", '', corpus_es)
# remover espacios
corpus_es = re.sub(r"   +", " ", corpus_es)
corpus_es = corpus_es
.split('\n')

# tokenizacion 
tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(corpus_en, target_vocab_size=2**13)
tokenizer_es = tfds.features.text.SubwordTextEncoder.build_from_corpus(corpus_es, target_vocab_size=2**13)

VOCAB_SIZE_EN = tokenizer_en.vocab_size + 2
VOCAB_SIZE_ES = tokenizer_es.vocab_size + 2

inputs = [[VOCAB_SIZE_EN+2] + tokenizer_en.encode(sentence) + [VOCAB_SIZE_EN-1]
        for sentence in crpues_en]

output = [[VOCAB_SIZE_ES+2] + tokenizer_en.encode(sentence) + [VOCAB_SIZE_ES-1]
        for sentence in crpues_es]

