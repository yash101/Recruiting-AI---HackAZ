import os
import sys
import argparse
import json

import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.models import Model

ap = argparse.ArgumentParser()

ap.add_argument('-f', '--file', required=True, help='File to preprocess')

argp = vars(ap.parse_args())

data = None

with open(argp['file'], 'r') as f:
	data = json.load(f)

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/'
TEXT_DATA_DIR = BASE_DIR + '/Dataset/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100

texts = []

for resume in data:
	texts.append(resume["resume_text"])
	
embeddingsIdx = {}

f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
	values = line.split()
	word = values[0]
	coefficients = np.asarray(values[1:], dtype='float32')
	embeddingsIdx[word] = coefficients
f.close()

print('STAGE 2: Process input text dataset')

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

wordIdx = tokenizer.word_index

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]

nbWords = min(MAX_NB_WORDS, len(wordIdx))
embeddingMatrix = np.zeros((nbWords + 1, EMBEDDING_DIM))

for word, i in wordIdx.items():
	if i > MAX_NB_WORDS:
		continue

	embeddingVector = embeddingsIdx.get(word)
	if embeddingVector is not None:
		embeddingMatrix[i] = embeddingVector

embeddingLayer = Embedding(nbWords + 1,
			EMBEDDING_DIM,
			weights=[embeddingMatrix],
			input_length=MAX_SEQUENCE_LENGTH,
			trainable = False)

sequenceInput = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embed = embeddingLayer(sequenceInput)

model = Model(input=sequenceInput,output=embed)
print(model.predict(data[:1000])[0][0])
