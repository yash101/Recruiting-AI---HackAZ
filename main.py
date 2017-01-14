import os
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.models import Model

import sys

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/'
TEXT_DATA_DIR = BASE_DIR + '/Dataset/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


print('STAGE 1: Indexing vectors for words')

embeddingsIdx = {}

f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
	values = line.split()
	word = values[0]
	coefficients = np.asarray(values[1:], dtype='float32')
	embeddingsIdx[word] = coefficients
f.close()

print('Indexed %s vectors for individual words' % len(embeddingsIdx))





print('STAGE 2: Process input text dataset')

texts = []
labelsIdx = {}
labels = []

for name in sorted(os.listdir(TEXT_DATA_DIR)):
	print("Checking: %s" % name)
	path = os.path.join(TEXT_DATA_DIR, name)
	print("--> Checking: %s" % path)
	if os.path.isdir(path):
		print("!!! Is Directory")
		labelId = len(labelsIdx)
		labelsIdx[name] = labelId

		for fname in sorted(os.listdir(path)):
			fpath = os.path.join(path, fname)
			if sys.version_info < (3,):
				f = open(fpath)
			else:
				f = open(fpath, encoding='latin-1')
			texts.append(f.read())
			f.close()
			labels.append(labelId)

print('Found %s different texts' % len(texts))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

wordIdx = tokenizer.word_index
print('Found %s unique tokens' % len(wordIdx))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))

print('Tensor shape for data: ', data.shape)
print('Tensor shape for label: ', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nbValidationSamples = int(VALIDATION_SPLIT * data.shape[0])

xTrain = data[:-nbValidationSamples]
yTrain = labels[:-nbValidationSamples]
xVal = data[-nbValidationSamples:]
yVal = labels[-nbValidationSamples:]

print('STAGE 3: Preparing to embed matrix...')

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
			trainable=False)

print(embeddingMatrix)

print('STAGE 4: Training the model')

sequenceInput = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embeddedSequences = embeddingLayer(sequenceInput)

x = Conv1D(128, 5, activation='relu')(embeddedSequences)
x = MaxPooling1D(5)(x)

x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)

x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labelsIdx), activation='softmax')(x)

model = Model(sequenceInput, preds)

model.compile(loss='categorical_crossentropy',
		optimizer='rmsprop',
		metrics=['acc'])

model.fit(xTrain, yTrain, validation_data=(xVal, yVal), nb_epoch=4000, batch_size=128)
