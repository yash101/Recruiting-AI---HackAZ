import os
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.models import Model

import sys



'''
Directories for stuff
Constants
'''
BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/'
TEXT_DATA_DIR = BASE_DIR + '/Dataset/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def ai_system(input_text):
	##########################################################################


	#Stage 1
	print('STAGE 1: Indexing vectors for words')

	#Indices for embeddings
	embeddingsIdx = {}

	#Open the glove file
	f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
	#Each line from the glove file
	for line in f:
		#Words, no spaces
		values = line.split()
		word = values[0]
		coefficients = np.asarray(values[1:], dtype='float32')
		embeddingsIdx[word] = coefficients
	#Close the file
	f.close()

	#Number of words indexed :D
	print('Indexed %s vectors for individual words' % len(embeddingsIdx))

	##########################################################################

	print('STAGE 2.5: Tokenizing and forming data')

	tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
	tokenizer.fit_on_texts(input_text)
	sequences = tokenizer.texts_to_sequences(input_text)

	wordIdx = tokenizer.word_index
	print('Found %s unique tokens' % len(wordIdx))

	data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

	indices = np.arange(data.shape[0])
	np.random.shuffle(indices)
	data = data[indices]

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


	##########################################################################


	print('STAGE 4: Define the model')

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
	x = Dropout(0.1)(x)
	x = Dense(128, activation='relu')(x)
	preds = Dense(7, activation='softmax')(x)

	model = Model(sequenceInput, preds)


	##########################################################################


	print('STAGE 5: Load the weights, so we dont have to retrain')

	model.load_weights('amazing_weights.h5')

	answer = model.predict(data)
	answer = answer[0]
	largest = -1
	max_index = -1
	for index, value in enumerate(answer):
		if value > largest:
			largest = value
			max_index = index
	answer = max_index

	predicted = ""
	if answer == 0:
		predicted = "Aerospace"
	elif answer == 1:
		predicted = "Cullinary"
	elif answer == 2:
		predicted = "Entertainment"
	elif answer == 3:
		predicted = "Financial"
	elif answer == 4:
		predicted = "Medical"
	elif answer == 5:
		predicted = "Retail"
	elif answer == 6:
		predicted = "Tech"
	else:
		predicted = "Error"

	return predicted
	##########################################################################
	##########################################################################
	##########################################################################
	##########################################################################
	##########################################################################
	##########################################################################
	##########################################################################
	##########################################################################

