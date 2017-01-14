import json
import os
import numpy as np

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from keras.engine import Input
from keras.layers import Embedding, merge
from keras.models import Model

tokenize = lambda x: simple_preprocess(x)


def createEmbeddings(
	data_dir,
	embeddings_path='embeddings.npz',
	vocab_path='map.json',
	**params
):
	class SentenceGenerator(object):
		def __init__(self, dirname):
			self.dirname = dirname

		def __iter__(self):
			for fname in os.listdir(self.dirname):
				for line in open(os.path.join(self.dirname, fname)):
					yield tokenize(line)

	sentences = SentenceGenerator(data_dir)

	model = Word2Vec(sentences, **params)
	weights = model.syn0
	np.save(open(embeddings_path, 'wb'), weights)

	vocab = dict([(k, v.index) for k, v in model.vocab.items()])
	with open(vocab_path, 'w')  as f:
		f.write(json.dumps(vocab))


def loadVocab(vocab_path='map.json'):
	with open(vocab_path, 'r') as f:
		data = json.loads(f.read())
	word2idx = data
	idx2word = dict([(v, k) for k, v in data.items()])
	return word2idx, idx2word

def word2vecEmbeddingLayer(embeddings_path='embeddings.npz'):
	weights = np.load(open(embeddings_path, 'rb'))
	layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights])
	return layer

if __name__ == '__main__':
	data_path = os.environ['EMBEDDINGS_TEXT_PATH']
	createEmbeddings(data_path, size=100, min_count=5, window=5, sg=1, iter=25)

	word2idx, idx2word = loadVocab()

	input_a = Input(shape=(1,), dtype='int32', name='input_a')
	input_b = Input(shape=(1,), dtype='int32', name='input_b')

	embeddings = word2vecEmbeddingLayer()
	embedding_a = embeddings(input_a)
	embedding_b = embeddings(input_b)

	similarity = merge([embedding_a, embedding_b], mode='cos', dot_axes=2)
