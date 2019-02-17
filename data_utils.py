
from keras.preprocessing.text import Tokenizer
from NLImodel.settings import *
from NLImodel.data_loader import *

class XNLIDataset(object):
	def __init__(self, path='NLImodel/', max_vocab_size=None):
		self.train_dir = path+setting.train_dir
		self.dev_dir = path+setting.dev_dir

		self.max_vocab_size = max_vocab_size

		self.data_all = [None] * 6 #include original version of self.train_X, self.train_Y, self.train_Z, self.val_X, self.val_Y, self.val_Z
		
		self.train_X = None
		self.train_Y = None
		self.train_Z = None
		self.val_X = None
		self.val_Y = None
		self.val_Z = None	

		#used for Google Language Model
		self.dict = dict() #used for building selcted Word Embedding
		self.inv_dict = dict()
		self.full_dict = dict()
		self.inv_full_dict = dict()

		self.build_dataset()

	def build_dataset(self):
		
		print("Train file: ",self.train_dir)
		print("Dev file: ",self.dev_dir)
		#print("Embedding file: ",embeddings_name)
		
		train_data=[l.strip().split('\t') for l in open(self.train_dir, errors='ignore')]
		dev_data=[l.strip().split('\t') for l in open(self.dev_dir, errors='ignore')]

		self.full_dict = get_vocab(train_data + dev_data)

		if self.max_vocab_size is None:
		    self.max_vocab_size = len(self.full_dict) + 1
		
		for word, idx in self.full_dict.items():
			if idx < self.max_vocab_size:
				self.inv_dict[idx] = word
				self.dict[word] = idx
		    #self.full_dict[word] = idx
			self.inv_full_dict[idx] = word 

		#word_embeddings = get_embeddings(word_vocab, embeddings_name, setting.word_dim)

		self.data_all[0], self.data_all[1], self.data_all[2],   \
			self.data_all[3], self.data_all[4], self.data_all[5] = create_train_dev_set(train_data, dev_data, self.dict)
		
		self.train_X = [[w if w < self.max_vocab_size else self.max_vocab_size for w in sen] for sen in self.data_all[0]]
		self.train_Y = [[w if w < self.max_vocab_size else self.max_vocab_size for w in sen] for sen in self.data_all[1]]
		self.train_Z = [[w if w < self.max_vocab_size else self.max_vocab_size for w in sen] for sen in self.data_all[2]]
		self.val_X = [[w if w < self.max_vocab_size else self.max_vocab_size for w in sen] for sen in self.data_all[3]]
		self.val_Y = [[w if w < self.max_vocab_size else self.max_vocab_size for w in sen] for sen in self.data_all[4]]
		self.val_Z = [[w if w < self.max_vocab_size else self.max_vocab_size for w in sen] for sen in self.data_all[5]]	

		print('Dataset built !')
		print('Dict size:', len(self.dict), ' eg:', next(iter(self.dict.items())))
		print('Train size:', len(self.train_X), ' eg:', self.train_X[0])
		print('Dev size:', len(self.val_X))

 




'''      
		self.path = path
		self.train_path = path + '/train.txt'
		self.test_path = path + '/test.txt'
		self.vocab_path = path + '/vocab.txt'
		self.max_vocab_size = max_vocab_size

		print("Train file: ", self.train_path)
		print("Dev file: ", self.test_path)
		print("Vocab file: ", self.vocab_path)

		self.vocab, self.reverse_vocab = self._read_vocab()
		self.train_text, self.train_y = self.read_text(self.train_path)
		self.test_text, self.test_y = self.read_text(self.test_path)

		self.dict = dict()
		self.inv_dict = dict()
		self.full_dict = dict()
		self.inv_full_dict = dict()

		print('tokenizing...')
		
		# Tokenized text of training data
		self.tokenizer = Tokenizer()

		self.tokenizer.fit_on_texts(self.train_text)
		if max_vocab_size is None:
		    max_vocab_size = len(self.tokenizer.word_index) + 1

		self.dict['UNK'] = max_vocab_size
		self.inv_dict[max_vocab_size] = 'UNK'
		
		for word, idx in self.tokenizer.word_index.items():
		    if idx < max_vocab_size:
				self.inv_dict[idx] = word
				self.dict[word] = idx
		    self.full_dict[word] = idx
		    self.inv_full_dict[idx] = word 
		
		self.train_seqs = self.tokenizer.texts_to_sequences(self.train_text)
		self.train_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.train_seqs]
		
		self.test_seqs = self.tokenizer.texts_to_sequences(self.test_text)
		self.test_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.test_seqs]
 
		print('Dataset built !')

    def _read_vocab(self):
		with open(self.vocab_path, 'r') as f:
		    vocab_words = f.read().split('\n')
		    vocab = dict([(w,i) for i, w in enumerate(vocab_words)])
		    reverse_vocab = dict([(i,w) for w,i in self.vocab.items()])
		return vocab, reverse_vocab

    def read_text(self, path):
		""" Returns a list of text documents and a list of their labels"""
		data=[l.strip().split('\t') for l in open(path, errors='ignore')]

		data_list = []
		labels_list = []
		for line in data:
		    #print(data)
		    data_list.append([line[1],line[2]])
		    labels_list.append(line[0])

		return data_list, labels_list
'''