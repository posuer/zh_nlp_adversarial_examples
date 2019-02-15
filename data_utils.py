
from keras.preprocessing.text import Tokenizer


class XNLIDataset(object):
    def __init__(self, path='xnli', max_vocab_size=None):
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
        #只应该存应该待修改的句子 也就是后面那个句子
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
            data_list.append(line[1]+'\t'+line[2])
            labels_list.append(line[0])

        return data_list, labels_list