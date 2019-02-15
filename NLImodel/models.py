import numpy as np
import pandas as pd
import argparse
import settings
import os
import time
from settings import *
from data_loader import *
from keras.layers import *
from keras.activations import softmax
from keras.regularizers import l2
from keras.models import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.constraints import max_norm
from keras.models import load_model
#from keras import backend as K

np.random.seed(0)

class NLImodel(object):
	def __init__(self,setting=settings.Setting(),is_train=True):
		self.lr = setting.lr
		self.word_dim = setting.word_dim
		self.lstm_dim = setting.lstm_dim
		self.max_len = setting.max_len
		self.dense_dim = setting.dense_dim
		self.keep_prob = setting.keep_prob
		self.batch_size = setting.batch_size
		self.drop_prob = 1 - self.keep_prob
		self.optimizer = Adam(self.lr)
		self.epochs = setting.epochs
		#self.is_adv = is_adv
		self.is_train = is_train
		self.model = None
		
		self.model_dir = setting.model_dir #for load model
		self.train_dir = setting.train_dir
		self.dev_dir = setting.dev_dir
		self.embed_dir = setting.embed_dir

		print("Learning Rate: ",self.lr)
		if self.is_train:
			self.train_model()
		else:
			self.model = load_model(self.model_dir)
			print("Model Weights Loaded.")

	
	def predict(self, test_X, test_Y):
		assert len(test_X) >= 2 and len(test_Y) >= 2
		return self.model.predict([test_X, test_Y])

	def evaluate_accuracy(self, test_x, test_y, test_z):
		test_accuracy = 0.0
		pred_z = self.predict(test_x, test_y)

		for i in len(pred_z):
			test_accuracy += int(np.argmax(pred_z[i], axis=1) == np.argmax(test_z[i]))
		test_accuracy /= len(test_z)
		return test_accuracy

	def train_model(self, source_language='zh'):
		
		#train_name = self.train_dir+"multinli.train.%s.txt"%source_language
		#dev_name = self.dev_dir+"xnli_%s.txt"%source_language
		train_name = self.train_dir	
		dev_name = self.dev_dir
		embeddings_name = self.embed_dir
		print("Train file: ",train_name)
		print("Dev file: ",dev_name)
		print("Embedding file: ",embeddings_name)
		
		train_data=[l.strip().split('\t') for l in open(train_name, errors='ignore')]
		dev_data=[l.strip().split('\t') for l in open(dev_name, errors='ignore')]

		word_vocab = get_vocab(train_data + dev_data)	
		word_embeddings = get_embeddings(word_vocab, embeddings_name, setting.word_dim)

		train_X, train_Y, train_Z, \
			val_X, val_Y, val_Z	= create_train_dev_set(train_data, dev_data, word_vocab)
		
		if train_X is None:
			print("++++++ Unable to train model +++++++")
			return None	

			
		embedding_layer = Embedding(len(word_vocab), self.word_dim, weights=[word_embeddings], input_length=(self.max_len,), trainable=False)
		
		prem_input = Input(shape=(self.max_len,), dtype='int32')
		hypo_input = Input(shape=(self.max_len,), dtype='int32')
		
		prem = embedding_layer(prem_input)
		hypo = embedding_layer(hypo_input)
		
		lstm_layer = Bidirectional(CuDNNLSTM(self.lstm_dim, return_sequences=True))
		
		prem = GlobalMaxPooling1D()(lstm_layer(prem))
		hypo = GlobalMaxPooling1D()(lstm_layer(hypo))
		
		merged = Concatenate()([prem, hypo, submult(prem,hypo)])

		dense = Dropout(self.drop_prob)(merged)
		dense = Dense(self.dense_dim, kernel_initializer='uniform', activation='relu')(dense)
		dense = Dropout(self.drop_prob)(dense)
		preds = Dense(3, activation='softmax')(dense)
			
		model = Model([prem_input, hypo_input], [preds])

		model.compile(optimizer=self.optimizer, loss=['categorical_crossentropy'], metrics=['accuracy'])
		
		checkpoint_dir = "models/" + str(int(time.time())) + '/'
		
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		
		STAMP = 'lstm_%d_%d_%.2f' % (self.lstm_dim, self.dense_dim, self.drop_prob)
		
		filepath= checkpoint_dir + STAMP + "_{val_acc:.2f}.h5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

		lr_sched = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, cooldown=1, verbose=1)
		early_stopping = EarlyStopping(monitor='val_acc', patience=10)
		#K.clear_session()
		model.fit([train_X, train_Y], [train_Z], 
				validation_data = ([val_X, val_Y], [val_Z]),
				epochs=self.epochs, batch_size = self.batch_size, shuffle = True,
				callbacks=[checkpoint, lr_sched, early_stopping])	
		
		self.model = model


'''
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-src_lang', '--source_language', type=str, default='zh', help='source_language')
	params = parser.parse_args()
	
	setting = settings.Setting()

	Tars = NLImodel(setting)
	Tars.train_model(params.source_language)
'''