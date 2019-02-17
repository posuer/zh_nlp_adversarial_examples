import sys

class Setting(object):
	def __init__(self):
		self.lr = 0.0007#0.0002
		self.word_dim = 300
		self.lstm_dim = 110#110 0.3941#120 0.3784 #0.3843#130 0.3686#150 0.3451#170 0.3804#180 0.3843#220 0.3471#200 0.3471#256
		self.max_len = 30
		self.dense_dim =  128
		self.keep_prob = 0.6
		self.batch_size = 1024
		self.epochs = 50
    		# your directory here
		self.model_dir = "NLImodel/models/1550117241/lstm_110_128_0.40_0.63.h5"
		self.train_dir = "NLImodel/multinli.train.zh.txt"
		self.dev_dir = "NLImodel/xnli_zh.txt"
		self.embed_dir = "../../wiki.zh.vec"

labels = {'neutral':0, 'entailment':1, 'contradiction':2}
labels_reverse = {0:'neutral', 1:'entailment', 2:'contradiction'}

#lr0.001 lstm_dim110 
#392702/392702 [==============================] - 19s 48us/step - 
#loss: 0.8167 - acc: 0.6333 - val_loss: 0.9281 - val_acc: 0.5529

#lr0.0002 lstm_dim110
#loss: 0.8668 - acc: 0.6010 - val_loss: 0.9407 - val_acc: 0.5529

#lr0.0007 lstm_dim110
#loss: 0.8354 - acc: 0.6234 - val_loss: 0.9247 - val_acc: 0.5608

#lr0.0007 lstm_dim50
#loss: 0.8501 - acc: 0.6126 - val_loss: 0.9313 - val_acc: 0.5529
