import numpy as np
import tensorflow as tf
import embedding_utils
import pickle 
from keras.preprocessing.sequence import pad_sequences

MAX_VOCAB_SIZE = 50000

'''load counter-fitted words '''
embedding_matrix = np.load(('aux_files/embeddings_counter_%d.npy' %(MAX_VOCAB_SIZE)))
#missed = np.load(('aux_files/missed_embeddings_counter_%d.npy' %(MAX_VOCAB_SIZE)))
print("debug")
c_ = -2*np.dot(embedding_matrix.T , embedding_matrix)
a = np.sum(np.square(embedding_matrix), axis=0).reshape((1,-1))
b = a.T
dist = a+b+c_
np.save(('aux_files/dist_counter_%d.npy' %(MAX_VOCAB_SIZE)), dist)

# Try an example
with open('aux_files/dataset_%d.pkl' %MAX_VOCAB_SIZE, 'rb') as f:
    dataset = pickle.load(f)
src_word = dataset.dict['good']
neighbours, neighbours_dist = embedding_utils.pick_most_similar_words(src_word, dist)
print('Closest words to `good` are :')
result_words = [dataset.inv_dict[x] for x in neighbours]
print(result_words)
 