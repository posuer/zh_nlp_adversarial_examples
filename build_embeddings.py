
import numpy as np
import pickle
import os

import embedding_utils
import data_utils

MAX_VOCAB_SIZE = 50000
DATA_PATH = 'NLImodel/'
EMBEDDING_PATH = 'cc.zh.300.vec'
COUNTER_FITTED = 'counter_fitted_vectors.txt'

if not os.path.exists('aux_files'):
	os.mkdir('aux_files')

xnli_dataset = data_utils.XNLIDataset(path=DATA_PATH, max_vocab_size=MAX_VOCAB_SIZE)

# save the dataset
with open(('aux_files/dataset_%d.pkl' %(MAX_VOCAB_SIZE)), 'wb') as f:
    pickle.dump(xnli_dataset, f)


# create the glove embeddings matrix (used by the classification model)
glove_model = embedding_utils.loadEmbeddingModel(EMBEDDING_PATH )
glove_embeddings, _ = embedding_utils.create_embeddings_matrix(glove_model, xnli_dataset.dict, xnli_dataset.full_dict)
# save the glove_embeddings matrix
np.save('aux_files/embeddings_glove_%d.npy' %(MAX_VOCAB_SIZE), glove_embeddings)

# Load the counterfitted-vectors (used by our attack)
glove2 = embedding_utils.loadEmbeddingModel(COUNTER_FITTED)
# create embeddings matrix for our vocabulary
counter_embeddings, missed = embedding_utils.create_embeddings_matrix(glove2, xnli_dataset.dict, xnli_dataset.full_dict)

# save the embeddings for both words we have found, and words that we missed.
np.save(('aux_files/embeddings_counter_%d.npy' %(MAX_VOCAB_SIZE)), counter_embeddings)
np.save(('aux_files/missed_embeddings_counter_%d.npy' %(MAX_VOCAB_SIZE)), missed)
print('All done')