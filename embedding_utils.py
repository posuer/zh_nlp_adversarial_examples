import numpy as np

def loadEmbeddingModel(File):
    print ("Loading Embedding Model")
    f = open(File,'r')
    model = {}
    for line in f:
        row = line.strip().split(' ')
        word = row[0]
        #print(word)
        embedding = np.array([float(val) for val in row[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model


def create_embeddings_matrix(embedding_model, dictionary, full_dictionary, d=300):
    MAX_VOCAB_SIZE = len(dictionary)
    # Matrix size is 300
    embedding_matrix = np.zeros(shape=((d, MAX_VOCAB_SIZE+1)))
    cnt  = 0
    cnt2 = 0 
    unfound = []
    
    for w, i in dictionary.items():
        if not w in embedding_model:
            cnt += 1
            #if cnt < 10:
            # embedding_matrix[:,i] = embedding_model['UNK']
            unfound.append(i)
        else:
            cnt2 += 1
            embedding_matrix[:, i] = embedding_model[w]
    print('Number of not found words = ', cnt)
    print('Number of found words = ', cnt2)
    return embedding_matrix, unfound

def pick_most_similar_words(src_word, dist_mat, ret_count=10, threshold=None):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    dist_order = np.argsort(dist_mat[src_word,:])[1:1+ret_count]
    dist_list = dist_mat[src_word][dist_order]
    if dist_list[-1] == 0:
        return [], []
    mask = np.ones_like(dist_list)
    if threshold is not None:
        mask = np.where(dist_list < threshold)
        return dist_order[mask], dist_list[mask]
    else:
        return dist_order, dist_list