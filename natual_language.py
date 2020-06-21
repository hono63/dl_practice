"Chapter 2. Natual Language Processing."

import numpy as np
import matplotlib.pyplot as plt
import uuid

from common.layers import Affine, Sigmoid, SoftmaxWithLoss
from common.optimizer import SGD
from common import utils
from common import config
from dataset import spiral 

config.GPU = True

def preprocessing(text):
    "コーパスを作成する"
    words = text.lower().replace(".", " .").split(" ")
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id.keys():
            #new_id = str(uuid.uuid4())
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[word] for word in words])
    return corpus, word_to_id, id_to_word

def create_co_matrix(corpus, vocab_size, window_size=1):
    "共起行列を作成する"
    length = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
        for widx in range(max(idx-window_size, 0), min(idx+window_size+1, length)):
            if widx == idx:
                continue
            word_id2 = corpus[widx]
            co_matrix[word_id, word_id2] += 1
    return co_matrix

def most_similar(query, word_to_id, id_to_word, word_matrix, top):
    "get the most similar words"
    qid = word_to_id.get(query.lower())
    qvec = word_matrix[qid]
    if qid is None:
        return
    print("\n[query] " + query)
    ###
    vocab_size = len(id_to_word)
    similarities = np.zeros(vocab_size, dtype=np.float)
    for wid in id_to_word.keys():
        similarities[wid] = cos_similarity(qvec, word_matrix[wid])
    ###
    count = 0
    for i in reversed(similarities.argsort()): # argsortはnumpy配列を小さい順にsortしたindexを返す
        if i == qid:
            continue
        print(" " + id_to_word[i] + ": " + str(similarities[i]))
        count += 1
        if count >= top:
            return

def cos_similarity(x, y, eps=1e-8):
    "calculate cosine similarity"
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)
    return np.dot(nx, ny)

def ppmi(co_matrix, verbose=False, eps=1e-8):
    "calculate Positive Pointwize Mutual Information 正の相互情報量"
    M = np.zeros_like(co_matrix, dtype=np.float32)
    N = np.sum(co_matrix) # コーパスの単語数
    Cn = np.sum(co_matrix, axis=0)
    total = co_matrix.shape[0] + co_matrix.shape[1]
    count = 0
    for i in range(co_matrix.shape[0]):
        for j in range(co_matrix.shape[1]):
            Cxy = co_matrix[i, j]
            pmi = np.log2(Cxy * N / (Cn[i] * Cn[j]) + eps)
            M[i, j] = max(0, pmi)
            if verbose and total > 100:
                count += 1
                if count % (total//100) == 0:
                    print("%.1f%% done" % (100*count/total))
    return M

np.set_printoptions(precision=3)
if __name__=="__main__":
    corpus, word_to_id, id_to_word = preprocessing('You say goodbye and I say hello.')
    #print(word_to_id.items())
    co_matrix = create_co_matrix(corpus, len(word_to_id), window_size=1)
    #print(cos_similarity(co_matrix[word_to_id["you"]], co_matrix[word_to_id["i"]]))
    most_similar("you", word_to_id, id_to_word, co_matrix, 5)
    W = ppmi(co_matrix, verbose=True)
    print(co_matrix)
    print("-"*50)
    print(W)
    ### 特異値分解 Singular Value Deomposition. X = USV^T
    U, S, V = np.linalg.svd(W)
    print("-"*50)
    print(co_matrix[0])
    print(W[0])
    print(U[0]) # この行列の1,2列目だけを使う
    for word, wid in word_to_id.items():
        plt.annotate(word, (U[wid, 0], U[wid, 1]))
    plt.scatter(U[:,0], U[:,1], alpha=0.5)
    plt.show()