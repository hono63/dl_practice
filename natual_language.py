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

if __name__=="__main__":
    corpus, word_to_id, id_to_word = preprocessing('You say goodbye and I say hello.')
    print(word_to_id.items())
    co_matrix = create_co_matrix(corpus, len(word_to_id), window_size=1)
    print(co_matrix)
    