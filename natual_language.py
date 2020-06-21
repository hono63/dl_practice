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
    words = text.lower().replace(".", " .").split(" ")
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = str(uuid.uuid4())
            word_to_id[word] = new_id
            id_to_word[new_id]   = word
    corpus = np.array(list(id_to_word.keys()))
    return corpus, word_to_id, id_to_word

if __name__=="__main__":
    corpus, word_to_id, id_to_word = preprocessing('You said goodbye and I say hello.')
    print(corpus)
    