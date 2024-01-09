# author: yzc

import gensim.models as word2vec
from gensim.models.word2vec import LineSentence
import os

from utils.configure_util import ConfLoader


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding='utf-8'):
                yield line.split()


def train_word2vec(datasetPath, outVector, params):
    sentences = LineSentence(datasetPath)
    # size 为向量维度，window 为词向量上下文最大距离，min_count 需要计算词向量的最小词频 sg为1是Skip-Gram模型）
    model = word2vec.Word2Vec(sentences, vector_size=params["max_length"], sg=1, window=5, min_count=1, workers=4)
    model.wv.save_word2vec_format(outVector, binary=False)


def load_word2vec_model(w2vPath):
    model = word2vec.KeyedVectors.load_word2vec_format(w2vPath, binary=False)
    return model


def calculate_most_similar(model, word):
    similar_words = model.most_similar(word)
    print(word)
    for term in similar_words:
        print(term[0], term[1])


if __name__ == '__main__':
    params = ConfLoader("./conf.yaml")
    datasetPath = "texts/segmented_text.txt"
    outVector = 'embedding.vector'

    train_word2vec(datasetPath, outVector, params)
    model = load_word2vec_model(outVector)
    calculate_most_similar(model, "绿色")
