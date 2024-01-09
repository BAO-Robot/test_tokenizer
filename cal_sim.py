# author: yzc

import jieba
import torch
from tokenizers import Tokenizer
import numpy as np
from train_word2vec import load_word2vec_model
from torch.nn.functional import cosine_similarity


def padding(vector, max_length):
    paddedVectors = torch.zeros(len(vector), max_length, dtype=torch.int)
    for k, i in enumerate(vector):
        length = min(len(i), max_length)
        paddedVectors[k, :length] = torch.tensor(i[:length])
    return paddedVectors


def get_sentence_matrix(wordList, model):
    mat = []
    for word in wordList:
        word_vec = model.get_vector(word)
        mat.append(word_vec)
    return torch.tensor(np.asarray(mat))


if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("my_tokenizer.json")
    w2v = load_word2vec_model("./embedding.vector")

    type_get = "帮我把红色小球拿到紫色方块右上角"
    imge_get = ["左上角", "右下角", "黄色小球", "黄色方块", "绿色方块", "紫色小球", "蓝色方块", "左侧"]

    type_in = list(jieba.cut(type_get))
    imge_in_list = []
    for i in imge_get:
        imge_in = list(jieba.cut(i))
        imge_in_list.append(imge_in)

    print(type_in)
    print(imge_in_list)

    type_in_sentence_mat = get_sentence_matrix(type_in, w2v)
    imge_in_sentence_mat_list = []
    for i in imge_in_list:
        imge_in_sentence_mat = get_sentence_matrix(i, w2v)
        imge_in_sentence_mat_list.append(imge_in_sentence_mat)

    res_list = []
    for imge_in_sentence_mat in imge_in_sentence_mat_list:
        mat1 = type_in_sentence_mat
        mat2 = imge_in_sentence_mat

        similarity_mat = cosine_similarity(mat1[:, None, :], mat2[None, :, :], dim=2)
        res, _ = similarity_mat.max(dim=0)
        res = res.view(1, -1)
        res = res.sum() / len(res[0])
        res_list.append(res)

    print(imge_get[torch.argmax(torch.tensor(res_list))])
