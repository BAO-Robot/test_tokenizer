from tokenizers import Tokenizer, normalizers, models, pre_tokenizers, trainers
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity

from utils.configure_util import ConfLoader


def train_tokenizer(params):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))  # 用 [UNK] 替换未知字符
    tokenizer.normalizer = normalizers.Sequence(  # 按一定规则标准化：NFD：统一 Unicode 编码
        [normalizers.NFD(), normalizers.Lowercase()]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()  # 使用 空格 来分割
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=params["vocab_size"], special_tokens=special_tokens)  # 分成 10000 类

    tokenizer.train(['./texts/text.txt', './texts/key_sentences.txt'], trainer=trainer)  # 用于训练的文本
    tokenizer.save('my_tokenizer.json')


def padding(vector, max_length):
    paddedVectors = torch.zeros(len(vector), max_length, dtype=torch.int)
    for k, i in enumerate(vector):
        length = min(len(i), max_length)
        paddedVectors[k, :length] = torch.tensor(i[:length])
    return paddedVectors


class Embedding(nn.Module):
    def __init__(self, params):
        super(Embedding, self).__init__()
        self.dim = params["hidden_dim"]
        self.device = params["device"]
        self.word_em = nn.Embedding(num_embeddings=params["vocab_size"], embedding_dim=self.dim).to(self.device)

    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        we = self.word_em(input_ids)
        return we * torch.sqrt(torch.tensor(self.dim)).to(self.device)


if __name__ == "__main__":
    params = ConfLoader("conf.yaml")
    embed = Embedding(params)
    # word2vec = KeyedVectors.load_word2vec_format("", binary=True)

    train_tokenizer(params)
    tokenizer = Tokenizer.from_file("my_tokenizer.json")
    # sentence = tokenizer.get_vocab().keys()
    # print(sentence)

    s = "帮我拿黄色小球"  # "帮我把红色小球拿到紫色方块右上角"
    tokens = tokenizer.encode(s)
    print(tokens.ids, tokens.tokens)



    # type_input = torch.tensor(tokens.ids).unsqueeze(dim=0)
    # type_input_raw_length = len(type_input[0])
    # type_input = padding(type_input, params["max_length"])
    # type_input_vec = embed.forward(type_input)
    # print(type_input_vec.shape)
    #
    # img_get = ["左上角", "右下角", "黄色小球", "黄色方块", "绿色方块", "紫色小球", "蓝色方块", "左侧"]
    # img_input_vecs = []
    # img_input_raw_length = []
    # for i in img_get:
    #     output = tokenizer.encode(i)
    #     img_input_vecs.append(output.ids)
    #     img_input_raw_length.append(len(output.ids))
    # img_input_vecs = padding(img_input_vecs, params["max_length"])
    # img_input_vec = embed.forward(img_input_vecs)
    # print(img_input_vec.shape)
    #
    # # for i in img_input_vec:
    # mat = cosine_similarity(type_input_vec[0][:, None, :], img_input_vec[0][None, :, :], dim=2)
    # print(mat.shape)
    # vector = mat[:type_input_raw_length, :img_input_raw_length[0]]
    # print(vector)
    # print(torch.argmax(vector))
