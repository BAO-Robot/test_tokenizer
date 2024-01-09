# author: yzc

from tokenizers import Tokenizer, normalizers, models, pre_tokenizers, trainers
from utils.configure_util import ConfLoader


def train_tokenizer(params):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))  # 用 [UNK] 替换未知字符
    tokenizer.normalizer = normalizers.Sequence(  # 按一定规则标准化：NFD：统一 Unicode 编码
        [normalizers.NFD(), normalizers.Lowercase()]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()  # 使用 空格 来分割
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=params["vocab_size"], special_tokens=special_tokens)

    tokenizer.train(['./texts/dataset/the_three_body_problem.txt',
                            './texts/dataset/key_sentences.txt',
                            './texts/dataset/in_the_name_of_people.txt',
                            './texts/dataset/essay.txt'], trainer=trainer)  # 用于训练的文本
    tokenizer.save('my_tokenizer.json')


if __name__ == "__main__":
    params = ConfLoader("conf.yaml")

    train_tokenizer(params)
    tokenizer = Tokenizer.from_file("my_tokenizer.json")

    s = "帮我拿黄色小球"  # "帮我把红色小球拿到紫色方块右上角"
    tokens = tokenizer.encode(s)
    print(tokens.ids, tokens.tokens)
