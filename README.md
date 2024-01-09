# test_tokenizer
测试切词器、word2vec 等组件

### 文件说明：

- `texts`: 训练时用到的文本
  - `dataset`: 用于训练切词器的数据集，原始数据集
  - `jieba_dict.txt`: jieba 的词典，可以替换为自定义的词典
  - `segmented_text.txt`: 切好词之后的数据集，用于训练 word2vec
- `utils`: 加载超参数组件
- `cal_sim.py`: 计算句子之间相似度
- `conf.yaml`: 配置文件
- `embedding.vector`: 训练好的词向量
- `my_tokenizer.json`: 训练好的 tokenizer
- `sentence_segment.py`: 文本切词，生成 `segmented_text.txt` 文件
- `tokenizer.py`: 训练切词器
- `train_word2vec.py`: 训练词向量

### Quik Start

1. 将准备好的文本放入 `texts/dataset` 路径下
2. 运行 `sentence_segment.py`
3. 运行 `train_word2vec.py`
4. 运行 `cal_sim.py`

### TODO

封装代码

---
Author: yzc
