# author: yzc

import jieba
import re


def get_sentences(filePath):
    sentences = []
    text = ""
    with open(filePath, encoding='utf-8') as f:
        for line in f:
            sentences.extend(re.split(r'[！？。]+', line))
    return sentences


def segment_sentences(sentences, savePath):
    format_sentences = []
    remove_chars = '[·’!"#$%&\'()*+,-./:;<=>?@，。?★、…（）【】《》？“”‘’！[\\]^_`{|}~]+'
    for i in range(len(sentences)):
        string = re.sub(remove_chars, "", sentences[i])
        format_sentences.append(string)

    fileTrainSeg = []
    for i in range(len(format_sentences)):
        fileTrainSeg.append([' '.join(jieba.cut(format_sentences[i], cut_all=False))])

    with open(savePath, 'wb') as fw:
        for i in range(len(fileTrainSeg)):
            if fileTrainSeg[i][0] == "\n":
                continue
            fw.write(fileTrainSeg[i][0].strip().encode('utf-8'))
            fw.write("\n".encode('utf-8'))


if __name__ == "__main__":
    filePaths = ['texts/dataset/the_three_body_problem.txt',
                 'texts/dataset/in_the_name_of_people.txt',
                 'texts/dataset/key_sentences.txt',
                 'texts/dataset/essay.txt']
    savePath = 'texts/segmented_text.txt'

    total_sentences = []
    for filePath in filePaths:
        sentences = get_sentences(filePath)
        total_sentences.extend(sentences)

    segment_sentences(total_sentences, savePath)

