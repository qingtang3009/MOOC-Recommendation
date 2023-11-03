# -*- coding:utf-8 -*-
import json
import numpy as np
from collections import Counter


def load_stopword():
    stopwords_path = r'./data/stopwords.txt'
    with open(stopwords_path, 'r', encoding='utf-8') as r:
        list_ = r.readlines()
    stopwords = []
    for word in list_:
        stopwords.append(word.strip())
    return stopwords


def load_data(path):
    with open(path, 'r', encoding='utf-8') as r:
        dic = json.load(r)
    return dic


stopwords = load_stopword()

dic = load_data(r'./data/ids_texts_seg.json')


word_bag = []
texts_words_list = []
for value in dic.values():
    texts_words_list.append(value.split())
    for word in value.split():
        if word not in stopwords:
            word_bag.append(word)
print(len(texts_words_list))

seg_dic = dict(zip(dic.keys(), texts_words_list))

counter = Counter(word_bag)
counter_ = sorted(counter.items(), key=lambda x: x[1], reverse=True)
vocab = []
for tuple in counter_:
    vocab.append(tuple[0])
print(vocab)
print("The lenth of the word bag before cutting off: {}".format(len(vocab)))
vocab = vocab[:10000]
print(vocab)
print("The lenth of the word bag after cutting off: {}".format(len(vocab)))

list_ = []
for value in seg_dic.values():
    bag_vector = np.zeros(len(vocab))
    for i, word_ in enumerate(vocab):
        for word in value:
            if word == word_:
                bag_vector[i] = bag_vector[i] + 1
    list_.append(list(bag_vector))
    # print("句子{}，词向量{}".format(key, np.array(bag_vector)))

dic_ = dict(zip(seg_dic.keys(), list_))
json_file = json.dumps(dic_, indent=4, ensure_ascii=False)
save_path = r'./data/sparse_vectors.json'
with open(save_path, 'w', encoding='utf-8') as w:
    w.write(json_file)



