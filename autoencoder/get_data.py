# -*- coding:utf-8 -*-
import json
import re
import jieba


def pre_processing(raw_data, ids_texts_path):
    with open(raw_data, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    keys = []
    data = []

    for line in lines:
        list_ = line.split(",")
        str_text = (list_[3])[11:-1]
        str = re.sub(r'\\n|\\t|\\r|&nbsp;|</p>|●|<p>|<span>|<br/>|</strong>|<strong>|<div>|<span|</span>|<tr>|<td>|'
                       r'<tbody>|<br />|<table>|<a>|</a>|<b>|</b>|<ul>|<li>|</li>|</ul>|</tr>|</td>|</tbody>|</table>|'
                       r'</div>|<h2>|</h2>|About this course'
                       r'<p class=\\"MsoNormal\\" style=\\"text-indent:18.0pt;\\"> style=\\"font-size:9.0pt;font-family:宋体;\\">|'
                       r'<p style=\\"color:#666666;font-family:Aria|'
                       r'style=\\"background-color:#FFFFFF;line-height:24px;font-family:宋体;font-size:16px;\\">“|'
                       r'style=\\"background-color:#FFFFFF;color:#333333;line-height:24px;font-family:宋体;font-size:16px;\\">|'
                       r'<p class=\\\"MsoNormal\\\" style=\\\"text-indent:18.0pt;\\\"> style=\\\"font-size:9.0pt;font-family:宋体;\\\">|'
                       r'<p class=\\\"MsoNormal\\\" style=\\\"text-align:justify;text-indent:28.0pt;\\\">|'
                       r'style=\\\"background-color:#FFFFFF;line-height:24px;font-family:宋体;font-size:16px;\\\">|'
                       r'<p class=\\\"MsoNormal\\\">', '', str_text)
        str_name = (list_[0])[8:-1]
        data.append(str)
        keys.append(str_name)

    ids_texts = dict(zip(keys, data))

    json_file = json.dumps(ids_texts, indent=4, ensure_ascii=False)
    with open(ids_texts_path, 'w', encoding='utf-8') as w:
        w.write(json_file)

    return ids_texts


def text2words(ids_texts_path, ids_texts_seg_path):
    with open(ids_texts_path, 'r', encoding='utf-8') as f:
        ids_texts = json.load(f)
    print(ids_texts)
    words = []

    for text in ids_texts.values():
        words_ = jieba.cut(text, cut_all=False)
        words.append(" ".join(words_))

    ids_texts_seg = dict(zip(ids_texts.keys(), words))

    json_file = json.dumps(ids_texts_seg, indent=4, ensure_ascii=False)
    with open(ids_texts_seg_path, 'w', encoding='utf-8') as w:
        w.write(json_file)

    return ids_texts_seg


data_path = r'.\data\course.json'
ids_texts_save_path = r'.\data\ids_texts.json'
# ids_texts = pre_processing(data_path, ids_texts_save_path)

ids_texts_path = r'.\data\ids_texts_hand_made.json'
ids_texts_seg_save_path = r'.\data\ids_texts_seg.json'
text2words(ids_texts_path, ids_texts_seg_save_path)

