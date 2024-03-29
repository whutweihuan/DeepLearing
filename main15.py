# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/3/31  22:41
"""
# 生成一个字典,将数据保存在 mydata/en.txt 中

import numpy as np
from collections import Counter, Set
import re

import re
import numpy as np
import unicodedata
import string
import html

RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(
    chr(768), chr(769), chr(832), chr(833), chr(2387),
    chr(5151), chr(5152), chr(65344), chr(8242)), re.UNICODE)
RE_RESERVED_CHAR_FILTER = re.compile(r'[¶¤«»]', re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}]'.format(re.escape(string.punctuation)), re.UNICODE)

LEFT_PUNCTUATION_FILTER = """!%&),.:;<=>?@\\]^_`|}~"""
RIGHT_PUNCTUATION_FILTER = """"(/<=>@[\\^_`{|~"""
NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE)


class Tokenizer():
    """Manager tokens functions and charset/dictionary properties"""

    def __init__ (self, chars, max_text_length = 128):
        self.PAD_TK, self.UNK_TK = "¶", "¤"
        self.chars = (self.PAD_TK + self.UNK_TK + chars)

        self.PAD = self.chars.find(self.PAD_TK)
        self.UNK = self.chars.find(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode (self, text):
        """Encode text to vector"""

        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
        text = " ".join(text.split())
        encoded = []

        for item in text:
            index = self.chars.find(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)

    def decode (self, text):
        """Decode vector to text"""

        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)
        decoded = text_standardize(decoded)

        return decoded

    def remove_tokens (self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "")


def text_standardize (text):
    """Organize/add spaces around punctuation marks"""

    if text is None:
        return ""

    text = html.unescape(text).replace("\\n", "").replace("\\t", "")

    text = RE_RESERVED_CHAR_FILTER.sub("", text)
    text = RE_DASH_FILTER.sub("-", text)
    text = RE_APOSTROPHE_FILTER.sub("'", text)
    text = RE_LEFT_PARENTH_FILTER.sub("(", text)
    text = RE_RIGHT_PARENTH_FILTER.sub(")", text)
    text = RE_BASIC_CLEANER.sub("", text)

    text = text.lstrip(LEFT_PUNCTUATION_FILTER)
    text = text.rstrip(RIGHT_PUNCTUATION_FILTER)
    # text = text.translate(str.maketrans({c: f" {c} " for c in string.punctuation}))
    text = NORMALIZE_WHITESPACE_REGEX.sub(" ", text.strip())

    return text


DATAPATH = "C:\\Users\\weihuan\\Desktop\\IAM"


def standard_mytext ():
    with open(DATAPATH + "\\ascii\\lines.txt") as f:
        text = f.read()
        text = text_standardize(text)
        print(text)
        return text


def write_dic ():
    # 首先处理数据,解析的是文本，统计词的个数，构建词典

    label = []
    # with open(DATAPATH + "\\ascii\\lines.txt") as f:
    with open("mydata\\en0.txt") as f:
        for i in range(23):
            f.readline()
        while True:
            line = f.readline()
            if line:
                line = line.split(' ')

                label.append(line[8].strip('\n').split('|'))
            else:
                break;
    label = [item.lower() for sub in label for item in sub]
    cnt = Counter(label).most_common(12000)

    pattern = re.compile(r".*?[\d,#\"'\(\):;\?\!\.\-]+.*?")
    all_words = [item[0] for item in cnt]
    np.savetxt('.\mydata\en1.txt', np.array(all_words), fmt = "%s")


def write_dic_ch ():
    # 首先处理数据,解析的是文本，统计词的个数，构建词典

    label = []
    lines = []
    # with open(DATAPATH + "\\ascii\\lines.txt") as f:
    with open("mydata\\en0.txt") as f:
        for i in range(23):
            f.readline()
        while True:
            line = f.readline()
            if line:
                line = line.split(' ')

                label.append(line[8].strip('\n').split('|'))
            else:
                break;

    label = [item.lower() for sub in label for item in sub]
    label = ' '.join(label)
    print(label)

    cnt = Counter(label)

    print(cnt)
    print(len(cnt))
    keys = [k[0] for k in cnt.items()]
    print(keys)
    return

    pattern = re.compile(r".*?[\d,#\"'\(\):;\?\!\.\-]+.*?")
    all_words = [item[0] for item in cnt]
    np.savetxt('.\mydata\en1.txt', np.array(all_words), fmt = "%s")


def findblankbug ():
    pat = re.compile(r'[a-z0-9]+')


def handel_hwd2 ():
    text = ''
    with open(r'C:\Users\weihuan\Desktop\data\trainlabels.txt', encoding = 'utf-8') as f:
        for line in f:
            text += line.split('|')[1].strip('\n')
    with open(r'C:\Users\weihuan\Desktop\data\testlabels.txt', encoding = 'utf-8') as f:
        for line in f:
            text += line.split('|')[1].strip('\n')
    text += 'abcdefghijklmnopkrstuvwxyz'
    cnt = Counter(text.lower())
    words = [w[0] for w in cnt.most_common()]
    words = ''.join(sorted(words))
    print(words)

if __name__ == '__main__':
    # write_dic_ch()
    handel_hwd2()
    pass
