# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/3/31  22:41
"""
# 生成一个字典,将数据保存在 mydata/en.txt 中

import numpy as np
from collections import Counter,Set
import re

DATAPATH = "C:\\Users\\weihuan\\Desktop\\IAM"

# 首先处理数据,解析的是文本，统计词的个数，构建词典

label = []
with open(DATAPATH + "\\ascii\\lines.txt") as f:
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
cnt = Counter(label).most_common(5000)
# print(cnt)
# print(len(cnt))
# token =[',','.','*','?','!','"',"'",':',';','-','0','1','2','3','4','5','6','7','8','9']
pattern = re.compile(r".*?[\d,#\"'\(\):;\?\!\.\-]+.*?")

all_words = [item[0] for item in cnt if pattern.match(item[0]) == None]

# print(len(cnt))
print(all_words)
# print(len(all_words))
np.savetxt('.\mydata\en.txt',np.array(all_words),fmt="%s")