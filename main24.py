# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/4/17  0:45
"""

import io
import os
import codecs
import struct
import numpy as np
from PIL import  Image

# 本程序将 hwdb2.0 图片转化为 png 格式图片和标签

gen_test_data = True  # 如果产生测试图片写 True, 训练集写 False

datapath = r'C:\Users\weihuan\Desktop\data2'
TRAIN_SOURCE_PATH = os.path.join(datapath,'HWDB2.2Train')
TRAIN_SAVE_PATH = os.path.join(datapath,'train')

TEST_SOURCE_PATH = os.path.join(datapath,'HWDB2.2Test')
TEST_SAVE_PATH = os.path.join(datapath,'test')

bad_image = False

step = 40000
iter = 0


img_root = TEST_SOURCE_PATH if gen_test_data else TRAIN_SOURCE_PATH
label_txt = 'testlabels.txt' if gen_test_data else 'trainlabels.txt'
file_num = len(os.listdir(img_root))
savepath = TEST_SAVE_PATH if gen_test_data else TRAIN_SAVE_PATH
labelwriter = open(os.path.join(datapath,label_txt),'w+',encoding = 'utf-8')

for fname in os.listdir(img_root):
    iter = iter + 1
    print('[{}/{}]'.format(iter,file_num))
    fname_root = os.path.join(img_root, fname)
    with codecs.open(fname_root, mode = 'rb') as fin:
        while True:
            try:
                dgrlhsize = fin.read(4)
                dgrlhsize = struct.unpack("I", dgrlhsize)[0]
            except :
                break
            format_code = fin.read(8)
            illuslen = dgrlhsize - 36
            illuslen = fin.read(illuslen)
            codetype = fin.read(20)
            codelen = fin.read(2)
            codelen = struct.unpack('h', codelen)[0]
            bitspp = fin.read(2)

            # 图片高宽
            pageHei = fin.read(4)
            pageHei = struct.unpack('I', pageHei)[0]
            pageWid = fin.read(4)
            pageWid = struct.unpack('I', pageWid)[0]

            # 图片行数
            lineNumber = fin.read(4)
            lineNumber = struct.unpack('I', lineNumber)[0]

            for i in range(lineNumber):
                # 读取标签
                charNumber = (struct.unpack('I', fin.read(4))[0])
                label = fin.read(charNumber * codelen)
                label = label.replace(b'\x00',b'')
                label = label.replace(b'\xff',b'')
                try:
                    label = str(label,('gbk')).encode('utf-8').decode('utf-8')
                except  Exception as e:
                    print('error' + str(e))
                    bad_image = True

                # 读取图像
                lineTop = struct.unpack('I', fin.read(4))[0]
                lineLeft = struct.unpack('I', fin.read(4))[0]
                lineHei = struct.unpack('I', fin.read(4))[0]
                lineWid = struct.unpack('I', fin.read(4))[0]

                # 读取并转为灰度图像
                bytes_image = fin.read(lineHei* lineWid)
                image = Image.frombytes(mode = 'L' ,data = bytes_image,size = (lineWid,lineHei))

                if not bad_image:
                    step += 1
                    image_name = str(step).rjust(9,'0')+'.png'
                    savename = os.path.join(savepath,image_name)
                    labelwriter.writelines(image_name +'|'+label+'\n')
                    image.save(savename)
                bad_image = False
labelwriter.close()
