# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/1/18  21:15
"""

import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# 读入一张灰度图的图片
im = Image.open('./1.png').convert('L')
# 将其转换为一个矩阵
im = np.array(im, dtype = 'float32')

# print(im.shape[0],im.shape[1])
# plt.imshow(im.astype('uint8'),cmap='gray')
# plt.show()

# pytorch tensor
im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))

# 定义一个算子对其进行轮廓检测
conv1 = nn.Conv2d(1, 1, 3, bias = False)

# 轮廓检测算子
sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype = 'float32')
# 适配卷积的输入输出
sobel_kernel = sobel_kernel.reshape(1, 1, 3, 3)
# 给卷积的 kernel 赋值
conv1.weight.data = torch.from_numpy(sobel_kernel)

# 作用在图片上
edge1 = conv1(Variable(im))
# 将输出转化为图片格式
edge1 = edge1.data.squeeze().numpy()
plt.imshow(edge1, cmap = 'gray')
plt.show()

pool1 = nn.MaxPool2d(2, 2)
smallIm = pool1(Variable(im))
smallIm = smallIm.data.squeeze().numpy()
plt.imshow(smallIm, cmap = 'gray')
plt.show()
