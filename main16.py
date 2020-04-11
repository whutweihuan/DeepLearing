# -*- coding: utf-8 -*-
"""
author: weihuan
date: 2020/4/1  23:41
"""
# 测试 resize 图片 的函数
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

H = 32
W = 1024
N_IMAGE = 2

def read_img (img_path, desired_size = (H, W)):
    img = cv2.imread(img_path, 0)
    img_resize, crop_cc = resize_image(img, desired_size)
    img_resize = Image.fromarray(img_resize)
    # img_tensor = transform(img_resize)
    return img_resize


def resize_image (image, desired_size ):
    ''' Helper function to resize an image while keeping the aspect ratio.
    Parameter
    ---------

    image: np.array
        The image to be resized.
    desired_size: (int, int)
        The (height, width) of the resized image
    Return
    ------
    image: np.array
        The image of size = desired_size
    bounding box: (int, int, int, int)
        (x, y, w, h) in percentages of the resized image of the original
    '''
    size = image.shape[:2]
    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0]) / size[0]
        ratio_h = float(desired_size[1]) / size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x * ratio) for x in size])
        image = cv2.resize(image, (new_size[1], new_size[0]))
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = image[0][0]
    if color < 230:
        color = 230
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = float(color))
    crop_bb = (left / image.shape[1], top / image.shape[0], (image.shape[1] - right - left) / image.shape[1],
               (image.shape[0] - bottom - top) / image.shape[0])
    image[image > 230] = 255
    return image, crop_bb

for i in range(N_IMAGE):

    path = 'mydata/a01-000u-0'+ str(i) +'.png'
    img = read_img(path)
    # print(img)
    img.show()
    # print(np.array(img).shape)
# print (Image.fromarray(img))

