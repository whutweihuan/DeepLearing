# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/4/18  22:59
"""
import cv2
import numpy as np
from PIL import Image
import os

# 测试二值图片阈值


def OTSU (img_gray):
    max_g = 0
    suitable_th = 0
    th_begin = 0
    th_end = 256
    for threshold in range(th_begin, th_end):
        bin_img = img_gray > threshold
        bin_img_inv = img_gray <= threshold
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if 0 == fore_pix:
            break
        if 0 == back_pix:
            continue

        w0 = float(fore_pix) / img_gray.size
        u0 = float(np.sum(img_gray * bin_img)) / fore_pix
        w1 = float(back_pix) / img_gray.size
        u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix
        # intra-class variance
        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        if g > max_g:
            max_g = g
            suitable_th = threshold

    return suitable_th

if __name__ == '__main__':
    root = 'mydata/images/'
    fname = '000000002'
    # fname = 'a01-000u-00'
    # fname = 'a01-000u-01'
    path = os.path.join(root,fname+'.png')
    image = cv2.imread(path)
    Image.fromarray(image).show()
    # input()
    thed = OTSU(image)
    # image[image>thed] = 255
    # image[image<thed] = 0

    # image = cv2.medianBlur(image, 5)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    Image.fromarray(image).show()
    Image.fromarray(image).save(os.path.join(root,fname+'-mdianBlur.png'))
    
    