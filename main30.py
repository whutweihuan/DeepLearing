# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/5/25  17:02
"""
# 仿射变换前后
import torchvision
from torchvision import transforms
from PIL import Image

im = Image.open('.\\mydata\\images\\000000001.png')

im1 = transforms.ColorJitter(brightness=0.8)(im)
im1.save('.\\mydata\\images\\brightness.png')

im2 = transforms.ColorJitter(contrast = 0.8)(im)
im2.save('.\\mydata\\images\\contrast.png')

im3 =  transforms.ColorJitter(saturation = 0.8)(im)
im3.save('.\\mydata\\images\\saturation.png')

im4 = transforms.RandomAffine((-2, 2), fillcolor=255)(im)
im4.save('.\\mydata\\images\\affine.png')

# im5 = transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value=0)(im)
# im5.save('.\\mydata\\images\\erase.png')

