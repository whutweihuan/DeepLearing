# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/1/10  13:38
"""
# https://bastings.github.io/annotated_encoder_decoder/
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# we will use CUDA if it is available
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:",USE_CUDA)
print(DEVICE)
















