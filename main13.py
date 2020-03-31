# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/3/30  23:25
"""
# 正式开始项目
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

DATAPATH = "C:\\Users\\weihuan\\Desktop\\IAM"


class TrainData(Dataset):
    def __init__ (self, transform = None, target_transform = None):
        self.path = []  # 图片的路径
        self.label = []  # 图片代表的字符
        self.transform = transform
        self.target_transform = target_transform

        # 首先处理数据,解析的是文本，统计词的个数，构建词典
        with open(DATAPATH + "\\ascii\\lines.txt") as f:
            for i in range(23):
                f.readline()
            while True:
                line = f.readline()
                if line:
                    line = line.split(' ')
                    self.path.append(line[0])
                    self.label.append(' '.join(line[8].strip('\n').split('|')))
                else:
                    break;

    def __getitem__ (self, item):
        path = DATAPATH + "\\lines\\lines\\lines_all\\" + self.path[item] + ".png"
        img = Image.open(path).convert('RGB').resize((280,32),Image.BILINEAR)
        # img.show()
        label = self.label[item]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__ (self):
        return len(self.path)


class BidirectionalLSTM(nn.Module):
    def __init__ (self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional = True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward (self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        # print(T,b,h)
        # input('--------------------2')
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class Encoder(nn.Module):
    '''
        CNN+BiLstm做特征提取
    '''

    def __init__ (self, imgH, nc, nh):
        super(Encoder, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 64x16x50
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 128x8x25
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),  # 256x8x25
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 1), padding = (0, 1)),  # 256x4x26
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),  # 512x4x26
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 3), (2, 1), (0, 1)),  # 512x2x26
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True))  # 512x1x25
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh))

    def forward (self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        # print(b,c,h,w)
        # input('-----------2-----------')
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features calculate
        encoder_outputs = self.rnn(conv)  # seq * batch * n_classes// 25 × batchsize × 256（隐藏节点个数）

        return encoder_outputs

class Decoder(nn.Module):
    '''
        decoder from image features
    '''
    def __init__(self, nh=256, nclass=13, dropout_p=0.1, max_length=71):
        super(Decoder, self).__init__()
        self.hidden_size = nh
        self.decoder = Attentiondecoder(nh, nclass, dropout_p, max_length)

    def forward(self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result

class Attentiondecoder(nn.Module):
    """
        采用attention注意力机制，进行解码
    """
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=71):
        super(Attentiondecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # calculate the attention weight and weight * encoder_output feature
        embedded = self.embedding(input)         # 前一次的输出进行词嵌入
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)        # 上一次的输出和隐藏状态求出权重, 主要使用一个linear layer从512维到71维，所以只能处理固定宽度的序列
        attn_applied = torch.matmul(attn_weights.unsqueeze(1),
                                 encoder_outputs.permute((1, 0, 2)))      # 矩阵乘法，bmm（8×1×56，8×56×256）=8×1×256

        output = torch.cat((embedded, attn_applied.squeeze(1) ), 1)       # 上一次的输出和attention feature做一个融合，再加一个linear layer
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)                         # just as sequence to sequence decoder

        output = F.log_softmax(self.out(output[0]), dim=1)          # use log_softmax for nllloss
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))

        return result




if __name__ == '__main__':
    td = TrainData(transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               # transforms.Normalize
                                           ]))
    dataloader = DataLoader(td,batch_size=8,shuffle=True, num_workers=4)
    # print(len(dataloader))
    encoder = Encoder(32,3,256)
    for i,(x,y) in enumerate(dataloader):
       encoder_output =   encoder(x)
    # print('Done')
    # print(len(dataloader.dataset))
    #    print(encoder_output.shape)
       print((i+1) / len(dataloader.dataset)*100)
       # input('-----------')
    #     print(x,y)
    #   print(x)
    # print(dataloader.dataset[0][0].shape)
