# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/5/25  21:46
"""
# encoder decoder没有attention的训练，目前没有成功

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from IPython.core.display import display
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import re
import logging
import logging.config
from collections import Counter
import matplotlib.ticker as ticker
import matplotlib.image as imgplt

logging.config.fileConfig('logging.conf', defaults={
                          'logfilename': 'log11.txt'})

# create logger
logger = logging.getLogger('nice')

DATAPATH = "C:\\Users\\weihuan\\Desktop\\data"
SAVE_MODLE_NAME = '06-encoder_hwdb.pt'
SAVE_MODLE_NAME2 = '06-decoder_hwdb.pt'

EOS = '<EOS>'
SOS = '<SOS>'
BLK = '<BLK>'
UNK = '<UNK>'

MAX_LENGTH = 128
BATCH = 32
TEACH_FORCING_PROB = 0.5
N_EPOCH = 1000
LEARNING_RATE = 0.00001
IMG_WIDTH = 280
IMG_HEIGHT = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')


class Lang():
    def __init__(self):
        # self.words = np.loadtxt('./mydata/en1.txt', dtype = np.str, encoding = 'utf-8')
        self.w2d = {}
        # self.d2w = {}
        self.w2d[BLK] = 0
        self.w2d[SOS] = 1
        self.w2d[EOS] = 2
        self.w2d[UNK] = 3
        self.lab2v = {}
        with open(r'mydata/hdwn2_char.txt', encoding='utf-8') as f:
            alpha = f.read()

        for i, word in enumerate(alpha):
            self.w2d[word] = i + 4
        self.d2w = {value: key for key, value in self.w2d.items()}

    def word2index(self, s):
        try:
            return self.w2d[s]
        except:
            return self.w2d[UNK]

    def index2word(self, i):
        return self.d2w[i]

    # label such as like
    # a move to stop mr. gaitskell from
    def label2vec(self, label, lens, real_label=False):
        # if label in self.lab2v:
        #     return torch.tensor(self.lab2v[label])
        word_list = [item for item in label]
        vec = []
        vec.append(self.word2index(SOS))
        vec.extend(self.word2index(w) for w in word_list)
        vec.append(self.word2index(EOS))
        if len(vec) < lens and real_label == False:  # 填充空格，对齐操作
            vec.extend([self.word2index(BLK)] * (lens - len(vec)))

        return torch.tensor(vec)

    def size(self):
        return len(self.w2d)

    def batchlabels2vec(self, blabels):

        lens = max(len(s) for s in blabels)
        # print(lens)
        # input("lens----------")
        lens += 2
        bl = torch.ones(len(blabels), lens, dtype=torch.long)
        for i in range(len(blabels)):
            bl[i, :] = self.label2vec(blabels[i], lens)
        return bl


word_lang = Lang()


class TrainData(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.path = []  # 图片的路径
        self.label = []  # 图片代表的字符
        self.transform = transform
        self.target_transform = target_transform

        # 首先处理数据,解析的是文本，统计词的个数，构建词典
        # with open(DATAPATH + "\\ascii\\lines.txt") as f:
        # with open("mydata\\en0.txt", encoding = 'utf-8') as f:
        with open(DATAPATH + "\\trainlabels.txt", encoding='utf-8') as f:
            for line in f:
                self.path.append(line.split('|')[0])
                self.label.append(line.split('|')[1].strip('\n').lower())
            # print(le)
            # print(self.label)
            # input()

    def __getitem__(self, item):
        path = DATAPATH + "\\train\\" + self.path[item]
        # img = Image.open(path).convert('RGB').resize((280, 32), Image.BILINEAR)
        # print(path)
        # input()

        img = self.read_img(path)

        label = self.label[item]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.path)

    def read_img(self, img_path, desired_size=(IMG_HEIGHT, IMG_WIDTH)):
        img = cv2.imread(img_path, 0)
        img_resize, crop_cc = self.resize_image(img, desired_size)
        # img_resize.show()
        img_resize = Image.fromarray(img_resize)  # 得到 PIL 图片
        # img_resize.show()
        img_tensor = self.transform(img_resize)
        # return img_tensor
        return img_tensor

    def OTSU(self, img_gray):
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

    def resize_image(self, image, desired_size):
        # thed = self.OTSU(image)
        # image[image > thed] = 255
        # image[image < thed] = 0
        # image = cv2.medianBlur(image, 5)

        size = image.shape[:2]
        #
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

        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=float(color))
        crop_bb = (left / image.shape[1], top / image.shape[0], (image.shape[1] - right - left) / image.shape[1],
                   (image.shape[0] - bottom - top) / image.shape[0])

        # image = cv2.medianBlur(image, 5)
        return image, crop_bb


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        # (input_size,hidden_size,num_layers)
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
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

    def __init__(self, imgH, nc, nh):
        super(Encoder, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2,2),  # 64x16x50   W/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 128x8x25  W/4
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),  # 256x8x25
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 256x4x25  W/4 + 1
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),  # 512x4x25
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d( (2, 2), (2, 1), (0, 1)),  # 512x2x25  W/4 + 2
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True))  # 512x1x25   W/4 + 1

        self.rnn = nn.Sequential(
            # 第一个参数 指输入向量大小，这里指的是一个图片每一列的通道数
            # 第二个参数 是指隐藏层的向量大小
            # 第三个参数 是指 通过Liner转化的目标大小
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        # print(batch,channel,hight,width)
        # input('-----------2-----------')
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        # [width, batch, channel]  ==> [WIDTH / 4 + 1, batch, 512]
        conv = conv.permute(2, 0, 1)
        # rnn features calculate
        # seq * batch * n_classes// 25 × batchsize × 256（隐藏节点个数）
        encoder_outputs = self.rnn(conv)

        return encoder_outputs


class Decoder(nn.Module):
    def __init__(self, hidden_size=256, output_size=word_lang.size()):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, len(input), -1)
        # print(output.shape)
        # input()


        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self,batchsize):
        return torch.zeros(1, batchsize, self.hidden_size, device=device)


def get_transform(phase="train"):
    transfrom_PIL_list = [
        transforms.RandomAffine((-2, 2), fillcolor=255),
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5),
        transforms.ColorJitter(saturation=0.5),

    ]
    transfrom_tensor_list = [
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value=0),
    ]
    if phase == "train":
        transform = transforms.Compose([
            transforms.RandomApply(transfrom_PIL_list),
            transforms.ToTensor(),
            transforms.RandomApply(transfrom_tensor_list),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]),
        ])
    return transform


# https://blog.csdn.net/dss_dssssd/article/details/83990511
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


encoder = Encoder(IMG_HEIGHT, 1, 256).to(device)
encoder.apply(weight_init)
decoder = Decoder().to(device)
decoder.apply(weight_init)
ctc_loss = torch.nn.CTCLoss(blank=word_lang.word2index(BLK)).to(device)
# encoder_optimizer = torch.optim.RMSprop(encoder.parameters(), lr = LEARNING_RATE)
# decoder_optimizer = torch.optim.RMSprop(decoder.parameters(), lr = LEARNING_RATE)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
# encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr = LEARNING_RATE)
# decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr = LEARNING_RATE)

encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min', factor=0.8, patience=3,
                                                               verbose=False)
td = TrainData(transform=get_transform('train'))
td2 = TrainData(transform=get_transform('test'))


def train():
    encoder.train()
    decoder.train()
    # encoder.load_state_dict(torch.load(SAVE_MODLE_NAME))
    train_dataloader = DataLoader(
        td, batch_size=BATCH, shuffle=True, num_workers=4)
    try:
        encoder.load_state_dict(torch.load(SAVE_MODLE_NAME))
        decoder.load_state_dict(torch.load(SAVE_MODLE_NAME2))
        print("读取模型成功，开始训练")
    except:
        print("未能读取模型，重新开始训练")

    for epoch in range(N_EPOCH):
        loss_total = .0
        # print(train_dataloader.dataset)
        # input()
        for iter, (x, y) in enumerate(train_dataloader):
            # print(train_dataloader.dataset)
            # input()
            # torchvision.transforms.ToPILImage()(x[0]).show()
            label = word_lang.batchlabels2vec(y)
            x, label = x.to(device), label.to(device)
            encoder_output = encoder(x).to(device)  # seq * batch * dic_len

            cost = 0.0
            # print('111111111111')
            # input()
            decoder_input = label[:, 0].to(device, dtype=torch.int64)
            # hidden = decoder.initHidden(len(label))
            hidden = encoder_output[-1].unsqueeze(0)

            teachingforce = True if random.random() < TEACH_FORCING_PROB else False
            attentions = []
            for di in range(1, label.shape[1]):
                decoder_output, hidden = decoder(decoder_input, hidden)
                decoder_input = label[:, di].to(
                    device) if teachingforce else decoder_output.data.topk(1)[1].squeeze()
                # attentions.append(attention)

                cost += F.nll_loss(decoder_output, label[:, di])

            loss = cost
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()

            loss_total += loss.item()
            encoder_optimizer.step()
            decoder_optimizer.step()
            if (iter + 1) % 10 == 0:
                logger.info("[{}/{}][{}/{}] loss: {}".format(epoch + 1, N_EPOCH, iter + 1, len(train_dataloader),
                                                             loss_total / 10))
                loss_total = .0
            if (iter + 1) % 100 == 0:
                torch.save(encoder.state_dict(), SAVE_MODLE_NAME)
                torch.save(decoder.state_dict(), SAVE_MODLE_NAME2)
        torch.save(encoder.state_dict(), SAVE_MODLE_NAME)
        torch.save(decoder.state_dict(), SAVE_MODLE_NAME2)



import  Levenshtein
# print(Levenshtein.distance("今天中午吃饭了吗","今天中午刚吃了海鲜"))

def testMyImage(path=None,label=None):
    # print('-' * 80)
    if path == None:
        path = input('输入图片路径: ')
    image = 0
    try:
        image = td2.read_img(path)
        img = cv2.imread(path, 0)
        img_resize, crop_cc = td2.resize_image(
            img, desired_size=(IMG_HEIGHT, IMG_WIDTH))
        # display(Image.fromarray(img_resize).resize((720, 60)))

        # x = imgplt.imread(path)
        # image = get_transform('test')(image)
        # plt.imshow(x)
    except:
        print('输入图片路径有误')
        return

    try:
        encoder.load_state_dict(torch.load(SAVE_MODLE_NAME))
        decoder.load_state_dict(torch.load(SAVE_MODLE_NAME2))
        # print("读取模型成功，开始识别")
    except:
        print("读取模型失败")
        return
    # print(image.shape)
    image = image.unsqueeze(0).to(device)

    # label = word_lang.batchlabels2vec(y)
    # x, label = x.to(device), label.to(device)
    # seq * batch * dic_len  => 71 * 1 * 5600
    encoder_output = encoder(image).to(device)
    decoder_input = torch.tensor([word_lang.word2index(SOS)], device=device)
    hidden = decoder.initHidden(1)

    pred = []
    attentions = []
    for di in range(1, MAX_LENGTH):
        decoder_output, hidden, attention = decoder(
            decoder_input, hidden, encoder_output)
        # decoder_input = label[:, di].to(device) if teacher == False else decoder_output.data.topk(1)[1].squeeze()
        # attentions.append(attention)
        wi = decoder_output.data.topk(1)[1].squeeze(-1)
        pred.append(word_lang.index2word(wi.item()))
        decoder_input = wi
        attentions.append(attention.squeeze().cpu().data.numpy())
        if wi.item() == word_lang.word2index(EOS) or wi.item() == word_lang.word2index(BLK):
            break
    attentions = np.array(attentions)
    # print(np.array(attentions).shape)
    # showAttention(attentions)
    # print(attentions.shape)
    # print(y[0])
    # print(''.join(pred))
    pred=pred[:-1]
    if label != None:
        temp = 1- Levenshtein.distance(''.join(pred),label)/len(label)
        if(temp>0.5):
            print(temp)
        # print(''.join(pred))
        # print(label)
        # print('-' * 80)
        # print()
    # return ''.join(pred)
    # print('-' * 80)

# def test_image_help(path)

if __name__ == '__main__':
    # evaluate()
    # testMyImage()
    train()
    # print('hello world!')
    # evaluate()
    # print('123')
    # a = word_lang.label2vec("浔阳地僻无音乐,终岁不闻丝竹声。",30)
    # print(a)
    pass
