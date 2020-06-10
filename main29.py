# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/5/24  23:52
"""
# ctc hwdb2.2
import Levenshtein
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
import random
import logging
import logging.config
from  collections import Counter

logging.config.fileConfig('logging.conf', defaults={
                          'logfilename': 'log10.txt'})

# create logger
logger = logging.getLogger('nice')

DATAPATH = "C:\\Users\\weihuan\\Desktop\\data\\"

EOS = '<EOS>'
SOS = '<SOS>'
BLK = '<BLK>'
UNK = '<UNK>'

MAX_LENGTH = 20
BATCH = 64
TEACH_FORCING_PROB = 0.5
N_EPOCH = 10000
LEARNING_RATE = 0.0001
IMG_WIDTH = 280
IMG_HEIGHT = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Lang():
    def __init__ (self):
        # self.words = np.loadtxt('./mydata/hdwn2_char.txt', dtype = np.str, encoding = 'utf-8')
        with open('./mydata/hdwn2_char.txt',encoding = 'utf8') as f:
            self.words = f.read()
        self.w2d = {}
        # self.d2w = {}
        self.w2d[SOS] = 0
        self.w2d[EOS] = 1
        self.w2d[BLK] = 2
        self.w2d[UNK] = 3
        self.lab2v = {}
        for i, word in enumerate(self.words):
            self.w2d[word] = i + 4
        self.d2w = {value: key for key, value in self.w2d.items()}

    def word2index (self, s):
        try:
            return self.w2d[s]
        except:
            return self.w2d[UNK]

    def index2word (self, i):
        try:
            return self.d2w[i]
        except:
            return UNK

    # label such as like
    # 他了。第三，扃扉就枕
    def label2vec (self, label, lens):
        # if label in self.lab2v:
        #     return torch.tensor(self.lab2v[label])
        word_list = [w for w in label]
        vec = []
        vec.append(self.word2index(SOS))
        vec.extend(self.word2index(w) for w in word_list)
        vec.append(self.word2index(EOS))
        if len(vec) < lens:  # 填充空格，对齐操作
            vec.extend([self.word2index(BLK)] * (lens - len(vec)))

        return torch.tensor(vec)

    def size (self):
        return len(self.w2d)

    def batchlabels2vec (self, blabels):

        lens = max(len(s) for s in blabels)
        # print(lens)
        # input("lens----------")
        lens += 2
        bl = torch.ones(len(blabels), lens, dtype = torch.long)
        for i in range(len(blabels)):
            bl[i, :] = self.label2vec(blabels[i], lens)
        return bl


word_lang = Lang()


class TrainData(Dataset):
    def __init__ (self, transform = None, target_transform = None):
        self.path = []  # 图片的路径
        self.label = []  # 图片代表的字符
        self.transform = transform
        self.target_transform = target_transform

        # 首先处理数据,解析的是文本，统计词的个数，构建词典
        # with open(DATAPATH + "\\ascii\\lines.txt") as f:
        with open(DATAPATH+"trainlabels.txt", encoding = 'utf-8') as f:
            lines = f.readlines()
            self.path = [item.split('|')[0] for item in lines]
            self.label = [item.split('|')[1].strip('\n') for item in lines]
            # self.label = torch.tensor(self.label)

    def __getitem__ (self, item):
        path = DATAPATH + "train\\" + self.path[item]
        # img = Image.open(path).convert('RGB').resize((280, 32), Image.BILINEAR)
        img = self.read_img(path)
        # img = np.asarray(img)
        # print(torch.from_numpy(img))
        # print(img)
        # return
        # Image.fromarray(img).show()
        # img.show()
        label = self.label[item]

        # if self.transform is not None:
        #     # img = self.transform(img)
        #     img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        # label = word_lang.label2vec(label)

        return img, label

    def __len__ (self):
        return len(self.path)

    def read_img (self, img_path, desired_size = (IMG_HEIGHT, IMG_WIDTH)):
        img = cv2.imread(img_path, 0)
        img_resize, crop_cc = self.resize_image(img, desired_size)
        # img_resize.show()
        img_resize = Image.fromarray(img_resize)  # 得到 PIL 图片
        # img_resize.show()
        img_tensor = self.transform(img_resize)
        # return img_tensor
        return img_tensor

    def resize_image (self, image, desired_size):
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
        thed = self.OTSU(image)
        image[image > thed] = 255
        image[image < thed] = 0
        image = cv2.medianBlur(image, 5)


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

    def OTSU (self, img_gray):
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

class BidirectionalLSTM(nn.Module):
    def __init__ (self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        # (input_size,hidden_size,num_layers)
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
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 64x16x50   W/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 128x8x25  W/4
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),  # 256x8x25
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 256x4x25  W/4 + 1
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),  # 512x4x25
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 512x2x25  W/4 + 2
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True))  # 512x1x25   W/4 + 1

        self.rnn = nn.Sequential(
            # 第一个参数 指输入向量大小，这里指的是一个图片每一列的通道数
            # 第二个参数 是指隐藏层的向量大小
            # 第三个参数 是指 通过Liner转化的目标大小
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, word_lang.size()))

    def forward (self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        # print(batch,channel,hight,width)
        # input('-----------2-----------')
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [width, batch, channel]  ==> [WIDTH / 4 + 1, batch, 512]
        # rnn features calculate
        encoder_outputs = self.rnn(conv)  # seq * batch * n_classes// 25 × batchsize × 256（隐藏节点个数）

        return encoder_outputs


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
def weight_init (m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# print(label2vec('#'))
encoder = Encoder(IMG_HEIGHT, 1, 256).to(device)
encoder.apply(weight_init)
# decoder = decoderV2(256, word_lang.size(), dropout_p = 0.1).to(device)

loss_fn = torch.nn.CTCLoss(blank = word_lang.word2index(BLK)).to(device)
# loss_fn = torch.nn.nll_loss().to(device)
encoder_optimizer = torch.optim.RMSprop(encoder.parameters(), lr = LEARNING_RATE)

encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min', factor = 0.8, patience = 3,
                                                               verbose = False)
td = TrainData(transform = get_transform('train'))


def train ():
    train_dataloader = DataLoader(td, batch_size = BATCH, shuffle = True)
    try:
        encoder.load_state_dict(torch.load('05-encoder_ctc.pt'))
        print("读取模型成功，开始训练")
    except:
        print("未能读取模型，重新开始训练")
    encoder.train()
    for epoch in range(N_EPOCH):
        loss_total = .0
        for iter, (x, y) in enumerate(train_dataloader):
            label = word_lang.batchlabels2vec(y)
            x, label = x.to(device), label.to(device)
            encoder_output = encoder(x)  # seq * batch * dic_len

            # CTC loss
            # 第一个参数是指 编码器输出的结果 seq * batch * dic_len
            preds_size = Variable(torch.IntTensor([encoder_output.size(0)] * encoder_output.shape[1]))
            label_len = Variable(torch.IntTensor(
                [len(v) - Counter(v.cpu().data.numpy())[word_lang.word2index(BLK)] for v in label]))

            loss = loss_fn(encoder_output, label, preds_size, label_len)
            encoder.zero_grad()
            loss.backward()
            loss_total += loss.item()
            encoder_optimizer.step()
            if (iter + 1) % 10 == 0:
                logger.info("[{}/{}][{}/{}] loss: {}".format(epoch + 1, N_EPOCH, iter + 1, len(train_dataloader),
                                                       loss_total / 10))
                loss_total = .0
        torch.save(encoder.state_dict(), '05-encoder_ctc.pt')


def train_notebook (lr):
    global encoder_optimizer
    encoder_optimizer = torch.optim.RMSprop(encoder.parameters(), lr = lr)

    train()


def evaluate ():
    encoder.load_state_dict(torch.load('05-encoder_ctc.pt'))
    encoder.eval()
    test_dataloader = DataLoader(td, batch_size = 1, shuffle = False)

    for iter, (x, y) in enumerate(test_dataloader):
        label = word_lang.batchlabels2vec(y)
        x, label = x.to(device), label.to(device)
        encoder_output = encoder(x)  # seq * batch * dic_len  => 71 * 1 * 5600
        _, preds = encoder_output.max(2)      # preds 71 * 1
        preds = preds.transpose(1, 0).contiguous().view(-1)
        # for di in range
        text = [word_lang.index2word(i) for i in preds.cpu().data.numpy()]
        print(y[0])
        text = [item for item in text if item != '<SOS>' and item != '<EOS>']
        print(''.join(text) + '\n\n\n')
        if iter % 100 == 0:
            input('-----')

def plot_loss(logname, savename, start=0):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # data = np.genfromtxt('log3.txt',delimiter=',',dtype=None)
    # print(data)
    with open(logname) as f:
        text = f.readlines()
    # print(text)
    y = [item.split(': ')[-1].strip('\n') for item in text]
    try:
        y = y[start:]
    except:
        print('Index error')
    y = y[10:]
    y = [float(i) for i in y]

    # print(y)
    # plt.figure(figsize=(10 ,10))
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # 设置轴的位置
    # ax.spines['left'].set_position('center')
    # 设置轴的颜色
    # ax.spines['right'].set_color('none')
    # 设置轴的位置
    # ax.spines['bottom'].set_position('center')
    # 设置轴的颜色
    # ax.spines['top'].set_color('none')
    # print(t)
    # plt.axis['xzero'].set_visible(True)

    ax.tick_params(axis='x', colors='r')
    ax.tick_params(axis='y', colors='r')

    ax.spines['left'].set_color('red')
    ax.spines['bottom'].set_color('red')

    ax.set_xlabel('iter', fontsize=30)
    ax.set_ylabel('loss', fontsize=30)
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('red')

    ax.plot(y, label='hwdb 中文识别 ctc loss')
    ax.legend(loc=0, prop={'size': 30})
    plt.show()
    fig.savefig('checkpoint\\' + savename)


def testMyImage(path=None,label=None):
    encoder.eval()
    # print('-' * 80)
    if path == None:
        path = input('输入图片路径: ')
    image = 0
    # print('111')
    try:
        image = td.read_img(path)
        img = cv2.imread(path, 0)
        img_resize, crop_cc = td.resize_image(
            img, desired_size=(IMG_HEIGHT, IMG_WIDTH))
        # display(Image.fromarray(img_resize).resize((720, 60)))

        # x = imgplt.imread(path)
        # image = get_transform('test')(image)
        # plt.imshow(x)
        # print('1111')
    except:
        print('输入图片路径有误')
        return

    try:
        encoder.load_state_dict(torch.load("05-encoder_ctc.pt"))
        # decoder.load_state_dict(torch.load(SAVE_MODLE_NAME2))
        # print("读取模型成功，开始识别")
    except Exception as e:
        print("读取模型失败")
        # print(e)
        return
    # print(image.shape)
    image = image.unsqueeze(0).to(device)

    # label = word_lang.batchlabels2vec(y)
    # x, label = x.to(device), label.to(device)
    # seq * batch * dic_len  => 71 * 1 * 5600
    encoder_output = encoder(image).to(device)
    # decoder_input = torch.tensor([word_lang.word2index(SOS)], device=device)
    # hidden = decoder.initHidden(1)

    _, preds = encoder_output.max(2)  # preds 71 * 1
    preds = preds.transpose(1, 0).contiguous().view(-1)
    # print(preds)


    # for di in range
    text = [word_lang.index2word(i) for i in preds.cpu().data.numpy()]
    # print(y[0].replace('|', ' '))
    text = [item for item in text if item != '<SOS>' and item != '<EOS>' and item != '<BLK>']
    # print(''.join(text).replace('|', ' ') + '\n\n\n')


    pred = ''.join(text)
    if label != None:
        temp = 1- Levenshtein.distance(pred,label)/len(label)
        if(temp>0.5):
            print(temp)
    # print(pred)
    # print(label)
    # print('-'* 80 )
    # attentions = []
    # for di in range(1, MAX_LENGTH):
    #     decoder_output, hidden, attention = decoder(
    #         decoder_input, hidden, encoder_output)
    #     # decoder_input = label[:, di].to(device) if teacher == False else decoder_output.data.topk(1)[1].squeeze()
    #     # attentions.append(attention)
    #     wi = decoder_output.data.topk(1)[1].squeeze(-1)
    #     pred.append(word_lang.index2word(wi.item()))
    #     decoder_input = wi
    #     attentions.append(attention.squeeze().cpu().data.numpy())
    #     if wi.item() == word_lang.word2index(EOS) or wi.item() == word_lang.word2index(BLK):
    #         break
    # attentions = np.array(attentions)
    # print(np.array(attentions).shape)
    # showAttention(attentions)
    # print(attentions.shape)
    # print(y[0])
    # print(''.join(pred))
    # pred=pred[:-1]
    # if label != None:
    #     temp = 1- Levenshtein.distance(''.join(pred),label)/len(label)
    #     if(temp>0.5):
    #         print(temp)
        # print(''.join(pred))
        # print(label)
        # print('-' * 80)
        # print()
    # return ''.join(pred)
    # print('-' * 80)


if __name__ == '__main__':
    train()
