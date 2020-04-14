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
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

DATAPATH = "C:\\Users\\weihuan\\Desktop\\IAM"

EOS = '<EOS>'
SOS = '<SOS>'
BLK = '<BLK>'
UNK = '<UNK>'

MAX_LENGTH = 20
BATCH = 32
TEACH_FORCING_PROB = 0.5
N_EPOCH = 10
LEARNING_RATE = 0.0001
IMG_WIDTH = 512
IMG_HEIGHT = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class Lang():
    def __init__ (self):
        self.words = np.loadtxt('./mydata/en1.txt', dtype = np.str)
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
        return self.d2w[i]

    def label2vec (self, label, lens):
        # if label in self.lab2v:
        #     return torch.tensor(self.lab2v[label])
        word_list = label.split('|')
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

        lens = max(len(s.split('|')) for s in blabels)
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
        with open("mydata\\en0.txt") as f:
            for i in range(23):
                f.readline()
            while True:
                line = f.readline()
                if line:
                    line = line.split(' ')
                    self.path.append(line[0])
                    self.label.append(line[8].strip('\n').lower())
                else:
                    break;
            # self.label = torch.tensor(self.label)

    def __getitem__ (self, item):
        path = DATAPATH + "\\lines\\lines\\lines_all\\" + self.path[item] + ".png"
        # img = Image.open(path).convert('RGB').resize((280, 32), Image.BILINEAR)
        img = self.read_img(path)
        # img.show()
        label = self.label[item]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        # label = word_lang.label2vec(label)

        return img, label

    def __len__ (self):
        return len(self.path)

    def read_img (self, img_path, desired_size = (IMG_HEIGHT,IMG_WIDTH)):
        img = cv2.imread(img_path, 0)
        img_resize, crop_cc = self.resize_image(img, desired_size)
        img_resize = Image.fromarray(img_resize)  # 得到 PIL 图片
        # img_tensor = self.transform(img_resize)
        # return img_tensor
        return img_resize

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

    def getFirst (self):
        path = DATAPATH + "\\lines\\lines\\lines_all\\" + self.path[0] + ".png"
        # img = Image.open(path).convert('RGB').resize((280, 32), Image.BILINEAR)
        img = self.read_img(path)

        return img


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
        # self.cnn = nn.Sequential(
        # nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 64x64x512
        # nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 128x32x256
        # nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 256x16x128
        # nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
        # nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 1), padding = (0, 1)),  # 256x8x129
        # nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 512x4x130
        # nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 512x2x131
        # nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True))  # 512x1x130
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
            BidirectionalLSTM(nh, nh, nh))

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


class Decoder(nn.Module):
    '''
        decoder from image features
    '''

    def __init__ (self, nh = 256, nclass = word_lang.size(), dropout_p = 0.1, max_length = IMG_WIDTH / 4 + 1):
        super(Decoder, self).__init__()
        self.hidden_size = nh
        self.decoder = Attentiondecoder(nh, nclass, dropout_p, max_length)

    def forward (self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

    def initHidden (self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result


class Attentiondecoder(nn.Module):
    """
        采用attention注意力机制，进行解码
    """

    def __init__ (self, hidden_size, output_size, dropout_p = 0.1, max_length = IMG_WIDTH / 4 + 1):
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

    def forward (self, input, hidden, encoder_outputs):
        # calculate the attention weight and weight * encoder_output feature
        embedded = self.embedding(input)  # 前一次的输出进行词嵌入
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden[0]), 1)),
            dim = 1)  # 上一次的输出和隐藏状态求出权重, 主要使用一个linear layer从512维到71维，所以只能处理固定宽度的序列
        attn_applied = torch.matmul(attn_weights.unsqueeze(1),
                                    encoder_outputs.permute((1, 0, 2)))  # 矩阵乘法，bmm（8×1×56，8×56×256）=8×1×256

        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)  # 上一次的输出和attention feature做一个融合，再加一个linear layer
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)  # just as sequence to sequence decoder

        output = F.log_softmax(self.out(output[0]), dim = 1)  # use log_softmax for nllloss
        return output, hidden, attn_weights

    def initHidden (self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))

        return result


class DecoderRNN(nn.Module):
    """
        采用RNN进行解码
    """
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))

        return result


class AttentiondecoderV2(nn.Module):
    """
        采用seq to seq模型，修改注意力权重的计算方式
    """
    """
        hidden_size 是隐藏向量的大小，是一个超参数
        output_size 是字典的大小，用于进行词嵌入      
    """
    def __init__ (self, hidden_size, output_size, dropout_p = 0.1):
        super(AttentiondecoderV2, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        # embedding, 输入词典单词个数，以及维度
        # 输出 Output: (*, embedding_dim), where * is the input shape
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # test
        self.vat = nn.Linear(hidden_size, 1)
    """
        input 是指上一次 decoder 的输出, 是一个 batch 维的向量，每个数字代表词的在字典的位置
        hidden 是指隐藏层的大小
        encoder_output 是指 encoder 输出的上下文向量，是对图片编码的结果
    """
    def forward (self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)  # 前一次的输出进行词嵌入
        embedded = self.dropout(embedded)

        # test
        batch_size = encoder_outputs.shape[1]
        # alpha = hidden + encoder_outputs  # 特征融合采用+/concat其实都可以
        alpha = hidden + encoder_outputs  # 特征融合采用+/concat其实都可以
        alpha = alpha.view(-1, alpha.shape[-1])
        attn_weights = self.vat(torch.tanh(alpha))  # 将encoder_output:batch*seq*features,将features的维度降为1
        attn_weights = attn_weights.view(-1, 1, batch_size).permute((2, 1, 0))
        attn_weights = F.softmax(attn_weights, dim = 2)

        # attn_weights = F.softmax(
        #     self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)        # 上一次的输出和隐藏状态求出权重

        attn_applied = torch.matmul(attn_weights,
                                    encoder_outputs.permute((1, 0, 2)))  # 矩阵乘法，bmm（8×1×56，8×56×256）=8×1×256

        # 向量合并
        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)  # 上一次的输出和attention feature，做一个线性+GRU
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim = 1)  # 最后输出一个概率
        return output, hidden, attn_weights

    def initHidden (self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))

        return result


class decoderV2(nn.Module):
    '''
        decoder from image features
    '''

    def __init__ (self, nh = 256, nclass = word_lang.size(), dropout_p = 0.1):
        super(decoderV2, self).__init__()
        self.hidden_size = nh
        self.decoder = AttentiondecoderV2(nh, nclass, dropout_p)

    def forward (self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

    def initHidden (self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result

def get_transform (phase = "train"):
    transfrom_PIL_list = [
        transforms.RandomAffine((-2, 2), fillcolor = 255),
        transforms.ColorJitter(brightness = 0.5),
        transforms.ColorJitter(contrast = 0.5),
        transforms.ColorJitter(saturation = 0.5),

    ]
    # transfrom_tensor_list = [
    #     transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value=0),
    # ]
    if phase == "train":
        transform = transforms.Compose([
            transforms.RandomApply(transfrom_PIL_list),
            transforms.ToTensor(),
            # transforms.RandomApply(transfrom_tensor_list),
            transforms.Normalize(
                mean = [0.5],
                std = [0.5]),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.5],
                std = [0.5]),
        ])
    return transform

 # 1. 根据网络层的不同定义不同的初始化方式
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

# print(label2vec('#'))
encoder = Encoder(IMG_HEIGHT, 1, 256).to(device)
encoder.apply(weight_init)
# decoder = decoderV2(256, word_lang.size(), dropout_p = 0.1).to(device)
decoder = decoderV2(256, word_lang.size(), dropout_p = 0.1).to(device)
loss_fn = torch.nn.NLLLoss().to(device)
# loss_fn = torch.nn.nll_loss().to(device)
encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr = LEARNING_RATE)
decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr = LEARNING_RATE)
# encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = LEARNING_RATE)
# decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr = LEARNING_RATE)

# class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
#  verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
# optimer指的是网络的优化器
# mode (str) ，可选择‘min’或者‘max’，min表示当监控量停止下降的时候，学习率将减小，max表示当监控量停止上升的时候，学习率将减小。默认值为‘min’
# factor 学习率每次降低多少，new_lr = old_lr * factor
# patience=10，容忍网路的性能不提升的次数，高于这个次数就降低学习率
# verbose（bool） - 如果为True，则为每次更新向stdout输出一条消息。 默认值：False
# threshold（float） - 测量新最佳值的阈值，仅关注重大变化。 默认值：1e-4
# cooldown： 减少lr后恢复正常操作之前要等待的时期数。 默认值：0。
# min_lr,学习率的下限
decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min',factor=0.8, patience=5, verbose=False)
encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min',factor=0.8, patience=5, verbose=False)
td = TrainData(transform = get_transform('train33'))


def train ():
    dataloader = DataLoader(td, batch_size = BATCH, shuffle = False)
    encoder.load_state_dict(torch.load('./encoder.pt'))
    decoder.load_state_dict(torch.load('./decoder.pt'))

    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = True
        d.requires_grad = True

    encoder.train()
    decoder.train()


    losslist = []
    cnt = 0
    total_loss = 0.0
    # for epoch in range(N_EPOCH):
    for epoch in range(1):
        for iter, (x, y) in enumerate(dataloader):
            # print(y)
            # # print(word_lang.batchlabels2vec(y))
            # input('-----------')
            y = word_lang.batchlabels2vec(y)

            x, y = x.to(device), y.to(device)
            decoder_hidden = decoder.initHidden(y.shape[0]).to(device)
            loss = 0.0
            encoder_output = encoder(x).to(device)  # 卷积提取图片特征，W/4+1 x Batch x hidden

            # encoder_optimizer.zero_grad()
            # decoder_optimizer.zero_grad()

            # print(encoder_output.size())
            # input('--------------')

            decoder_input = y[:, 0].to(device)
            teach_forcing = True if random.random() < TEACH_FORCING_PROB else False
            if teach_forcing:
                for di in range(1,y.shape[1]):  # 每次预测一个字符

                    # print(y)
                    # input()

                    decode_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_output)
                    decode_output, decoder_hidden, decoder_attention = decode_output.to(device), \
                                                                       decoder_hidden.to(device), \
                                                                       decoder_attention.to(device)
                    loss += loss_fn(decode_output, y[:, di].to(device))

                    # print(decode_output.size())
                    decoder_input = y[:, di]


            else:
                for di in range(1,y.shape[1]):  # 每次预测一个字符

                    decode_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_output)

                    decode_output, decoder_hidden, decoder_attention = decode_output.to(device), \
                                                                       decoder_hidden.to(device), \
                                                                       decoder_attention.to(device)
                    # print(decode_output.size())
                    loss += loss_fn(decode_output, y[:, di].to(device))

                    topv, topi = decode_output.data.topk(1)
                    decoder_input = topi.squeeze(1)
                    # print(decoder_input)
            total_loss += loss.item()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            print(loss)
            # print(decode_output)
            # encoder_optimizer.zero_grad()
            # decoder_optimizer.zero_grad()
            # loss.backward()
            # encoder_optimizer.step()
            # decoder_optimizer.step()
            cnt = cnt + 1
            if cnt % 10 == 0:
                losslist.append(total_loss / 10)
                decoder_scheduler.step(total_loss / 10)
                encoder_scheduler.step(total_loss / 10)
                print("epoch:{:>6.2f}% progress:{:>6.2f}% loss: {:.6f}".format((epoch + 1) / N_EPOCH * 100,
                                                                               (iter + 1) / len(dataloader) * 100,
                                                                               total_loss / 10))
                total_loss = 0.0
    torch.save(encoder.state_dict(), 'encoder.pt')
    torch.save(decoder.state_dict(), 'decoder.pt')


# 对模型进行测试
def evl ():

    encoder.load_state_dict(torch.load('./encoder.pt'))
    decoder.load_state_dict(torch.load('./decoder.pt'))

    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False

    encoder.eval()
    decoder.eval()

    test_loader = DataLoader(td, batch_size = 1, shuffle = True)

    for batch_iter ,(x, y) in enumerate(test_loader):
        if batch_iter == 100:
            break

        decoded_words = []
        decoded_label = []
        target = word_lang.batchlabels2vec(y).to(device)
        x, target = x.to(device), target.to(device)  # x: BATCH x chanel x W x H, y: BATCH x max len of this batch
        encoder_output = encoder(x).to(device)  # 卷积提取图片特征，W/4+1 x Batch x hidden
        # print(encoder_output.shape)
        # print(target.shape)
        # input()
        decoder_input = target[:, 0].to(device)  # BATCH
        decoder_hidden = decoder.initHidden(1).to(device)  # 1 X BATCH x hidden
        # print(decoder_input.shape,decoder_hidden.shape)
        # input()

        # for di in range(1, MAX_LENGTH):
        for di in range(1,len(y[0].split('|'))):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output)
            decoder_output, decoder_hidden, decoder_attention = decoder_output.to(device), \
                                                                decoder_hidden.to(device), \
                                                                decoder_attention.to(device)

            # decoder_attentions[di - 1] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze(1)
            decoder_input = ni
            idx = ni.item()
            # if idx == word_lang.word2index(EOS) and False:
            #     decoded_words.append(EOS)
            #     decoded_label.append(idx)
            #     break
            # else:
            decoded_words.append(word_lang.index2word(idx))
            decoded_label.append(idx)
        print(' '.join(y[0].split('|')))
        print(' '.join(decoded_words), '\n\n')


if __name__ == '__main__':
    for i in range(N_EPOCH):
        train()
        evl()



