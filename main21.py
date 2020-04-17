# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/4/16  6:54
"""
# 尝试实现论文算法，失败告终
# class Decoder2(nn.Module):
#     def __init__ (self, hiddensize = 256, dic_size = word_lang.size()):
#         super(Decoder2, self).__init__()
#         self.hidden_size = hiddensize
#         self.embed = nn.Embedding(dic_size, hiddensize)
#         self.fc1 = nn.Linear(2 * hiddensize, hiddensize)
#         self.lstm = nn.LSTM(hiddensize * 2, hiddensize)
#         self.fc2 = nn.Linear(2 * hiddensize, dic_size)
#         self.fc_att = nn.Linear(2 * hiddensize, 1)
#         self.dropout = nn.Dropout(0.1)
#         self.fc_lstm = nn.Linear(2 * hiddensize, hiddensize)
#
#     def forward (self, lstm_hc, encoder_output, factlabel, att_apply):
#         '''词嵌入'''
#         factlabel = self.embed(factlabel)
#         embed = self.dropout(factlabel)
#
#         ''' 输入 lstm'''
#         lstm_input = torch.cat((embed.unsqueeze(0), att_apply), dim = -1)
#         lstm_output, hc = self.lstm(lstm_input, lstm_hc)
#
#         ''' 生成alpha权重 '''
#         alpha = torch.cat((encoder_output, lstm_output.expand(len(encoder_output),  # ====> len * batch * 2hiddensize
#                                                               lstm_output.shape[1], lstm_output.shape[2])), -1)
#         attn_weights = torch.tanh(self.fc_att(alpha)).permute(1, 2, 0)  # ====> batch * 1 * seq_len
#         attn_weights = F.softmax(attn_weights, dim = -1)  # ====> batch * 1 * seq_len
#
#         '''权重和原文相乘得到加权结果'''
#         att_apply = torch.bmm(  # =====>  batch * 1 * hidden
#             attn_weights,  # =====>batch * 1 * seq_len
#             encoder_output.permute(1, 0, 2)  # =====> batch * seq_len * hidden
#         )
#
#         out = torch.cat((att_apply.permute(1, 0, 2), lstm_output), -1)  # =====>  1 * batch * 2-hidden
#         out = self.fc2(out.squeeze(0))
#         out = torch.tanh(out)
#         return F.log_softmax(out, -1), lstm_hc, attn_weights, att_apply.permute(1, 0, 2)
#
#     def initHidden (self, batch_size):
#         h = Variable(torch.zeros(1, batch_size, self.hidden_size, device = device))
#         # c = Variable(torch.zeros(1, batch_size, self.hidden_size, device = device))
#         return h