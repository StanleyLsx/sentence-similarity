# -*- coding: utf-8 -*-
# @Time : 2020/10/29 23:35 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : bert_esim.py
# @Software: PyCharm
from abc import ABC
from torch import nn
from transformers import BertModel
import torch.nn.functional as F
import torch


class BertwwmEsimModel(nn.Module, ABC):
    def __init__(self, device, num_classes=2):
        super().__init__()
        self.device = device
        self.dropout = 0.5
        self.embedding_dim = 768
        self.hidden_dim = 200
        self.linear_size = 128
        self.num_classes = num_classes
        self.bilstm1 = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(self.hidden_dim * 8, self.hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(8 * self.hidden_dim, num_classes)
        self.dropout = nn.Dropout(self.dropout)
        self.bertwwm_model = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').to(device)
        for param in self.bertwwm_model.parameters():
            param.requires_grad = False

    @staticmethod
    def soft_align_attention(x1, x2, mask1, mask2):
        # attention: batch_size * seq_len * seq_len
        mask1 = mask1.eq(0)
        mask2 = mask2.eq(0)
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))
        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)

        # x_align: batch_size * seq_len * hidden_size
        return x1_align, x2_align

    @staticmethod
    def submul(x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    @staticmethod
    def apply_multiple(x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, s1_ids, s1_masks, s2_ids, s2_masks):
        with torch.no_grad():
            s1_hidden = self.bertwwm_model(s1_ids, attention_mask=s1_masks)[0].to(self.device)
            s2_hidden = self.bertwwm_model(s2_ids, attention_mask=s2_masks)[0].to(self.device)
        # input encoding
        o1, _ = self.bilstm1(s1_hidden)
        o2, _ = self.bilstm1(s2_hidden)
        # local inference modeling
        q1_align, q2_align = self.soft_align_attention(o1, o2, s1_masks, s2_masks)

        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        q1_compose, _ = self.bilstm2(q1_combined)
        q2_compose, _ = self.bilstm2(q2_combined)

        # Aggregate
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        x = self.dropout(x)
        logits = self.linear(x)
        probabilities = F.softmax(logits, dim=-1)
        return logits, probabilities
