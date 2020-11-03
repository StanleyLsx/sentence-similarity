# -*- coding: utf-8 -*-
# @Time : 2020/10/29 23:35 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : bert_esim_model.py
# @Software: PyCharm
from abc import ABC
from torch import nn
from transformers import BertModel
import torch.nn.functional as F
import torch


class BertwwmModel(nn.Module, ABC):
    def __init__(self, device, num_classes=2):
        super().__init__()
        self.device = device
        self.dropout = 0.5
        self.embedding_dim = 768
        self.num_classes = num_classes
        self.linear = nn.Linear(3 * self.embedding_dim, num_classes)
        self.dropout = nn.Dropout(self.dropout)
        self.bertwwm_model = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').to(device)
        for param in self.bertwwm_model.parameters():
            param.requires_grad = False

    def forward(self, q_ids, q_masks, r_ids, r_masks):
        with torch.no_grad():
            q_hidden = self.bertwwm_model(q_ids, attention_mask=q_masks)[0].to(self.device)
            r_hidden = self.bertwwm_model(r_ids, attention_mask=r_masks)[0].to(self.device)
        q_average = F.avg_pool1d(q_hidden.transpose(1, 2), q_hidden.size(1)).squeeze(-1)
        r_average = F.avg_pool1d(r_hidden.transpose(1, 2), r_hidden.size(1)).squeeze(-1)
        combined = torch.cat([q_average, r_average, q_average-r_average], -1)
        dropout_results = self.dropout(combined)
        logits = self.linear(dropout_results)
        probabilities = F.softmax(logits, dim=-1)
        return logits, probabilities









