# -*- coding: utf-8 -*-
# @Time : 2020/10/28 21:13 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py 
# @Software: PyCharm
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm
import torch


class DataPrecessForSentence(Dataset):
    """
    文本处理
    """

    def __init__(self, df_data, logger):
        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.max_sequence_length = 103
        self.q_ids, self.q_masks, self.r_ids, self.r_masks, self.labels = self.prepare_data(df_data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.q_ids[idx], self.q_masks[idx], self.r_ids[idx], self.r_masks[idx], self.labels[idx]

    def encode(self, text):
        padding_id = self.tokenizer.pad_token_id
        token_inputs = self.tokenizer(text, max_length=self.max_sequence_length, truncation=True)
        input_ids = token_inputs['input_ids']
        input_masks = token_inputs['attention_mask']
        padding_length = self.max_sequence_length - len(input_ids)
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        return input_ids, input_masks

    def prepare_data(self, df_data):
        input_ids_q, input_masks_q = [], []
        input_ids_r, input_masks_r = [], []
        for _, row in tqdm(df_data.iterrows()):
            query, reply = row.query, row.reply
            query_input_ids, query_input_masks = self.encode(query)
            reply_input_ids, reply_input_masks = self.encode(reply)

            input_ids_q.append(query_input_ids)
            input_masks_q.append(query_input_masks)

            input_ids_r.append(reply_input_ids)
            input_masks_r.append(reply_input_masks)

        labels = df_data['label'].values
        ids_q = torch.LongTensor(input_ids_q)
        masks_q = torch.LongTensor(input_masks_q)
        ids_r = torch.LongTensor(input_ids_r)
        masks_r = torch.LongTensor(input_masks_r)
        labels = torch.LongTensor(labels)
        return ids_q, masks_q, ids_r, masks_r, labels
