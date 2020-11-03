# -*- coding: utf-8 -*-
# @Time : 2020/10/28 21:13 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : datasets.py
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
        self.s1_ids, self.s1_masks, self.s2_ids, self.s2_masks, self.labels = self.prepare_data(df_data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.s1_ids[idx], self.s1_masks[idx], self.s2_ids[idx], self.s2_masks[idx], self.labels[idx]

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
        input_ids_1, input_masks_1 = [], []
        input_ids_2, input_masks_2 = [], []
        for _, row in tqdm(df_data.iterrows()):
            sent1, sent2 = row.sentence1, row.sentence2
            sent1_ids, sent1_masks = self.encode(sent1)
            sent2_ids, sent2_masks = self.encode(sent2)

            input_ids_1.append(sent1_ids)
            input_masks_1.append(sent1_masks)

            input_ids_2.append(sent2_ids)
            input_masks_2.append(sent2_masks)

        labels = df_data['label'].values
        ids_1 = torch.LongTensor(input_ids_1)
        masks_1 = torch.LongTensor(input_masks_1)
        ids_2 = torch.LongTensor(input_ids_2)
        masks_2 = torch.LongTensor(input_masks_2)
        labels = torch.LongTensor(labels)
        return ids_1, masks_1, ids_2, masks_2, labels
