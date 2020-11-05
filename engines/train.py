# -*- coding: utf-8 -*-
# @Time : 2020/10/28 23:38 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py 
# @Software: PyCharm
import pandas as pd
from torch.utils.data import DataLoader
from engines.data import DataPrecessForSentence
from engines.models.sentence_bert import BertwwmModel
from engines.models.esim import EsimModel
from engines.models.bert_esim import BertwwmEsimModel
from engines.utils.metrics import correct_predictions, cal_metrics
from transformers.optimization import AdamW
from tqdm import tqdm
import torch
import time


def evaluate(logger, device, model, criterion, dev_data_loader):
    """
    验证集评估函数，分别计算f1、precision、recall
    """
    model.eval()
    start_time = time.time()
    loss_sum = 0.0
    correct_preds = 0
    all_predicts = []
    all_labels = []
    with torch.no_grad():
        for step, (s1_ids, s1_masks, s2_ids, s2_masks, labels) in enumerate(tqdm(dev_data_loader)):
            s1_ids, s1_masks, s2_ids, s2_masks, labels = s1_ids.to(device), s1_masks.to(device), s2_ids.to(device), \
                                                         s2_masks.to(device), labels.to(device)
            logits, probabilities = model(s1_ids, s1_masks, s2_ids, s2_masks)
            loss = criterion(logits, labels)
            loss_sum += loss.item()
            correct_preds += correct_predictions(probabilities, labels)
            predicts = torch.argmax(probabilities, dim=1)
            all_predicts.extend(predicts.cpu())
            all_labels.extend(labels.cpu())
    val_time = time.time() - start_time
    val_loss = loss_sum / len(dev_data_loader)
    val_accuracy = correct_preds / len(dev_data_loader.dataset)
    val_measures = cal_metrics(all_predicts, all_labels)
    val_measures['accuracy'] = val_accuracy
    # 打印验证集上的指标
    res_str = ''
    for k, v in val_measures.items():
        res_str += (k + ': %.3f ' % v)
    logger.info('loss: %.5f, %s' % (val_loss, res_str))
    logger.info('time consumption of evaluating:%.2f(min)' % val_time)
    return val_measures, all_predicts


def train(device, logger):
    # 定义各个参数
    batch_size = 128
    epoch = 15
    learning_rate = 0.0004
    patience = 3
    print_per_batch = 100

    train_file = 'datasets/train.csv'
    val_file = 'datasets/dev.csv'
    train_data = pd.read_csv(train_file, encoding='utf-8')
    val_data = pd.read_csv(val_file, encoding='utf-8')

    # 交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()

    best_f1 = 0.0

    train_data_manger = DataPrecessForSentence(train_data, logger)
    logger.info('train_data_length:{}\n'.format(len(train_data_manger)))
    train_loader = DataLoader(train_data_manger, shuffle=True, batch_size=batch_size)

    val_data_manger = DataPrecessForSentence(val_data, logger)
    logger.info('val_data_length:{}\n'.format(len(val_data_manger)))
    val_loader = DataLoader(val_data_manger, shuffle=False, batch_size=batch_size)

    model = EsimModel(device).to(device)
    params = list(model.parameters())
    optimizer = AdamW(params, lr=learning_rate)
    # 定义梯度策略
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=0)
    for i in range(epoch):
        train_start = time.time()
        logger.info('epoch:{}/{}'.format(i + 1, epoch))
        loss, loss_sum = 0.0, 0.0
        correct_preds = 0
        model.train()
        for step, (s1_ids, s1_masks, s2_ids, s2_masks, labels) in enumerate(tqdm(train_loader)):
            s1_ids, s1_masks, s2_ids, s2_masks, labels = s1_ids.to(device), s1_masks.to(device), s2_ids.to(device), \
                                                     s2_masks.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, probabilities = model(s1_ids, s1_masks, s2_ids, s2_masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct_preds += correct_predictions(probabilities, labels)
            # 打印训练过程中的指标
            if step % print_per_batch == 0 and step != 0:
                predicts = torch.argmax(probabilities, dim=1)
                measures = cal_metrics(predicts.cpu(), labels.cpu())
                res_str = ''
                for k, v in measures.items():
                    res_str += (k + ': %.3f ' % v)
                logger.info('training step: %5d, loss: %.5f, %s' % (step, loss, res_str))
        train_time = (time.time() - train_start) / 60
        train_accuracy = correct_preds / len(train_loader.dataset)
        scheduler.step(train_accuracy)
        logger.info('time consumption of training:%.2f(min)' % train_time)
        logger.info('start evaluate model...')
        val_measures, val_label_results = evaluate(logger, device, model, criterion, val_loader)

        patience_counter = 0
        if val_measures['f1'] >= best_f1 and val_measures['f1'] > 0.70:
            best_f1 = val_measures['f1']
            logger.info('find the new best model with f1: %.3f' % best_f1)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logger('Early stopping: patience limit reached, stopping...')
            break
