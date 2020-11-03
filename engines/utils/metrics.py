# -*- coding: utf-8 -*-
# @Time : 2020/9/9 6:14 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : metrics.py
# @Software: PyCharm
from sklearn import metrics


def cal_metrics(predicts, targets):
    """
    指标计算
    """
    average = 'binary'
    precision = metrics.precision_score(predicts, targets, average=average)
    recall = metrics.recall_score(predicts, targets, average=average)
    f1 = metrics.f1_score(predicts, targets, average=average)
    return {'precision': precision, 'recall': recall, 'f1': f1}


def correct_predictions(output_probabilities, targets):
    _, out_classes = output_probabilities.max(dim=1)
    correct = out_classes == targets
    correct_nums = correct.sum()
    return correct_nums.item()


def search_f1(y_true, y_pred):
    best_f1 = 0
    best_threshold = 0
    for i in range(30, 60):
        threshold = i / 100
        y_pred_bin = (y_pred > threshold).astype(int)
        score = metrics.f1_score(y_true, y_pred_bin, average='binary')
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return best_f1, best_threshold
