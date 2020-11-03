# -*- coding: utf-8 -*-
# @Time : 2020/10/31 19:25 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : test.py 
# @Software: PyCharm
from tqdm import tqdm
import time
import torch


def test(logger, device, model, test_loader):
    """
    运行测试集
    """
    label_results = []
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for step, (q_ids, q_masks, r_ids, r_masks, labels) in enumerate(tqdm(test_loader)):
            q_ids, q_masks, r_ids, r_masks, labels = q_ids.to(device), q_masks.to(device), r_ids.to(device), \
                                                     r_masks.to(device), labels.to(device)
            logits, probabilities = model(q_ids, q_masks, r_ids, r_masks)
            predicts = torch.argmax(probabilities, dim=1)
            label_results.extend(predicts.cpu())
    test_time = time.time() - start_time
    logger.info('time consumption of testing:%.2f(min)' % test_time)
    return label_results
