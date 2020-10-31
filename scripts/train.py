#!/usr/bin/env python
# encoding: utf-8
"""
@author: HeWenYong
@contact: 1060667497@qq.com
@software: Pycharm
@file: train.py
@time: 2020/10/27 16:34
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scripts.mydataset import MyDataset
from scripts.model import MyLinearRegression
import torch
# 设置numpy数组显示数据完全输出
np.set_printoptions(threshold=None)
# 设置dataframe列数显示无限制
pd.set_option("display.max_columns", None)

filepath_x = r"../data/train_x.csv"  # 数据路径
filepath_y = r"../data/train_y.csv"
dataset = MyDataset(filepath_x, filepath_y)
train_loader = DataLoader(dataset=dataset,
                          shuffle=True,
                          batch_size=16,
                          num_workers=0
)

# 定义一个model对象
model = MyLinearRegression(dataset.x_data.shape[1], dataset.y_data.shape[1], bias=True)


def train(total_epoch, lr=0.01):

    """ 训练函数 """
    for epoch in range(total_epoch):
        # iteration = 0
        for data in train_loader:
            # 1.得到训练数据
            train_x, train_y = data

            # 2.前向传播
            y_predict = model(train_x)

            # 3.计算损失
            loss = model.loss(y_predict, train_y.view(train_y.shape[0], -1).numpy())

            # 4.反向传播
            grads = model.backward(train_x, y_predict, train_y.numpy())  # 反向
            if epoch % 10 == 0:
                # print(type(sum_grad[0]))
                print("Epoch: {}/{}  Loss: {}".format(epoch + 1,
                                                      total_epoch,
                                                      loss))
            # iteration += 1
            # 5.参数更新
            model.updateParameters(lr, grads)


if __name__ == "__main__":
    total_epoch = 1000
    train(total_epoch, lr=0.0001)

