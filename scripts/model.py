#!/usr/bin/env python
# encoding: utf-8
"""
@author: HeWenYong
@contact: 1060667497@qq.com
@software: Pycharm
@file: model.py
@time: 2020/10/27 17:01
"""

import torch
from torch import nn
import numpy as np


class MyLinearRegression(nn.Module):
    """
        实现了adagrad算法的线性回归模型
    """

    def __init__(self, in_features, out_features, bias=False):
        super(MyLinearRegression, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self._init_parameters()  # 初始化参数
        self.count = 0  # sumgrad只考虑前10次的梯度和作为adagrada更新参数

    def _init_parameters(self):
        """ 初始化模型参数 """
        self.weight = np.random.rand(self.in_features, self.out_features)  # 随机初始化
        if self.bias:
            self.bias = np.zeros([self.out_features, 1])  # bias
        else:
            self.bias = None
        self.sum_grad = np.zeros((self.in_features, 1))

    def forward(self, train_x):
        # 前向传播函数，把输入变为输出结果
        return np.dot(train_x, self.weight)

    def loss(self, output, y):
        # 损失函数
        return np.sum((output - y)**2) / output.shape[0]

    def backward(self, train_x, output, y):
        """反向传播计算各参数的梯度"""
        # 制造一个长度为特征个数的array，保存每一个weight在更新过程中的平方和
        # 计算梯度(这里先不考虑bias的情况)
        grad_w = ((np.dot(train_x.T, (output - y))) / train_x.shape[0])
        if self.count <= 10:
            # sumgrad只考虑前10次的梯度和作为adagrada更新参数，可以加快参数的更新
            self.sum_grad += grad_w ** 2  # adagrad算法核心
            self.count += 1  #
        # self.sum_grad += grad_w ** 2  # adagrad算法核心
        return grad_w

    def updateParameters(self, lr, grads):
        """ 使用Adagrad更新参数 """
        self.weight = self.weight - ((lr*grads) / np.sqrt(self.sum_grad))





