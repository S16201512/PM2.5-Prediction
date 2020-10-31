#!/usr/bin/env python
# encoding: utf-8
"""
@author: HeWenYong
@contact: 1060667497@qq.com
@software: Pycharm
@file: mydataset.py
@time: 2020/10/27 15:38
"""
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class MyDataset(Dataset):
    """
        本文件是定义一个继承Dataset的类。
            - 功能：完成数据集的进一步处理和封装。
    """
    def __init__(self, filepath_x, filepath_y):
        self.x_data = self.readcsvFile(filepath_x).values  # 读取数据集
        self.y_data = self.readcsvFile(filepath_y).values
        self._len = self.x_data.shape[0]  # 计算数据样本数
        self.dataProcess()  # 数据归一化

    def __getitem__(self, index):
        """ 得到单个样本 """
        return self.x_data[index, :], self.y_data[index]

    def __len__(self):
        """ 得到数据集的数量 """
        return self._len

    def readcsvFile(self, filepath):
        """ 读取指定路径文件 """
        with open(filepath, encoding='big5') as f:
            return pd.read_csv(f, index_col=0)

    def dataProcess(self):
        """ 数据预处理操作
            根据需求进行处理
        """
        # 对输入特征归一化
        for row in range(self.x_data.shape[0]):
            # 方法一：（方差归一化）
            # 遍历一个batch_size的每一行，并进行归一化
            # row_mean = np.mean(self.x_data, axis=0)  # 每一列中的最小值(即每个维度在本batch_size中的最小值)
            # row_std = np.std(self.x_data, axis=0)  # 每一列中的最大值(即每个维度在本batch_size中的最大值)
            # self.x_data[row, :] = (self.x_data[0, :] - row_mean) / row_std

            # 方法二：（最值归一化）
            row_max = np.max(self.x_data, axis=0)  # 每一列中的最小值(即每个维度在本batch_size中的最小值)
            row_min = np.std(self.x_data, axis=0)  # 每一列中的最大值(即每个维度在本batch_size中的最大值)
            self.x_data[row, :] = (self.x_data[0, :] - row_min) / (row_max - row_min)

        # 对label归一化
        for row in range(self.y_data.shape[0]):
            row_max = np.max(self.y_data, axis=0)  # 每一列中的最小值(即每个维度在本batch_size中的最小值)
            row_min = np.std(self.y_data, axis=0)  # 每一列中的最大值(即每个维度在本batch_size中的最大值)
            self.y_data[row, :] = (self.y_data[0, :] - row_min) / (row_max - row_min)