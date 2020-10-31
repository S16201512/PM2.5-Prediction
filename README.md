# PM2.5预测
## 简介
    通过手动实现深度学习的基本流程完成PM2.5的预测，主要有：
        - Dataset类对数据集的封装
        - Dataloader对提取数据操作进行封装
        - 实现线性回归
        - 实现神经网络的正向传播
        - 实现神经网络的反向传播
        - 实现Adagrad算法进行参数更新

## 数据集
    处理之前的训练集和测试集
        - train.csv
        - test.csv
    处理之后的训练集和训练集标签
    处理流程请见提供的doc文档
        - train_x.csv
            - 维度（5760，162）
                - 5760表示样本数
                - 162表示总特征数
        - train_y.csv
                - （5760，1）
                - 对应5760条样本的PM2.5值，即标签

## 文件夹介绍
    - data：数据集
    - scripts：程序代码
        - model.py：实现线性回归模型
        - mydataset.py：实现数据集的封装和处理
        - train.py：训练开始函数，程序入口
    Readme.md介绍文档
 
## 运行方法
    python ./scripts/train.py
    
## 结果图
![损失结果](https://github.com/S16201512/PM2.5-Prediction/blob/master/result.png)