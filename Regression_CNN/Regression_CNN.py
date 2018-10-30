#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:18:22 2018

@author: ubuntu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
import numpy as np

class RegNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,10)
        self.fc2 = nn.Linear(10,1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def train():
    
    # data load
    x = torch.linspace(-1,1,100)
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    
    x = x.reshape(-1,1)   # size从一维的[100]变为二维[100,1]，因为model只能处理二维列数据
    y = y.reshape(-1,1)   # 尽量用reshape替代view,因为view需要输入一个整片tensor，如不整片就需要contiguous处理
                          # 而4.0版之后的reshape没有这个额外要求
    # model load
    model = RegNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.4)
    
    # training define
    num_epoch = 200
    
    # start training
    
    '''本例子输入1列特征，数据量比较少。
       如果lr = 0.01, 学习的速度比较慢，需要10000个epoch才能得到比较好的拟合度
       如果lr = 0.2, 学习的速度非常快，只需要200个epoch就能得到较好的拟合度
       如果lr = 0.4, 学习的速率更快，但很快就发现loss过大，这可能是因为学习速度太大，导致梯度
    '''
    loss_list = []
    for i in range(num_epoch):
        optimizer.zero_grad()
        
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.detach().numpy())
        
    plt.figure()
    plt.plot(np.arange(0,num_epoch), loss_list)
    plt.title('Loss')
    plt.show()
    
    # predict
    plt.figure()
    plt.title('Original vs Predict')
    plt.scatter(x,y)
    plt.plot(x.detach().numpy(), model(x).detach().numpy())
    plt.show()
    print(loss)
    

if __name__ == '__main__':
    train()




