#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 08:25:53 2018

@author: ubuntu

1. 任意class都有一个默认名字，可通过str(type(self))获得
2. 函数time.strftime()用于组合str和time生成一个字符串
3. model的save就是保存state_dict:  torch.save(model.state_dict(),name)
   model的load就是加载state_dict:  model.load_state_dict(torch.load(path))

"""

import torch
import time

class BasicModule(torch.nn.Module):
    
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name=str(type(self))   # 定义一个新属性model_name ='<class '__main__.BasicModule'>'
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        
    def save(self, name=None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth') #生成保存路径和文件名
        torch.save(self.state_dict(), name)  # 基于name，保存模型的状态参数
        return name


class Flat(torch.nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)
    
if __name__=='__main__':
    net = BasicModule()  # 创建一个net做测试
    print(net.model_name)  # 可以看到默认model_name属性是一个奇怪的名字
    net.save()  # 此时save到目标路径checkpoints文件夹
