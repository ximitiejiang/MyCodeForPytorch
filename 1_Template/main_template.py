#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:37:56 2018

@author: ubuntu
"""
from data.dataset import DogCat   # 导入数据class
from config import opt  # 导入默认配置class

def train():
    # 1. 定义模型
    
    
    # 2. 定义数据
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.val_data_root, train=False, test=False)
    
    
    pass

def val():
    pass

def test():
    pass

def help():
    pass

if __name__=='__main__':
    opt = DefaultConfig()
    opt.train_data_root

    train()
    