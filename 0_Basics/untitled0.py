#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 08:49:35 2018

@author: ubuntu
"""

from torch.utils.data import Dataset

class mydata(Dataset):
    '''创建一个基础版数据集，继承自pytorch的Dataset(不继承是不是也没问题)
    
    '''
    def __init__(self,root):
        super().__init__()
        
    
    def __getitem__(self):
        pass
    
    def __len__(self):
        pass