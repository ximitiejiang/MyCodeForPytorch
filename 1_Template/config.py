#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:55:43 2018

@author: ubuntu
"""

class DefaultConfig(object):  # 这个object作用是？？？
    
    model = 'resnet'
    
    train_data_root = '/home/ubuntu/MyDatasets/DogsVSCats/train/'
    test_data_root = '/home/ubuntu/MyDatasets/DogsVSCats/test/'
    load_model_path = 'checkpoints/model.pth'
    
    batch_size = 128
    use_gpu = True
    num_workers = 2
    
    max_epoch = 10
    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4

opt = DefaultConfig()
    