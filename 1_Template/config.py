#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:55:43 2018

@author: ubuntu
"""

class DefaultConfig(object):  # 这个object作用是？？？
    
    model = 'vgg'
    
    train_data_root = '/home/ubuntu/MyDatasets/DogsVSCats/train/'
    test_data_root = '/home/ubuntu/MyDatasets/DogsVSCats/test/'
    load_model_path = 'checkpoints/model.pth'
    
    batch_size = 8
    use_gpu = True
    num_workers = 2
    
    max_epoch = 4
    lr = 0.01
    lr_decay = 0.95
    weight_decay = 1e-4
    
    print_freq = 20    # 每n个batch打印一次

opt = DefaultConfig()
    