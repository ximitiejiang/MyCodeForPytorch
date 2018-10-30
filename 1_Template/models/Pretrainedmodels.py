#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 12:49:27 2018

@author: ubuntu
"""


import torch.nn as nn
from torchvision import models

class PretrainedModels():
    
    def __init__(self, model_name, num_classes=2):
        self.model_name = model_name
        self.num_classes = num_classes
        
        # vgg16_pretrained
        if self.model_name == 'vgg16':
            
            model = models.vgg16(pretrained=True)
            
            for param in model.parameters(): # 取出每一个参数tensor
                param.requires_grad = False  # 原始模型的梯度锁定
                
            in_fc = model.fc.in_features
            model.fc = nn.Linear(in_fc, num_classes) # 替换最后一层fc
            self.parameters_to_update = model.fc.parameters() 
        
        # resnet18_pretrained
        elif self.model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            
            for param in model.parameters(): # 取出每一个参数tensor
                param.requires_grad = False  # 原始模型的梯度锁定
                
            in_fc = model.fc.in_features
            model.fc = nn.Linear(in_fc, num_classes) # 替换最后一层fc
            self.parameters_to_update = model.fc.parameters() 
                
            
    def parameters(self):        
        return self.parameters_to_update       


if __name__=='__main__':
    
    model = PretrainedModels('vgg16', num_classes=2)
    print(model)
#    for p in model.parameters():
#        print(p.data.)

