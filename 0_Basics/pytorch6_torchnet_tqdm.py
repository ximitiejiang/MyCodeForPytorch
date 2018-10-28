#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:30:34 2018

@author: ubuntu

"""

'''--------------------------------------------------------
Q. 如何使用torchnet的组件?

需要预安装torchnet:  pip3 install torchnet
    - torchnet帮助文件： https://tnt.readthedocs.io/en/latest/
    - torchnet.meter: 用来记录模型性能数据
    - torchnet.logger：把记录数据登录发布到visdom
用了torchnet后，似乎不用显式使用visdom去画图了！！！
-----------------------------------------------------------
'''
# 导入库文件
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
# 创建平均值量表
loss_meter = tnt.meter.AverageValueMeter()
# 创建混淆矩阵量表
confusion_meter = tnt.meter.ConfusionMeter(10)
# 创建plotlogger线条记录器
loss_logger = VisdomPlotLogger('line', win='loss', opts={'title':'Loss'}, port=8097, server='localhost')
# 创建通用记录器(多用于记录混淆矩阵量表信息)
acc_logger = VisdomLogger('heatmap', win='acc', opts={'title': 'Confusion matrix','columnnames': list(range(10)),'rownames': list(range(10))})

# 添加数据进量表
loss_meter.add(loss.item()) 
confusion_meter.add(outputs.detach(), labels.detach())  # outputs = N个样本*K个类，labels = N个样本或NxK的独热编码
                                                        # 生成K*K的confusion表，横坐标是真实值(ground truth), 纵坐标是预测值

# 基于confusion meter计算精度： 
# 注意confusion meter.value()是代表[[00,01],[10,11]]的形式，而显示的matrix原点在左下角，显示为[[10,11],[00,01]
# 所以计算精度时是用矩阵00,11的正确值，除以总和。
accuracy = 100. * (confusion_meter.value()[0][0] + confusion_meter.value()[1][1]) / (confusion_meter.value().sum())

# 记录器记录数据(跟visdom相比，无需设置append update, 因为默认就是更新)
loss_logger.log(epoch,loss_meter.value()[0])
confusion_matrix.add(outputs.detach(), labels.detach())


'''--------------------------------------------------------
Q. 如何在深度神经网络中应用torchnet?
-----------------------------------------------------------
'''
# ---------- 0. 导入库文件：主要分1/2/3/4步实施 ----------
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger# 其中VisdomPlotLogger用于plot各种线条，VisdomLogger用于绘制confusion matrix

# ---------- 1. 创建2种量表meter，和记录器logger ----------
loss_meter = tnt.meter.AverageValueMeter()
confusion_matrix = tnt.meter.ConfusionMeter(10)

loss_logger = VisdomPlotLogger('line', win='loss', opts={'title':'Loss'}, port=8097, server='localhost')
acc_logger = VisdomLogger('heatmap', win='acc', opts={'title': 'Confusion matrix','columnnames': list(range(10)),'rownames': list(range(10))})
    
for epoch in range(num_epoch):

# ---------- 2. 重置meter和matrix ----------
    loss_meter.reset()
    confusion_matrix.reset()
    
    for inputs, labels in tqdm(trainloader):
        
        optimizer.zero_grad()
        outputs = model(inputs)  
        loss = criterion(outputs, labels) 
        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()
        
# ---------- 3. 记录每个batch的meter(增加loss)和matrix(增加outputs/labels) ----------
        loss_meter.add(loss.item())   
        confusion_matrix.add(outputs.detach(), labels.detach())
      
    
# ---------- 4. 更新logger到visdom ----------
    loss_logger.log(epoch,loss_meter.value()[0])
    acc_logger.log(confusion_matrix.value())




'''--------------------------------------------------------
Q. 如何应用tqdm?

用于对iterable对象进行包装，提供图形化动态进度条
-----------------------------------------------------------
'''
import time
from tqdm import tqdm
text = ""
for char in tqdm(["a", "b", "c", "d"]):  # tqdm()包装list
    text = text + char
    time.sleep(0.5)
    
# 对于有的iterable object，由于无法统计总数，tqdm就无法显示进度条，
# 而只能显示速度。所以尽可能把tqdm安置在有len的对象前面
# 