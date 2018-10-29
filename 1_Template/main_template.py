#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:37:56 2018

@author: ubuntu

"""
import numpy as np
import visdom
from data.dataset import DogCat   # 导入数据类
from config import opt  # 导入配置类的对象
#import models        # 导入包，不能直接调用模块，因为模块没导入，还需要在init里边再预导入模块
from models.Alexnet import AlexNet # 导入模型类
from models.Resnet import ResNet34

import torch
import torch.nn as nn
#from torchvision import datasets

from torchnet import meter
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from tqdm import tqdm

import time


def test(**kwargs):  # 测试模块还没调试
    '''test函数一般放在train/val函数后边，沿用已经训练好的模型，思路跟val函数一样
       同时定义一个model加载条件分支，以便使用已经保存的现有model做测试预测
    '''
    # configure model
    model = getattr(models, opt.model)().eval() 
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # data
    train_data = DogCat(opt.test_data_root,test=True)
    test_dataloader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    results = []
    for ii,(data,path) in tqdm(enumerate(test_dataloader)):
        input = data.to(opt.device)
        score = model(input)
        probability = t.nn.functional.softmax(score,dim=1)[:,0].detach().tolist()
        # label = score.max(dim = 1)[1].detach().tolist()
        
        batch_results = [(path_.item(),probability_) for path_,probability_ in zip(path,probability) ]

        results += batch_results
    write_csv(results,opt.result_file)

    return results



def val(model, dataloader, num_classes, vis):
    # 验证模式：model不需要反向传播梯度计算
    model.eval()  
    
    # 初始化量表和记录器
    val_confusion_meter = meter.ConfusionMeter(num_classes)
    val_confusion_logger = VisdomLogger('heatmap', win='v_cf', 
                              opts={'title': 'Validating confusion matrix','columnnames': list(range(num_classes)),'rownames': list(range(num_classes))}, 
                              port=8097, server='localhost')
    # 对每个batch的数据进行预测
    since = time.time()
    for ii, (val_inputs, val_labels) in enumerate(dataloader):
        
        if opt.use_gpu:
            val_inputs = val_inputs.cuda()
            val_labels = val_labels.cuda()
        
        val_outputs = model(val_inputs)
        
        val_confusion_meter.add(val_outputs.detach(), val_labels.detach())

    model.train()
    time_elapsed = time.time() - since

    val_confusion_logger.log(val_confusion_meter.value())
    accuracy = 100. * (val_confusion_meter.value()[0][0] + val_confusion_meter.value()[1][1]) / (val_confusion_meter.value().sum())
    print('Validating complete in {:.0f}m {:.0f}s, validate accuracy: {}'.format(time_elapsed // 60, time_elapsed % 60, accuracy))


def train(**kwargs):
    # 初始化可视化环境
    vis = visdom.Visdom(env='main')    
    # 1. 定义模型
    num_classes = 2
    
#    model = AlexNet(num_classes= num_classes)
    model = ResNet34(num_classes = num_classes)
    
#    from torchvision import models
#    model = models.alexnet(pretrained =True)
    
    
    # 2. 定义数据
    train_data = DogCat(opt.train_data_root, train=True)

    val_data = DogCat(opt.train_data_root, train=False, test=False)

    
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size = opt.batch_size,
                                                   shuffle = True,
                                                   num_workers = opt.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_data, 
                                                 batch_size = opt.batch_size,
                                                 shuffle = False,
                                                 num_workers = opt.num_workers)
    # 3. 定义损失函数,优化器,设备
    criterion = torch.nn.CrossEntropyLoss()
#    optimizer = torch.optim.Adam(model.parameters(),
#                                 lr = opt.lr,
#                                 weight_decay = opt.weight_decay)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if opt.use_gpu:
#        if torch.cuda.device_count() > 1:
#            model = nn.DataParallel(model)
        #model.to(device)
        model.cuda()
        #print('model.device: {}'.format(model.device()))
        
    # 4. 统计指标初始化
    '''使用torchnet.meter:
    '''
    loss_meter = meter.AverageValueMeter()     # 创建平均值量表
    confusion_meter = meter.ConfusionMeter(num_classes)  # 创建混淆量表
    loss_logger = VisdomPlotLogger('line', win='loss',   # 创建记录仪
                                   opts={'title':'Train Loss'}, 
                                   port=8097, server='localhost')
    confusion_logger = VisdomLogger('heatmap', win='conf', 
                              opts={'title': 'Training Confusion matrix','columnnames': list(range(num_classes)),'rownames': list(range(num_classes))}, 
                              port=8097, server='localhost')
    
#    previous_loss =1e100
    
    # 5. 训练
    since = time.time()
    for epoch in tqdm(range(opt.max_epoch)):
        
        loss_meter.reset()       # 每个epoch重置average loss
        confusion_meter.reset() # 每个epoch重置confusion matrix
        print('\n')
        print('epoch position: {} / {} ...'.format(epoch+1, opt.max_epoch))
        for ii, (inputs, labels) in enumerate(train_dataloader):
            
            if opt.use_gpu:
                #inputs = inputs.to(device)
                #labels = labels.to(device)                
                inputs = inputs.cuda()
                labels = labels.cuda()
                
            
            optimizer.zero_grad()
            
            outputs = model(inputs)             # 计算输出在每个类的概率
            loss = criterion(outputs, labels)
            
            loss.backward()                     # 计算
            optimizer.step()                    #  

            # 更新平均损失和混淆矩阵
            loss_meter.add(loss.item())  # loss_meter是对每个batch的平均loss累加
            confusion_meter.add(outputs.detach(), labels.detach()) # accurary的另一种表达为混淆矩阵
            
            
#            if ii%opt.print_freq == opt.print_freq - 1:  # 比如9/14/19个batch, 每5个batch打印，此时余数=4，
#                vis.plot('loss', loss_meter.value()[0])  # 
#                vis.line(X=,Y=,win=,name=,update='append')
        
        loss_logger.log(epoch, loss_meter.value()[0])
        confusion_logger.log(confusion_meter.value())
        
        accuracy = 100. * (confusion_meter.value()[0][0] + confusion_meter.value()[1][1]) / (confusion_meter.value().sum())
        vis.line(X=np.array(epoch).reshape(1), Y=np.array(accuracy).reshape(1), win='cur',opts={'title':'Train accuracy'}, update='append')
        
#        model.save() # 每个epoch保存一次
        
#        # validate and visualize
#        val_cm,val_accuracy = val(model,val_dataloader)
#
#        vis.plot('val_accuracy',val_accuracy)
#        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
#                    epoch = epoch,loss = loss_meter.value()[0],val_cm = str(val_cm.value()),train_cm=str(confusion_meter.value()),lr=lr))
        
        # update learning rate
#        if loss_meter.value()[0] > previous_loss:          
#            lr = lr * opt.lr_decay
#            # 第二种降低学习率的方法:不会有moment等信息的丢失
#            for param_group in optimizer.param_groups:
#                param_group['lr'] = lr
#        print('epoch:{}, lr{}'.format(epoch, lr))
#        previous_loss = loss_meter.value()[0]
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    # 验证集的计算：验证集基于训练集训练完成后model的state_dict，所以是共享了训练结果的    
    val(model, val_dataloader, num_classes, vis)

    # 测试集的计算: 测试集基于训练集训练完成后model的state_dict，所以是共享了训练结果的 
#    test(model)
    
    
def train_pt():
    pass

    

def help():
    pass

if __name__=='__main__':

    train()
    