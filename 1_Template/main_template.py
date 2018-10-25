#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:37:56 2018

@author: ubuntu


2. 函数形参写法的功能和特点?
def test(**kwargs):

"""

import visdom
from data.dataset import DogCat   # 导入数据类
from config import opt  # 导入配置类的对象
#import models        # 导入包，不能直接调用模块，因为模块没导入，还需要在init里边再预导入模块
from models.Alexnet import AlexNet # 导入模型类
import torch
import torch.nn as nn
from torchvision import datasets

from torchnet import meter
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from tqdm import tqdm


def test(**kwargs):
    opt._parse(kwargs)

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



def val(model, dataloader):
    
    model.eval()  # validation模式下，model不需要反向传播梯度计算
    
    confusion_meter = meter.ConfusionMeter(2)
    # 对每个batch的数据进行预测
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_meter.add(score.detach().squeeze(), label.type(t.LongTensor))

    model.train()
    
    cm_value = confusion_meter.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_meter, accuracy


def train(**kwargs):
    # 初始化可视化环境
    # vis = visdom.Visdom(env='template')  # 用torchnet似乎就不用显式调用visdom
    
    # 1. 定义模型
    model = AlexNet(num_classes= 10)
    
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
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = opt.lr,
                                 weight_decay = opt.weight_decay)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if opt.use_gpu:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    #model.to(device)
    model.cuda()
        
    # 4. 统计指标初始化
    '''使用torchnet.meter:
    '''
    loss_meter = meter.AverageValueMeter()     # 创建平均值仪表
    confusion_meter = meter.ConfusionMeter(2)  # 创建混淆仪表
    # 创建线条记录器和通用记录器(记录混淆矩阵)
    loss_logger = VisdomPlotLogger('line', win='loss',  
                                   opts={'title':'Loss'}, 
                                   port=8097, server='localhost')
    confusion_logger = VisdomLogger('heatmap', win='acc', 
                              opts={'title': 'Confusion matrix','columnnames': list(range(10)),'rownames': list(range(10))}, 
                              port=8097, server='localhost')
    
    previous_loss =1e100
    
    # 5. 训练
    
    for epoch in range(opt.max_epoch):
        
        loss_meter.reset()       # 每个epoch重置average loss
        confusion_meter.reset() # 每个epoch重置confusion matrix
        
        for ii, (inputs, labels) in tqdm(enumerate(train_dataloader)):
            
            if opt.use_gpu:
                #inputs = inputs.to(device)
                #labels = labels.to(device)                
                inputs = inputs.cuda()
                labels = labels.cuda()
                
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # 更新平均损失和混淆矩阵
            loss_meter.add(loss.item())  # loss_meter是对每个batch的平均loss累加
            confusion_meter.add(outputs.detach(), labels.detach()) # accurary的另一种表达为混淆矩阵
            
#            if ii%opt.print_freq == opt.print_freq - 1:  # 比如9/14/19个batch, 每5个batch打印，此时余数=4，
#                vis.plot('loss', loss_meter.value()[0])  # 
#                vis.line(X=,Y=,win=,name=,update='append')
        
        loss_logger.log(epoch, loss_meter.value()[0])
        confusion_logger.log(confusion_meter.value())
        
        model.save() # 每个epoch保存一次
        
#        # validate and visualize
#        val_cm,val_accuracy = val(model,val_dataloader)
#
#        vis.plot('val_accuracy',val_accuracy)
#        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
#                    epoch = epoch,loss = loss_meter.value()[0],val_cm = str(val_cm.value()),train_cm=str(confusion_meter.value()),lr=lr))
        
        # update learning rate
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        previous_loss = loss_meter.value()[0]



def help():
    pass

if __name__=='__main__':

    train()
    