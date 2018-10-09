#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:03:15 2018

@author: suliang

来自pytorch官方英文教程

简单模型进行保存与载入的方式

"""

# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建模型
model = TheModelClass()

# 初始化优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 打印模型的state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# 打印优化器的state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


import os
path = '~/SaveAndLoad/state_dict.pth'  # 待获得完整路径的目录名要以～开头
fullpath = os.path.expanduser(path)   # 获得完整路径名
# 保存模型的参数
torch.save(model.state_dict(), fullpath) # 需要指定一个新pth文件存放参数 
# 重新载入模型参数
model = TheModelClass()
model.load_state_dict(torch.load(fullpath))
model.eval()


# 保存整个模型
PATH = '/Users/suliang/MyCodeForPytorch/SaveAndLoad/model_save.pth'
model2 = TheModelClass()  # 创建一个模型
torch.save(model, PATH)  # 保存整个模型
model.eval()


# 保存checkpoints
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, PATH)
    
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()



    
    