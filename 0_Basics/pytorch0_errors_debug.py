#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:11:18 2018

@author: ubuntu
"""

'''--------------------------------------------------------
Q. pytorch报错：Expected object of type torch.FloatTensor 
but found type torch.cuda.FloatTensor for argument #2 'weight'

1. 代码报错位置：
> /home/ubuntu/suliang_git/MyCodeForPytorch/0_Basics/test_module.py(116)<module>()
    114         optimizer.zero_grad()
    115         # 计算每个batch的输出/损失/预测
--> 116         outputs = model(inputs)
3   117         loss = criteria(outputs, labels)
    118         _, preds = torch.max(outputs, 1)

2. 问题解析：输入到model去的变量为cuda变量，但系统期望一个cpu变量
但实际情况是，我已设置device = cuda：0，且已设置inputs.to(device).
查了下inputs的device属性inputs.device, 发现返回的是cpu形式，但我确实是把data和label
都设置成cuda形式过了。会过头去查看代码发现：
        inputs.to(device)
        labels.to(device)
这种写法不会改变变量的属性，因为to命令不是inplace方式，需要赋值操作，改为：
        inputs = inputs.to(device)
        labels = labels.to(device)
但注意model是不需要这么写的，model可以直接写成model.to(device)
-----------------------------------------------------------
'''
# 错误写法
inputs.to(device)
labels.to(device)
# 正确写法
inputs = inputs.to(device)
labels = labels.to(device)



'''--------------------------------------------------------
Q. GPU跟CPU速度差多少？如何提高训练速度去部署到多个GPU上运行模型？

用实例对比CPU和GPU的性能差别：基于CIFAR10数据集，训练resnet18迁移模型
img数据集50000张训练图片，每张3x224x224(resize之后), batch_size=4, 结果如下：
- cpu：1个epoch耗时38m42s
- gpu：1个epoch耗时1m3s，GPU比CPU快了30倍，网络越复杂，GPU比CPU优势越大
需要区分把model转为cuda model和把tensor转为cuda tensor的区别
- model.to(device)        # model的转换一次就可以，不是inplace也ok
- data = data.to(device)  # data的转换每个data都需要，且因为不是inplace所以要赋值
-----------------------------------------------------------
'''
# 基于cpu
device = torch.device('cpu')
model.to(device)
# 基于gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #cuda起始编号为0,而不是第0个cuda
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model,device_ids=[0, 1]) # 如果不写ids则默认使用全部GPU，需注意编号与前句device对应
model.to(device)


'''--------------------------------------------------------
Q. 如何实验验证数据分配给不同GPU了？

通过在model内部嵌入代码打印model的input，可以发现：
- 在inputs = input.to(device) 这句话时基于device的cuda个数，就把input分成几份
- 分成几份inputs就会调用几次model，从而打印几次model内部代码
- 但最终的outputs = model(inputs)输出又会拼接model内部的几个outputs生成一个outputs输出
参考：https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html?highlight=dataparallel
-----------------------------------------------------------
'''
# 简单验证
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    # Our model
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
    
    
'''--------------------------------------------------------
Q. 为什么model生成的outputs不能直接使用，且报错RuntimeError: Expected object of 
type torch.LongTensor but found type torch.cuda.LongTensor for argument #2 'other'

- 需要正确认识outputs的组成
- 需要二次处理outputs的内容
- 需要仔细区分torch.max()函数与python的max函数区别：torch.max()返回2个tensor,
  第一个tensor是最大值即概率值，第二个tensor是最大值位置即分类的位置
-----------------------------------------------------------
'''
# 比如10分类问题，model的outputs类似如下
labels = torch.tensor([2,7,9,1])
outputs = torch.randn(4,10)
_, preds = torch.max(outputs, 1)      # 取torch.max()函数第二个返回tensor即分类位置
accuracy = torch.sum(preds == labels.data) # labels不能直接跟preds进行运算因为preds为long tensor

    
'''--------------------------------------------------------
Q. 为什么使用torchnet的confusion meter时报错说输入matrix的size不对？
AssertionError: number of predictions does not match size of confusion matrix

confusion_logger.log(confusion_meter.value())
-----------------------------------------------------------
'''
# 原因在于创建confusion_meter时需要定义的唯一参数是n_class，他需要跟模型最终输出的n_class匹配
confusion_meter = meter.ConfusionMeter(n_class) 


'''--------------------------------------------------------
Q. 为什么一个简单Alexnet使用Adam作为求解器loss爆炸了，？
-----------------------------------------------------------
'''
# 原来的optimizer，batch_size=128
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.95)

# 换成SGD把lr改小，并且batch_size = 8，loss就收缩正常了
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


'''--------------------------------------------------------
Q. 如何在调试时查看神经网络各层输出和各个参数的梯度？
-----------------------------------------------------------
'''
# 查看各层输出：通过n/s两个ipdb命令进入相应层，查看outputs, outputs.size()
outputs.size()
# 查看各层输出： 
for n, p in model.named_parameters(): print(n, p.data.std(), p.grad.data.std())



'''--------------------------------------------------------
Q. 为什么一个已经调通的程序，在重启电脑后重新运行报如下错误：
ConnectionRefusedError: [Errno 111] Connection refused
-----------------------------------------------------------
'''
# 根本原因是程序里边定义了visdom的服务server,这需要在运行前启动visdom server
python -m visdom.server

# 也有用如下方式启动服务的，区别在于用的nohup python避免挂起式运行，并且&代表后台运行
nohup python -m visdom.server &


'''--------------------------------------------------------
Q. 为什么原来在mac上良好运行的代码在ubuntu不能运行报错：[Errno 2] No such file or directory

文件打不开或者文件路径错误
-----------------------------------------------------------
'''
# 原来的root
root = '/Home/ubuntu/MyDatasets/DogsVSCats/train/dog.7014.jpg'

# 通过如下命令查看目录发现：目录写得有问题，mac支持/Home/，而ubuntu支持/home/，大小写不同
import os
path = '~/MyDatasets/DogsVSCats/train/dog.7014.jpg'  # 待获得完整路径的目录名要以～开头
fullpath = os.path.expanduser(path)   # 获得完整路径名
print(fullpath)



'''--------------------------------------------------------
Q. 为什么在命令行运行文件报错：
$ python demo.py example.jpg

Traceback (most recent call last):
  File "demo.py", line 10, in <module>
    import matplotlib.pyplot as plt
ImportError: No module named 'matplotlib'
但相关库文件matplotlib.pyplot我已经安装，并且在ipdb单句调试都是可以的
-----------------------------------------------------------
'''
# 奇葩的问题，解决方案是把运行语句的python改为用python3：
$ python3 demo.py example.jpg



'''--------------------------------------------------------
Q. 为什么运行cuda没问题，但运行cupy的时候会报错：
cuda driver version is insufficient to cuda runtime version?
-----------------------------------------------------------
'''
# 检查了下cuda drvier version(即nvidia driver version)为384.130,
# 而cuda9.2要求driver version >= 396.37 必须等于或者高于他所要求的drive version
# 于是卸载当前显卡驱动，安装了410.48版本的driver version，然后正常了。


'''--------------------------------------------------------
Q. 为什么之前定义好的包和模块无法实现导入，会报错：
ModuleNotFoundError: No module named 'utils'？
-----------------------------------------------------------
'''
# 对于一个待运行的程序，要导入包和模块，需要满足如下条件：
#   1. 需求文件需要放在根目录
#   2. 被导入的包和模块需要内建一个__init__文件
#   3. 需要把spyder右上角的working dictory设置到该根目录(这一步往往漏掉导致无法导入)
#      (该working dictory往往不会跟随鼠标点击文件夹一起变换,所以有时候需要手动切换)

