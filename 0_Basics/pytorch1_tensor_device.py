#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:12:02 2018

@author: ubuntu
"""

'''--------------------------------------------------------
Q. 如何生成tensor？
- tensor跟array基本没区别
------------------------------------------------------------
'''
a = torch.FloatTensor([[1, 2, 3], [4, 5, 6]]) # 浮点数tensor

b = torch.IntTensor(2, 4).zero_()  # 整数tensor

c = torch.Tensor([[1,2,3],[4,5,6]])  # 浮点数tensor简化新建

d = torch.ones(2,3)  # 全1tensor
d = torch.zeros(2,3) # 全0tensor
d = torch.eye(3,3)   # 主对角线全1tensor
d = torch.arange(0,10,2)      # 从0-10取值, 间隔2
d = torch.linspace(0,10,5)  # 从0-10取值, 取5份

d = torch.rand(2,3)   # 随机-0-1之间的均匀分布
d = torch.randn(2,3)  # 随机-标准0-1正态分布(均值0，方差1)
d = torch.randperm(6) # 随机0-n的整数排列 

# 如果是单元素tensor，可通过item直接转成python数据
d = torch.Tensor([1])
d1 = d.item()


'''--------------------------------------------------------
Q. 对tensor有哪些属性可用？
- dtype属性
- 基本属性
- 设备属性
-----------------------------------------------------------
'''
import torch
# dtype属性: 有8种dtype可用，包括int32,float32,...
a = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32, requires_grad=True) # 只有dtype=float才能梯度计算
b = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.int32)  
print(a[1][1])    # 切片
# 基本属性
a.requires_grad
a.grad
a.data   # tensor数据，浅复制
a.item   # 
a.size()
a.shape
a.shape[1]
# 设备属性
device = torch.device('cuda',0)   # 返回一个设备对象，但不是指定运行的设备
device = torch.device('cuda:1')   # 另一种写法
print(device)


'''--------------------------------------------------------
Q. 如何对tensor进行切片?
-----------------------------------------------------------
'''
import torch
x = torch.FloatTensor([[1, 0, 3], [4, 5, 6]]) 
print(x[1][2])  #类似于对list的原始多维切片

x[0][1] = 8   # 可以直接切片赋值
print(x)

print(x[:,1])  # 跟numpy一样的高级切片方式
print(x[:,1].size())

z = x > 1   # 条件筛选切片: 返回符合条件=1的0-1矩阵
z = x[x>3]  # 条件筛选切片: 返回符合条件的值
z = x[x!=0]


'''--------------------------------------------------------
Q. 如何对tensor内的元素进行计算以及相关函数？
- 基本上所有tensor都支持两类接口: t1.func(), t.func(a)，其中t1.func()这种后置式的更方便常用
- 函数名以下划线结尾的都是inplace方式，即会修改原始tensor，比如a.zero_(), b.abs_()
-----------------------------------------------------------
'''
# 计算清零
a = torch.FloatTensor([[1,-2,3],[-4,5,-6]])
a.zero_()  # 用0填充tensor, 只有一种带后缀的方式，修改原tensor
print(a)

# 计算size()
print(a.shape)    # 跟numpy一样，用shape不用加括号，最简洁
print(a.shape[1])  # 获得tensor的形状的列数
print(a.size()[1])  # 获得tensor的size形状,跟shape一样，但要带括号没shape简洁

# 计算绝对值/平方根/除法/对数
b = torch.FloatTensor([[1,-2,3],[-4,5,-6]])
z = b.abs()   # 不带后缀，不修改原tensor
z = b.abs_()  # 带后缀，修改原tensor
c = torch.Tensor([4,16])
z = c.sqrt()  # 开方
z = c.div(2)  # 除法
z = c.exp()   # e的指数
z = c.log()   # 对数
z = c.pow(2)  # 幂次
z = c**2      # 平方
z = c + c     # 加法
z = c*c       # 按位相乘
z = c.mul(c)  # 按位相乘

# 计算点积
a = torch.FloatTensor([[1,-2,3],[-4,5,-6]])
b = torch.FloatTensor([[1,2],[2,1],[0,1]])
c = a.mm(b) # 点积
print(c)

# 格式转换
a = torch.FloatTensor([[1,-2,3],[-4,5,-6]])
an = a.numpy()  # 把tensor转化为numpy
at = torch.from_numpy(an)  # 把numpy转化为tensor
b = a.tolist()  # tensor to list

an[0,0] = 10   # tensor与array共享内存，所以任何一个变化会引起另一个变化
print(an)
print(at)

# 矩阵的转秩
b = a.t()  # tensor转秩
print(a)
print(b)

# 在图像处理中，有采用transpose([1,2,0])把tensor的CxHxW转换成图像的HxWxC

# 计算求和/求平均/求最大
a = torch.FloatTensor([[1,-2,3],[-4,5,-6]])
b = a.sum()  # 求和
c = a.mean() # 求平均
d = a.min(dim=0)
d = a.max(dim=1)  # 求最大,y方向（把数据拍成一个y轴）
print(d[0][0], d[1][0])
# 注意max的输出结果很特殊，第一行是最大值列表，第二行是最大值标签列表


# 取整/求商/取余
a = torch.FloatTensor([1.75,3.1415])
b = a.round()  # 四舍五入
b = a.ceil()   # 上取整
b = a.floor()  # 下取整
b = a%2

# 截断
a = torch.FloatTensor([[1,-2,3],[-4,5,-6]])
b = a.clamp(2,4)  # 2,4之间的值，超出的则取2，4

# 调整tensor的形状
a = torch.arange(0,6)
e = a.reshape(2,3) # 跟numpy的reshape一样好用
e = a.reshape(-1,1)  # 生成一列
b = a.view(2,3)   # view相当于python中的reshape()
c = a.view(-1,2)  # view相当于python中的reshape()

d = b.unsqueeze(1)  # 待测试

# tensor转标量: 往往需要指定dim, 相当于numpy中的axis
# dim(axis)等于哪个轴，该轴变为1，也就是沿着该轴挤压(或叫沿着该轴坍缩)
# 比如(2,3,2) dim=1就会变成(2,1,2)
a = torch.FloatTensor([[1,-2,3],[-4,5,-6]])
z = a.mean()  # 均值
z = a.sum(dim=1)   # 求和
z = a.median()  # 中位数
z = a.mode()    # 众数
z = a.var()    # 方差
z = a.std()    # 标准差 = 方差的开方


'''--------------------------------------------------------
Q. 如何在pytorch中调整张量的维度，即实现手动广播法则？
- 虽然pytorch跟numpy一样有广播法则对维度不一样的计算进行广播
但最好先定义好扩维或者降维，避免混乱
-----------------------------------------------------------
'''
a = torch.ones(3,2)
b = torch.zeros(2,3,1)
a.expend(2,3,2)

x = torch.linspace(-1, 1, 5)   # 初始tensor: 一维，5
y1 = torch.unsqueeze(x, 1)     # 扩维：二维，5x1

z1 = x.reshape(-1,1)
z = x.view(-1,1)               # 扩维：二维，5x1，view跟reshape功能一样



'''--------------------------------------------------------
Q. 对cuda有那些属性可用?
-----------------------------------------------------------
'''
# 定义一个设备
device = torch.device('cuda:1')
print(device)
# 设置当前设备
torch.cuda.set_device(1)
torch.cuda.current_device()
# 设备数，设备能力，设备可用
torch.cuda.device_count()
torch.cuda.get_device_capability(0)
torch.cuda.get_device_name(0)
torch.cuda.is_available()

