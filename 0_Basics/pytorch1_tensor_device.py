#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:12:02 2018

@author: ubuntu
"""

import numpy as np
import torch

'''--------------------------------------------------------
Q1. 创建list/array
------------------------------------------------------------
'''
lst = []
lst.append(3)      # 只能加入一个集成元素
lst.extend([3,4,5]) # 加入多个元素
lst.pop()

a = np.array([1,2,3])
a = np.zeros((2,3))
a = np.ones((2,3))
a = np.eye(3)

a = range(5)
a = np.arange(1,10,2)  # 1-10之间间隔2 (最常用，写法跟切片是统一的,a,b,c代表起点终点和间隔)
a = np.linspace(1,2,11) #1-2之间，总计11个数(所以是n-1等分)

a = np.random.randn(9,9) # 随机标准正态分布(均值0,方差1)，常用于深度学习
a.std()
a.mean()

'''--------------------------------------------------------
Q2. 创建dict
------------------------------------------------------------
'''
d = {'a':2,'b':3}
d.get('a',0)  # 获得key的值，如果没有则返回0
d.keys()
d.values()
d.items()

'''--------------------------------------------------------
Q2. 获得list/array的尺寸
------------------------------------------------------------
'''
lst = [[1,2,3],[4,5,6]]
arr = np.array(lst)

len(lst)    # 尺寸
arr.shape   # 形状
arr.size    # 元素个数

'''--------------------------------------------------------
Q3. 切片与筛选
------------------------------------------------------------
'''
a = np.array([[2,3,1,5],[5,3,7,1]])
a[:,1:4:2]  # a:b:c代表从第a个位置到第b个位置(不包含第b个位置)，间隔c

index = np.where(np.array(a)>0) # 筛选大于小于某值的index, 可适用array,对list可以转换使用


'''--------------------------------------------------------
Q4. 排序/取最大最小值
------------------------------------------------------------
'''
# list的排序，sort(), argsort(), 另有一个sorted() 
a = [2,11,5,7,1]
a.sort()    # 在原数上排序inplace方式
a.argsort()

sorted(a)[::-1]   # 通过取数得到小到大排序，和大到小排序

# numpy的排序： 需要sort() argsort()
a = np.array([[2,1,5,3,7,2,7,4],[7,4,0,1,9,4,8,4]])
a.sort(axis=1)
a.argsort()

np.sort(a,axis=1)    # numpy也有对应函数，但似乎没有必要
np.argsort(a,axis=1) # numpy也有对应函数，但似乎没有必要

# tensor的排序
t1 = torch.tensor([2,4,1,5,3,7,0,9])
t1.sort()  # tensor有sort()，但argsort()不能用

# dict的排序
price = {'ACME': 45.23,'AAPL': 612.78,'IBM': 205.55,'HPQ': 37.2,'FB': 10.75}
sorted_price = sorted(price.values())                   # 只对value排序
sorted_price = sorted(zip(price.values(),price.keys())) # 同时对value和key排序

# 最大值可以排序后获得，也可以用函数max()
a = np.array([[2,1,5,3,7,2,7,4],[7,4,0,1,9,4,8,4]])
a.max(axis=1)

'''--------------------------------------------------------
Q4. 统计出现频次？
------------------------------------------------------------
'''
# list统计频次
lst = [1,4,5,1,7,8,1,9,7]
lst.count(1)

# array不能直接用count()方法。
# 办法1,ravel()之后转list然后用count()
arr= np.array([[1,4,2,6,8],[4,1,2,7,1]])
arr.ravel().tolist().count(1)
# 办法2,从别的模块collections导入counter()函数


'''--------------------------------------------------------
Q4. 获得不重复元素？以及基于不重复元素的并集和交集
------------------------------------------------------------
'''
lst1 = [1,4,5,1,7,8,1,9,7]
lst2 = [21,32,90,45,7,8,1,9,7]
set(lst1)  # 不重复元素

set(lst1).union(set(lst2))  # 并集
set(lst1)^set(lst2)         # 差集
                            # 交集 = 并集^差集


'''--------------------------------------------------------
Q4. 类型获取与类型转换
------------------------------------------------------------
'''
lst = [1,2]
arr = np.array(lst)
dic = {'a':1,'b':2}
ten = torch.tensor(lst)
ten1 = torch.tensor([2.])

type(arr)  # 数据类型查看：适用于任何
arr.dtype  # 元素类型查看
isinstance(arr, np.float32)  # 元素类型判断，返回True/False
arr.astype(np.float32)       # 元素类型转换(深度学习常用)
np.asscalar(torch.tensor(5)) # 张量转标量
l1 = arr.tolist()            # numpy转list
t1 = torch.from_numpy(arr)   # numpy转torch
n1 = t1.numpy()              # torch转numpy

scalar1 = ten1.item()        # 单元素tensor转标量，用.item()


'''--------------------------------------------------------
Q4. 如何做浅复制和深复制？
------------------------------------------------------------
'''
import copy
old = [4,1,3,['age',10]]
new = copy.deepcopy(old)  # 要做到完全深度拷贝，不影响源数据，唯一方法是deepcopy()
new[3].append(100)
print(old, new)

# 另一种深复制：直接取到元素，元素作为不可变对象，肯定是创建新内存
new1 = [old[0],old[1],old[2],[old[3][0],old[3][1]]] # 此处如果取old[3]也不行，因为old[3]是list而不是不可变对象
old[3][1]=100
new1  # 可以看到虽然old[3][1]变了，但new1没有受影响



'''--------------------------------------------------------
Q4. 如何产生随机数？
------------------------------------------------------------
'''
a = np.random.randn(2,3)  # 正态分布随机矩阵

b = np.arange(10)
np.random.choice(b)      # 从数据中随机抽取一个数

np.random.seed(5)        # 定义一次随机数产生过程
np.random.randn(1)       # 如果之前有seed, 产生的随机数都相同


'''--------------------------------------------------------
Q4. 如何进行堆叠和展平？
------------------------------------------------------------
'''
a = np.array([[2,1,4,5],[4,2,7,1]])
a.ravel()     # 在原数据上直接展平
a.flatten()   # 展平，但不影响原数据

b0=np.array([[1,2],[3,4]])
b1=np.array([[5,6],[7,8]])
np.concatenate((b0,b1),axis=0)  # 最核心常用的堆叠命令(axis可以控制堆叠方向)
# 另外还有几个堆叠命令 np.hstack(), np.vstack(), np.stack()


'''--------------------------------------------------------
Q4. 如何调整列顺序和如何调整维度顺序？
------------------------------------------------------------
'''
# 调整列顺序：有个非常简单的技巧
a = np.array([[1,2,3],[4,5,6]])
a[(1,2,0),:]  # 把1,2列提前，0列放最后。很重要的技巧，还没发现其他实现类似功能的简洁方法

# 调整维度顺序：套用公式，与调整列顺序方法居然异曲同工
np.random.seed(1)
b = np.random.randn(2,3,4)
c = b.transpose(1,2,0)   # 把维度1,维度2提前，维度0放最后
                         # 最通用：transpose()适用于所有array/tensor类型 
d = torch.from_numpy(b).permute(1,2,0)  # permute()功能跟transpose一致，但只适用于tensor
c.shape  


'''--------------------------------------------------------
Q4. 如何改变tensor维度？
------------------------------------------------------------
'''
a = torch.tensor([1,2,3,4,5,6])
b = a.reshape(2,3)  # 变维度，可用于array/tensor...
b = a.view(-1,1)    # 变维度，只能用于tensor

c = a.unsqueeze(0)  # 增加第0维度
d = c.squeeze(0)    # 挤掉第0维度





# =============================================================================
# 专门针对pytorch
# =============================================================================

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
a.item() # 用于把单元素tensor转化为数值标量,比如一个int值 
a.size() # 等同于shape，可用a.size(0),a.size(1)分别去每个维度的大小。
a.shape  # 维度
a.shape[1]  #第2个维度
data = data.to(device)  # tensor送到设备
label = label.to(device)  # tensor送到设备


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
Q. 如何对tensor进行堆叠?
-----------------------------------------------------------
'''
import torch
import numpy as np
# array的堆叠(参考本部分前面)
b0 = np.array([[1,2],[3,4]])
b1 = np.array([[5,6],[7,8]])
b2 =np.concatenate((b0,b1),axis=0)

# tensor堆叠采用类似的cat()函数
t0 = torch.tensor([1,2,3])  
t1 = torch.tensor([4,5,6])
t2 = torch.cat((t0,t1),-1)   # 由于tensor size = 1是一维的，也就只能进行axis=0的堆叠
                             # 而不能进行更高维的axis =1 的操作。即使转换成array的(3,)也是一维的
# tensor正常堆叠操作跟numpy的concatenate()一样，axis=0(行循环), axis=1(列循环)
c0 = torch.tensor(b0)
c1 = torch.tensor(b1)
c2 = torch.cat((c0,c1),1)  # axis = 1为列变换方向，即列堆叠


'''--------------------------------------------------------
Q. 如何对tensor广播式堆叠相加/相乘
pytorch中还没有实现广播原则，所以对于需要广播机制的，则要手动通过repeat()函数完成
-----------------------------------------------------------
'''
a = torch.arange(10).repeat(5,1)  # t.repeat(5,1)表示把原来的数字看成一个元素，堆叠成5行1列



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
device = torch.device('cuda:1') #device有两个属性:device.index, device.type
print(device)
# 设置当前设备
torch.cuda.set_device(1)     # 输入GPU的index 
torch.cuda.current_device()  # 输出GPU的index
# 设备数，设备能力，设备可用
torch.cuda.device_count()
torch.cuda.get_device_capability(0)
torch.cuda.get_device_name(0)
torch.cuda.is_available()


'''--------------------------------------------------------
Q. 对view和reshape函数的认识？
参考：https://blog.csdn.net/jacke121/article/details/80824575

group1: transpose跟permute的功能基本一样，用于对高维矩阵进行各种转秩
group2: view和reshape的功能基本一样，用于把高维矩阵展平
group_m: 这个函数介于group1-group2之间，group1做完会有not contiguous的问题，
         需要使用contiguous()函数把tensor空间连续化，否则view函数无法使用
         但如果group2使用reshape函数就没有这个问题
-----------------------------------------------------------
'''
# group 1的使用
import torch
x = torch.ones(2,4)
y = x.transpose(1,0)  # 把第1个维度跟第0个维度交换，即为把size [m,n]变为[n,m]
z = x.permute(1,0)    # 跟transpose一样

# group2的使用
import torch
x = torch.ones(2,4)
y = x.view(-1,1)           # view改变矩阵维度形式，变为n行1列 (这是常用的layer输入)

x = torch.arange(1,10,1)
y = x.view(-1,1)           # view的功能2：把1维单行变为2维单列。因为1维数据model/layer不能处理

# group_m的介入
import torch
x = torch.ones(2,4)
x.is_contiguous()

y = x.transpose(1,0)      # group1函数之后，发现contiguous=False
y.is_contiguous()

y.view(-1,1)              # group2函数用view报错：因为view需要tensor的内存是整块的
y.contiguous().view(-1,1) # group2函数用contiguous + view修正不报错

y.reshape(-1,1)           # group2函数用reshape正确：它大致相当于 tensor.contiguous().view()


