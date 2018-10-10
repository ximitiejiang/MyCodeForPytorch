#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:37:07 2018

@author: suliang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:36:50 2018

@author: suliang

参考pytorch英文原教程
实现图像风格迁移 neural style transfer（神经风格迁移）

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义图片size： 如果有GPU可以渲染更大的图片，否则就处理小图片
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

# 图片变换器
loader = transforms.Compose([
    transforms.Resize((imsize,imsize)),  # 缩放图片到128x128
    transforms.ToTensor()])     # 转换为tensor


def image_loader(image_name):
    image = Image.open(image_name)       # 打开
    image = loader(image).unsqueeze(0)   # 缩放，转换tensor，增加维度
    return image.to(device, torch.float) #  


'''
载入图片：包括打开，缩放，转换tensor, 增加维度
'''
style_img = image_loader("./images/picasso.jpg")
content_img = image_loader("./images/dancing.jpg")

print(style_img.size())
print(content_img.size())

'''
显示加载的图片
'''
unloader = transforms.ToPILImage()  # 把tensor恢复成图片

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # 克隆
    image = image.squeeze(0)      # resize BxCxHxW to CxHxW
    image = unloader(image)       # CxHxW转为HxWxC, 类似transpose(1,2,0)操作
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')


'''
定义内容损失函数和风格损失函数
'''
class ContentLoss(nn.Module): # 内容损失函数类：继承ContentLoss

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):  # 风格损失函数类：继承StyleLoss

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


'''
建立模型
'''
cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


