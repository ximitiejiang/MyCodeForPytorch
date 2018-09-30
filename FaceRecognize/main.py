#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:00:26 2018

@author: suliang

该实例来自pytorch官方英文教程
识别脸部特征

"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


'''
读入数据
'''
landmarks_frame = pd.read_csv('/Users/suliang/MyDatasets/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]  # 第65行的某张图
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()  # 取出该行数据
landmarks = landmarks.astype('float').reshape(-1, 2) # 分成2列，分别就是x坐标和y坐标

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

def show_landmarks(image, landmarks):
    """
    该函数同时显示图片和标记
    """
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()

show_landmarks(io.imread(os.path.join('/Users/suliang/MyDatasets/faces/', img_name)),
               landmarks)
plt.show()


'''
创建自定义数据集
'''
class FaceLandmarksDataset(Dataset):
    """
    创建数据集
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample