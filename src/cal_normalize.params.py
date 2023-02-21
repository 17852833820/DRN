

import time

import numpy as np
import shutil

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
from os.path import exists, join, split

import data_transforms as transforms

class SegList(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        #data = [Image.open(join(self.data_dir, self.image_list[index]))]
        data = [Image.open(join(self.image_list[index]))]#Image对象2048，1024
        if self.label_list is not None:
            data.append(Image.open(
                join(self.data_dir, self.label_list[index])))
        data = list(self.transforms(*data))
        if self.out_name:
            # if self.label_list is None:
                # data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, "demovideo", 'back_image_path.txt') #读取图像路径列表list'./datasets/testimg/test_images.txt'
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())
 
if __name__ == '__main__':
    data_dir="./datasets/back_test"
    dataset = SegList(data_dir, transforms.Compose([ transforms.ToTensor(), ]), list_dir=None, out_name=True)

    print(getStat(dataset))
