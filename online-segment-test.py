#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import math
import os
import shutil
import sys
import threading
import time
from os.path import exists, join, split

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import src.data_transforms   as transforms
import src.drn as drn
import warnings
warnings.filterwarnings("ignore")
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT,filename="./run_log/cityscape_drn38_4gLTE_report_bus_0001.log",filemode="w")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]
"""用于训练的语义分割模型"""
class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        #根据modelname从drn加载对应模型 
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return self.softmax(y), x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param
"""构造数据集"""
class SegList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms,
                 out_name=False):
        self.list_dir = data_dir 
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        #data = [Image.open(join(self.data_dir, self.image_list[index]))]#Image对象2048，1024
        data = [Image.open(self.image_list[index])]
        if self.label_list is not None:
            # data.append(Image.open(join(self.data_dir, self.label_list[index])))
            data.append(Image.open(self.label_list[index]))
        data = list(self.transforms(*data))#对原始图像像素和label像素执行totensor变换 
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])#加入原始图像路径
        return tuple(data)#img，label，name

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir,    "demovideo", 'images_path.txt') #读取图像路径列表list'./datasets/testimg/test_images.txt'
        label_path = join(self.list_dir,  "demovideo", 'labels_path.txt')#读取语义分割label真实标签路径列表list'./datasets/testimg/test_labels.txt'
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):#如果存在GT
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)#图像和GT一一对应
"""计算语义分割精确度"""
def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    score = correct.float().sum(0).mul(100.0 / correct.size(0))
    return score.data
def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    np.seterr(divide='ignore',invalid='ignore')
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def save_encode_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].transpose(1,2,0).astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind].split("/")[-1])
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)
def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind].split("/")[-1] )
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_colorful_images(predictions, filenames, output_dir, palettes):
   """
   Saves a given (B x C x H x W) into an image file.
   If given a mini-batch tensor, will save the tensor as a grid of images.
   """
   for ind in range(len(filenames)):
       im = Image.fromarray(palettes[predictions[ind].squeeze()])
       fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
       out_dir = split(fn)[0]
       if not exists(out_dir):
           os.makedirs(out_dir)
       im.save(fn)

def select_params(time,tables,bandwidth):#,complexity,difference):
    """tables:fps.B.time acc.V.com.diff"""
    complexitys_circle=0.2
    differences_circle=0.2
    data=tables
    bandwidth=float(bandwidth)
    # """1. complexity """
    # data=tables[(tables["complexity"]>complexity-complexitys_circle)&(tables["complexity"]<complexity+complexitys_circle)]
    # """2. difference"""
    # data=data[(data["difference"]>difference-differences_circle)&(data["difference"]<difference+differences_circle)]
    """3. V """
    # data=data[(data["V"]*17+(17-(data["fps"]))*20)<float(bandwidth)]#bandwidth: KByte
    data=data[(data["V"]*17+(17)*5)<float(bandwidth)]#bandwidth: KByte
    """4. acc"""
    max_acc=data["acc"].max()
    data=data[data["acc"]==max_acc]
    if data.empty:
        fps=3.0
        B=0.001
    else:
        fps=data.iat[0,0]
        B=float(data.iat[0,1])
    return fps,B
from functools import reduce

def str2float(s):
    b = s
    front = reduce(lambda x, y: y + x * 10, map(int, a))
    a = 0
    if 'e' in b:
        for i in b:
            a += 1
            if i == 'e':
                c = b[a+1:]
                middle = reduce(lambda x,y:y+x*10,map(int,c))
                b = b[:a-1]
                buttom = reduce(lambda x,y:y+x*10,map(int,b))
                result = (front + buttom / 10 ** (len(b))) / 10 ** middle
                return result
    else:
        buttom = reduce(lambda x, y: y + x * 10, map(int, b))
        result = front + buttom / 10 ** (len(b))
        return result

def get_train_params(fps,B):
    flag=str(str(int(fps))+","+str(float(B)))
    dict={
        "17,2.5e-05":65,
        "17,0.0002":35,
        "17,6e-05":45,
        "17,0.001":15,
        "17,1e-05":85,
        "17,2e-05":80,
        "17,2.2e-05":70,
        "17,0.0005":30,
        "17,0.001":25,
        "15,0.0002":40,
         "15,0.0001":45,
          "15,9e-05":50,
        "15,7e-05":55,
        "15,5e-05":60,
        "15,3.5e-05":70,
        "15,2.5e-05":75,
        "15,2e-05":80,
         "15,1e-05":85,
        "12,0.0002":40,
         "12,0.0001":45,
          "12,9e-05":50,
        "12,7e-05":55,
        "12,5e-05":60,
           "12,2e-05":80,
        "12,2.5e-05":75,
         "12,1e-05":85,
        "10,0.001":25,
        "10,0.0005":30,
        "10,2e-05":80,
        "10,1e-05":85,
        "10,0.002":25,
        "10,0.0001":70,
        "10,0.0002":65,
        "10,0.0003":45,
        "10,0.0004":40,
        "8,0.0004":40,
        "8,0.0003":45,
        "8,0.0002":65,
        "8,0.0001":70,
        "8,2e-05":75,
        "8,1e-05":85,
        "8,0.001":25,
        "5,0.001":25,
        "5,1e-05":85,
        "5,2e-05":75,
        "5,0.0004":40,
        "5,0.0003":45,
        "5,0.0002":65,
        "5,0.0001":70,
        "5,0.000101":70,
        "3,0.0001":70,
        "3,0.0002":65,
        "3,0.0003":45,
        "3,0.0004":40,
        "3,2e-05":75,
        "3,1e-05":85,
        "3,0.001":25
    }
    init_I_qp=dict[flag]
    table1_threshold=np.zeros((3,2))
    table1_threshold[1][0]=max(init_I_qp-15,0)
    table1_threshold[1][1]=min(100,init_I_qp-15+30)
    table1_threshold[0][0]=max(0,init_I_qp-10)
    table1_threshold[1][0]=min(100,init_I_qp-10+30)
    table1_threshold[2][0]=max(0,init_I_qp-20)
    table1_threshold[2][0]=min(100,init_I_qp-20+30)
    rate_dict={
        "17,2.5e-05":0.86,
        "17,0.0002":0.8,
        "17,6e-05":0.85,
        "17,0.001":0.25,
        "17,1e-05":0.99999,
        "17,2e-05":0.98,
        "17,2.2e-05":0.98,
        "17,0.0005":0.8,
        "17,0.001":0.7,
        "15,0.0002":0.82,
        "15,0.0001":0.85,
        "15,9e-05":0.87,
        "15,7e-05":0.9,
          "15,5e-05":0.95,
             "15,3.5e-05":0.96,
          "15,2.5e-05":0.97,
          "15,2e-05":0.98,#0.99
          "15,1e-05":0.99999,#0.99999
        "12,0.0002":0.83,
        "12,0.0001":0.85,
        "12,9e-05":0.87,
        "12,7e-05":0.9,
          "12,5e-05":0.95,
          "12,2.5e-05":0.97,
          "12,2e-05":0.985,
          "12,1e-05":0.99999,
        "10,0.001":0.6,
        "10,0.0005":0.7,
        "10,2e-05":0.95,
        "10,1e-05":0.99999,
        "10,0.002":0.5,
        "10,0.0001":0.92,
        "10,0.0002":0.9,
        "10,0.0003":0.9,
        "10,0.0004":0.85,
        "8,0.0004":0.7,
        "8,0.0003":0.8,
        "8,0.0002":0.88,
        "8,0.0001":0.95,
        "8,2e-05":0.97,
        "8,1e-05":0.99999,
        "8,0.001":0.55,
        "5,0.001":0.55,
        "5,1e-05":0.9995,
        "5,2e-05":0.97,
        "5,0.0004":0.7,
        "5,0.0003":0.78,
        "5,0.0002":0.85,
        "5,0.0001":0.9,
        "5,0.000101":0.95,
        "3,0.0001":0.92,
        "3,0.0002":0.85,
        "3,0.0003":0.78,
        "3,0.0004":0.7,
        "3,2e-05":0.97,
        "3,1e-05":0.9995,
        "3,0.001":0.55
    }
    table_rate=rate_dict[flag]
    return init_I_qp,table1_threshold,table_rate
def get_trace(trace_file):
    trace=[]
    iter=0
    mean_trace_second=[]
    std_trace_second=[]
    test=[]
    for line in trace_file:
        # trace.append(float(line.strip().split(',')[3])/1000)    #FCC
        trace.append(float(line.strip().split(' ')[4])/1000)#KByte LTE
        iter+=1
    # for line in trace_file: #wifi
    #     if(float(line.strip().split(',')[0])==0): continue
    #     trace.append(float(line.strip().split(',')[0])/1024)#KByte
    #     iter+=1
    trace_file.close()
    return trace
def test(eval_data_loader, model, num_classes,
         output_dir='pred', has_gt=True, save_vis=True):
    from src.selector import Selector
    import numpy as np
    model.eval()
    hist = np.zeros((num_classes, num_classes))
    criterion = nn.NLLLoss2d(ignore_index=255)
    criterion.cuda()
    import re
    import cv2 as cv
    import pandas as pd
    import pickle
    from PIL import Image
    from DIS import Flow, accuracy_np, load_label, map_index, psnr, warp_flow
    from pyjpeg import PyJPEG, blockify, deblockify
    qtables = torch.load('./pth/jpeg_qtables.pth') #shape：100，3，8，8     #读取jpeg量化表
    search_tables=pd.read_hdf("./youtube_drn22_log/table_youtube_drn22.hdf5",key="df",mode="r")
    trace_file=open("./datasets/network_trace/4gLTE/report_bus_0001.log")
    trace=get_trace(trace_file)

    prev_gray = None
    rec_image=None
    jpeg = PyJPEG(True)
    iter_fps=(1/17.0)
    upload_fps=0
    I_number=0
    sum_size=upload_num=sum_acc=sum_sizejpeg=num=offload_num=sum_accjpeg=0
    """init params"""
    fps=3
    init_I_qp=25
    selector = Selector(B=0.001, region=1,pattern='jpeg')#初始化，B
    for iter, (image, label, name) in enumerate(eval_data_loader):   
        # if iter<3758:continue
        torch.cuda.empty_cache()
        _, filename = os.path.split(name[0])
        flow_var = np.load('/media/xxd/新加卷/Projects/DRN/DIS/flow/' + filename.replace(".png", '.npy')).astype(np.float32)    # 读取filename对应名字的光流
        row_size=os.path.getsize( name[0])/1024#KB
        cur_img = image # 0-1 rgb
        cur_gray = Flow().tensor2gray(cur_img) # 转灰度图（1024，2048）
        if prev_gray is not None :
            if  iter_fps<(upload_fps)<(iter_fps+(1/17.0)) or iter%17==0 or fps==17:
                upload_fps+=(1.0/fps)
                type="P"
                offload_num+=1
                Offloading=True
                prev_gray = cur_gray
                cur_img=jpeg.rgb_to_ycbcr_jpeg(cur_img.float().div(1/255))#0-255 yuv
                """1. 使用jpeg压缩并重建 """
                #传播量化表1
                table_indexs1 = warp_flow(table_indexs1, flow_var) # H,W,C对量化表索引进行传播
                selected_tables = selector.extract_table1(table_indexs1) # 通过传播后的索引提取量化表
                q1 = selected_tables.clone() 
                compressed = jpeg.compress(cur_img.clone().float().div(255), q1,is_residual=False) # 指定 QP使用torchjpeg压缩当前帧:DCT变换+量化，返回量化后的矩阵
                dct_var = jpeg.dequantize(compressed, q1, normalize=False) # 反量化得到dct系数
                """jpeg acc"""
                s = dct_var.clone()
                dct_var.requires_grad = True
                image = jpeg.idct(dct_var.clone(),is_residual=False) # 反dct得到原始图像 0-1 yuv
                image=image.float().div(1/255).clamp(0,255)#1,c,h,w0-255 yuv
                image=jpeg.ycbcr_to_rgb_jpeg(image).clamp(0,255)#(-6,285)
                image = jpeg.normalize(image.div(255)) # 归一化
                final = model(image)[0] # 输入到网络计算
                _, pred = torch.max(final, 1) 
                loss_jpeg = criterion(final, label.cuda().data)
                pred = pred.cpu().data.numpy()
                save_output_images(pred, name, output_dir='/media/xxd/新加卷/Projects/DRN/run_log/pred_img/cityscape_drn38_4gLTE/')
                pred_jpeg = load_label('/media/xxd/新加卷/Projects/DRN/run_log/pred_img/cityscape_drn38_4gLTE/' + '/' + name[0].split("/")[-1])
                acc_jpeg = accuracy_np(pred_jpeg, label[0].cpu().data.numpy())
                #反DCT变换
                rec_image1 = jpeg.idct(torch.tensor(dct_var.clone().detach().requires_grad_(True).int(),dtype=torch.float32),is_residual=False) # 逆dct得到重建图像rec（RGB）1,3,h,w 0-1
                rec_image1=rec_image1.float().div(1/255).clamp(0,255)#0-255 yuv
                    
                """2. 使用H264原理帧间压缩"""
                #1.光流传播，将上一帧的重建图像映射到当前帧
                rec_image=jpeg.rgb_to_ycbcr_jpeg(rec_image)#0-255,yuv
                cur_map_image=map_index(rec_image[0].permute(1,2,0),-flow_var).permute(2,0,1)#C,H,W(0-255)
                #2. 计算残差 (pix level)
                residual_jpeg=rec_image1-cur_map_image
                residual_img=cur_img[0]-cur_map_image#C,H,W -255,255
                #3. 对残差进行jpeg压缩
                # 选择量化表2

                table_indexs2=selector.choose_table2(q1,iter,jpeg,rec_image1.clone().float().div(255),cur_img.float().div(255),compressed,residual_jpeg.float().div(255),residual_img.float().div(255).unsqueeze(0),cur_map_image.float().div(255).unsqueeze(0))
                selected_tables2 = selector.extract_table2(table_indexs2) # 通过传播后的索引提取量化表
                q2 = selected_tables2.clone() 
                # 对残差进行压缩1
                compressed_residual=jpeg.compress(residual_img.float().div(255).unsqueeze(0),q2,is_residual=True)#
                size = jpeg.torchjpeg_compress(compressed_residual.clone()) # 计算压缩后图片的体积
                jpeg_size=jpeg.torchjpeg_compress(compressed.clone())
                sum_sizejpeg+= jpeg_size
                if size>jpeg.torchjpeg_compress(compressed.clone()): # 计算压缩后图片的体积
                    I_number+=1
                    type="I"
                    image=rec_image1
                    rec_image=jpeg.ycbcr_to_rgb_jpeg(rec_image1).clamp(0,255)#0-255 grb   
                    size=jpeg.torchjpeg_compress(compressed.clone())
                else:
                #上载残差和量化表2
                    #图像重建"""
                    dct_var_residual=jpeg.dequantize(compressed_residual,q2,normalize=False)
                    residual=jpeg.idct(dct_var_residual,is_residual=True).float().div(1/255)#-255,255,C,H,W   decode error:(-28,38,mean:0.97)
                    image=residual+cur_map_image
                    image=image.clamp(0,255)#0-255 yuv
                    rec_image=jpeg.ycbcr_to_rgb_jpeg(image).clamp(0,255)#0-255 rgb
                # save_encode_images(rec_image.clone().cpu().data.numpy(),name,output_dir='/media/xxd/新加卷/Projects/DRN/run_log/encode_img')
                """jpeg acc"""

                """3.计算当前帧量化表1"""
                dct_var = jpeg.dct(image.clone().float().div(255),is_residual=False) # DCT得到dct系数
                s = dct_var.clone()
                dct_var.requires_grad = True
                image = jpeg.idct(dct_var,is_residual=False) # 反dct得到原始图像0-1 yuv
                image=image.float().div(1/255).clamp(0,255)#1,c,h,w0-255 yuv
                image=jpeg.ycbcr_to_rgb_jpeg(image).clamp(0,255).div(255)#(-6,285)
                image = jpeg.normalize(image) # 归一化
                final = model(image)[0] # 输入到网络计算
                _, pred = torch.max(final, 1) 
                loss = criterion(final, label.cuda().data)
                loss.backward()
                g = dct_var.grad.clone() # 得到关于残差dct系数的梯度g
                pred = pred.cpu().data.numpy()
                """压缩 后残差体积"""
                sum_size += size
                upload_num += 1 
                save_output_images(pred, name, output_dir='/media/xxd/新加卷/Projects/DRN/run_log/pred_img/cityscape_drn38_4gLTE/')
                pred = load_label('/media/xxd/新加卷/Projects/DRN/run_log/pred_img/cityscape_drn38_4gLTE/'+ '/' + name[0].split("/")[-1])
                acc = accuracy_np(pred, label[0].cpu().data.numpy())
                if type=="I":
                    acc=acc_jpeg
                    pred=pred_jpeg
                sum_acc += acc
                """ cal index i  using jpeg qp=90 rec image"""
                q = qtables[89].reshape(1, 3, 1, 8, 8) # 第一帧使用jpeg（Q=85）量化表
                compressed = jpeg.compress(cur_img.clone().float().div(255), q,is_residual=False) # 指定 QP使用torchjpeg压缩当前帧:DCT变换+量化，返回量化后的矩阵
                dct_var = jpeg.dequantize(compressed, q, normalize=False) # 反量化得到dct系数            
                s = dct_var.clone()
                dct_var.requires_grad = True
                image = jpeg.idct(dct_var,is_residual=False) # 反dct得到原始图像0-1 yuv
                image=image.float().div(1/255).clamp(0,255)#1,c,h,w0-255 yuv
                image=jpeg.ycbcr_to_rgb_jpeg(image).clamp(0,255).div(255)#(-6,285)
                image = jpeg.normalize(image) # 归一化
                final = model(image)[0] # 输入到网络计算
                loss = criterion(final, label.cuda().data)
                loss.backward()
                g = dct_var.grad.clone() # 得到关于残差dct系数的梯度g
                table_indexs1 = selector.choose_table(s, g) # 根据dct系数和grad选择量化表
            else:
                type="P"
                Offloading=False
                size=0
                prev_gray = cur_gray
                pred = warp_flow(pred, flow_var) # 对segment results应用光流传播到下一帧
                pred.resize(1,1024,2048)
                save_output_images(pred, name, output_dir='/media/xxd/新加卷/Projects/DRN/run_log/pred_img/cityscape_drn38_4gLTE/')
                pred = load_label('/media/xxd/新加卷/Projects/DRN/run_log/pred_img/cityscape_drn38_4gLTE/'+ '/' + name[0].split("/")[-1])
                acc = accuracy_np(pred, label[0].cpu().data.numpy())
                sum_acc+=acc
                sum_size+=size
            iter_fps+=(1.0/17.0)



           
        """帧内压缩(原始jpeg压缩)"""
        if prev_gray is None :# or p < PSNR_B: # 第一帧或psnr小于 门限（即需要上载的帧的处理）
            upload_fps=(1.0/fps)
            torch.cuda.empty_cache()
            type="I"
            I_number+=1
            Offloading=True
            q = qtables[init_I_qp].reshape(1, 3, 1, 8, 8) # 第一帧使用jpeg（Q=85）量化表
            prev_gray = cur_gray
            glitch = prev_gray.copy()#
 
            """    DCT变换 +量化 (DC处理)   """
            cur_img=jpeg.rgb_to_ycbcr_jpeg(cur_img.float().div(1/255))#0-255,yuv
            compressed = jpeg.compress(cur_img.clone().float().div(255), q,is_residual=False) # 指定 QP使用torchjpeg压缩当前帧:DCT变换+量化，返回量化后的矩阵
            dct_var = jpeg.dequantize(compressed, q, normalize=False) # 反量化得到dct系数
            s = dct_var.clone()
            dct_var.requires_grad = True

            """ 反DCT变换"""
            image_idct = jpeg.idct(dct_var,is_residual=False) # 逆dct得到重建图像rec（RGB）
            rec_image=image_idct.float().div(1/255)#0-255 yuv
            rec_image=rec_image.clamp(0,255)#1,c,h,w (14-266)yuv
            rec_image=jpeg.ycbcr_to_rgb_jpeg(rec_image).clamp(0,255)#(-6,285)
            # save_encode_images(rec_image.clone().cpu().data.numpy(),name,output_dir='/media/xxd/新加卷/Projects/DRN/run_log/encode_img')
            image = jpeg.normalize(rec_image.float().div(255)) # 正则化
            """上载"""
            final = model(image)[0] # 输入到网络计算
            _, pred = torch.max(final, 1) 
            loss = criterion(final, label.cuda().data)
            loss.backward()
            g = dct_var.grad.clone() # 得到关于dct系数的梯度g
            pred = pred.cpu().data.numpy()
            """压缩后整张图像的体积"""
            size = jpeg.torchjpeg_compress(compressed.clone()) # 计算压缩后图片的体积
            sum_size += size#KB
            upload_num += 1
            save_output_images(pred, name, output_dir='/media/xxd/新加卷/Projects/DRN/run_log/pred_img/cityscape_drn38_4gLTE/')
            #save_colorful_images(pred, name, '/media/xxd/新加卷/Projects/DRN/save-fps/pred_img' + '_color/B={0}-fps={1}'.format(B,fps),TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
            pred = load_label('/media/xxd/新加卷/Projects/DRN/run_log/pred_img/cityscape_drn38_4gLTE/' + '/' + name[0].split("/")[-1])
            acc = accuracy_np(pred, label[0].cpu().data.numpy())
            sum_acc += acc
            """ cal index i  using jpeg qp=90 rec image"""
            q = qtables[89].reshape(1, 3, 1, 8, 8) # 第一帧使用jpeg（Q=85）量化表
            compressed = jpeg.compress(cur_img.clone().float().div(255), q,is_residual=False) # 指定 QP使用torchjpeg压缩当前帧:DCT变换+量化，返回量化后的矩阵
            dct_var = jpeg.dequantize(compressed, q, normalize=False) # 反量化得到dct系数            
            s = dct_var.clone()
            dct_var.requires_grad = True
            image = jpeg.idct(dct_var,is_residual=False) # 反dct得到原始图像0-1 yuv
            image=image.float().div(1/255).clamp(0,255)#1,c,h,w0-255 yuv
            image=jpeg.ycbcr_to_rgb_jpeg(image).clamp(0,255).div(255)#(-6,285)
            image = jpeg.normalize(image) # 归一化
            final = model(image)[0] # 输入到网络计算
            loss = criterion(final, label.cuda().data)
            loss.backward()
            g = dct_var.grad.clone() # 得到关于残差dct系数的梯度g
            table_indexs1 = selector.choose_table(s, g) # 根据dct系数和grad选择量化表
            table_indexs2=table_indexs1
        
        
        """log"""
        logger.info("Frame id :{0}, Row size:{1} KB, Type: {2}, Offloading:{3}, compressed size:{4}".format(iter+1,row_size,type,Offloading,size))
        import numpy as np
        logger.info("Loss:{0},Acc:{1}".format(loss,acc))
        if (iter+1)%17==0 and iter>0:
            print("The {0} s I frame rate:{1}/17".format(int(iter/17),I_number))
            """update fps and B"""
            # complexity,difference=cal_com_diff()
            fps,B=select_params(iter+1,search_tables[search_tables["time"]==((iter+1)%17+1)],trace[(int((iter+1)/17))%len(trace)])#,complexity,difference)
            
            init_I_qp,table1_threshold,table_rate=get_train_params(fps,B)
            logger.info("fps:{0},B:{1}".format(fps,B))
            selector.update_params(B,table1_threshold,table_rate)
            upload_fps=(1.0/fps)
            iter_fps=(1/17.0)
            sum_acc=0
            sum_size=0
            I_number=0
           
        num += 1
        model.zero_grad()
        if has_gt:
            label = label[0].numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
    print('average sum size: ', sum_size/num/30)#KB
    print('average upload size: ', sum_size/upload_num)
    print('JPEG average upload size: ', sum_sizejpeg/(upload_num-1))
    print('sum jpeg acc: ', sum_acc/num)
    print('average upload jpeg size: ', sum_accjpeg/(num-1))
    print('upload img num:', upload_num)
    print('bias', selector.calc_bais1())
    print('average sum size with bias', (sum_size + upload_num * selector.calc_bais1())/num/30)

    if has_gt: #val 有ground truth
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)
def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase
    single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                          pretrained=False)
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()

    data_dir = args.data_dir
    dataset = SegList(data_dir, phase, transforms.Compose([#totensor变换 
        transforms.ToTensor(),
        # normalize,
    ]), out_name=True)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )
    cudnn.benchmark = True
    # optionally resume from a checkpoint
    out_dir = 'test_output_{}_{}'.format(args.arch, phase)#输出文件夹'drn_d_22_000_test'
    mAP = test(test_loader, model, args.classes, save_vis=True,
                   has_gt= args.with_gt, output_dir=out_dir)
    logger.info('mAP: %f', mAP)
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-cmd', choices=['train', 'test'])
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                        help='output path for training checkpoints')
    parser.add_argument('--save_iter', default=1, type=int,
                        help='number of training iterations between'
                             'checkpoint history saves')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--load-release', dest='load_rel', default=None)
    parser.add_argument('--phase', default='val')
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--bn-sync', action='store_true')
    parser.add_argument('--with-gt', action='store_true')
    args = parser.parse_args()

    assert args.classes > 0

    #print(' '.join(sys.argv))
    #print(args)

    if args.bn_sync:
        drn.BatchNorm = batchnormsync.BatchNormSync
    return args


def main():
    args = parse_args()
    if args.cmd == 'test':
        test_seg(args)


if __name__ == '__main__':
    main()
