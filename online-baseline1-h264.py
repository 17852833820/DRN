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
'''try:rage Acc:97.34599319689428
Average size:929.6069417238002
    from modules import batchnormsync
except ImportError:
    pass'''
fps=17.0
qp=95
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT,filename="./run_log/youtube_drn38_4gLTE_baseline1.log",filemode="w")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
#                     level=logging.DEBUG,
#                     filename='test-B={}.log'.format(B),
#                     filemode='a')


CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)


TRIPLET_PALETTE = np.asarray([
    [0, 0, 0, 255],
    [217, 83, 79, 255],
    [91, 192, 222, 255]], dtype=np.uint8)


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

"""?????????????????????????????????"""
class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        #??????modelname???drn?????????????????? 
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

"""???????????????"""
class SegList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        #data = [Image.open(join(self.data_dir, self.image_list[index]))]#Image??????2048???1024
        data = [Image.open(self.image_list[index])]
        if self.label_list is not None:
            # data.append(Image.open(join(self.data_dir, self.label_list[index])))
            data.append(Image.open(self.label_list[index]))
        data = list(self.transforms(*data))#????????????????????????label????????????totensor?????? 
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])#????????????????????????
        return tuple(data)#img???label???name

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir,    "demovideo", 'back_test_images_path.txt') #????????????????????????list'./datasets/testimg/test_images.txt'
        label_path = join(self.list_dir,  "demovideo", 'back_test_labels_path.txt')#??????????????????label????????????????????????list'./datasets/testimg/test_labels.txt'
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):#????????????GT
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)#?????????GT????????????

"""???????????????"""
class SegListMS(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]#??????????????????
        w, h = data[0].size
        if self.label_list is not None:
            data.append(Image.open(
                join(self.data_dir, self.label_list[index])))#??????lebel?????? 
        # data = list(self.transforms(*data))
        out_data = list(self.transforms(*data))#???????????????????????????totensor?????????normalize??????
        ms_images = [self.transforms(data[0].resize((int(w * s), int(h * s)),
                                                    Image.BICUBIC))[0]
                     for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)

"""val"""
def validate(val_loader, model, criterion, eval_score=None, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        if eval_score is not None:
            score.update(eval_score(output, target_var), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {score.val:.3f} ({score.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                score=score))

    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))

    return score.avg

"""?????????????????????value????????????"""
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

"""???????????????????????????"""
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

"""train  model"""
def train(train_loader, model, criterion, optimizer, epoch,
          eval_score=None, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()

        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)[0]#????????????????????????
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        if eval_score is not None:
            scores.update(eval_score(output, target_var), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=scores))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers  #?????????
    crop_size = args.crop_size
    """model loading"""
    #??????DRN???????????????????????????
    single_model = DRNSeg(args.arch, args.classes, None,
                          pretrained=True)
    #????????????????????????????????? 
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()#??????GPU
    """define loss function (criterion) """
    criterion = nn.NLLLoss2d(ignore_index=255)
    criterion.cuda()
    """Data loading code"""
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t = []
    #???????????????????????????
    if args.random_rotate > 0:
        t.append(transforms.RandomRotate(args.random_rotate))
    if args.random_scale > 0:
        t.append(transforms.RandomScale(args.random_scale))
    t.extend([transforms.RandomCrop(crop_size),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              normalize])
    val_loader = torch.utils.data.DataLoader(
        SegList(data_dir, 'val', transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]), list_dir=args.list_dir),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    cudnn.benchmark = True
    """evaluate"""
    if args.evaluate:
        validate(val_loader, model, criterion, eval_score=accuracy)
        return


def adjust_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


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
        qp=20
    else:
        fps=data.iat[0,0]
        qp=float(data.iat[0,1])
    return (fps+1),int(qp)
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
def plot_gradient(g,cur_img,jpeg,alpha=0.5,dir="test.png"):
            from pyjpeg import PyJPEG, blockify, deblockify
            import cv2
            g_copy=blockify(g,8)
            sum=torch.sum(torch.abs(g_copy[0,2,:,:,:]),dim=[1,2])#num
            g_abs=torch.abs(sum).numpy()
            bins=np.linspace(np.min(g_abs),np.max(g_abs),255)
            diagitized=torch.from_numpy(np.digitize(g_abs,bins).reshape(1,1,32768,1,1)).repeat(1,3,1,8,8)
            diagitized[0,1,:,:,:]=0
            diagitized[0,2,:,:,:]=0
            diagitized=deblockify(torch.tensor(diagitized,dtype=torch.float32),(1024,2048)).numpy()
            out=cv2.addWeighted(diagitized[0].transpose(1,2,0).astype(np.float32),alpha,jpeg.ycbcr_to_rgb_jpeg(cur_img)[0].numpy().transpose(1,2,0),1-alpha,0)
            cv2.imwrite(dir,cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
def test(eval_data_loader, model, num_classes,
         output_dir='pred', has_gt=True, save_vis=True):
    # candidate = torch.load('./pth/jpeg_qtables.pth')
    # tables = candidate[85 - 1:95]
    """???????????????????????????"""
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
    qtables = torch.load('./pth/jpeg_qtables.pth') #shape???100???3???8???8     #??????jpeg?????????
    search_tables=pd.read_hdf("./youtube_drn22_log/table_youtube_drn22_baseline1.hdf5",key="df",mode="r")
    trace_file=open("./datasets/network_trace/4gLTE/report_bus_0001.log")
    trace=get_trace(trace_file)
    prev_gray = None
    rec_image=None
    jpeg = PyJPEG(True)
    iter_fps=(1/17.0)
    upload_fps=0
    I_number=0
    sum_size=upload_num=sum_acc=sum_sizejpeg=num=offload_num=sum_accjpeg=0  
    import re
    import cv2 as cv
    from PIL import Image
    from DIS import Flow, accuracy_np, load_label, map_index, psnr, warp_flow
    from pyjpeg import PyJPEG, blockify, deblockify
    compressed_size=[]
    accuracys=[]
    """init params"""
    fps=3
    qp=50
    for iter, (image, label, name) in enumerate(eval_data_loader):
        torch.cuda.empty_cache()
        _, filename = os.path.split(name[0])
        flow_var = np.load('/media/xxd/?????????/Projects/DRN/DIS/back_test_flow/' + filename.replace(".png", '.npy')).astype(np.float32)    # ??????filename?????????????????????
        row_size=os.path.getsize( name[0])/1024#KB
        cur_img = image # 0-1 rgb
        cur_gray = Flow().tensor2gray(cur_img) # ???????????????1024???2048???
        # jpeg_size=acc_jpeg=loss_jpeg=loss=0
        if prev_gray is not None :
            if  iter_fps<(upload_fps)<(iter_fps+(1/17.0)) or iter%17==0 or fps==17:
                upload_fps+=(1.0/fps)
                type="P"
                offload_num+=1
                Offloading=True
                prev_gray = cur_gray
                cur_img=jpeg.rgb_to_ycbcr_jpeg(cur_img.float().div(1/255))#0-255 yuv
                """2. ??????H264??????????????????"""
                #1.????????????????????????????????????????????????????????????
                rec_image=jpeg.rgb_to_ycbcr_jpeg(rec_image)#0-255,yuv
                cur_map_image=map_index(rec_image[0].permute(1,2,0),-flow_var).permute(2,0,1)#C,H,W(0-255)
                #2. ???????????? (pix level)
                residual_img=cur_img[0]-cur_map_image#C,H,W -255,255
                #3. ???????????????jpeg??????
                # ???????????????2
                q2 = qtables[qp-1].reshape(1, 3, 1, 8, 8) # ???????????????jpeg???Q=85????????????
                # ?????????????????????1
                compressed_residual=jpeg.compress(residual_img.float().div(255).unsqueeze(0),q2,is_residual=True)#
                size = jpeg.torchjpeg_compress(compressed_residual.clone()) # ??????????????????????????????
                jpeg_size=jpeg.torchjpeg_compress(compressed.clone())
                sum_sizejpeg+= jpeg_size
                #????????????????????????2
                #????????????"""
                dct_var_residual=jpeg.dequantize(compressed_residual,q2,normalize=False)
                residual=jpeg.idct(dct_var_residual,is_residual=True).float().div(1/255)#-255,255,C,H,W   decode error:(-28,38,mean:0.97)
                image=residual+cur_map_image
                image=image.clamp(0,255)#0-255 yuv
                rec_image=jpeg.ycbcr_to_rgb_jpeg(image).clamp(0,255)#0-255 rgb
                # save_encode_images(rec_image.clone().cpu().data.numpy(),name,output_dir='/media/xxd/?????????/Projects/DRN/save-fps-drn22/baseline1/fps={0}/encode_img/B={1}-fps={2}'.format(int(fps),B,fps))
                """jpeg acc"""

                """3.????????????????????????1"""
                dct_var = jpeg.dct(image.clone().float().div(255),is_residual=False) # DCT??????dct??????
                s = dct_var.clone()
                dct_var.requires_grad = True
                image = jpeg.idct(dct_var,is_residual=False) # ???dct??????????????????0-1 yuv
                image=image.float().div(1/255).clamp(0,255)#1,c,h,w0-255 yuv
                image=jpeg.ycbcr_to_rgb_jpeg(image).clamp(0,255).div(255)#(-6,285)
                image = jpeg.normalize(image) # ?????????
                final = model(image)[0] # ?????????????????????
                #pred=label
                _, pred = torch.max(final, 1) 
                loss = criterion(final, label.cuda().data)
                loss.backward()
                g = dct_var.grad.clone() # ??????????????????dct???????????????g
                pred = pred.cpu().data.numpy()
                """?????? ???????????????"""
                compressed_size.append(size)
                sum_size += size
                upload_num += 1
                # if fps==17:
                save_output_images(pred, name, output_dir='/media/xxd/?????????/Projects/DRN/run_log/pred_img/youtube_drn38_4gLTE/baseline1/')
                #save_colorful_images(pred, name, '/media/xxd/?????????/Projects/DRN/save-fps/pred_img' + '_color/B={0}-fps={1}'.format(B,fps),TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
                #next_pred = load_label('/media/xxd/?????????/Projects/DRN/save-drn38/baseline1/fps={0}/pred_img/fps={1}'.format(int(fps),fps) + '/' + name[0].split("/")[-1])
                pred = load_label('/media/xxd/?????????/Projects/DRN/run_log/pred_img/youtube_drn38_4gLTE/baseline1/'+ '/' + name[0].split("/")[-1])
                # else:
                #     pred.resize(1024,2048)
                acc = accuracy_np(pred, label[0].cpu().data.numpy())
                accuracys.append(acc)
                sum_acc += acc
            else:
               type="P"
               Offloading=False
               size=0
               prev_gray = cur_gray
               pred.resize(1,1024,2048)
               save_output_images(pred, name, output_dir='/media/xxd/?????????/Projects/DRN/run_log/pred_img/youtube_drn38_4gLTE/baseline1/')
               preda = load_label('/media/xxd/?????????/Projects/DRN/run_log/pred_img/youtube_drn38_4gLTE/baseline1/'+ '/' + name[0].split("/")[-1])
               acc = accuracy_np(preda, label[0].cpu().data.numpy())
               sum_acc+=acc
               sum_size+=size
            iter_fps+=(1.0/17.0)



           
        """????????????(??????jpeg??????)"""
        if prev_gray is None :# or p < PSNR_B: # ????????????psnr?????? ??????????????????????????????????????????
            upload_fps=(1.0/fps)
            torch.cuda.empty_cache()
            type="I"
            I_number+=1
            Offloading=True
            q = qtables[qp-1].reshape(1, 3, 1, 8, 8) # ???????????????jpeg???Q=85????????????
            prev_gray = cur_gray
            glitch = prev_gray.copy()#
 
            """    DCT?????? +?????? (DC??????)   """
            cur_img=jpeg.rgb_to_ycbcr_jpeg(cur_img.float().div(1/255))#0-255,yuv
            compressed = jpeg.compress(cur_img.clone().float().div(255), q,is_residual=False) # ?????? QP??????torchjpeg???????????????:DCT??????+?????????????????????????????????
            dct_var = jpeg.dequantize(compressed, q, normalize=False) # ???????????????dct??????
            s = dct_var.clone()
            dct_var.requires_grad = True

            """ ???DCT??????"""
            image_idct = jpeg.idct(dct_var,is_residual=False) # ???dct??????????????????rec???RGB???
            rec_image=image_idct.float().div(1/255)#0-255 yuv
            rec_image=rec_image.clamp(0,255)#1,c,h,w (14-266)yuv
            rec_image=jpeg.ycbcr_to_rgb_jpeg(rec_image).clamp(0,255)#(-6,285)
            # save_encode_images(rec_image.clone().cpu().data.numpy(),name,output_dir='/media/xxd/?????????/Projects/DRN/save-drn2/fps={0}/encode_img/B={1}-fps={2}'.format(int(fps),B,fps))
            #Image.fromarray(rec_image[0].detach().numpy().transpose(1,2,0).astype(np.uint8)).show()#rec image
            image = jpeg.normalize(rec_image.float().div(255)) # ?????????
            """??????"""
            final = model(image)[0] # ?????????????????????
            _, pred = torch.max(final, 1) 
            loss = criterion(final, label.cuda().data)
            loss.backward()
            g = dct_var.grad.clone() # ????????????dct???????????????g
            pred = pred.cpu().data.numpy()
            last_pred=pred
            next_pred=pred
            """??????????????????????????????"""
            size = jpeg.torchjpeg_compress(compressed.clone()) # ??????????????????????????????
            compressed_size.append(size)
            sum_size += size#KB
            upload_num += 1
            save_output_images(pred, name, output_dir='/media/xxd/?????????/Projects/DRN/run_log/pred_img/youtube_drn38_4gLTE/baseline1/')
            #save_colorful_images(pred, name, '/media/xxd/?????????/Projects/DRN/save-fps/pred_img' + '_color/B={0}-fps={1}'.format(B,fps),TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
            pred = load_label('/media/xxd/?????????/Projects/DRN/run_log/pred_img/youtube_drn38_4gLTE/baseline1/'+ '/' + name[0].split("/")[-1])
            acc = accuracy_np(pred, label[0].cpu().data.numpy())
            accuracys.append(acc)
            sum_acc += acc
        
        
        """log"""
        logger.info("Frame id :{0}, Row size:{1} KB, Type: {2}, Offloading:{3}, compressed size:{4}".format(iter+1,row_size,type,Offloading,size))
        import numpy as np
        logger.info("Loss:{0},Acc:{1}".format(loss,acc))
        if (iter+1)%17==0 and iter>0:
            logger.info("Average Acc:   {0}   Average Size:    {1}        in time:  {2}".format(sum_acc/17.0,sum_size/17.0,int(iter/17)))
            print("The {0} s I frame rate:{1}/17".format(int(iter/17),I_number))
            fps,qp=select_params(iter+1,search_tables[search_tables["time"]==((iter+1)%17+1)],trace[(int((iter+1)/17))%len(trace)])#,complexity,difference)
            logger.info("fps:{0},qp:{1}".format(fps,qp))
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
        end = time.time()
     
    print('average sum size: ', sum_size/num/30)#KB
    print('average upload size: ', sum_size/upload_num)
    print('JPEG average upload size: ', sum_sizejpeg/(upload_num-1))
    print('sum jpeg acc: ', sum_acc/num)
    print('average upload jpeg size: ', sum_accjpeg/(num-1))
    print('upload img num:', upload_num)
    # print('bias', selector.calc_bais1())
    # print('average sum size with bias', (sum_size + upload_num * selector.calc_bais1())/num/30)

    if has_gt: #val ???ground truth
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))

    # workers = [threading.Thread(target=resize_one, args=(i, j))
    #            for i in range(tensor.size(0)) for j in range(tensor.size(1))]

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    # for i in range(tensor.size(0)):
    #     for j in range(tensor.size(1)):
    #         out[i, j] = np.array(
    #             Image.fromarray(tensor_cpu[i, j]).resize(
    #                 (w, h), Image.BILINEAR))
    # out = tensor.new().resize_(*out.shape).copy_(torch.from_numpy(out))
    return out

"""test ms"""
def test_ms(eval_data_loader, model, num_classes, scales,
            output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    num_scales = len(scales)
    for iter, input_data in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        if has_gt:
            name = input_data[2]
            label = input_data[1]
        else:
            name = input_data[1]
        h, w = input_data[0].size()[2:4]
        images = [input_data[0]]
        images.extend(input_data[-num_scales:])
        # pdb.set_trace()
        outputs = []
        for image in images:
            image_var = Variable(image, requires_grad=False, volatile=True)
            final = model(image_var)[0]
            outputs.append(final.data)
        final = sum([resize_4d_tensor(out, w, h) for out in outputs])
        # _, pred = torch.max(torch.from_numpy(final), 1)
        # pred = pred.cpu().numpy()
        pred = final.argmax(axis=1)
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_images(pred, name, output_dir)
            save_colorful_images(pred, name, output_dir + '_color',
                                 CITYSCAPE_PALETTE)
        if has_gt:
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    #for k, v in args.__dict__.items():
        #print(k, ':', v)
    #?????????????????????????????????
    single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                          pretrained=False)
    #????????????????????? 
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()

    data_dir = args.data_dir
    '''info = json.load(open(join(data_dir, 'info.json'), 'r'))#std???mean size???2,3???
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])#normalie class?????????
    scales = [0.5, 0.75, 1.25, 1.5, 1.75]#???????????? '''
    '''if args.ms:
        dataset = SegListMS(data_dir, phase, transforms.Compose([transforms.ToTensor(),normalize,#totensor class?????????????????????totensor???normalize??????????????????????????????????????? compose???
        ]), scales, list_dir=args.list_dir)
    else:'''
    dataset = SegList(data_dir, phase, transforms.Compose([#totensor?????? 
        transforms.ToTensor(),
        # normalize,
    ]), list_dir=args.list_dir, out_name=True)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    out_dir = 'test_output_{}_{:03d}_{}'.format(args.arch, start_epoch, phase)#???????????????'drn_d_22_000_test'
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix
    if args.ms:
        out_dir += '_ms'

    if args.ms:
        mAP = test_ms(test_loader, model, args.classes, save_vis=True,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)
    else:
        mAP = test(test_loader, model, args.classes, save_vis=True,
                   has_gt= args.with_gt, output_dir=out_dir)
    logger.info('mAP: %f', mAP)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-cmd', choices=['train', 'test'])
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('-l', '--list-dir', default=None,
                        help='List dir to look for train_images.txt etc. '
                             'It is the same with --data-dir if not set.')
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
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
    parser.add_argument('--ms', action='store_true',
                        help='Turn on multi-scale testing')
    parser.add_argument('--with-gt', action='store_true')
    parser.add_argument('--test-suffix', default='', type=str)
    args = parser.parse_args()

    assert args.classes > 0

    #print(' '.join(sys.argv))
    #print(args)

    if args.bn_sync:
        drn.BatchNorm = batchnormsync.BatchNormSync

    return args



def main():
    args = parse_args()
    if args.cmd == 'train':
        train_seg(args)
    elif args.cmd == 'test':
        test_seg(args)


if __name__ == '__main__':
    main()
