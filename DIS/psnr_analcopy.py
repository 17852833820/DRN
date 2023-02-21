import sys
sys.path.append("../..")
from DIS import Flow
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import re
from test_DIS import warp_flow
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
# raw 97.1855
num = 1016
label = cv.imread(f"frankfurt_000000_001016_rawcolor.png")


def load_label(path):
    target = Image.open(path)
    target = torch.as_tensor(np.array(target), dtype=torch.int64)
    return target.numpy()


def accuracy(pred, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    # _, pred = out.max(1)
    pred = pred.reshape(1, -1)
    target = target.reshape(1, -1)
    correct = pred == target
    correct = correct[target != 255]
    correct = correct.flatten()
    score = np.sum(correct) * (100.0 / correct.size)
    return score

images = open('test_images.txt')
files = images.readlines()

# while files:
#     f = files.pop(0)
#     dir_name, filename = os.path.split(f)
#     print(filename[:-1])
#     if not os.path.exists(dir_name):
#         os.makedirs(dir_name)
#     pattern = re.compile(r'(.*?)_\d{6}_(\d{6}).*?\.png\n')
#     dir, num = pattern.findall(filename)[0]
#     num = int(num)
#     flow_var = np.load(f'flow/' + filename[:-1].replace(f'_leftImg8bit.png', '.npy'))
#     standard = load_label(f'../datasets/testimg/gtFine/val/{dir}/' + filename[:-1].replace(f'{num:06d}_leftImg8bit.png', f'{num+9:06d}_gtFine_trainIds.png'))
#     for index, i in enumerate(range(num, num + 9)):
#         flow = Flow()
#         cur = flow.tensor2gray(flow.open(f'../datasets/testimg/leftImg8bit_sequence/val/{dir}/' + filename[:-1].replace(f'{num:06d}_leftImg8bit.png', f'{num + 9:06d}_leftImg8bit.png')))
#         label = load_label(f'drn_d_22_000_test/leftImg8bit_sequence/val/{dir}/' + filename[:-1].replace(f'{num:06d}_leftImg8bit.png', f'{i:06d}_leftImg8bit.png'))
#         prev = flow.tensor2gray(flow.open(f'../datasets/testimg/leftImg8bit_sequence/val/{dir}/' + filename[:-1].replace(f'{num:06d}_leftImg8bit.png', f'{i:06d}_leftImg8bit.png')))
#         glitch = prev.copy()
#         for j in range(index, 9):
#             label = warp_flow(label, flow_var[j])
#             glitch = warp_flow(glitch, flow_var[j])
#         print(psnr(cur, glitch))
#         print(accuracy(label, standard))
#         input()


psnr = torch.zeros(167, 9)
acc = torch.zeros(167, 9)
plain_acc = torch.zeros(167)
with open('test_images_fix.txt') as f:
    total = f.readlines()
pattern = re.compile(r'(.*?)_\d{6}_(\d{6}).*?\.png')
d = {}
for i in total:
    _, f = os.path.split(i[:-1])
    dir, num = pattern.findall(f)[0]
    if d.get(dir):
        d[dir].append(f)
    else:
        d[dir] = [f]

# datasets = os.listdir('../datasets/testimg/leftImg8bit/val')
for dir in d:
    files = d[dir]
    pattern = re.compile(f'{dir}' + r'_\d{6}_(\d{6}).*\.png')
    for fi, f in enumerate(files):
        print(fi, f)
        num = int(pattern.findall(f)[0])
        standard = load_label(f'../datasets/testimg/gtFine/val/{dir}/' + f.replace('leftImg8bit.png', 'gtFine_trainIds.png'))
        label = load_label(f'drn_d_22_000_test/leftImg8bit_sequence/val/{dir}/' + f)
        plain_acc[fi] = accuracy(label, standard)
        for index, i in enumerate(range(num-9, num)[::-1]):
            flow = Flow()
            label = load_label(f'drn_d_22_000_test/leftImg8bit_sequence/val/{dir}/' + f.replace(f'{num:06d}_leftImg8bit.png', f'{i:06d}_leftImg8bit.png'))
            for n in range(i, num+1):
                img = flow.open(f'../datasets/testimg/leftImg8bit_sequence/val/{dir}/' + f.replace(f'{num:06d}_leftImg8bit.png', f'{n:06d}_leftImg8bit.png'))
                res = flow.propagate(img)
                if not res:
                    flow.update(img, label)
            psnr[fi, index] = flow.psnr
            acc[fi, index] = accuracy(flow.label, standard)
torch.save(plain_acc, 'plain_acc.pth')
torch.save(psnr, 'psnr.pth')
torch.save(acc, 'acc.pth')
