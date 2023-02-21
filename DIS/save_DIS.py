import sys
sys.path.append("../")
from DIS import Flow
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import os
import re
datasets = os.listdir('../datasets/testimg/leftImg8bit/val')
import numpy as np
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
#     flow_var = np.zeros([8, 1024, 2048, 2])
#     for index, i in enumerate(range(num, num + 8-1)):
#         flow = Flow()
#         prev = flow.open(f'../datasets/testimg/leftImg8bit_sequence/val/{dir}/' + filename[:-1].replace(f'{num:06d}_leftImg8bit.png', f'{i:06d}_leftImg8bit.png'))
#         flow.update(prev)
#         cur = flow.open(f'../datasets/testimg/leftImg8bit_sequence/val/{dir}/' + filename[:-1].replace(f'{num:06d}_leftImg8bit.png', f'{i + 1:06d}_leftImg8bit.png'))
#         flow.propagate(cur)
#         flow_var[index] = flow.flow
#     np.save(f'flow/' + filename[:-1].replace(f'_leftImg8bit.png', '.npy'), flow_var)
#     for i in range(8):
#         files.pop(0)
    # input()
for dir in datasets:
    files = os.listdir(f'../datasets/testimg/leftImg8bit/val/{dir}')
    pattern = re.compile(f'{dir}' + r'_\d{6}_(\d{6}).*\.png')
    for fi, f in enumerate(files):
        print(fi, f)
        if f != 'frankfurt_000001_044413_leftImg8bit.png':
            continue
        num = int(pattern.findall(f)[0])
        flow_var = np.zeros([29, 1024, 2048, 2])
        flow = Flow()
        for index, i in enumerate(range(num - 19, num+11)):
            cur = flow.open(f'../datasets/testimg/leftImg8bit_sequence/val/{dir}/' + f.replace(f'{num:06d}_leftImg8bit.png', f'{i:06d}_leftImg8bit.png'))
            res = flow.propagate(cur)
            if not res:
                flow.update(cur)
            else:
                flow_var[index-1] = flow.flow
        np.save(f'flow/' + f.replace(f'_leftImg8bit.png', '.npy'), flow_var)
