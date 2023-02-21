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
import torchvision.transforms as transforms
# raw 97.1855
num = 1016
label = cv.imread(f"frankfurt_000000_001016_rawcolor.png")

f_image = open('./test_images_fix.txt', 'w')
f_labels = open('./test_labels_fix.txt', 'w')
flow_var = np.zeros([9, 1024, 2048, 2])
sets = os.listdir('./flow')#read flow
for f in sets:
    flow_var[:8] = np.load('flow/'+f)
    pattern = re.compile(r'(.*?)_\d{6}_(\d{6})\.npy')
    print(f)
    dir, num = pattern.findall(f)[0]
    num = int(num)
    flow = Flow()
    prev = flow.open(f'../datasets/testimg/leftImg8bit_sequence/val/{dir}/' + f.replace(f'{num:06d}.npy', f'{num+8:06d}_leftImg8bit.png'))
    flow.update(prev)
    cur = flow.open(f'../datasets/testimg/leftImg8bit_sequence/val/{dir}/' + f.replace(f'{num:06d}.npy', f'{num+9:06d}_leftImg8bit.png'))
    flow.propagate(cur)
    flow_var[8] = flow.flow
    f_image.write(f'leftImg8bit_sequence/val/{dir}/' + f.replace(f'{num:06d}.npy', f'{num+9:06d}_leftImg8bit.png') + '\n')
    f_labels.write(f'gtFine/val/{dir}/' + f.replace(f'{num:06d}.npy', f'{num+9:06d}_gtFine_trainIds.png') + '\n')
    np.save('flow/'+f, flow_var)
f_image.close()
f_labels.close()
# datasets = os.listdir('../datasets/testimg/leftImg8bit/val')
# for dir in datasets:
#     files = os.listdir(f'../datasets/testimg/leftImg8bit/val/{dir}')
#     pattern = re.compile(f'{dir}' + r'_\d{6}_(\d{6}).*\.png')
#     for f in files:
#         # input('wait')
#         print(f)
#         num = int(pattern.findall(f)[0])
#         flag = 0
#         for index, i in enumerate(range(num-5, num)[::-1]):
#             flow = Flow()
#             for n in range(i, num+1):
#                 img = flow.open(f'../datasets/testimg/leftImg8bit_sequence/val/{dir}/' + f.replace(f'{num:06d}_leftImg8bit.png', f'{n:06d}_leftImg8bit.png'))
#                 res = flow.propagate(img)
#                 if not res:
#                     flow.update(img)
#             if flow.psnr < 25 and index < 4:
#                 print(flow.psnr, index)
#                 flag = 1
#             if flag == 1:
#                 break
#         if flag == 1:
#             continue
#         print('success', flow.psnr, index)
#         for index, i in enumerate(range(num - 9, num)):
#             raw_img = f'leftImg8bit_sequence/val/{dir}/' + f.replace(f'{num:06d}_leftImg8bit.png', f'{i:06d}_leftImg8bit.png') + '\n'
#             label = f'gtFine/val/{dir}/' + f.replace('leftImg8bit.png', 'gtFine_trainIds.png') + '\n'
#             f_image.write(raw_img)
#             f_labels.write(label)

