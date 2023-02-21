import sys
sys.path.append("../")
from DIS import Flow
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import os
import re
import os.path as osp
datasets = ['./datasets/back_test/test_raw']
import numpy as np
images = open('./datasets/back_test/demovideo/back_test_images_path.txt')
files = images.readlines().sort()
# for dir in datasets:
files = os.listdir(f'./datasets/back_test/test_raw')
allnum=len(files)
files=sorted(files)
pattern = re.compile(r'(\d{5}).*\.png')
flow = Flow()
for fi, f in enumerate(files):
    print(fi, f)
    num = int(pattern.findall(f)[0])
    flow_var = np.zeros([ 1024, 2048, 2])
    cur = flow.open(osp.join(f"./datasets/back_test/test_raw/",f))
    res = flow.propagate(cur)
    if not res:
        flow.update(cur)
    else:
        flow_var= flow.flow#num 2 xiangduiyu num 1 de flow cunzai flow_var[num-1=1]
#np.save('/media/weisgroup/新加卷/yingying/drn/DIS/flow/' + f"{dir}/"+f.replace("leftImg8bit.png",".npy"), flow_var)
    np.save('/media/xxd/新加卷/Projects/DRN/DIS/back_test_flow/' + f"{f}".replace(".png",".npy"), flow_var)
    print('save{0}'.format(f))
