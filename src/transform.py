import torch
import os
torch.hub.set_dir('./model')
cur_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(cur_path)
import torch.nn as nn 
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
from score import SegmentationMetric
from visualize import get_color_pallete
from torchvision import transforms as T
from torchvision.transforms import functional as F
import numpy as np
import torch_dct as dct

class selfModel(nn.Module):
    def __init__(self):
        super(selfModel, self).__init__()

    def forward(self, input):
        image = Decompress()(input[0], input[1])
        
class SemTransform(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        image = Compress()(image)

        target = np.array(target).astype('int32')
        target[target == 255] = -1
        target = torch.from_numpy(target).long()
        return image, target

if __name__ == "__main__":
    c = Compress()
    from PIL import Image
    img = Image.open('b.png').convert('RGB')
    img = np.array(img)
    img= F.to_tensor(img)
    img = c(img)
    # dataset = torchvision.datasets.VOCSegmentation('./datasets', '2012', 'val', download=False, transforms=SemTransform())
    # dataloader = data.DataLoader(dataset=dataset)
    # for i, (image, target) in enumerate(dataloader):
    #     print(i, image[0].shape, target.shape)
    #     os._exit(0)