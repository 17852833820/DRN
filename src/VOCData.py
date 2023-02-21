import os

import numpy as np
import torch
from PIL import Image


class VOCSegmentation(object):
    NUM_CLASS = 21
    def __init__(self, root='datasets/VOCdevkit', transform=None, base_size=520, crop_size=480):
        
        root = os.path.join(root, 'VOC2012')
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size

        _image_dir = os.path.join(root, 'JPEGImages')
        _mask_dir = os.path.join(root, 'SegmentationClass')


        self.images = []
        self.masks = []
        with open(os.path.join(root, 'ImageSets/Segmentation', 'val.txt')) as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")
                assert os.path.isfile(_mask)
                self.masks.append(_mask)
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = np.array(img)
        img = self.transform(img)
        mask = Image.open(self.masks[index])
        mask = self._mask_transform(mask)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0

if __name__ == '__main__':
    dataset = VOCSegmentation()

    
    


