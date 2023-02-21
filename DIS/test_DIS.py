import os

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.transforms.functional import to_pil_image, to_tensor


def accuracy_np(pred, target):
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

class Flow():
    def __init__(self, init_frame=None, bound=30):
        self.bound = bound
        self.prev_gray = None
        self.flow = None
        self.glitch = None
        self.label = None
        self.comp = None
        self.inst = cv.DISOpticalFlow.create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.inst.setFinestScale(2)
        self.inst.setGradientDescentIterations(12)
        self.inst.setPatchSize(8)
        self.psnr = None
        # self.inst.setVariationalRefinementAlpha(10)
        # self.inst.setVariationalRefinementDelta(10)
        # self.inst.setVariationalRefinementGamma(5)
        # self.inst.setVariationalRefinementIterations(32)
        if init_frame:
            self.update(init_frame)
    
    def open(self, file):
        im = to_tensor(Image.open(file))
        return im.unsqueeze(0) 

    def update(self, frame, label=None, comp=None):
        self.prev_gray = self.tensor2gray(frame)
        self.glitch = self.prev_gray.copy()
        self.label = label
        self.comp = comp
    """传播光流"""
    def propagate(self, frame):
        if self.prev_gray is None:
            return False
        cur_gray = self.tensor2gray(frame)

        flow = self.flow
        if flow is not None:
            #warp previous flow to get an initial approximation for the current flow:
            flow = self.inst.calc(self.prev_gray, cur_gray, warp_flow(flow, flow))
        else:
            flow = self.inst.calc(self.prev_gray, cur_gray, None)

        self.glitch = warp_flow(self.glitch, flow)

        # if self.comp is not None:
        #     self.comp = warp_flow(self.comp, flow)
        self.psnr = psnr(cur_gray, self.glitch)
        # if sim < self.bound:
        #     return False
        # cv.imshow('glitch', self.glitch)
        if self.label is not None:
            self.label = warp_flow(self.label, flow)
            # cv.imshow('labels', self.labels)
        # for i in range(1024):
        #     for j in range(2048):
        #         if np.all(self.labels[i, j] == 0):
        #             if 900>i>100 and 1900>j>100:
        #                 print('cood:', i, j)
        # input('end')
        # cv.imshow('labels', self.labels)
        # cv.waitKey(0)
        self.prev_gray = cur_gray
        self.flow = flow
        return True

    def tensor2gray(self, image):
        image = to_pil_image(image.squeeze(0))
        image = np.asarray(image)
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        return image

    def opencv2pil(self, array):
        return Image.fromarray(cv.cvtColor(array, cv.COLOR_BGR2RGB))

def load_label(path):
    target = Image.open(path)
    target = torch.as_tensor(np.array(target), dtype=torch.int64)
    return target.numpy()


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def map_index(im, flow, mode='bilinear'):
    """Performs a backward warp of an image using the predicted flow.
    numpy version

    Args:
        im: input image. ndim=2, 3 or 4, [[num_batch], height, width, [channels]]. num_batch and channels are optional, default is 1.
        flow: flow vectors. ndim=3 or 4, [[num_batch], height, width, 2]. num_batch is optional
        mode: interpolation mode. 'nearest' or 'bilinear'
    Returns:
        warped: transformed image of the same shape as the input image.
    """
    # assert im.ndim == flow.ndim, 'The dimension of im and flow must be equal '
    flow=torch.tensor(flow)
    flag = 4
    if im.ndim == 2:
        height, width = im.shape
        num_batch = 1
        channels = 1
        im = im[np.newaxis, :, :, np.newaxis]
        flow = flow[np.newaxis, :, :]
        flag = 2
    elif im.ndim == 3:
        height, width,channels = im.shape
        num_batch = 1
        im = im[np.newaxis, :, :]
        flow = flow[np.newaxis, :, :]
        flag = 3
    elif im.ndim == 4:
        num_batch,height, width,channels = im.shape
        flag = 4
    else:
        raise AttributeError('The dimension of im must be 2, 3 or 4')

    max_x = width - 1
    max_y = height - 1
    zero = 0

    # We have to flatten our tensors to vectorize the interpolation
    im_flat = torch.reshape(im, [-1, channels])
    flow_flat = torch.reshape(flow, [-1, 2])

    # Floor the flow, as the final indices are integers
    #flow_floor = torch.floor(flow_flat).type(torch.int32)
    flow_floor = (flow_flat).type(torch.int32)

    # Construct base indices which are displaced with the flow
    pos_x = torch.tile(torch.arange(width), [height * num_batch])
    grid_y = torch.tile(torch.arange(height).unsqueeze(1), [1, width])
    pos_y = torch.tile(torch.reshape(grid_y, [-1]), [num_batch])

    x = flow_floor[:, 0]
    y = flow_floor[:, 1]

    x0 = pos_x + x
    y0 = pos_y + y

    x0 = torch.clip(x0, zero, max_x)
    y0 = torch.clip(y0, zero, max_y)

    dim1 = width * height
    batch_offsets = torch.arange(num_batch) * dim1
    base_grid = torch.tile(batch_offsets.unsqueeze(1), [1, dim1])
    base = torch.reshape(base_grid, [-1])

    base_y0 = base + y0 * width

    if mode == 'nearest':
        idx_a = base_y0 + x0
        warped_flat = im_flat[idx_a]
    elif mode == 'bilinear':
        # The fractional part is used to control the bilinear interpolation.
        bilinear_weights = flow_flat - torch.floor(flow_flat)

        xw = bilinear_weights[:, 0]
        yw = bilinear_weights[:, 1]

        # Compute interpolation weights for 4 adjacent pixels
        # expand to num_batch * height * width x 1 for broadcasting in add_n below
        wa =( (1 - xw) * (1 - yw)).unsqueeze(1)# top left pixel
        wb = ((1 - xw) * yw).unsqueeze(1)# bottom left pixel
        wc =( xw * (1 - yw)).unsqueeze(1) # top right pixel
        wd = (xw * yw).unsqueeze(1) # bottom right pixel

        x1 = x0 + 1
        y1 = y0 + 1

        x1 = torch.clip(x1, zero, max_x)
        y1 = torch.clip(y1, zero, max_y)

        base_y1 = base + y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        warped_flat = wa * Ia + wb * Ib + wc * Ic + wd * Id
    warped = torch.reshape(warped_flat, [num_batch, height, width, channels])

    if flag == 2:
        warped = torch.squeeze(warped)
    elif flag == 3:
        warped = torch.squeeze(warped, axis=0)
    else:
        pass
    warped = warped.type(torch.uint8)

    return warped#H,W,C



def warp_flow(img, flow):#光流传播flow(1024,2048,2)上一帧的重建图像映射到当前帧
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)#第一通道（即[ h e i g h t , w i d t h , 0 ] [height, width, 0][height,width,0]）表示图像在x xx方向的偏移方向和大小。
    #这里的x xx方向是水平方向，即图像数组中的行向量方向
    flow[:, :, 1] += np.arange(h)[:,np.newaxis]#表示图像在y yy方向的偏移方向和大小。这里的y yy方向是竖直方向，即图像数组中的列向量方向。
    res = cv.remap(img, flow, None, cv.INTER_NEAREST)
    # res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res#H,W,C

if __name__ == '__main__':
    # base_dir = "../datasets/testimg/leftImg8bit_sequence/val/frankfurt/"
    # num = 557
    f = Flow()
    # i = 0
    # img = f.open(base_dir + f"frankfurt_000000_{i + num:06d}_leftImg8bit.png")
    # f.update(img)
    # i = 1
    # img2 = f.open(base_dir + f"frankfurt_000000_{i + num:06d}_leftImg8bit.png")
    # f.propagate(img2)
    # select_tables = torch.randint(1, 5, (int(1024/8), int(2048/8)))
    # tables = torch.kron(select_tables, torch.ones(8, 8))
    base_dir = "../datasets/testimg/demovideo/"
    num = 1
    for i in range(100):
        img = f.open(base_dir+f"stuttgart_00_000000_{i+num:06d}_leftImg8bit.png")
        # labels = cv.imread(f"../datasets/testimg/stuttgart_00/stuttgart_00_000000_{i:06d}_leftImg8bit.png")
        color = cv.imread(f"./results/stuttgart_00_color/stuttgart_00_000000_{i:06d}_leftImg8bit.png")
        print(color.size)
        input()
        r = f.propagate(img)
        if not r:
            # f.update(img, labels)
            f.update(img)

    cv.destroyAllWindows()
