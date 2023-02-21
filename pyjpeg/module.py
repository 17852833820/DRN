import os

import numpy as np
import torch
import torch.nn as nn
import torchjpeg.codec
from PIL import Image
from torchjpeg.dct import block_dct, block_idct, blockify, deblockify
from torchvision import transforms

#info = {"std": [0.1829540508368939, 0.18656561047509476, 0.18447508988480435], "mean": [0.29010095242892997, 0.32808144844279574, 0.28696394422942517]}#cityscape
info = {"std": [0.2726677, 0.30549198, 0.31206694], "mean": [0.37415853, 0.40829375, 0.38689488]}
import time


class PyJPEG(object):
    def __init__(self, to_ycbcr=True):
        self.real_size = None
        self.to_ycbcr = to_ycbcr    
        self.dc_table = {}
        self.ac_table = {}
        dc1_lens = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        dc1_eles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.dc_table[0] = self.gen_huffman_tables(dc1_lens, dc1_eles)

        ac1_lens = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125]
        ac1_eles = [1, 2, 3, 0, 4, 17, 5, 18, 33, 49, 65, 6, 19, 81, 97, 7, 34, 113, 20, 50, 129, 145, 161, 8, 35, 66, 177, 193, 21, 82, 209, 240, 36, 51, 98, 114, 130, 9, 10, 22, 23, 24, 25, 26, 37, 38, 39, 40, 41, 42, 52, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 131, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250]
        self.ac_table[0] = self.gen_huffman_tables(ac1_lens, ac1_eles)

        dc2_lens = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        dc2_eles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.dc_table[1] = self.dc_table[2] = self.gen_huffman_tables(dc2_lens, dc2_eles)
        

        ac2_lens = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119]
        ac2_eles = [0, 1, 2, 3, 17, 4, 5, 33, 49, 6, 18, 65, 81, 7, 97, 113, 19, 34, 50, 129, 8, 20, 66, 145, 161, 177, 193, 9, 35, 51, 82, 240, 21, 98, 114, 209, 10, 22, 36, 52, 225, 37, 241, 23, 24, 25, 26, 38, 39, 40, 41, 42, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 130, 131, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 226, 227, 228, 229, 230, 231, 232, 233, 234, 242, 243, 244, 245, 246, 247, 248, 249, 250]
        self.ac_table[1] = self.ac_table[2] = self.gen_huffman_tables(ac2_lens, ac2_eles)        

        self.zigzag = torch.tensor([ 0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5, 12, 19,
                                26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28, 35, 42, 49, 56,
                                57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31,
                                39, 46, 53, 60, 61, 54, 47, 55, 62, 63], dtype=torch.int32).long()
        
        self.bits_map = {0: ''}
        for i in range(20):
            for n in range(2**i, 2**(i+1)):
                self.bits_map[n] = f'{n:0{i+1}b}'
                self.bits_map[-n] = f'{2**(i+1)-1-n:0{i+1}b}'

    def padding(self, image):
        height, width = self.real_size
        if height % 8 != 0 or width % 8 != 0:
            padding_bottom = 8 - height % 8
            padding_right = 8 - width % 8
            pad = nn.ConstantPad2d((0, padding_right, 0, padding_bottom), 255)
            image = pad(image)             
        return image  

    def depadding(self, image):
        image = image[:, :, :self.real_size[0], :self.real_size[1]]
        return image
    #rgb图像转换为yuv
    def rgb_to_ycbcr_jpeg(self, image):
        matrix = np.array(
            [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]],
            dtype=np.float32).T
        matrix = torch.from_numpy(matrix)
        shift = torch.tensor([0., 128., 128.])
        if image.is_cuda:
            matrix=matrix.cuda()
            shift=shift.cuda()
        image = image.permute(0, 2, 3, 1)
        # image = image.unsqueeze(0)
        # print(image.shape, matrix.shape)
        result = torch.tensordot(image, matrix, dims=1) + shift
        # result = torch.from_numpy(result)
        result = result.permute(0, 3, 1, 2)
        return result

    def ycbcr_to_rgb_jpeg(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        matrix = np.array(
            [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
            dtype=np.float32).T
        matrix = torch.from_numpy(matrix)
        shift = torch.tensor([0., -128., -128.])
        if tensor.is_cuda:
            matrix=matrix.cuda()
            shift=shift.cuda()
        result = torch.tensordot(tensor + shift, matrix, dims=1)
        result = result.permute(0, 3, 1, 2)
        return result
    #dct离散傅里叶变换
    def dct(self, spatial,is_residual=False):
        spatial *= 255#逆归一化还原原始像素
        # if self.to_ycbcr:
        #     spatial = self.rgb_to_ycbcr_jpeg(spatial)
        #如果是帧内压缩，需要将其变为-128-128范围，如果是帧间压缩，对残差做dct变换，不需要变换范围
        if is_residual:
            spatial/=2
        elif self.to_ycbcr:
            #spatial = self.rgb_to_ycbcr_jpeg(spatial)
            spatial -= 128#从 0-255变成-128-128
        real_size = (spatial.shape[2], spatial.shape[3])
        self.real_size = real_size   
        spatial = self.padding(spatial)#1，3，1024，2048    b

        padding_size = (spatial.shape[2], spatial.shape[3])
        im_blocks = blockify(spatial, 8)#分成8*8的块，一共（1024/8）（2048/8）块
        dct_blocks = block_dct(im_blocks)#对每个block做dct      c
        dct = deblockify(dct_blocks, padding_size)#组合     d
        return dct     
    def residual_dct(self,spatial):
        spatial*=255
        if self.to_ycbcr:
            spatial = self.rgb_to_ycbcr_jpeg(spatial)
        dct_blocks = block_dct(spatial)#对每个block做dct
        return dct_blocks
    def idct(self, coeff,is_residual=False):
        #self.t6=time.time()
        padding_size = (coeff.shape[2], coeff.shape[3])
        dct_blocks = blockify(coeff, 8)
        im_blocks = block_idct(dct_blocks)
        spatial = deblockify(im_blocks, padding_size)
        spatial = self.depadding(spatial)
        if is_residual:
            spatial*=2
            spatial = spatial.clamp(-255, 255)
        elif self.to_ycbcr:
            spatial += 128
            spatial = spatial.clamp(0, 255)
            #spatial = self.ycbcr_to_rgb_jpeg(spatial)
        # if self.to_ycbcr:
        #     spatial = self.ycbcr_to_rgb_jpeg(spatial)
        spatial = spatial / 255
        #self.t7=time.time()
        #print("IDCT time{0}".format(self.t7-self.t6))
        return spatial           
    # def residual_compress(self,residual,tables=None):
        
    #     residual_blocks=blockify(residual,8)#残差块
    #     #对残差块进行jpeg编码 
    #     dct_blocks=self.residual_dct(residual_blocks)
    #     padding_size = (residual.shape[2], residual.shape[3])
    #     quantized_blocks = torch.round(dct_blocks / tables)#执行8*8量化，得到量化块
    #     quantized = deblockify(quantized_blocks, padding_size)#组合量化block
    #     return quantized
    def compress(self,image, tables=None,is_residual=False):#image size(1,3,1024,2048)RGB
        #self.t1=time.time()
        dct = self.dct(image,is_residual)#得到DCT系数
        #self.t2=time.time()
        #print("DCT time{0}".format(self.t2-self.t1))
        #size=jpeg.torchjpeg_compress(dct) # 计算压缩后图片的体积
        padding_size = (dct.shape[2], dct.shape[3])
        dct_blocks = blockify(dct, 8)       # d
        quantized_blocks = torch.round(dct_blocks / tables)#执行8*8量化，得到量化块  e
        quantized = deblockify(quantized_blocks, padding_size)#组合量化block    e
        #self.t3=time.time()
        #print("quantized time{0}".format(self.t3-self.t2))
        return quantized

    def decompress(self, quantized, tables=None, normalize=False):
        padding_size = (quantized.shape[2], quantized.shape[3])
        quantized_blocks = blockify(quantized, 8)
        dequantized_blocks = quantized_blocks * tables
        dct_blocks = deblockify(dequantized_blocks, padding_size)
        image = self.idct(dct_blocks)
        if normalize:
            image = self.normalize(image)
        return image

    def dequantize(self, quantized, tables=None, normalize=False):
        #self.t4=time.time()
        padding_size = (quantized.shape[2], quantized.shape[3])    #e
        quantized_blocks = blockify(quantized, 8)     
        dequantized_blocks = quantized_blocks * tables    #f
        dct_blocks = deblockify(dequantized_blocks, padding_size)
        #self.t5=time.time()
        #print("dequantize time{0}".format(self.t5-self.t4))
        return dct_blocks

    def normalize(self, tensor):
        tensor = tensor.squeeze(0)
        tensor = transforms.Normalize(mean=info['mean'], std=info['std'])(tensor)
        tensor = tensor.unsqueeze(0)
        return tensor

    def self_compress(self, dct, gradient):
        padding_size = (dct.shape[2], dct.shape[3])
        dct_blocks = blockify(dct, 8)
        quantized_blocks = torch.round(dct_blocks / tables)
        quantized = deblockify(quantized_blocks, padding_size)
        return quantized

    def self_decompress(self, quantized, tables=None, normalize=False):
        padding_size = (quantized.shape[2], quantized.shape[3])
        quantized_blocks = blockify(quantized, 8)
        dequantized_blocks = quantized_blocks * tables
        dct_blocks = deblockify(dequantized_blocks, padding_size)
        image = self.idct(dct_blocks)
        if normalize:
            image = self.normalize(image)
        return image
    
    def gen_huffman_tables(self, lengths, elements):
        table = {}
        lengths = list(lengths)
        bit = None
        for index in range(len(lengths)):
            l = index + 1
            for i in range(lengths[index]):
                if bit is None:
                    bit = -1
                v = elements.pop(0)
                bit += 1
                table[v] = f'{bit:0{l}b}'
                # print(v, f'{bit:0{l}b}')
                # input()
                # print(v, l, f'{bit:0{l}b}')
            if bit is not None:
                bit = 2*(bit+1) - 1
        # print(table)
        return table

    def encode_huffman(self, tensor, subsampling=False):
        total_len = ''
        last_dc = [0, 0, 0]
        end = ['1010', '00', '00']
        if subsampling:
            mcu_size = [16, 8, 8]
        else:
            mcu_size = [8, 8, 8]
        Y_coeff  = blockify(tensor[0].unsqueeze(0).unsqueeze(0), mcu_size[0])
        U_coeff  = blockify(tensor[1].unsqueeze(0).unsqueeze(0), mcu_size[1])
        V_coeff  = blockify(tensor[2].unsqueeze(0).unsqueeze(0), mcu_size[2])
        for j in range(Y_coeff.shape[2]):           
            if subsampling:
                mcus = [Y_coeff[0, 0, j, :8, :8], Y_coeff[0, 0, j, :8, 8:], Y_coeff[0, 0, j, 8:, :8], Y_coeff[0, 0, j, 8:, 8:],
                        U_coeff[0, 0, j], 
                        V_coeff[0, 0, j]
                        ]
                mcu_indexs = [0, 0, 0, 0, 1, 2]

            else:
                mcus = [Y_coeff[0, 0, j, :8, :8], U_coeff[0, 0, j, :8, :8], V_coeff[0, 0, j, :8, :8],
                        #Y_coeff[0, 0, j, :8, 8:], U_coeff[0, 0, j, :8, 8:], V_coeff[0, 0, j, :8, 8:],
                        #Y_coeff[0, 0, j, 8:, :8], U_coeff[0, 0, j, 8:, :8], V_coeff[0, 0, j, 8:, :8],
                        #Y_coeff[0, 0, j, 8:, 8:], U_coeff[0, 0, j, 8:, 8:], V_coeff[0, 0, j, 8:, 8:]
                        ]
                mcu_indexs = [0, 1, 2, 
                            #0, 1, 2, 0, 1, 2
                            ]

            for i, tensor in zip(mcu_indexs, mcus):
                block = tensor.flatten()[self.zigzag]
                print(block)
                dc = block[0] - last_dc[i]
                bits = self.bits_map[int(dc)]
                last_dc[i] = block[0]
                total_len += self.dc_table[i][len(bits)] + bits
                print(len(total_len), dc, self.dc_table[i][len(bits)], bits)
                input()
                zeros_num = 0
                for k in range(1, 64):
                    ac = block[k]
                    if ac == 0:
                        zeros_num += 1
                    else:
                        while zeros_num > 15:
                            total_len += self.ac_table[i][0xF0]
                            print(len(total_len), ac, self.ac_table[i][0xF0])
                            zeros_num -= 16
                        bits = self.bits_map[int(ac)]
                        l = len(bits)
                        total_len += self.ac_table[i][(zeros_num << 4) + l]
                        total_len += bits
                        print(len(total_len), ac, self.ac_table[i][(zeros_num << 4) + l], bits)
                        input()
                        zeros_num = 0
                total_len += end[i]
                print(len(total_len), end[i])
                zeros_num = 0 
        return total_len

    def len_encode_huffman(self, tensor, subsampling=False):  
        total_len = 0
        last_dc = [0, 0, 0]
        end = ['1010', '00', '00']
        if subsampling:
            mcu_size = [16, 8, 8]
        else:
            mcu_size = [8, 8, 8]
        Y_coeff  = blockify(tensor[:, 0].unsqueeze(0), mcu_size[0])
        U_coeff  = blockify(tensor[:, 1].unsqueeze(0), mcu_size[1])
        V_coeff  = blockify(tensor[:, 2].unsqueeze(0), mcu_size[2])
        for j in range(Y_coeff.shape[2]):           
            if subsampling:
                mcus = [Y_coeff[0, 0, j, :8, :8], Y_coeff[0, 0, j, :8, 8:], Y_coeff[0, 0, j, 8:, :8], Y_coeff[0, 0, j, 8:, 8:],
                        U_coeff[0, 0, j], 
                        V_coeff[0, 0, j]
                        ]
                mcu_indexs = [0, 0, 0, 0, 1, 2]

            else:
                mcus = [Y_coeff[0, 0, j, :8, :8], U_coeff[0, 0, j, :8, :8], V_coeff[0, 0, j, :8, :8],
                        #Y_coeff[0, 0, j, :8, 8:], U_coeff[0, 0, j, :8, 8:], V_coeff[0, 0, j, :8, 8:],
                        #Y_coeff[0, 0, j, 8:, :8], U_coeff[0, 0, j, 8:, :8], V_coeff[0, 0, j, 8:, :8],
                        #Y_coeff[0, 0, j, 8:, 8:], U_coeff[0, 0, j, 8:, 8:], V_coeff[0, 0, j, 8:, 8:]
                        ]
                mcu_indexs = [0, 1, 2, 
                            #0, 1, 2, 0, 1, 2
                            ]

            for i, tensor in zip(mcu_indexs, mcus):
                block = tensor.flatten()[self.zigzag]
                # print(block)
                dc = block[0] - last_dc[i]
                bits = self.bits_map[int(dc)]
                last_dc[i] = block[0]
                total_len += len(self.dc_table[i][len(bits)] + bits)
                # print(len(total_len), dc, self.dc_table[i][len(bits)], bits)
                # input()
                zeros_num = 0
                for k in range(1, 64):
                    ac = block[k]
                    if ac == 0:
                        zeros_num += 1
                    else:
                        while zeros_num > 15:
                            total_len += len(self.ac_table[i][0xF0])
                            # print(len(total_len), ac, self.ac_table[i][0xF0])
                            zeros_num -= 16
                        bits = self.bits_map[int(ac)]
                        l = len(bits)
                        total_len += len(self.ac_table[i][(zeros_num << 4) + l])
                        total_len += len(bits)
                        # print(len(total_len), ac, self.ac_table[i][(zeros_num << 4) + l], bits)
                        # input()
                        zeros_num = 0
                total_len += len(end[i])
                # print(len(total_len), end[i])
                zeros_num = 0 
        return total_len/1024/8


    def torchjpeg_compress(self, tensor, path='./tmp/test_comp.jpg'):#计算压缩后 图像体积 大小 tensor（１，３，ｈ，ｗ）
        dimensions = torch.tensor([[1024, 2048],
                                  [1024, 2048],
                                  [1024, 2048]], dtype=torch.int32)
        coefficients = torch.ShortTensor(1, 3, 1024//8, 2048//8, 8, 8)#1,3,128,256,8,8
        tensor = torchjpeg.dct.blockify(tensor, 8)#1,3,block num,8,8
        coefficients[:] = tensor.reshape(1, 3, 1024//8, 2048//8, 8, 8)#将原始图像 重新划分block#1,3,128,256,8,8
        quantization = torch.ones(3, 8, 8, dtype=torch.int16)
        torchjpeg.codec.write_coefficients(path, dimensions, quantization, coefficients[:, 0], coefficients[:, 1:])#UV:coefficients[:, 1:](1,2,128,256,8,8)
        return os.path.getsize(path)/1024#KByte

if __name__ == "__main__":
    # from torchvision.transforms.functional import to_tensor
    # im = to_tensor(Image.open("./img/frankfurt_000000_000294_leftImg8bit.png"))
    # im = im.unsqueeze(0)  
    # # # print(im)
    jpeg = PyJPEG(True)
    # q = torch.load('./pth/plain_q_tables.pth')
    # qtables = q[:, 200, :]
    # compressed = jpeg.compress(im.clone(), qtables.reshape(1, 3, 1, 8, 8))
    # bits = jpeg.huffman(compressed)
    # print(bits/1024/8)
    import torchjpeg.codec
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    # im = to_tensor(Image.open("./img/frankfurt_000000_000294_leftImg8bit.png"))
    # im = im.unsqueeze(0) 
    # im = to_tensor(Image.open("./img/frankfurt_000000_000294_leftImg8bit.png"))
    # im.save()
    dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients("./img/vs_test.jpg")
    # torchjpeg.codec.write_coefficients('./img/test_jpeg90.jpg', dimensions, quantization, Y_coefficients, CbCr_coefficients)
    # input()
    Y_coefficients = Y_coefficients.reshape(1, 1, Y_coefficients.shape[1]*Y_coefficients.shape[2], 8, 8).float()
    Y_coefficients = deblockify(Y_coefficients, (1024, 2048))
    Y_coefficients = Y_coefficients[0, 0]
    CbCr_coefficients = CbCr_coefficients.reshape(1, 2, CbCr_coefficients.shape[1]*CbCr_coefficients.shape[2], 8, 8).float()
    CbCr_coefficients = deblockify(CbCr_coefficients, (1024, 2048))
    U_coefficients = CbCr_coefficients[0, 0]
    V_coefficients = CbCr_coefficients[0, 1]
    total = jpeg.encode_huffman(Y_coefficients, U_coefficients, V_coefficients, subsampling=False)
    print(len(total))

    from struct import unpack

    from JPEGDecoder import JPEGDecoder
    img = JPEGDecoder("./img/vs_test.jpg")
    data = img.decode()
    num = 0
    last = None
    for i in unpack("B"*100, data[:100]):
        if last == 0xFF and i == 0x00:
            continue
        print(f'{i:08b}', f'{num*8+1}-{(num+1)*8}')
        if f'{i:08b}' != total[:8]:
            print('error')
            break
        total = total[8:]
        num += 1
        last = i
    # # Y_coefficients = Y_coefficients.reshape(1, Y_coefficients.shape[1]*Y_coefficients.shape[2], 8, 8)
    # test = torch.zeros(1, 10, 8, 8)
    # test[0, 0] = Y_coefficients[0, 0, 0]
    # test[0, 1] = Y_coefficients[0, 0, 1]
    # test[0, 2] = Y_coefficients[0, 1, 0]
    # test[0, 3] = Y_coefficients[0, 1, 1]
    # test[0, 4] = CbCr_coefficients[0, 0, 0]

    # a = jpeg.encode_huffman(test, 0)
    # print(a)
    # input()
    # CbCr_coefficients = CbCr_coefficients.reshape(2, CbCr_coefficients.shape[1]*CbCr_coefficients.shape[2], 8, 8)
    # # print(CbCr_coefficients[0, 0])
    # a += jpeg.single_huffman(CbCr_coefficients[0].unsqueeze(0), 1)
    # a += jpeg.single_huffman(CbCr_coefficients[1].unsqueeze(0), 1)
    # print(a/8/1024)
    # torchjpeg.codec.write_coefficients('save.jpg', dimensions, quantization, Y_coefficients, CbCr_coefficients)
    # r = jpeg.decompress(dct, torch.ones(1, 3, 1, 8, 8))
    # print(r)
    # jpeg = PyJPEG(True)
