import re

import huffman
import torch

from pyjpeg import blockify, deblockify


class Selector(object):
    def __init__(self, B=0.9e-4, pattern='jpeg', region=1):
        self.B = B#每个像素点的最大loss bound
        self.region = region
        self.region_size = region * 8
        self.table_rate=0.99
        self.table1_threshold=((80,100),(80,100),(80,100))
        if pattern == 'jpeg':#使用jpeg量化表 
            self.tables1Y = torch.load('./pth/jpeg_qtables.pth')[self.table1_threshold[0][0]:self.table1_threshold[0][1]]
            self.tables1U = torch.load('./pth/jpeg_qtables.pth')[self.table1_threshold[1][0]:self.table1_threshold[1][1]]
            self.tables1V = torch.load('./pth/jpeg_qtables.pth')[self.table1_threshold[2][0]:self.table1_threshold[2][1]]
 
            self.tables2 = torch.load('./pth/jpeg_qtables.pth')#[30:100]#num，3，8，8
            #print(self.tables.shape)
        else:
            self.tables = torch.load('./pth/plain_q_tables.pth')[15:25]
        self.choices1Y = self.tables1Y.shape[0]#可选量化表的数量
        self.choices1U = self.tables1U.shape[0]#可选量化表的数量
        self.choices1V = self.tables1V.shape[0]#可选量化表的数量

        self.choices2 = self.tables2.shape[0]#可选量化表的数量
        self.prob1Y = torch.zeros(self.choices1Y)
        self.prob1U = torch.zeros(self.choices1U)
        self.prob1V = torch.zeros(self.choices1V)

        self.prob2 = torch.zeros(self.choices2)#每个量化表的概率
    def update_params(self,B,table1_threshold,table_rate):
            self.B=B
            self.table1_threshold=table1_threshold
            self.table_rate=table_rate
    def choose_table2_baseline3(self,s,g,qp):
            import numpy as np
            h, w = s.shape[2:]
            g = blockify(g, 8 * self.region)#1，3，32768，8，8
            grad_sum=torch.sum(g,dim=[3,4])
            table_indexs = torch.zeros(3, g.shape[2])#（3，block数量）为每个block选择一个量化等级
            for i in range(3):#yuv每个通道
                media=torch.median((grad_sum[0][i][:]))
                for index in range(grad_sum.shape[2]):
                    if grad_sum[0][i][index]>media:
                        table_indexs[i][index]=min(99,qp+5)
                    else:
                        table_indexs[i][index]=max(0,qp-5)
            table_indexs = table_indexs.reshape(1, 3, -1, 1, 1)

            table_indexs=table_indexs.repeat(1, 1, 1, self.region_size, self.region_size)#1,3,num,8,8

            #table_indexs[0,:,:,0,0]=mean_dt.reshape(3,1).repeat(1, s.shape[2])
            table_indexs = deblockify(table_indexs, (h, w))#1,3,h,w
            table_indexs = table_indexs.squeeze(0).permute(1, 2, 0).numpy()
            return table_indexs #H,W,3 每个block中每个像素点的量化等级相同
    def choose_table2(self,q1,iter,jpeg,rec_image1,image,compressed,residual_jpeg,residual_img,cur_map_image):#jpeg,table_indexs1,cur_img,compressed,residual_img
        """
        input:
                table_index1: array,H,W,3
                image :tensor,1,3,H,W(0-1 yuv)
                compressed:tensor,1,3,H,W
                residual_img:tensor,1,3,H,W(-1,1)
        """
        """残差压缩和重建,为每个block选择量化表 """
        #import time
        #t1=time.time()
        h,w=residual_img.shape[2:]
        #residual_img=blockify(residual_img,8*self.region).cuda()#1,3,num,8,8
        compressed=blockify(compressed,8*self.region)
        compressed[0,:,:,0,0]=0
        cur_map_image=cur_map_image
        block_num=int(int(h/(8*self.region))*int(w/(8*self.region)))
        #cur_map_image=blockify(cur_map_image.float(),8*self.region).cuda()
        # table_index1=torch.from_numpy(table_index1).permute(2,0,1).unsqueeze(0)#1,3,H,W
        # table_index1=blockify(table_index1,8*self.region).cuda().squeeze(0)#.permute(1,0,2,3)#block num,3,8,8
        table_indexs = torch.zeros(block_num,3)#（block数量,3,8,8）
        filters=torch.arange(0, self.choices2, 1).reshape(self.choices2,1,1).repeat(1,3,block_num)#choices,3,num
        dct = jpeg.dct(residual_img.clone(),is_residual=True)# 1,C,H,W
        dct_jpeg=jpeg.dct(residual_jpeg.clone(),is_residual=True)
        e=(self.tables2).reshape(self.choices2, 1, 3, 1,8, 8).repeat(1, 1,1,block_num, self.region, self.region)#choices,1,3,num,8,8
        e=torch.cat([deblockify(e[level],(h,w)) for level in range(len(e))],dim=0).reshape(len(e),1,3,h,w)#(choices,1,3,h,w)
        quantized_blocks =torch.cat( [torch.round(dct[0] / e[level][0] ) for level in range(len(filters))],dim=0).reshape(len(filters),3,h,w)#(choices,3,h,w)
        dequantized_blocks=torch.cat([ quantized_blocks[index] * e[index][0] for index in range(len(filters))],dim=0).reshape(len(filters),3,h,w)#choices,3,h,w
        #residuals=torch.cat([jpeg.idct(dequantized_blocks[index].unsqueeze(0),is_residual=True) for index in range(len(filters))],dim=0).reshape(len(filters),3,h,w)#choice,8,8
        #errors=torch.cat([image[0]-(residuals[index]+cur_map_image[0]) for  index in range(len(filters))],dim=0).reshape(len(filters),3,h,w).cuda()#choices,3,h,w
        #errors=torch.cat([blockify(errors[level].unsqueeze(0),self.region*8) for level in range(len(filters))],dim=0).reshape(len(filters),1,3,int((h/8)*(w/8)),self.region*8,self.region*8)
        errors=torch.cat([dequantized_blocks[index]-dct[0] for index in range(len(filters))],dim=0).reshape(self.choices2,3,h,w)
        errors=torch.cat([blockify(errors[level].unsqueeze(0),self.region*8) for level in range(len(filters))],dim=0).reshape(len(filters),1,3,block_num,self.region*8,self.region*8)
        image=blockify(image,8*self.region)
        rec_image1=blockify(rec_image1,8*self.region)
        quantized_blocks=torch.cat([blockify(quantized_blocks[level].unsqueeze(0),self.region*8) for level in range(len(filters))],dim=0).reshape(len(filters),1,3,block_num,self.region*8,self.region*8)
        volume=torch.stack([torch.sum(torch.abs(quantized_blocks[level,0,:,:]),dim=[2,3]) for level in range(len(filters))])#choices,3,num
        #error_jpeg=image-rec_image1#1,3,num,8,8
        #error_jpeg=torch.sum(torch.abs(error_jpeg[0,:,:]),dim=[2,3])#3,num
        error_jpeg=blockify(dct-dct_jpeg,self.region*8)#1,3,num,8,8
        error_jpeg=torch.sum(torch.abs(error_jpeg[0,:,:]),dim=[2,3])
        block_compressed_sum=torch.sum(torch.abs(compressed[0,:,:]),dim=[2,3])#3,num
        error_sum=torch.stack([torch.sum(torch.abs(errors[level,0,:,:]),dim=[2,3]) for level in range(len(filters))])#choices,3,num
        #t2=time.time()
        #print("sum time:{0}".format(t2-t1))
        mode1=torch.le(error_sum[:,:,:]-error_jpeg[:][:].reshape(1,3,block_num).repeat(len(filters),1,1),0) #<= choices,3,num
        #import numpy as np
        #mode1=torch.ge(filters,torch.from_numpy(np.array([40])).reshape(1,1).repeat(3,int((h/8)*(w/8))).cuda())
        mode2=torch.le(volume-block_compressed_sum.reshape(1,3,block_num).repeat(len(filters),1,1),0)
        #print("mode1:{0},mode2:{1}".format((mode1==True).sum(),(mode2==True).sum()))
        #filter2=torch.masked_select(filters[],mode2)
        #mode3=torch.le(torch.stack([volume[f1,:,:] for f1 in filter1])-block_compressed_sum.reshape(1,3,int((h/8)*(w/8))).repeat(len(filter1),1,1),0)
        #filter3=torch.masked_select(filter1,mode3)
        mode=mode1*mode2
        num1=num2=num3=0
        rate=[]
        for i in range(3):
            for j in range(block_num):
                filter1=torch.masked_select(filters[:,i,j],mode1[:,i,j])#choices
                #filter1=filters
                rate.append(len(filter1))
                if (mode1[:,i,j]==True).sum()>0 : # 1
                    mode2=torch.le(torch.stack([volume[level,i,j] for level in filter1])-block_compressed_sum[i,j],0)
                    if (mode2==True).sum()==0:# 1 not 2
                        num1+=1
                        table_indexs[j][i]=filter1[torch.argmin(torch.abs(torch.stack([volume[f1,i,j] for f1 in filter1])),dim=0)]
                    else:# 1 and 2
                        num2+=1
                        filter2=torch.masked_select(filter1,mode2)
                        table_indexs[j][i]=filter2[0]
                else:
                    num3+=1
                    table_indexs[j][i]=torch.argmin(torch.abs(error_sum[:,i,j]),dim=0)#num,3
                #if table_indexs[j][i]*1.1<100:table_indexs[j][i]*=1.1
                #else:table_indexs[j][i]=99
                table_indexs[j][i]*=self.table_rate
                #table_indexs[j][i]=torch.max(( table_indexs[j][i]-10),0)
        means=torch.mean(table_indexs,dim=0)
        # for i in range(3):
        #     for j in range(block_num):
        #             table_indexs[j][i]=(table_indexs[j][i]-means[i])*0.8+means[i]
        import numpy as np
        #print("rate:{0}".format(np.mean((np.array(rate)))))
        #print("num1:{0},num2:{1},num3:{2}".format(num1,num2,num3))
        #t3=time.time()
        #print("for time:{0}".format(t3-t2))
        table_indexs=table_indexs.unsqueeze(0).permute(0,2,1).reshape(1,3,block_num,1,1).repeat(1,1,1,self.region*8,self.region*8)#1,3,num
        table_indexs=deblockify(table_indexs,(h,w))
        table_indexs=table_indexs.squeeze(0).permute(1,2,0).numpy()#h,w,3
        return table_indexs
    def choose_table(self, s, g):#选择量化表
        h, w = s.shape[2:]
        s = blockify(s, 8 * self.region)#分区，分区大小为（8*regin）（8*regin）
        g = blockify(g, 8 * self.region)#1，3，32768，8，8
        table_indexs = torch.zeros(3, s.shape[2])#（3，block数量）为每个block选择一个量化等级
        for i in range(3):#yuv每个通道
            #s_repeat = s[0, i].repeat(self.choices1, 1, 1, 1)#（choices，32768，8，8）
            # e = s_repeat * torch.round(s_repeat / self.tables1[:, i].reshape(-1, 1, 8 * self.region, 8 * self.region))
            if i==0:
                tables=self.tables1Y
                choice=self.choices1Y
            elif i==1:
                tables=self.tables1U
                choice=self.choices1U
            else:
                tables=self.tables1V
                choice=self.choices1V
            e = (tables[:, i] / 2).reshape(choice, 1, 8, 8).repeat(1, s.shape[2], self.region, self.region)#(choices,block num,region,region) q/2
            if e.is_cuda:
                s=s.cuda()
                g=g.cuda()
            e *= g[0, i]#(block num,region,region)每个量化表 每个block 每个像素点的q都乘以对应像素点的梯度：（q/2）*grad=B/M
            e = torch.sum(torch.abs(e), dim=[2, 3])#（choices，block num）每个block量化步骤造成的最坏损失<=每个block最大loss bound B
            table_indexs[i] = torch.argmin(torch.abs(e - self.B*self.region**2), dim=0)#选体积最小的量化表：每个block的量化等级（3，block num）
            #if i==1 or i==2:
              #  table_indexs[i]= table_indexs[i]*0.4
        #mean_dt=torch.mean(torch.abs(table_indexs),dim=[1])#3
        table_indexs = table_indexs.reshape(1, 3, -1, 1, 1)

        table_indexs=table_indexs.repeat(1, 1, 1, self.region_size, self.region_size)#1,3,num,8,8

        #table_indexs[0,:,:,0,0]=mean_dt.reshape(3,1).repeat(1, s.shape[2])
        table_indexs = deblockify(table_indexs, (h, w))#1,3,h,w
        table_indexs = table_indexs.squeeze(0).permute(1, 2, 0).numpy()
        return table_indexs #H,W,3 每个block中每个像素点的量化等级相同

    def extract_table1(self, table_indexs):
        h, w = table_indexs.shape[0], table_indexs.shape[1]
        table_indexs = torch.from_numpy(table_indexs).permute(2, 0, 1).unsqueeze(0)
        table_indexs = blockify(table_indexs, 8*self.region)
        table_indexs = table_indexs.reshape(1, 3, table_indexs.shape[2], -1)
        table_indexs = table_indexs.mode(dim=3)[0].squeeze(0)
        for index, value in zip(table_indexs[0], table_indexs[1]):
            self.prob1Y[int(index)] += value
        selected_tables = torch.zeros(1, 3, table_indexs.shape[1], 8, 8)
        for i in range(3):
            if i==0:
                tables=self.tables1Y
            elif i==1:
                tables=self.tables1U
            else:
                tables=self.tables1V
            selected_tables[0, i] = tables[table_indexs[i].long(), i]
        selected_tables = selected_tables.repeat(1, 1, 1, self.region, self.region)
        selected_tables = deblockify(selected_tables, (h, w))
        selected_tables = blockify(selected_tables, 8)

        return selected_tables
    def extract_table2(self, table_indexs):
        h, w = table_indexs.shape[0], table_indexs.shape[1]
        table_indexs = torch.from_numpy(table_indexs).permute(2, 0, 1).unsqueeze(0)
        table_indexs = blockify(table_indexs, 8*self.region)
        table_indexs = table_indexs.reshape(1, 3, table_indexs.shape[2], -1)
        table_indexs = table_indexs.mode(dim=3)[0].squeeze(0)
        for index, value in zip(table_indexs[0], table_indexs[1]):
            self.prob2[int(index)] += value
        selected_tables = torch.zeros(1, 3, table_indexs.shape[1], 8, 8)
        for i in range(3):
            selected_tables[0, i] = self.tables2[table_indexs[i].long(), i]
        selected_tables = selected_tables.repeat(1, 1, 1, self.region, self.region)
        selected_tables = deblockify(selected_tables, (h, w))
        selected_tables = blockify(selected_tables, 8)

        return selected_tables
    def calc_bais1(self):
        indexs = []
        counts = []
        for i in range(self.choices1Y):
            indexs.append(i)
            counts.append(self.prob1Y[i].item())
        huf = huffman.codebook(zip(indexs, counts))

        self.prob1Y = self.prob1Y/self.prob1Y.sum()
        total_len = 0
        for i in range(self.choices1Y):
            total_len = self.prob1Y[i] * len(huf[i])
        total_len *= 1024*2048*3/(8*self.region)**2/8/1024
        return total_len
    def calc_bais2(self):
        indexs = []
        counts = []
        for i in range(self.choices2):
            indexs.append(i)
            counts.append(self.prob2[i].item())
        huf = huffman.codebook(zip(indexs, counts))

        self.prob2 = self.prob2/self.prob2.sum()
        total_len = 0
        for i in range(self.choices2):
            total_len = self.prob2[i] * len(huf[i])
        total_len *= 1024*2048*3/(8*self.region)**2/8/1024
        return total_len
# def choose_table(s, g, B, region=1):
#     # candidate = torch.load('./pth/plain_q_tables.pth')
#     # tables = candidate[20:25]
#     # tables = torch.zeros(3, 3, 8, 8)
#     # tables[0] = candidate[84-1]
#     # tables[1] = candidate[87-1]
#     # tables[2] = candidate[90-1]
#     s = blockify(s, 8 * region)
#     g = blockify(g, 8 * region)
#     selected_indexs = torch.zeros(1, 3, s.shape[2] / region ** 2, 8 * region, 8 * region)
#     for i in range(3):
#         # print(s[0, i].shape, tables[:, i].shape)
#         s_repeat = s[0, i].repeat(tables.shape[0], 1, 1, 1)
#         # e = s_repeat * torch.round(s_repeat / tables[:, i].reshape(-1, 1, 8, 8))
#         e = (tables[:, i] / 2).reshape(tables.shape[0], 1, 8, 8).repeat(1, s.shape[2], 1, 1)
#         e *= g[0, i]
#         e = torch.sum(torch.abs(e), dim=[2, 3])
#         selected_indexs = torch.argmin(torch.abs(e - B), dim=0)
#         # selected_tables[0, i] = tables[selected_indexs, i]
#     return selected_indexs

if __name__ == '__main__':
    selector = Selector(B=0.9e-4, region=2)
    s = torch.load('./pth/dct_var0.pth')
    g = torch.load('./pth/grad_var0.pth')
    #根据梯度选择量化表
    table_indexs = selector.choose_table(s, g)
    selector.extract_table(table_indexs)
