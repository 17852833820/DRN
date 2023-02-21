# 环境

建议创建虚拟环境，需安装修改的torchjpeg包，pip install torchjpeg-0.9.20-cp38-cp38-linux_x86_64.whl
详细核心代码注释见segment-stac-prop.py文件的第491行test文件，所有segment-*.py均是test函数有所不同

# 文件说明
数据集文件夹为 ./datasets/testimg/
        testing/test_images.txt  测试用的图像路径列表
        testing/test_labels.txt 测试用的语义分割GT的路径列表
        testing/leftImg8bit 测试用的图像
        testing/gtFine 测试用的语义分割GT

DIS/：用于计算密集光流
        test_DIS.py: 实现计算光流
                    'Flow '
                    'warp_flow'
                    'psnr'
                    'accuracy_np'
                    'load_label'
        DIS/flow/:存储的光流文件
pyjpeg/：jpeg编解码模块，包含分块 ，dct变换和逆变换
        'blockify':
        'deblockify':
        'block_dct':
        'block_idct':
        'PyJPEG':
torchjpeg-0.9.20: c++ jpeg编解码器接口，只用于linux系统

selector.py ：量化表选择器，实现根据梯度选择量化表
drn:语义分割DRN模型
data_transforms: 数据转换模块，实现随机裁剪、缩放、旋转、翻转、正则化、padding等
pretrained :语义分割预训练模型
                ./drn_d_22_cityscapes.pth

pth/jpeg_qtables.pt jpeg量化表


segment-raw.py: 原始drn
segment-grace-prop.py: 带帧间传播的grace
segment-grace.py: 固定普通grace
segment-jpeg.py: 带帧间传播的jpeg
segment-jpeg-prop.py: 固定普通jpeg
segment-stac-prop.py: stac

segment-test.py: 进行一些测试验证实验
segment-test-new.py：ours 

# 运行
## 运行stac 
```python segment-stac-prop.py test -d ./datasets/testimg/ -c 19 --arch drn_d_22 --pretrained ./drn_d_22_cityscapes.pth --phase test --batch-size 1 --with-gt```
```      
        cmd                 test/train
        -d                  数据集文件夹 
        -l
        -c                  classes numbers
        -s                  crop-size, default=0
        --arch              选择网络结构类型：'C','D'，对应model_name
        -- batch-size       input batch size for training (default: 64)
        --step              default=200
        --epoch             number of epochs to train 
        --lr                default: 0.01
        --lr-mode           default"step
        --momentum              SGD momentum (default: 0.9)
        --weight-decay      weight decay (default: 1e-4)
        -e
        --resume            path to latest checkpoint (default: none)
        --pretrained        use pre-trained model预训练模型路径
        --save_path         output path for training checkpoints
        --save_iter         number of training iterations between'
                             'checkpoint history saves'
        -j                  进程数量，默认8
        --load-release
        --phase             默认val
        --random-scale      默认0 随机缩放
        --random-rotate     默认0 随机旋转
        --bn-sync
        --ms                与创建datasets有关，dataset模式选择
        --with-gt           是否有ground truth
        --test-suffix```



