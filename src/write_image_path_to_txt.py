# -*- coding: utf-8 -*-
# 生成文件夹中所有文件的路径到txt
import os
def listdir(path, list_name):  # 传入存储的list
    print(1)
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
 
list_name=[]
path='../datasets/back_test/compare-test/qp28'  #文件夹路径
print(os.listdir(path))
listdir(path,list_name)
print(list_name)
list_name.sort()
 
with open('../datasets/back_test/compare-test/qp28.txt','w') as f:     #要存入的txt
    write=''
    for i in list_name:
        write=write+str(i)+'\n'
    f.write(write)