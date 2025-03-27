import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

old_dir = './LEVIR-CD/test/label'
new_dir = './LEVIR-CD_256/test/label'

if not os.path.exists(new_dir):
    os.makedirs(new_dir)

file_list = os.listdir(old_dir)

# print('图片数量：',len(file_list))
for i in tqdm(range(len(file_list))):
    file = file_list[i]
    # type = file.split('_')[0]
    # file_num = int(file.split('_')[1].split('.')[0])
    file_path = old_dir + '/' + file
    # print(file_path)
    # img = cv2.imread(file_path)
    img = cv2.imread(file_path, flags=0)
    # img1 = img[0:256,0:256,:]
    # print(img1.shape)
    # cv2.imshow('img1',img1)
    # cv2.waitKey(0)
    num = 1
    for i in range(4):
        for j in range(4):
            img_small = img[i * 256: (i + 1) * 256, j * 256: (j + 1) * 256]
            new_name = file.split('.')[0] + '_' + str(num) + '.png'
            num += 1
            cv2.imwrite(new_dir + '/' + new_name, img_small)