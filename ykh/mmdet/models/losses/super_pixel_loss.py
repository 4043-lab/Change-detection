
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from ..builder import LOSSES

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()

    def compute_similarity_matrix(self, x, sp, num):
        # print(sp.shape)
        # print('x')
        # print(x.shape)
        # print(x)

        sp = sp.repeat(1, x.shape[1], 1, 1).cuda()  # B, C, H, W
        mean_values = torch.zeros([x.shape[0], num[0], x.shape[1]], dtype=torch.float32).cuda()
        # similarity_matrix = torch.zeros([x.shape[0], num[0], num[0]], dtype=torch.float32)
        one=torch.ones(x.shape[0],x.shape[1],x.shape[2],x.shape[3])

        # 计算超像素平均值
        for i in range(x.shape[0]):
            for j in range(num[i]):
                mean_values[i][j] = torch.mean(x[i][sp[i] == j], dim=0)
                # print(torch.sum(one[i][sp[i] == j],dim=0))
                # print("i:{},j:{}".format(i,j))
                # print(mean_values[i][j])
        # 计算相似性矩阵
        mean_values.nan_to_num_(nan=0.0)
        tensor = mean_values.unsqueeze(2)
        # print("tensor")
        # print(tensor)
        diff = tensor - tensor.permute(0, 2, 1, 3)
        # print("diff")
        # print(diff)
        # print(torch.norm(diff, dim=-1))
        similarity_matrix = 1 - (torch.norm(diff, dim=-1) / math.sqrt(x.shape[1]))
        # print('similarity_matrix')
        # print(similarity_matrix)

        return similarity_matrix

    def forward(self, input, feature, sp, num):
        feature = F.interpolate(feature, size=input.shape[-1], mode='bilinear', align_corners=True)
        # print(input.shape)
        # print(feature.shape)
        # print(sp.shape)
        # print(num)
        mat1 = self.compute_similarity_matrix(input, sp, num)
        mat2 = self.compute_similarity_matrix(feature, sp, num)
        # print(mat1)
        # print(mat2)
        # print(mat1.shape)
        # print(mat2.shape)

        # print(torch.max(mat1))
        # print(torch.max(mat2))

        # return torch.norm(mat1 - mat2) / num[0]
        return nn.L1Loss()(mat1, mat2)

def get_superpixel_label(img):
    superpixel = cv2.ximgproc.createSuperpixelSEEDS(image_width=img.shape[1], image_height=img.shape[0], image_channels=3, num_levels=10,
                                              num_superpixels=256)
    superpixel.iterate(img, 10)  # 迭代次数，越大效果越好
    label_superpixel = superpixel.getLabels()                       #获取超像素标签
    number_superpixel = superpixel.getNumberOfSuperpixels()         #获取超像素数目

    #分割可视化
    # mask_seeds = superpixel.getLabelContourMask()
    # mask_inv_seeds = cv2.bitwise_not(mask_seeds)
    # img_seeds = cv2.bitwise_and(img, img, mask=mask_inv_seeds)
    # color_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # color_img[:] = (0, 255, 0)
    # result_ = cv2.bitwise_and(color_img, color_img, mask=mask_seeds)
    # result = cv2.add(img_seeds, result_)
    # cv2.imwrite(r"/data1/ykh/seg/box2cd_CLIP/BoxInstSeg/imageMask.png", result)
    # print("save mask image")

    return label_superpixel, number_superpixel

#self_supervised_consistency_regularization_loss
@LOSSES.register_module()
class SuperPixelLoss(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(SuperPixelLoss, self).__init__()
        self.loss_weight=loss_weight

    def forward(self,image, feature, img_metas, imgName='filename1'):

        if self.loss_weight==0.0:
            return torch.tensor(0.0).cuda()
        sp_list = []
        num_list = []
        B=image.shape[0]
        image=nn.AvgPool2d(kernel_size=4,stride=4)(torch.abs(image))
        for i in range(B):
            #基于image 实现
            # img_tensor=image[i].cpu()
            # numpy_img=img_tensor.numpy().transpose((1,2,0))
            # numpy_img=(numpy_img*255).astype(np.uint8)
            # img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)

            #基于img_metes实现
            # print(img_metas)
            fillename1 = img_metas[i]['filename1']
            img1 = Image.open(fillename1)
            img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
            img1 = cv2.pyrDown(img1)
            img1 = cv2.pyrDown(img1)

            fillename2 = img_metas[i]['filename2']
            img2 = Image.open(fillename2)
            img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
            img2 = cv2.pyrDown(img2)
            img2 = cv2.pyrDown(img2)

            img = abs(img1-img2)

            # cv2.imwrite(r"/data1/ykh/seg/box2cd_CLIP/BoxInstSeg/image.png", img)
            # print("save image")

            sp,num=get_superpixel_label(img)
            sp_list.append(torch.tensor(sp))
            num_list.append(num)

        sp_list = torch.stack(sp_list,dim=0).unsqueeze(1)
        lossFunc = ConsistencyLoss()
        loss = lossFunc(image,feature,sp_list,num_list) * self.loss_weight

        return loss

if __name__ =="__main__":
    # tensor_img=torch.rand(4,3,256,256).cuda()
    # tensor_feature=torch.rand(4,256,64,64).cuda()
    # func=SuperPixelLoss()
    # loss=func(tensor_img,tensor_feature)
    # print(loss)

    fillename1 = r"D:\1Data\Python-code\DataSet\CD\BCDD\BCDD_luo\BCDD\train\A\7227.png"
    img1 = Image.open(fillename1)
    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img1 = cv2.pyrDown(img1)
    img1 = cv2.pyrDown(img1)

    fillename2 = r"D:\1Data\Python-code\DataSet\CD\BCDD\BCDD_luo\BCDD\train\B\7227.png"
    img2 = Image.open(fillename2)
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    img2 = cv2.pyrDown(img2)
    img2 = cv2.pyrDown(img2)

    img = abs(img1 - img2)

    # cv2.imwrite(r"C:\Users\Administrator\Desktop/image.png", img)
    # print("save image")