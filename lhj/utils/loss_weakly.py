import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()

    def compute_similarity_matrix(self, x, sp, num):
        # print(x.shape)
        # print(sp.shape)

        sp = sp.repeat(1, x.shape[1], 1, 1).cuda()  # B, C, H, W
        mean_values = torch.zeros([x.shape[0], num[0], x.shape[1]], dtype=torch.float32).cuda()
        # similarity_matrix = torch.zeros([x.shape[0], num[0], num[0]], dtype=torch.float32)

        # 计算超像素平均值
        for i in range(x.shape[0]):
            for j in range(num[i]):
                mean_values[i][j] = torch.mean(x[i][sp[i] == j], dim=0)
        # 计算相似性矩阵
        tensor = mean_values.unsqueeze(2)
        diff = tensor - tensor.permute(0, 2, 1, 3)
        # print(torch.norm(diff, dim=-1))
        similarity_matrix = 1 - (torch.norm(diff, dim=-1) / math.sqrt(x.shape[1]))

        # for b in range(x.shape[0]):
        #     print(b)
        #     for i in range(num[0]):
        #         for j in range(num[0]):
        #             if i != j:
        #                 # 直接使用保存的超像素平均值
        #                 pixels_i = mean_values[b][i]
        #                 pixels_j = mean_values[b][j]
        #                 # print((pixels_i - pixels_j).shape)
        #                 # 计算颜色相似性
        #                 similarity = 1 - (torch.norm(pixels_i - pixels_j) / pixels_i.shape[0])
        #                 # 更新相似性矩阵
        #                 similarity_matrix[b][i][j] = similarity
        return similarity_matrix

    def forward(self, input, feature, sp, num):
        feature = F.interpolate(feature, size=input.shape[-1], mode='bilinear', align_corners=True)
        # print(input.shape)
        # print(feature.shape)
        # print(sp.shape)
        # print(num)
        mat1 = self.compute_similarity_matrix(input, sp, num)
        mat2 = self.compute_similarity_matrix(feature, sp, num)
        # print(mat1.shape)
        # print(mat2.shape)

        # print(torch.max(mat1))
        # print(torch.max(mat2))

        # return torch.norm(mat1 - mat2) / num[0]
        return nn.L1Loss()(mat1, mat2)


class SaliencyLoss(nn.Module):
    def __init__(self):
        super(SaliencyLoss, self).__init__()

    def forward(self, feature, salient_class):
        # print(feature.shape)
        # print(salient_class.shape)

        loss = 0.0

        for i in range(feature.shape[0]):
            cls = 0 if salient_class[i][0] > salient_class[i][1] else 1  # 预测的类别
            total = torch.sum(feature[i] ** 2, dim=(-3, -2, -1)) / (feature.shape[-1])**2
            # unchanged = torch.sum(feature[i][0] ** 2, dim=(-2, -1)) / (feature.shape[-1])**2
            # changed = torch.sum(feature[i][1] ** 2, dim=(-2, -1)) / (feature.shape[-1])**2
            # print(total)
            # print(unchanged)
            # print(changed)

            salient_change = torch.sum(feature[i][cls] ** 2, dim=(-2, -1)) / (feature.shape[-1])**2

            loss += (torch.log(total) - torch.log(salient_change))
        return loss