import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2


class CrossAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.key1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        self.query2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.key2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        # self.gamma = nn.Parameter(torch.zeros(1))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=-1)

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU())  # conv_f

    def forward(self, input1, input2):
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1).view(batch_size, -1, height * width)
        k1 = self.key1(input1).view(batch_size, -1, height * width)
        v1 = self.value1(input1).view(batch_size, -1, height * width)
        a1 = self.sigmoid(self.linear(self.avgpool(input1)))

        q2 = self.query2(input2).view(batch_size, -1, height * width)
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)
        a2 = self.sigmoid(self.linear(self.avgpool(input2)))

        attn_matrix1 = torch.bmm(k1, q2.permute(0, 2, 1))
        attn_matrix1 = self.softmax(attn_matrix1)
        out1 = torch.bmm(attn_matrix1.permute(0, 2, 1), v1)
        out1 = out1.view(*input1.shape)
        # out1 = self.gamma * out1 + input1
        out1 = out1 * a1.expand_as(out1) + input1

        attn_matrix2 = torch.bmm(k2, q1.permute(0, 2, 1))
        attn_matrix2 = self.softmax(attn_matrix2)
        out2 = torch.bmm(attn_matrix2.permute(0, 2, 1), v2)
        out2 = out2.view(*input2.shape)
        # out2 = self.gamma * out2 + input2
        out2 = out2 * a2.expand_as(out2) + input2

        feat_sum = self.conv_cat(torch.cat([out1, out2], 1))
        return feat_sum


if __name__ == '__main__':
    input = torch.randn(8, 512, 32, 32).cuda()
    att = CrossAtt(in_channels=512, out_channels=512).cuda()
    output = att(input, input)
    print(output.shape)
