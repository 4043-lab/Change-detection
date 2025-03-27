import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

# class ChannelPool(nn.Module):
#     def forward(self, x):
#         return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
#
# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

# class SpatialGate(nn.Module):
#     def __init__(self):
#         super(SpatialGate, self).__init__()
#         kernel_size = 7
#         self.compress = ChannelPool()
#         self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.spatial(x_compress)
#         scale = F.sigmoid(x_out) # broadcasting
#         return x * scale
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

# channel-spatial features fusion module
class CSEF(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False),
        #     nn.Sigmoid()
        # )
        self.ConvRL = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Linear(channel // reduction, channel, bias=False)
        self.conv2 = nn.Linear(channel // reduction, channel, bias=False)

        # kernel_size = 7
        # self.compress1 = ChannelPool()
        # self.compress2 = ChannelPool()
        # self.spatial1 = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        # self.spatial2 = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        self.spatialAttention=SpatialAttention()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        b, c, _, _ = x1.size()
        #先进行自身的空间注意力

        x1 = x1 * self.spatialAttention(x1)+x1
        x2 = x2 * self.spatialAttention(x2)+x2


        # x1_compress = self.compress1(x1)
        # x1_out = self.spatial1(x1_compress)
        # scale = F.sigmoid(x1_out)  # broadcasting
        # x1 = x1 * scale
        #
        # x2_compress = self.compress2(x2)
        # x2_out = self.spatial2(x2_compress)
        # scale = F.sigmoid(x2_out)  # broadcasting
        # x2 = x2 * scale

        #再强化变化特征
        x = x1 + x2
        y = self.avg_pool(x).view(b, c)
        y = self.ConvRL(y)
        y1 = self.conv1(y).unsqueeze(0)
        y2 = self.conv2(y).unsqueeze(0)
        y_soft = torch.cat([y1,y2], dim=0)
        y_soft = torch.softmax(y_soft,dim=0)
        # print('y_soft', y_soft.size())
        y1 = y_soft[0].view(b, c, 1, 1)
        y2 = y_soft[1].view(b, c, 1, 1)
        # print('y1', y1.size())
        x1 = x1 * y1.expand_as(x1)
        x2 = x2 * y2.expand_as(x2)



        return x1 + x2


if __name__ == '__main__':
    input1 = torch.randn(2, 256, 128, 128) #2, 8, 256, 256
    input2 = torch.randn(2, 256, 128, 128)
    attn = CSEF(channel=256, reduction=2)
    output = attn(input1,input2)
    print(output.shape)