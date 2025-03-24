import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from model.idea4.resnet import resnet50

class decoder(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super(decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.maxpooling = nn.MaxPool2d(2)
        self.downsample = downsample

    def forward(self, x1, x2):
        if self.downsample:
            x1 = self.maxpooling(x1)
        x = torch.cat([x1, x2], dim=1)
        out = self.conv(x)
        return out

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.resnet = resnet50(pretrained=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4096, 2048, 1, 1, 0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = decoder(256+512, 512, downsample=True)
        self.decoder2 = decoder(512+1024, 1024, downsample=True)
        self.decoder3 = decoder(1024+2048, 2048, downsample=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(2048, 256, 1, bias=False)
        self.fc2 = nn.Conv2d(256, 2, 1, bias=False)

    # def forward(self, t1, t2):
    def forward_origin(self, t):
        t1 = t[0]
        t2 = t[1]
        x1_e1, x2_e1, x3_e1, x4_e1 = self.resnet(t1)
        x1_e2, x2_e2, x3_e2, x4_e2 = self.resnet(t2)

        y1 = torch.cat([x1_e1, x1_e2], dim=1)
        y2 = torch.cat([x2_e1, x2_e2], dim=1)
        y3 = torch.cat([x3_e1, x3_e2], dim=1)
        y4 = torch.cat([x4_e1, x4_e2], dim=1)

        y1 = self.conv1(y1)
        y2 = self.conv2(y2)
        y3 = self.conv3(y3)
        y4 = self.conv4(y4)

        # print(y1.shape)
        # print(y2.shape)
        # print(y3.shape)
        # print(y4.shape)

        out1 = self.decoder1(y1, y2)
        out2 = self.decoder2(out1, y3)
        out3 = self.decoder3(out2, y4)

        cam = F.relu(self.fc1(out3))
        cam = self.fc2(cam)
        out = self.avgpool(cam).squeeze(3).squeeze(2)

        cam = F.softmax(cam, dim=1)

        return cam, out1, out2, out3, out

    def forward_affine(self, t):
        t1 = t[0]
        t2 = t[1]

        rand1 = random.uniform(0, 1)
        rand2 = random.uniform(0, 1)
        rand3 = random.uniform(0, 1)
        b, c, h, w = t1.shape
        # 翻转
        if rand1 > 0.5:
            trf1 = transforms.RandomHorizontalFlip(p=1)
            untrf1 = transforms.RandomHorizontalFlip(p=1)
            t1 = trf1(t1)
            t2 = trf1(t2)
        # 旋转
        if 0.25 < rand2 <= 0.5:
            trf2 = transforms.RandomRotation((90, 90))
            untrf2 = transforms.RandomRotation((270, 270))
            t1 = trf2(t1)
            t2 = trf2(t2)
        elif 0.5 < rand2 <= 0.75:
            trf2 = transforms.RandomRotation((180, 180))
            untrf2 = transforms.RandomRotation((180, 180))
            t1 = trf2(t1)
            t2 = trf2(t2)
        elif 0.75 < rand2:
            trf2 = transforms.RandomRotation((270, 270))
            untrf2 = transforms.RandomRotation((90, 90))
            t1 = trf2(t1)
            t2 = trf2(t2)
        # 缩放
        if rand3 > 0.5:
            trf3 = transforms.Resize(int(h / 2))
            # untrf3 = transforms.Resize(h)
            t1 = trf3(t1)
            t2 = trf3(t2)


        x1_e1, x2_e1, x3_e1, x4_e1 = self.resnet(t1)
        x1_e2, x2_e2, x3_e2, x4_e2 = self.resnet(t2)

        y1 = torch.cat([x1_e1, x1_e2], dim=1)
        y2 = torch.cat([x2_e1, x2_e2], dim=1)
        y3 = torch.cat([x3_e1, x3_e2], dim=1)
        y4 = torch.cat([x4_e1, x4_e2], dim=1)

        y1 = self.conv1(y1)
        y2 = self.conv2(y2)
        y3 = self.conv3(y3)
        y4 = self.conv4(y4)

        # print(y1.shape)
        # print(y2.shape)
        # print(y3.shape)
        # print(y4.shape)

        out1 = self.decoder1(y1, y2)
        out2 = self.decoder2(out1, y3)
        out3 = self.decoder3(out2, y4)

        if rand3 > 0.5:
            out1 = F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=True)
            out2 = F.interpolate(out2, scale_factor=2, mode='bilinear', align_corners=True)
            out3 = F.interpolate(out3, scale_factor=2, mode='bilinear', align_corners=True)
        if rand2 > 0.25:
            out1 = untrf2(out1)
            out2 = untrf2(out2)
            out3 = untrf2(out3)
        if rand1 > 0.5:
            out1 = untrf1(out1)
            out2 = untrf1(out2)
            out3 = untrf1(out3)

        return out1, out2, out3

    def forward(self, t):
        cam, out1, out2, out3, out = self.forward_origin(t)
        out1_, out2_, out3_ = self.forward_affine(t)
        return cam, out1, out2, out3, out1_, out2_, out3_, out

if __name__ == '__main__':
    net = Network().cuda()
    # print(net)
    def getModelSize(model):
        param_size = 0
        param_sum = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_sum += param.nelement()
        buffer_size = 0
        buffer_sum = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            buffer_sum += buffer.nelement()
        all_size = (param_size + buffer_size) / 1024 / 1024
        print('模型总大小为：{:.3f}MB'.format(all_size))
        print('模型参数量为：{:.3f}M'.format(param_sum/1e6))
        # return (param_size, param_sum, buffer_size, buffer_sum, all_size)
    getModelSize(net)
    x = torch.rand([2, 4, 3, 256, 256]).cuda()
    y = net(x)
    for i in y:
        print(i.shape)
    # print(y.shape)