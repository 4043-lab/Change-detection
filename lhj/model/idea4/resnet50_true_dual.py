import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from model.idea4.resnet import resnet50

class dual_decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(dual_decoder, self).__init__()
        # self.conv = Bottleneck(in_ch, out_ch, downsample=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        out = x1 + x2
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

class decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
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

        self.dual_decoder3 = dual_decoder(2048, 1024)
        self.dual_decoder2 = dual_decoder(1024, 512)
        self.dual_decoder1 = dual_decoder(512, 256)

        self.decoder1 = decoder(2048+1024, 1024)
        self.decoder2 = decoder(1024+512, 512)
        self.decoder3 = decoder(512+256, 256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(256, 2, 1, bias=False)

    # def forward(self, t1, t2):
    def forward_origin(self, t):
        t1 = t[0]
        t2 = t[1]
        x1_e1, x2_e1, x3_e1, x4_e1 = self.resnet(t1)
        x1_e2, x2_e2, x3_e2, x4_e2 = self.resnet(t2)

        # print(x1_e1.shape)
        # print(x2_e1.shape)
        # print(x3_e1.shape)
        # print(x4_e1.shape)

        x4_d1 = x4_e1
        x3_d1 = self.dual_decoder3(x4_d1, x3_e1)
        x2_d1 = self.dual_decoder2(x3_d1, x2_e1)
        x1_d1 = self.dual_decoder1(x2_d1, x1_e1)
        x4_d2 = x4_e2
        x3_d2 = self.dual_decoder3(x4_d2, x3_e2)
        x2_d2 = self.dual_decoder2(x3_d2, x2_e2)
        x1_d2 = self.dual_decoder1(x2_d2, x1_e2)

        # print(x1_d1.shape)
        # print(x2_d1.shape)
        # print(x3_d1.shape)
        # print(x4_d1.shape)

        y1 = torch.cat([x1_d1, x1_d2], dim=1)
        y2 = torch.cat([x2_d1, x2_d2], dim=1)
        y3 = torch.cat([x3_d1, x3_d2], dim=1)
        y4 = torch.cat([x4_d1, x4_d2], dim=1)

        y1 = self.conv1(y1)
        y2 = self.conv2(y2)
        y3 = self.conv3(y3)
        y4 = self.conv4(y4)

        # print(y1.shape)
        # print(y2.shape)
        # print(y3.shape)
        # print(y4.shape)

        out1 = self.decoder1(y4, y3)
        out2 = self.decoder2(out1, y2)
        out3 = self.decoder3(out2, y1)

        # print(out1.shape)
        # print(out2.shape)
        # print(out3.shape)

        cam = self.fc(out3)
        out = self.avgpool(cam).squeeze(3).squeeze(2)

        cam = F.softmax(cam, dim=1)

        return cam, out

    def forward(self, t):
        cam, out = self.forward_origin(t)
        return cam, out
        # return out

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
    # print(net)