import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.

class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y

class conv_dy(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, padding):
        super(conv_dy, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.dim = int(math.sqrt(inplanes))
        squeeze = max(inplanes, self.dim ** 2) // 16

        self.q = nn.Conv2d(inplanes, self.dim, 1, stride, 0, bias=False)

        self.p = nn.Conv2d(self.dim, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(inplanes, squeeze, bias=False),
            SEModule_small(squeeze),
        )
        self.fc_phi = nn.Linear(squeeze, self.dim ** 2, bias=False)
        self.fc_scale = nn.Linear(squeeze, planes, bias=False)
        self.hs = Hsigmoid()

    def forward(self, x):
        r = self.conv(x)  # 先进行卷积，后对卷积结果加权
        b, c, _, _ = x.size()  # 获得batchsize和通道数
        y = self.avg_pool(x).view(b, c)  # 平均池化，self.avg_pool(x)尺寸为[b,c,1,1]，故加view
        y = self.fc(y)  # fc层
        phi = self.fc_phi(y).view(b, self.dim, self.dim)  # phi 即φ，这一步即为φ(x)
        scale = self.hs(self.fc_scale(y)).view(b, -1, 1, 1)  # hs 即Hsigmoid()激活函数
        r = scale.expand_as(r) * r  # 这里就是加权操作，从公式来看这里应该是A*W0，实际上这里是 A*W0*x，即把参数的获取和参数计算融合到一块

        out = self.bn1(self.q(x))  # q的话就是压缩通道数
        _, _, h, w = out.size()

        out = out.view(b, self.dim, -1)
        out = self.bn2(torch.matmul(phi, out)) + out  # 这里为什么加out呢，看论文像是设计成残差块？这里不太确定。
        out = out.view(b, -1, h, w)
        out = self.p(out) + r  # p是把通道拓展到输出空间，为什么要压缩再拓展，减少参数量
        return out

class Bottleneck(nn.Module):

    def __init__(self, in_ch, out_ch, downsample=True):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.down_stride = 2 if downsample else 1
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // self.expansion, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch // self.expansion),
            nn.ReLU(inplace=True)
        )
        self.conv3_1 = nn.Sequential(
            # nn.Conv2d(out_ch // self.expansion, out_ch // self.expansion, kernel_size=3, stride=self.down_stride, padding=1, bias=False),
            conv_dy(out_ch // self.expansion, out_ch // self.expansion, kernel_size=3, stride=self.down_stride, padding=1),
            nn.BatchNorm2d(out_ch // self.expansion),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(out_ch // self.expansion, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=self.down_stride, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.conv1_1(x)
        x = self.conv3_1(x)
        x = self.conv1_2(x)
        res = self.conv3_2(res)
        out = res + x
        out = self.relu(out)
        return out

class decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(decoder, self).__init__()
        self.conv = Bottleneck(in_ch, out_ch, downsample=False)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x = x1 + x2
        out = self.conv(x)
        return out

class Resnet_bottle(nn.Module):
    def __init__(self):
        super(Resnet_bottle, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res1 = Bottleneck(64, 128)
        self.res2 = Bottleneck(128, 256)
        # self.res3 = Bottleneck(256, 512)
        self.res3 = nn.Sequential(
            Bottleneck(256, 512),
            Bottleneck(512, 512, downsample=False),
            Bottleneck(512, 512, downsample=False)
        )
        self.res4 = Bottleneck(512, 512)

        self.decoder1 = decoder(1024, 512)
        self.decoder2 = decoder(512, 256)
        self.decoder3 = decoder(256, 128)
        self.decoder4 = decoder(128, 128)

        self.head = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)

        self.kaiming_init()

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, t):
        t1 = t[0]
        t2 = t[1]

        x1_e1 = self.stem(t1)
        x2_e1 = self.res1(x1_e1)
        x3_e1 = self.res2(x2_e1)
        x4_e1 = self.res3(x3_e1)
        x5_e1 = self.res4(x4_e1)
        x1_e2 = self.stem(t2)
        x2_e2 = self.res1(x1_e2)
        x3_e2 = self.res2(x2_e2)
        x4_e2 = self.res3(x3_e2)
        x5_e2 = self.res4(x4_e2)

        y1 = torch.cat([x1_e1, x1_e2], dim=1)
        y2 = torch.cat([x2_e1, x2_e2], dim=1)
        y3 = torch.cat([x3_e1, x3_e2], dim=1)
        y4 = torch.cat([x4_e1, x4_e2], dim=1)
        y5 = torch.cat([x5_e1, x5_e2], dim=1)

        out1 = self.decoder1(y5, y4)
        out2 = self.decoder2(out1, y3)
        out3 = self.decoder3(out2, y2)
        out4 = self.decoder4(out3, y1)

        out = self.head(out4)

        return out


if __name__ == '__main__':
    net = Resnet_bottle()
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
        # return (param_size, param_sum, buffer_size, buffer_sum, all_size)
    getModelSize(net)
    x = torch.ones([2, 8, 3, 256, 256])
    y = net(x)
    print(y.shape)