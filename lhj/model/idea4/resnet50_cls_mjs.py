import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.maxpooling = nn.MaxPool2d(2)
        self.downsample = downsample
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // self.expansion, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch // self.expansion),
            nn.ReLU(inplace=True)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(out_ch // self.expansion, out_ch // self.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch // self.expansion),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(out_ch // self.expansion, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample:
            x = self.maxpooling(x)
        res = x
        x = self.conv1_1(x)
        x = self.conv3_1(x)
        x = self.conv1_2(x)
        res = self.conv_res(res)
        out = res + x
        out = self.relu(out)
        return out

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.res1 = nn.Sequential(
            Bottleneck(64, 128),
            Bottleneck(128, 128, downsample=False),
            Bottleneck(128, 128, downsample=False)
        )
        self.res2 = nn.Sequential(
            Bottleneck(128, 256),
            Bottleneck(256, 256, downsample=False),
            Bottleneck(256, 256, downsample=False),
            Bottleneck(256, 256, downsample=False)
        )
        self.res3 = nn.Sequential(
            Bottleneck(256, 512),
            Bottleneck(512, 512, downsample=False),
            Bottleneck(512, 512, downsample=False),
            Bottleneck(512, 512, downsample=False),
            Bottleneck(512, 512, downsample=False),
            Bottleneck(512, 512, downsample=False)
        )
        self.res4 = nn.Sequential(
            Bottleneck(512, 1024),
            Bottleneck(1024, 1024, downsample=False),
            Bottleneck(1024, 1024, downsample=False),
        )

        self.mjs2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1, bias=False)
        )
        self.mjs3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1, bias=False)
        )
        self.mjs4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1, bias=False)
        )
        self.mjs5 = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1, bias=False)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.kaiming_init()

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def forward(self, t1, t2):
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

        # y1 = torch.cat([x1_e1, x1_e2], dim=1)
        y2 = torch.cat([x2_e1, x2_e2], dim=1)
        y3 = torch.cat([x3_e1, x3_e2], dim=1)
        y4 = torch.cat([x4_e1, x4_e2], dim=1)
        y5 = torch.cat([x5_e1, x5_e2], dim=1)

        # print(y1.shape)
        # print(y2.shape)
        # print(y3.shape)
        # print(y4.shape)
        # print(y5.shape)

        cam2 = self.mjs2(y2)
        cam3 = self.mjs3(y3)
        cam4 = self.mjs4(y4)
        cam5 = self.mjs5(y5)

        out2 = self.avgpool(cam2).squeeze(3).squeeze(2)
        out3 = self.avgpool(cam3).squeeze(3).squeeze(2)
        out4 = self.avgpool(cam4).squeeze(3).squeeze(2)
        out5 = self.avgpool(cam5).squeeze(3).squeeze(2)

        cam2 = F.softmax(cam2, dim=1)
        cam3 = F.softmax(cam3, dim=1)
        cam4 = F.softmax(cam4, dim=1)
        cam5 = F.softmax(cam5, dim=1)

        cam2 = F.interpolate(cam2, size=128, mode='bilinear', align_corners=True)
        cam3 = F.interpolate(cam3, size=128, mode='bilinear', align_corners=True)
        cam4 = F.interpolate(cam4, size=128, mode='bilinear', align_corners=True)
        cam5 = F.interpolate(cam5, size=128, mode='bilinear', align_corners=True)

        return cam2, cam3, cam4, cam5, out2, out3, out4, out5

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
    loss = nn.L1Loss()
    print(loss(y[0], y[1]))