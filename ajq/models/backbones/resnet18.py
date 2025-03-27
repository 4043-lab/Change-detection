import torch
import torch.nn as nn
import torch.nn.functional as F


class Basicblock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super(Basicblock, self).__init__()

        self.down_stride = 2 if downsample else 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=self.down_stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch ),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=self.down_stride, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.down_stride:
            res = self.downsample(res)
        out = res + x
        out = self.relu(out)
        return out


class decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(decoder, self).__init__()
        self.conv = Basicblock(in_ch, out_ch, downsample=False)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x = x1 + x2
        out = self.conv(x)
        return out


class Resnet_basic(nn.Module):
    def __init__(self):
        super(Resnet_basic, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.res1 = Bottleneck(64, 128)
        # self.res2 = Bottleneck(128, 256)
        # # self.res3 = Bottleneck(256, 512)
        # self.res3 = nn.Sequential(
        #     Bottleneck(256, 512),
        #     Bottleneck(512, 512, downsample=False),
        #     Bottleneck(512, 512, downsample=False)
        # )
        # self.res4 = Bottleneck(512, 512)

        self.layer1 = nn.Sequential(
            Basicblock(64, 64, downsample=False),
            Basicblock(64, 64, downsample=False)
        )
        self.layer2 = nn.Sequential(
            Basicblock(64, 128),
            Basicblock(128, 128, downsample=False)
        )
        # self.res3 = Bottleneck(256, 512)
        self.layer3 = nn.Sequential(
            Basicblock(128, 256),
            Basicblock(256, 256, downsample=False)
        )
        self.layer4 = nn.Sequential(
            Basicblock(256, 512),
            Basicblock(512, 512, downsample=False)
        )

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

    def forward(self, t1, t2):

        x1_e1 = self.maxpool(self.stem(t1))
        x2_e1 = self.layer1(x1_e1)
        x3_e1 = self.layer2(x2_e1)
        x4_e1 = self.layer3(x3_e1)
        x5_e1 = self.layer4(x4_e1)
        x1_e2 = self.maxpool(self.stem(t2))
        x2_e2 = self.layer1(x1_e2)
        x3_e2 = self.layer2(x2_e2)
        x4_e2 = self.layer3(x3_e2)
        x5_e2 = self.layer4(x4_e2)

        return x1_e1, x2_e1, x3_e1, x4_e1, x5_e1, x1_e2, x2_e2, x3_e2, x4_e2, x5_e2

        # y1 = torch.cat([x1_e1, x1_e2], dim=1)
        # y2 = torch.cat([x2_e1, x2_e2], dim=1)
        # y3 = torch.cat([x3_e1, x3_e2], dim=1)
        # y4 = torch.cat([x4_e1, x4_e2], dim=1)
        # y5 = torch.cat([x5_e1, x5_e2], dim=1)
        #
        # return y2, y3, y4, y5

        # out1 = self.decoder1(y5, y4)
        # out2 = self.decoder2(out1, y3)
        # out3 = self.decoder3(out2, y2)
        # out4 = self.decoder4(out3, y1)
        # out4 = F.interpolate(out4, scale_factor=2, mode='bilinear', align_corners=True)
        # out = self.head(out4)
        #
        # return out
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A = torch.randn(4, 3, 256, 256).to(device)
    B = torch.randn(4, 3, 256, 256).to(device)
    model = Resnet_basic().to(device)
    c1 = model(A, B)
