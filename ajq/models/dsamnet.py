# Implementation of
# Q. Shi, M. Liu, S. Li, X. Liu, F. Wang, and L. Zhang, “A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection,” IEEE Trans. Geosci. Remote Sensing, pp. 1–16, 2021, doi: 10.1109/TGRS.2021.3085870.

# The resnet implementation differs from the original work. See src/models/stanet.py for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import resnet


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // ratio, 1, bias=False),
            nn.ReLU())
        self.fc2 = nn.Conv2d(in_ch // ratio, in_ch, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        return F.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return F.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_ch, ratio=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_ch, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        y = self.ca(x) * x
        y = self.sa(y) * y
        return y


def get_norm_layer():
    # TODO: select appropriate norm layer
    return nn.BatchNorm2d


class Backbone(nn.Module):
    def __init__(self, in_ch, arch, pretrained=True, strides=(2, 1, 2, 2, 2)):
        super().__init__()

        if arch == 'resnet18':
            self.resnet = resnet.resnet18(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet34':
            self.resnet = resnet.resnet34(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet50':
            self.resnet = resnet.resnet50(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        else:
            raise ValueError

        if in_ch != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_ch,
                64,
                kernel_size=7,
                stride=strides[0],
                padding=3,
                bias=False
            )

        if not pretrained:
            self._init_weight()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        return x1, x2, x3, x4


class Decoder(nn.Module):
    def __init__(self, f_ch):
        super().__init__()
        self.dr1 = nn.Sequential(
            nn.Conv2d(64, 96, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU())
        self.dr2 = nn.Sequential(
            nn.Conv2d(128, 96, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU())
        self.dr3 = nn.Sequential(
            nn.Conv2d(256, 96, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU())
        self.dr4 = nn.Sequential(
            nn.Conv2d(512, 96, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU())

        self.conv_out = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, f_ch, 1, bias=False),
            nn.BatchNorm2d(f_ch),
            nn.ReLU()
        )


    def forward(self, feats):
        f1 = self.dr1(feats[0])
        f2 = self.dr2(feats[1])
        f3 = self.dr3(feats[2])
        f4 = self.dr4(feats[3])

        f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(f4, size=f1.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([f1, f2, f3, f4], dim=1)
        y = self.conv_out(x)

        return y


class DSLayer(nn.Sequential):
    def __init__(self, in_ch, out_ch, itm_ch, **convd_kwargs):
        super().__init__(
            nn.ConvTranspose2d(in_ch, itm_ch, kernel_size=3, padding=1, **convd_kwargs),
            nn.BatchNorm2d(itm_ch),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.ConvTranspose2d(itm_ch, out_ch, kernel_size=3, padding=1)
        )


class DSAMNet(nn.Module):
    def __init__(self, in_ch, out_ch, width=64, backbone='resnet18', ca_ratio=8, sa_kernel=7):
        super().__init__()

        self.backbone = Backbone(in_ch=in_ch, arch=backbone, strides=(1, 1, 2, 2, 1))
        self.decoder = Decoder(width)

        self.cbam1 = CBAM(64, ratio=ca_ratio, kernel_size=sa_kernel)
        self.cbam2 = CBAM(64, ratio=ca_ratio, kernel_size=sa_kernel)

        self.dsl2 = DSLayer(64, out_ch, 32, stride=2, output_padding=1)
        self.dsl3 = DSLayer(128, out_ch, 32, stride=4, output_padding=3)

        self.calc_dist = nn.PairwiseDistance(keepdim=True)

    def forward(self, t1, t2):
        f1 = self.backbone(t1)
        f2 = self.backbone(t2)

        y1 = self.decoder(f1)
        y2 = self.decoder(f2)

        y1 = self.cbam1(y1).permute(0, 2, 3, 1)
        y2 = self.cbam2(y2).permute(0, 2, 3, 1)

        dist = self.calc_dist(y1, y2).permute(0, 3, 1, 2)
        dist = F.interpolate(dist, size=t1.shape[2:], mode='bilinear', align_corners=True)

        ds2 = self.dsl2(torch.abs(f1[0] - f2[0]))
        ds3 = self.dsl3(torch.abs(f1[1] - f2[1]))

        return dist, ds2, ds3

