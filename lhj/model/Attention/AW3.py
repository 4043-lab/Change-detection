import numpy as np
import torch
from torch import nn
from torch.nn import init

# change-enhanced features fusion module
class AW3(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ConvRL = nn.Sequential(
            nn.Linear(channel * 2, channel * 2, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv_s = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.conv_t = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

        # self.init_weights()

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
        x1 = self.conv_t(x1)
        x2 = self.conv_s(x2)
        b, c, _, _ = x1.size()
        x = torch.cat([x1, x2], dim=1)
        y = self.avg_pool(x).view(b, c*2)
        y = self.ConvRL(y)
        # print(y.shape)
        y1 = torch.split(y, c, dim=-1)[0].unsqueeze(0)
        y2 = torch.split(y, c, dim=-1)[1].unsqueeze(0)
        # print(y1.shape)
        y_soft = torch.cat([y1, y2], dim=0)
        y_soft = torch.softmax(y_soft, dim=0)
        y1 = y_soft[0].view(b, c, 1, 1)
        y2 = y_soft[1].view(b, c, 1, 1)
        # print(y1)
        # print(y2)
        # print(y1+y2)
        x1_out = x1 * y1.expand_as(x1)
        x2_out = x2 * y2.expand_as(x2)
        return x1_out + x2_out


if __name__ == '__main__':
    input1 = torch.randn(2, 8, 256, 256)
    input2 = torch.randn(2, 8, 256, 256)
    print(input2)
    attn = AW(channel=8)
    output = attn(input1,input2)
    print(output.shape)
    # print(output)