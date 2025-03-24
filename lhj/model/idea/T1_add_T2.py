"# -- coding: UTF-8 --"
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):

    def __init__(self, channels, reduction_channels):
        super(SEModule, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            reduction_channels, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        #x_se = self.avg_pool(x)
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.fc1(x_se)
        x_se = self.ReLU(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()

class conv_e(nn.Module):
    def __init__(self, ch_in, ch_out, down=True):
        super(conv_e, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.se = SEModule(ch_out, ch_out // 4)
        self.down = down
        self.downsample = torch.nn.MaxPool2d(stride=2, kernel_size=2)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        res = out
        out = self.conv2(out)
        out = self.se(out)
        if self.down:
            res = self.downsample(res)
            out = self.downsample(out)
        out = out + res
        # out = self.ReLU(out)
        return out

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=2, size=256):
        super(UNet, self).__init__()

        self.size = size
        # self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # dual_encoder
        self.Conv1_e = conv_e(ch_in=img_ch, ch_out=32, down=False)
        self.Conv2_e = conv_e(ch_in=32, ch_out=64, down=True)
        self.Conv3_e = conv_e(ch_in=64, ch_out=128, down=True)
        self.Conv4_e = conv_e(ch_in=128, ch_out=256, down=True)

        #dual_decoder
        self.Conv3_d = conv_block(ch_in=256+128,ch_out=128)
        self.Conv2_d = conv_block(ch_in=128+64,ch_out=64)
        self.Conv1_d = conv_block(ch_in=64+32,ch_out=32)
        self.conv_out = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
        self.up_d = nn.Upsample(scale_factor=2)

    def forward(self, input):
        # x = input[0] + input[1]
        x = input[0]

        x1_e = self.Conv1_e(x)  #32
        x2_e = self.Conv2_e(x1_e)  #64
        x3_e = self.Conv3_e(x2_e)  #128
        x4_e = self.Conv4_e(x3_e)  #256

        x4_d = x4_e
        x3_d = self.Conv3_d(torch.cat((self.up_d(x4_d),x3_e),1))
        x2_d = self.Conv2_d(torch.cat((self.up_d(x3_d),x2_e),1))
        x1_d = self.Conv1_d(torch.cat((self.up_d(x2_d),x1_e),1))

        y = self.conv_out(x1_d)

        # return y
        return x1_d