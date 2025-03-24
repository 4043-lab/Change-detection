"# -- coding: UTF-8 --"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0, stride=1, bias=True),
            # SynchronizedBatchNorm2d(out_chn, momentum=bn_mom),
            nn.BatchNorm2d(out_chn),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)

class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0, stride=1, bias=True),
            # SynchronizedBatchNorm2d(out_chn, momentum=bn_mom),
            nn.BatchNorm2d(out_chn),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out

#DFM模块
class DF_Module(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=False):
        super(DF_Module, self).__init__()
        # if reduction:
        #     self.reduction = nn.Sequential(
        #         nn.Conv2d(dim_in, dim_in//2, kernel_size=1, padding=0),
        #         # SynchronizedBatchNorm2d(dim_in//2, momentum=bn_mom),
        #         nn.BatchNorm2d(dim_in//2),
        #         nn.ReLU(inplace=True),
        #     )
        #     dim_in = dim_in//2
        # else:
        #     self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            # SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            # SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        # if self.reduction is not None:
        #     x1 = self.reduction(x1)
        #     x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + self.conv2(x_add)
        return y

class NL_Block(nn.Module):
    def __init__(self, in_channels):
        super(NL_Block, self).__init__()
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        value = self.conv_v(x).view(batch_size, c, -1)
        value = value.permute(0, 2, 1)  # B * (H*W) * value_channels
        key = x.view(batch_size, c, -1)  # B * key_channels * (H*W)
        query = x.view(batch_size, c, -1)
        query = query.permute(0, 2, 1)
        sim_map = torch.matmul(query, key)  # B * (H*W) * (H*W)
        sim_map = (c ** -.5) * sim_map  # B * (H*W) * (H*W)  相当于除以Channel的平方
        sim_map = torch.softmax(sim_map, dim=-1)  # B * (H*W) * (H*W)  在(H*W)这个维度上softmax
        context = torch.matmul(sim_map, value)  # B * C * (H*W)
        context = context.permute(0, 2,
                                  1).contiguous()  # contiguous可以使permute() narrow() view() expand() transpose()不会改变元数据，在这好像没啥用
        context = context.view(batch_size, c, *x.size()[2:])  # B * C * H * W
        context = self.W(context)

        return context

#nlfpn模块
class NL_FPN(nn.Module):
    """ non-local feature parymid network"""

    def __init__(self, in_dim, reduction=True):
        super(NL_FPN, self).__init__()
        bn_mom = 0.0003
        if reduction:
            self.reduction = nn.Sequential(
                nn.Conv2d(in_dim, in_dim // 4, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_dim // 4, momentum=bn_mom),
                nn.ReLU(inplace=True),
            )
            self.re_reduction = nn.Sequential(
                nn.Conv2d(in_dim // 4, in_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_dim, momentum=bn_mom),
                nn.ReLU(inplace=True),
            )
            in_dim = in_dim // 4
        else:
            self.reduction = None
            self.re_reduction = None
        self.conv_e1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_e2 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 2, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_e3 = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_d3 = nn.Sequential(
            nn.Conv2d(in_dim * 4, in_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 2, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.nl3 = NL_Block(in_dim * 2)
        self.nl2 = NL_Block(in_dim)
        self.nl1 = NL_Block(in_dim)

        self.downsample_x2 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.upsample_x2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        if self.reduction is not None:
            x = self.reduction(x)
        e1 = self.conv_e1(x)  # C,H,W
        e2 = self.conv_e2(self.downsample_x2(e1))  # 2C,H/2,W/2
        e3 = self.conv_e3(self.downsample_x2(e2))  # 4C,H/4,W/4

        d3 = self.conv_d3(e3)  # 2C,H/4,W/4
        nl = self.nl3(d3)
        d3 = self.upsample_x2(torch.mul(d3, nl))  # 2C,H/2,
         # W/2
        d2 = self.conv_d2(e2 + d3)  # C,H/2,W/2
        nl = self.nl2(d2)
        d2 = self.upsample_x2(torch.mul(d2, nl))  # C,H,W
        d1 = self.conv_d1(e1 + d2)
        nl = self.nl1(d1)
        d1 = torch.mul(d1, nl)  # C,H,W
        if self.re_reduction is not None:
            d1 = self.re_reduction(d1)

        return d1

class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModul(in_channel=in_channel)  # 通道注意力模块
        self.Sam = SpatialAttentionModul(in_channel=in_channel)  # 空间注意力模块

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x

class ChannelAttentionModul(nn.Module):  # 通道注意力模块
    def __init__(self, in_channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(ChannelAttentionModul, self).__init__()
        # 全局最大池化
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 全局均值池化
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1.最大池化分支
        max_branch = self.MaxPool(x)
        # 送入MLP全连接神经网络, 得到权重
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        # 2.全局池化分支
        avg_branch = self.AvgPool(x)
        # 送入MLP全连接神经网络, 得到权重
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        # MaxPool + AvgPool 激活后得到权重weight
        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        x = Mc * x

        return x

class SpatialAttentionModul(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
        MaxPool = torch.max(x, dim=1).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
        AvgPool = torch.mean(x, dim=1)

        # 增加维度, 变成 [N, 1, H, W]
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        # 维度拼接 [N, 2, H, W]
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  # 获得特征图

        # 卷积操作得到空间注意力结果
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        # 与原图通道进行乘积
        x = Ms * x

        return x

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


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class BN_Re(nn.Module):
    def __init__(self, ch_out):
        super(BN_Re, self).__init__()
        self.BN_Re = nn.Sequential(
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.BN_Re(x)
        return x


class EGFDFN(nn.Module):
    def __init__(self, img_ch=3, output_ch=2, patch_size=256):
        super(EGFDFN, self).__init__()

        self.patch_size = patch_size
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # dual_encoder
        self.Conv1_e = conv_e(ch_in=img_ch, ch_out=16, down=False)
        self.Conv2_e = conv_e(ch_in=16, ch_out=32, down=True)
        self.Conv3_e = conv_e(ch_in=32, ch_out=64, down=True)
        self.Conv4_e = conv_e(ch_in=64, ch_out=128, down=True)
        self.Conv5_e = conv_e(ch_in=128, ch_out=256, down=True)

        # nlfpn
        self.nlf = NL_FPN(256,True)

        #dual_decoder
        self.Conv4_d = conv_block(ch_in=256+128,ch_out=128)
        self.Conv3_d = conv_block(ch_in=128+64,ch_out=64)
        self.Conv2_d = conv_block(ch_in=64+32,ch_out=32)
        self.Conv1_d = conv_block(ch_in=32+16,ch_out=16)
        self.up_d = nn.Upsample(scale_factor=2)

        # dfm
        self.dfm1 = DF_Module(dim_in=16,dim_out=16)
        self.dfm2 = DF_Module(dim_in=32,dim_out=32)
        self.dfm3 = DF_Module(dim_in=64,dim_out=64)
        self.dfm4 = DF_Module(dim_in=128,dim_out=128)
        self.dfm5 = DF_Module(dim_in=256,dim_out=256)

        # CBAM
        # self.cbam1_1 = CBAM(16)
        # self.cbam1_2 = CBAM(16)
        # self.cbam2_1 = CBAM(32)
        # self.cbam2_2 = CBAM(32)
        # self.cbam3_1 = CBAM(64)
        # self.cbam3_2 = CBAM(64)
        # self.cbam4_1 = CBAM(128)
        # self.cbam4_2 = CBAM(128)
        # self.cbam5_1 = CBAM(256)
        # self.cbam5_2 = CBAM(256)
        self.cbam1 = CBAM(16)
        self.cbam2 = CBAM(32)
        self.cbam3 = CBAM(64)
        self.cbam4 = CBAM(128)
        self.cbam5 = CBAM(256)

        self.Up5 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv6 = conv_block(ch_in=256, ch_out=256)
        self.Conv_1x1_6 = nn.Conv2d(256, output_ch, kernel_size=1, stride=1, padding=0)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128)
        self.Conv_1x1_5 = nn.Conv2d(128, output_ch, kernel_size=1, stride=1, padding=0)

        self.Up4 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1_4 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.Up3 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)
        self.Conv_1x1_3_edge = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_3_edge_ = BN_Re(ch_out=output_ch)
        self.Conv_1x1_3 = nn.Conv2d(34, output_ch, kernel_size=1, stride=1, padding=0)

        self.Up2 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16)
        self.Conv_1x1_2_edge = nn.Conv2d(16, output_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_2 = nn.Conv2d(18, output_ch, kernel_size=1, stride=1, padding=0)

    def encoder(self, x):
        x1_e1 = self.Conv1_e(x[0])
        x2_e1 = self.Conv2_e(x1_e1)
        x3_e1 = self.Conv3_e(x2_e1)
        x4_e1 = self.Conv4_e(x3_e1)
        x5_e1 = self.Conv5_e(x4_e1)
        x1_e2 = self.Conv1_e(x[1])
        x2_e2 = self.Conv2_e(x1_e2)
        x3_e2 = self.Conv3_e(x2_e2)
        x4_e2 = self.Conv4_e(x3_e2)
        x5_e2 = self.Conv5_e(x4_e2)
        # print(x[0].shape)
        # print(x1_e1.shape)
        # print(x2_e1.shape)
        # print(x3_e1.shape)
        # print(x4_e1.shape)
        # print(x5_e1.shape)

        x5_d1 = self.nlf(x5_e1)
        x5_d2 = self.nlf(x5_e2)

        x4_d1 = self.Conv4_d(torch.cat((self.up_d(x5_d1),x4_e1),1))
        x4_d2 = self.Conv4_d(torch.cat((self.up_d(x5_d2),x4_e2),1))
        x3_d1 = self.Conv3_d(torch.cat((self.up_d(x4_d1),x3_e1),1))
        x3_d2 = self.Conv3_d(torch.cat((self.up_d(x4_d2),x3_e2),1))
        x2_d1 = self.Conv2_d(torch.cat((self.up_d(x3_d1),x2_e1),1))
        x2_d2 = self.Conv2_d(torch.cat((self.up_d(x3_d2),x2_e2),1))
        x1_d1 = self.Conv1_d(torch.cat((self.up_d(x2_d1),x1_e1),1))
        x1_d2 = self.Conv1_d(torch.cat((self.up_d(x2_d2),x1_e2),1))

        # x5_d1 = self.cbam5_1(x5_d1)
        # x5_d2 = self.cbam5_2(x5_d2)
        # x4_d1 = self.cbam4_1(x4_d1)
        # x4_d2 = self.cbam4_2(x4_d2)
        # x3_d1 = self.cbam3_1(x3_d1)
        # x3_d2 = self.cbam3_2(x3_d2)
        # x2_d1 = self.cbam2_1(x2_d1)
        # x2_d2 = self.cbam2_2(x2_d2)
        # x1_d1 = self.cbam1_1(x1_d1)
        # x1_d2 = self.cbam1_2(x1_d2)

        x5_d1 = self.cbam5(x5_d1)
        x5_d2 = self.cbam5(x5_d2)
        x4_d1 = self.cbam4(x4_d1)
        x4_d2 = self.cbam4(x4_d2)
        x3_d1 = self.cbam3(x3_d1)
        x3_d2 = self.cbam3(x3_d2)
        x2_d1 = self.cbam2(x2_d1)
        x2_d2 = self.cbam2(x2_d2)
        x1_d1 = self.cbam1(x1_d1)
        x1_d2 = self.cbam1(x1_d2)

        dif_x5 = self.dfm5(x5_d1, x5_d2)
        dif_x4 = self.dfm4(x4_d1, x4_d2)
        dif_x3 = self.dfm3(x3_d1, x3_d2)
        dif_x2 = self.dfm2(x2_d1, x2_d2)
        dif_x1 = self.dfm1(x1_d1, x1_d2)

        return dif_x1, dif_x2, dif_x3, dif_x4, dif_x5



    def forward(self, input):
        # encoding path

        x1, x2, x3, x4, x5 = self.encoder(input)

        # decoding + concat path

        x5 = self.Up_conv6(x5)
        d6_out = self.Conv_1x1_6(x5)
        d6_out = F.interpolate(d6_out, scale_factor=16, mode='bilinear')

        x5 = self.Up5(x5)
        d5 = torch.cat((x5, x4), dim=1)
        d5 = self.Up_conv5(d5)
        d5_out = self.Conv_1x1_5(d5)
        d5_out = F.interpolate(d5_out, scale_factor=8, mode='bilinear')   #上采样为原图大小

        d4 = self.Up4(d5)
        d4 = torch.cat((d4, x3), dim=1)
        d4 = self.Up_conv4(d4)
        d4_out = self.Conv_1x1_4(d4)
        d4_out = F.interpolate(d4_out, scale_factor=4, mode='bilinear')

        d3 = self.Up3(d4)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.Up_conv3(d3)
        d3_edge1 = self.Conv_1x1_3_edge(d3)     #输出边界预测
        d3_edge = F.interpolate(d3_edge1, scale_factor=2, mode='bilinear')
        d3_edge_ = self.Conv_1x1_3_edge_(d3_edge1)  #通过BN和RELU后，特征融合，预测分割结果
        d3_ = torch.cat((d3_edge_, d3), dim=1)
        d3_out = self.Conv_1x1_3(d3_)
        d3_out = F.interpolate(d3_out, scale_factor=2, mode='bilinear')



        d2 = self.Up2(d3)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.Up_conv2(d2)
        d2_edge = self.Conv_1x1_2_edge(d2)
        d2_edge_ = self.Conv_1x1_3_edge_(d2_edge)
        d2_ = torch.cat((d2_edge_, d2), dim=1)
        d2_out = self.Conv_1x1_2(d2_)
        return d6_out, d5_out, d4_out, d3_out, d2_out, d3_edge, d2_edge