"# -- coding: UTF-8 --"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from model.idea3.DSConv import DSConv_pro

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
            nn.ReLU(inplace=True)
        )
        self.conv2_x = DSConv_pro(ch_out, ch_out, 9, 1.0, 0, True)
        self.conv2_y = DSConv_pro(ch_out, ch_out, 9, 1.0, 1, True)
        self.conv_cat = nn.Conv2d(3 * ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.down = down
        self.downsample = torch.nn.MaxPool2d(stride=2, kernel_size=2)
        self.ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        res = out
        x0 = self.conv2(out)
        x1 = self.conv2_x(out)
        x2 = self.conv2_y(out)
        x = torch.cat([x0, x1, x2], dim=1)
        out = self.conv_cat(x)
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
        self.Conv_3x3_3_edge = nn.Conv2d(32, output_ch, kernel_size=3, stride=1, padding=1)
        self.Conv_3x3_3_edge_ = BN_Re(ch_out=output_ch)
        self.Conv_1x1_3 = nn.Conv2d(34, output_ch, kernel_size=1, stride=1, padding=0)

        self.Up2 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16)
        self.Conv_3x3_2_edge = nn.Conv2d(16, output_ch, kernel_size=3, stride=1, padding=1)
        self.Conv_1x1_2 = nn.Conv2d(18, output_ch, kernel_size=1, stride=1, padding=0)

    def encoder(self, x1, x2):
        x1_e1 = self.Conv1_e(x1)
        x2_e1 = self.Conv2_e(x1_e1)
        x3_e1 = self.Conv3_e(x2_e1)
        x4_e1 = self.Conv4_e(x3_e1)
        x5_e1 = self.Conv5_e(x4_e1)
        x1_e2 = self.Conv1_e(x2)
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

        x5_d1 = x5_e1
        x5_d2 = x5_e2

        x4_d1 = self.Conv4_d(torch.cat((self.up_d(x5_d1),x4_e1),1))
        x4_d2 = self.Conv4_d(torch.cat((self.up_d(x5_d2),x4_e2),1))
        x3_d1 = self.Conv3_d(torch.cat((self.up_d(x4_d1),x3_e1),1))
        x3_d2 = self.Conv3_d(torch.cat((self.up_d(x4_d2),x3_e2),1))
        x2_d1 = self.Conv2_d(torch.cat((self.up_d(x3_d1),x2_e1),1))
        x2_d2 = self.Conv2_d(torch.cat((self.up_d(x3_d2),x2_e2),1))
        x1_d1 = self.Conv1_d(torch.cat((self.up_d(x2_d1),x1_e1),1))
        x1_d2 = self.Conv1_d(torch.cat((self.up_d(x2_d2),x1_e2),1))

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



    def forward(self, t1, t2):
        # encoding path

        x1, x2, x3, x4, x5 = self.encoder(t1, t2)

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
        d3_edge1 = self.Conv_3x3_3_edge(d3)     #输出边界预测
        d3_edge = F.interpolate(d3_edge1, scale_factor=2, mode='bilinear') #上采样到原图大小
        d3_edge_ = self.Conv_3x3_3_edge_(d3_edge1)  #通过BN和RELU后，特征融合，预测分割结果
        d3_ = torch.cat((d3_edge_, d3), dim=1)
        d3_out = self.Conv_1x1_3(d3_)
        d3_out = F.interpolate(d3_out, scale_factor=2, mode='bilinear')



        d2 = self.Up2(d3)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.Up_conv2(d2)
        d2_edge = self.Conv_3x3_2_edge(d2)
        d2_edge_ = self.Conv_3x3_3_edge_(d2_edge) #通过BN和RELU后，特征融合，预测分割结果
        d2_ = torch.cat((d2_edge_, d2), dim=1)
        d2_out = self.Conv_1x1_2(d2_)
        return d3_edge, d2_edge, d6_out, d5_out, d4_out, d3_out, d2_out

if __name__ == '__main__':
    net = EGFDFN().cuda()
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
    x1 = torch.ones([1, 3, 256, 256]).cuda()
    x2 = torch.ones([1, 3, 256, 256]).cuda()
    y = net(x1, x2)
    for i in y:
        print(i.shape)
    # print(y.shape)