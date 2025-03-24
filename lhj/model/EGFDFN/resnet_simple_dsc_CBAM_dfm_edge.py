import torch
import torch.nn as nn
import torch.nn.functional as F

from model.idea3.DSConv import DSConv_pro

class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chn, in_chn // 4, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(in_chn // 4),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_chn // 4, in_chn // 4, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(in_chn // 4),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_chn // 4, in_chn // 4, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(in_chn // 4),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_chn // 4, out_chn, kernel_size=1, padding=0, stride=1, bias=True),
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
            nn.Conv2d(in_chn, in_chn // 4, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(in_chn // 4),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_chn // 4, in_chn // 4, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(in_chn // 4),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_chn // 4, in_chn // 4, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(in_chn // 4),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_chn // 4, out_chn, kernel_size=1, padding=0, stride=1, bias=True),
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
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(dim_out * 2, dim_out, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(dim_out),
        #     nn.ReLU(inplace=True),
        # )

    def forward(self, x1, x2):
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = torch.cat([x_add, x_diff], dim=1)
        return y

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

class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModul(in_channel=in_channel)  # 通道注意力模块
        self.Sam = SpatialAttentionModul(in_channel=in_channel)  # 空间注意力模块

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x



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
        self.conv3_1_x = DSConv_pro(out_ch // self.expansion, out_ch // self.expansion, 9, 1.0, 0, True)
        self.conv3_1_y = DSConv_pro(out_ch // self.expansion, out_ch // self.expansion, 9, 1.0, 1, True)
        self.conv_cat = nn.Conv2d(3 * out_ch // self.expansion, out_ch // self.expansion, kernel_size=1, stride=1, padding=0, bias=False)

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

        x0 = self.conv3_1(x)
        x1 = self.conv3_1_x(x)
        x2 = self.conv3_1_y(x)
        x = torch.cat([x0, x1, x2], dim=1)
        x = self.conv_cat(x)

        x = self.conv1_2(x)
        res = self.conv_res(res)
        out = res + x
        out = self.relu(out)
        return out

class decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(decoder, self).__init__()
        # self.conv = Bottleneck(in_ch, out_ch, downsample=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x = x1 + x2
        out = self.conv(x)
        return out

class decoder_edge(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(decoder_edge, self).__init__()
        # self.conv = Bottleneck(in_ch, out_ch, downsample=False)
        self.conv_edge = nn.Conv2d(in_ch, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + 2, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.BR = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x = x1 + x2
        edge = self.conv_edge(x)
        # out = self.conv(torch.cat([x, edge], dim=1))
        out = self.conv(torch.cat([x, self.BR(edge)], dim=1))
        return out, edge

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
            Bottleneck(128, 128, downsample=False)
        )
        self.res2 = nn.Sequential(
            Bottleneck(128, 256),
            Bottleneck(256, 256, downsample=False)
        )
        self.res3 = nn.Sequential(
            Bottleneck(256, 512),
            Bottleneck(512, 512, downsample=False)
        )
        self.res4 = nn.Sequential(
            Bottleneck(512, 512),
            Bottleneck(512, 512, downsample=False)
        )

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        self.cbam5 = CBAM(512)

        self.dfm1 = DF_Module(dim_in=64,dim_out=64)
        self.dfm2 = DF_Module(dim_in=128,dim_out=128)
        self.dfm3 = DF_Module(dim_in=256,dim_out=256)
        self.dfm4 = DF_Module(dim_in=512,dim_out=512)
        self.dfm5 = DF_Module(dim_in=512,dim_out=512)

        self.decoder1 = decoder(1024, 512)
        self.decoder2 = decoder(512, 256)
        self.decoder3 = decoder_edge(256, 128)
        self.decoder4 = decoder_edge(128, 64)

        self.head = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)

        self.kaiming_init()

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, t1, t2):
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

        x1_e1 = self.cbam1(x1_e1)
        x1_e2 = self.cbam1(x1_e2)
        x2_e1 = self.cbam2(x2_e1)
        x2_e2 = self.cbam2(x2_e2)
        x3_e1 = self.cbam3(x3_e1)
        x3_e2 = self.cbam3(x3_e2)
        x4_e1 = self.cbam4(x4_e1)
        x4_e2 = self.cbam4(x4_e2)
        x5_e1 = self.cbam5(x5_e1)
        x5_e2 = self.cbam5(x5_e2)

        # y1 = torch.cat([x1_e1, x1_e2], dim=1)
        # y2 = torch.cat([x2_e1, x2_e2], dim=1)
        # y3 = torch.cat([x3_e1, x3_e2], dim=1)
        # y4 = torch.cat([x4_e1, x4_e2], dim=1)
        # y5 = torch.cat([x5_e1, x5_e2], dim=1)

        y1 = self.dfm1(x1_e1, x1_e2)
        y2 = self.dfm2(x2_e1, x2_e2)
        y3 = self.dfm3(x3_e1, x3_e2)
        y4 = self.dfm4(x4_e1, x4_e2)
        y5 = self.dfm5(x5_e1, x5_e2)

        # print(y1.shape)
        # print(y2.shape)
        # print(y3.shape)
        # print(y4.shape)
        # print(y5.shape)

        out1 = self.decoder1(y5, y4)
        out2 = self.decoder2(out1, y3)
        out3, edge3 = self.decoder3(out2, y2)
        out4, edge4 = self.decoder4(out3, y1)

        # print(out1.shape)
        # print(out2.shape)
        # print(out3.shape)
        # print(out4.shape)

        edge3 = F.interpolate(edge3, scale_factor=2, mode='bilinear')
        out = self.head(out4)

        # return out1, out2, out3, out4, out
        # return out
        return edge3, edge4, out


if __name__ == '__main__':
    net = Network().cuda()
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
    x1 = torch.ones([4, 3, 256, 256]).cuda()
    x2 = torch.ones([4, 3, 256, 256]).cuda()
    y = net(x1, x2)
    for i in y:
        print(i.shape)
    # print(y.shape)