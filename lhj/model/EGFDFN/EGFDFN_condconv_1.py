import torch
import torch.nn as nn
import torch.nn.functional as F

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

# DFM模块
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

# CBAM模块
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

class Attention(nn.Module):
    def __init__(self, in_planes, K, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Conv2d(in_planes, K, kernel_size=1, bias=False)  # 用1*1卷积实现公式中的权重矩阵R
        self.sigmoid = nn.Sigmoid()

        if (init_weight):
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)  # bs,dim,1,1
        att = self.net(att).view(x.shape[0], -1)  # bs,K
        return self.sigmoid(att)

class CondConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True, K=4,
                 init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention(in_planes=in_planes, K=K, init_weight=init_weight)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if (bias):
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None

        if (self.init_weight):
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planes, h, w = x.shape
        softmax_att = self.attention(x)  # bs,K  计算K个expert各自的权重
        x = x.view(1, -1, h, w)  # 所有batch通道拼在一起
        weight = self.weight.view(self.K, -1)  # K,-1
        aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups,
                                                              self.kernel_size, self.kernel_size)  # bs*out_p,in_p,k,k, 矩阵乘法

        if (self.bias is not None):
            bias = self.bias.view(self.K, -1)  # K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs,out_p
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)

        output = output.view(bs, self.out_planes, h // self.stride, w // self.stride)
        return output

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
            CondConv(in_planes=out_ch // self.expansion, out_planes=out_ch // self.expansion, kernel_size=3, stride=self.down_stride, padding=1, bias=False, K=16),
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

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            # CondConv(ch_in, ch_out, kernel_size=3, stride=1, padding=1, K=16),
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

        self.up = nn.Upsample(scale_factor=2)
        self.dual_decoder4 = Bottleneck(512+512, 512, downsample=False)
        self.dual_decoder3 = Bottleneck(512+256, 256, downsample=False)
        self.dual_decoder2 = Bottleneck(256+128, 128, downsample=False)
        self.dual_decoder1 = Bottleneck(128+64, 64, downsample=False)

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        self.cbam5 = CBAM(512)

        # dfm
        self.dfm1 = DF_Module(dim_in=64,dim_out=64)
        self.dfm2 = DF_Module(dim_in=128,dim_out=128)
        self.dfm3 = DF_Module(dim_in=256,dim_out=256)
        self.dfm4 = DF_Module(dim_in=512,dim_out=512)
        self.dfm5 = DF_Module(dim_in=512,dim_out=512)

        self.Up_conv6 = Bottleneck(512, 512, downsample=False)
        self.Conv_1x1_6 = nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0)

        self.Up5 = up_conv(ch_in=512, ch_out=512)
        self.Up_conv5 = Bottleneck(1024, 512, downsample=False)
        self.Conv_1x1_5 = nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = Bottleneck(512, 256, downsample=False)
        self.Conv_1x1_4 = nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = Bottleneck(256, 128, downsample=False)
        self.Conv_3x3_3_edge = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)
        self.Conv_3x3_3_edge_ = BN_Re(ch_out=2)
        self.Conv_1x1_3 = nn.Conv2d(128+2, 2, kernel_size=1, stride=1, padding=0)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = Bottleneck(128, 64, downsample=False)
        self.Conv_3x3_2_edge = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        self.Conv_1x1_2 = nn.Conv2d(64+2, 2, kernel_size=1, stride=1, padding=0)

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

        x5_d1 = x5_e1
        x5_d2 = x5_e2
        x4_d1 = self.dual_decoder4(torch.cat([self.up(x5_d1),x4_e1],dim=1))
        x4_d2 = self.dual_decoder4(torch.cat([self.up(x5_d2),x4_e2],dim=1))
        x3_d1 = self.dual_decoder3(torch.cat([self.up(x4_d1),x3_e1],dim=1))
        x3_d2 = self.dual_decoder3(torch.cat([self.up(x4_d2),x3_e2],dim=1))
        x2_d1 = self.dual_decoder2(torch.cat([self.up(x3_d1),x2_e1],dim=1))
        x2_d2 = self.dual_decoder2(torch.cat([self.up(x3_d2),x2_e2],dim=1))
        x1_d1 = self.dual_decoder1(torch.cat([self.up(x2_d1),x1_e1],dim=1))
        x1_d2 = self.dual_decoder1(torch.cat([self.up(x2_d2),x1_e2],dim=1))

        x1_d1 = self.cbam1(x1_d1)
        x1_d2 = self.cbam1(x1_d2)
        x2_d1 = self.cbam2(x2_d1)
        x2_d2 = self.cbam2(x2_d2)
        x3_d1 = self.cbam3(x3_d1)
        x3_d2 = self.cbam3(x3_d2)
        x4_d1 = self.cbam4(x4_d1)
        x4_d2 = self.cbam4(x4_d2)
        x5_d1 = self.cbam5(x5_d1)
        x5_d2 = self.cbam5(x5_d2)

        x1 = self.dfm1(x1_d1, x1_d2)
        x2 = self.dfm2(x2_d1, x2_d2)
        x3 = self.dfm3(x3_d1, x3_d2)
        x4 = self.dfm4(x4_d1, x4_d2)
        x5 = self.dfm5(x5_d1, x5_d2)

        x5 = self.Up_conv6(x5)
        d6_out = self.Conv_1x1_6(x5)
        d6_out = F.interpolate(d6_out, scale_factor=16, mode='bilinear', align_corners=True)

        x5 = self.Up5(x5)
        print(x5.shape,x4.shape)
        d5 = torch.cat((x5, x4), dim=1)
        d5 = self.Up_conv5(d5)
        d5_out = self.Conv_1x1_5(d5)
        d5_out = F.interpolate(d5_out, scale_factor=8, mode='bilinear', align_corners=True)   #上采样为原图大小

        d4 = self.Up4(d5)
        d4 = torch.cat((d4, x3), dim=1)
        d4 = self.Up_conv4(d4)
        d4_out = self.Conv_1x1_4(d4)
        d4_out = F.interpolate(d4_out, scale_factor=4, mode='bilinear', align_corners=True)

        d3 = self.Up3(d4)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.Up_conv3(d3)
        d3_edge1 = self.Conv_3x3_3_edge(d3)     #输出边界预测
        d3_edge = F.interpolate(d3_edge1, scale_factor=2, mode='bilinear', align_corners=True) #上采样到原图大小
        d3_edge_ = self.Conv_3x3_3_edge_(d3_edge1)  #通过BN和RELU后，特征融合，预测分割结果
        d3_ = torch.cat((d3_edge_, d3), dim=1)
        d3_out = self.Conv_1x1_3(d3_)
        d3_out = F.interpolate(d3_out, scale_factor=2, mode='bilinear', align_corners=True)

        d2 = self.Up2(d3)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.Up_conv2(d2)
        d2_edge = self.Conv_3x3_2_edge(d2)
        d2_edge_ = self.Conv_3x3_3_edge_(d2_edge) #通过BN和RELU后，特征融合，预测分割结果
        d2_ = torch.cat((d2_edge_, d2), dim=1)
        d2_out = self.Conv_1x1_2(d2_)
        return d6_out, d5_out, d4_out, d3_out, d2_out, d3_edge, d2_edge


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
    x = torch.ones([2, 4, 3, 256, 256])
    y5, y4, y3, y2, y1, e2, e1 = net(x)
    print(y5.shape)