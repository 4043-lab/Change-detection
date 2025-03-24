import torch
import torch.nn as nn
import torch.nn.functional as F
from model.idea2.ConvLSTM_new2 import ConvLSTM
from thop import profile, clever_format

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
            CondConv(out_ch // self.expansion, out_ch // self.expansion, kernel_size=3, stride=self.down_stride, padding=1, bias=False, K=4),
            nn.BatchNorm2d(out_ch // self.expansion),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(out_ch // self.expansion, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=self.down_stride, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.conv1_1(x)
        x = self.conv3_1(x)
        x = self.conv1_2(x)
        res = self.conv_res(res)
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

def pair_to_video(im1, im2, len):
    device = im1.device
    delta = 1.0 / (len - 1)
    steps = torch.arange(len, dtype=torch.float).view(1, -1, 1, 1, 1).to(device)
    interped = im1.unsqueeze(1) + ((im2 - im1) * delta).unsqueeze(1) * steps
    interped = interped.transpose(0,1)
    return interped

class Network(nn.Module):
    def __init__(self, seq_len=2, num_layers=2):
        super(Network, self).__init__()

        self.seq_len = seq_len

        self.CL1_forw = ConvLSTM(3, 64, (3, 3), 1, False, True, False)
        self.CL1_back = ConvLSTM(3, 64, (3, 3), 1, False, True, False)

        self.CL2_forw = ConvLSTM(64, 128, (3, 3), 1, True, True, False)
        self.CL2_back = ConvLSTM(64, 128, (3, 3), 1, True, True, False)

        self.CL3_forw = ConvLSTM(128, 256, (3, 3), 1, True, True, False)
        self.CL3_back = ConvLSTM(128, 256, (3, 3), 1, True, True, False)

        self.CL4_forw = ConvLSTM(256, 512, (3, 3), 1, True, True, False)
        self.CL4_back = ConvLSTM(256, 512, (3, 3), 1, True, True, False)

        self.CL5_forw = ConvLSTM(512, 512, (3, 3), 1, True, True, False)
        self.CL5_back = ConvLSTM(512, 512, (3, 3), 1, True, True, False)

        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.decoder1 = decoder(1024, 512)
        self.decoder2 = decoder(512, 256)
        self.decoder3 = decoder(256, 128)
        self.decoder4 = decoder(128, 128)

        self.head = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)

        self.kaiming_init()

    def downsample(self, x):
        b, t, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.pool(x)
        x = x.view(b, t, c, x.size(2), x.size(3))
        return x

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, t1, t2):
        # t1 = t[0]
        # t2 = t[1]

        video = pair_to_video(t1, t2, self.seq_len)
        # print(video.shape)

        video1_forw = self.CL1_forw(video, reverse=False)
        video1_back = self.CL1_back(video, reverse=True)
        video1_out = torch.cat([video1_forw[:,-1], video1_back[:,-1]], dim=1)
        y1 = video1_out

        video2_forw = self.CL2_forw(self.downsample(video1_forw), reverse=False)
        video2_back = self.CL2_back(self.downsample(video1_back), reverse=True)
        video2_out = torch.cat([video2_forw[:,-1], video2_back[:,-1]], dim=1)
        y2 = video2_out

        video3_forw = self.CL3_forw(self.downsample(video2_forw), reverse=False)
        video3_back = self.CL3_back(self.downsample(video2_back), reverse=True)
        video3_out = torch.cat([video3_forw[:,-1], video3_back[:,-1]], dim=1)
        y3 = video3_out

        video4_forw = self.CL4_forw(self.downsample(video3_forw), reverse=False)
        video4_back = self.CL4_back(self.downsample(video3_back), reverse=True)
        video4_out = torch.cat([video4_forw[:,-1], video4_back[:,-1]], dim=1)
        y4 = video4_out

        video5_forw = self.CL5_forw(self.downsample(video4_forw), reverse=False)
        video5_back = self.CL5_back(self.downsample(video4_back), reverse=True)
        video5_out = torch.cat([video5_forw[:,-1], video5_back[:,-1]], dim=1)
        y5 = video5_out

        # print(y1.shape)
        # print(y2.shape)
        # print(y3.shape)
        # print(y4.shape)
        # print(y5.shape)

        out1 = self.decoder1(y5, y4)
        out2 = self.decoder2(out1, y3)
        out3 = self.decoder3(out2, y2)
        out4 = self.decoder4(out3, y1)

        # print(out1.shape)
        # print(out2.shape)
        # print(out3.shape)
        # print(out4.shape)

        out = self.head(out4)

        return out1, out2, out3, out4, out


if __name__ == '__main__':
    net = Network(seq_len=4, num_layers=2).cuda()
    # def getModelSize(model):
    #     param_size = 0
    #     param_sum = 0
    #     for param in model.parameters():
    #         param_size += param.nelement() * param.element_size()
    #         param_sum += param.nelement()
    #     buffer_size = 0
    #     buffer_sum = 0
    #     for buffer in model.buffers():
    #         buffer_size += buffer.nelement() * buffer.element_size()
    #         buffer_sum += buffer.nelement()
    #     all_size = (param_size + buffer_size) / 1024 / 1024
    #     print('模型总大小为：{:.3f}MB'.format(all_size))
    #     print('模型参数量为：{:.3f}M'.format(param_sum/1e6))
    #     # return (param_size, param_sum, buffer_size, buffer_sum, all_size)
    # getModelSize(net)
    # # print(net)
    x = torch.ones([8, 3, 256, 256]).cuda()
    y = net(x,x)
    # print(y[-1].shape)

    # x = torch.ones([1, 3, 256, 256]).cuda()
    # flops, params = profile(net, inputs=(x,x))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("FLOPs: %s" % (flops))
    # print("params: %s" % (params))