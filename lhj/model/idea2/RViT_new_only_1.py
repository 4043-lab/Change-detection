import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.Transformer.RViT import RViT
from model.Transformer.RViT_new import RViT

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



class Network(nn.Module):
    def __init__(self, seq_len=4):
        super(Network, self).__init__()

        self.seq_len = seq_len
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.RViT = RViT(in_ch=64, seq_len=seq_len)
        self.conv_p2v = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = decoder(256, 128)
        self.decoder2 = decoder(128, 128)

        self.head = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)

        self.kaiming_init()

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def pair_to_video(self, im1, im2, len):
        device = im1.device
        delta = 1.0 / (len - 1)
        steps = torch.arange(len, dtype=torch.float).view(1, -1, 1, 1, 1).to(device)
        interped = im1.unsqueeze(1) + ((im2 - im1) * delta).unsqueeze(1) * steps
        interped = interped.transpose(0, 1)
        # return interped
        assert len >= 2
        video_list = []
        for i in range(1, len):
            dis = self.conv_p2v(torch.abs(interped[i] - interped[i - 1]))
            dis = dis.unsqueeze(0)
            video_list.append(dis)
        video_dis = torch.cat(video_list, dim=0)
        return video_dis

    def forward(self, t):
        t1 = t[0]
        t2 = t[1]

        x1_e1 = self.stem(t1)
        x1_e2 = self.stem(t2)

        video = self.pair_to_video(x1_e1, x1_e2, self.seq_len+1)
        # print(video.shape)
        rvit_out = self.RViT(video)
        v3 = rvit_out[0]
        v4 = rvit_out[1]
        v5 = rvit_out[2]

        out1 = self.decoder1(v5, v4)
        out2 = self.decoder2(out1, v3)
        out = F.interpolate(out2, scale_factor=4, mode='bilinear', align_corners=True)
        out = self.head(out)
        return out


if __name__ == '__main__':
    net = Network(seq_len=4).cuda()
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
    x = torch.ones([2, 2, 3, 256, 256]).cuda()
    y = net(x)
    print(y.shape)