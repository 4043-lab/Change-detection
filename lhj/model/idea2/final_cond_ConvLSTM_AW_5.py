import torch
import torch.nn as nn
import torch.nn.functional as F
from model.idea2.ConvLSTM_new2 import ConvLSTM
from model.Attention.AW import AW
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


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # model_load1 = 'result/idea2/resnet50_cond_up_4/LEVIR-CD-256/checkpoints/best_model_epoch069_f_score0.9122_iou0.8386.pth'
        # model_load1 = 'result/idea2/resnet50_cond_up_4/BCDD/checkpoints/best_model_epoch263_f_score0.9346_iou0.8772.pth'
        # model_load1 = 'result/idea2/resnet50_cond_up_4_2/SYSU-CD/checkpoints/best_model_epoch088_f_score0.8079_iou0.6777.pth'
        model_load1 = '../../result/idea2/resnet50_cond_up_4_2/SYSU-CD/checkpoints/best_model_epoch065_f_score0.8057_iou0.6746.pth'
        self.resnet = torch.load(model_load1).cuda()
        for name, param in self.resnet.named_parameters():
            param.requires_grad = False
        # model_load2 = 'result/idea2/ConvLSTM_new2_only_6/LEVIR-CD-256/checkpoints/best_model_epoch051_f_score0.9038_iou0.8245.pth'
        # model_load2 = 'result/idea2/ConvLSTM_new2_only_6/BCDD/checkpoints/best_model_epoch142_f_score0.9177_iou0.8480.pth'
        # model_load2 = 'result/idea2/ConvLSTM_new2_only_6/SYSU-CD/checkpoints/best_model_epoch006_f_score0.7899_iou0.6528.pth'
        model_load2 = '../../result/idea2/ConvLSTM_new2_only_6/SYSU-CD/checkpoints/best_model_epoch006_f_score0.7899_iou0.6528.pth'
        self.lstm = torch.load(model_load2).cuda()
        for name, param in self.lstm.named_parameters():
            param.requires_grad = False


        self.AW1 = AW(channel=512)
        self.AW2 = AW(channel=256)
        self.AW3 = AW(channel=128)
        self.AW4 = AW(channel=128)

        self.conv1 = Bottleneck(512, 256, downsample=False)
        self.conv2 = Bottleneck(256, 128, downsample=False)
        self.conv3 = Bottleneck(128, 128, downsample=False)
        self.conv4 = Bottleneck(128, 128, downsample=False)

        self.head = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)


    def forward(self, t1, t2):
        l1, l2, l3, l4, l_out = self.lstm(t1, t2)
        r1, r2, r3, r4, r_out = self.resnet(t1, t2)
        # print(l1.shape)
        # print(l2.shape)
        # print(l3.shape)
        # print(l4.shape)
        # print(l_out.shape)
        # print(r1.shape)
        # print(r2.shape)
        # print(r3.shape)
        # print(r4.shape)
        # print(r_out.shape)

        y1 = self.AW1(l1, r1)
        y2 = self.AW2(l2, r2)
        y3 = self.AW3(l3, r3)
        y4 = self.AW4(l4, r4)
        # print(y1.shape)
        # print(y2.shape)
        # print(y3.shape)
        # print(y4.shape)

        y1 = self.conv1(y1)
        y1 = F.interpolate(y1, scale_factor=2, mode='bilinear', align_corners=True, recompute_scale_factor=True)

        y2 = y1 + y2
        y2 = self.conv2(y2)
        y2 = F.interpolate(y2, scale_factor=2, mode='bilinear', align_corners=True, recompute_scale_factor=True)

        y3 = y2 + y3
        y3 = self.conv3(y3)
        y3 = F.interpolate(y3, scale_factor=2, mode='bilinear', align_corners=True, recompute_scale_factor=True)

        y4 = y3 + y4
        y4 = self.conv4(y4)

        out = self.head(y4)

        return [l_out, r_out, out]

if __name__ == '__main__':
    # net = ConvLSTM_video(seq_len=4, num_layers=2).cuda()
    # net = resnet_Cond().cuda()
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
    # print(net)
    x = torch.ones([8, 3, 256, 256]).cuda()
    y = net(x,x)
    print(y[-1].shape)

    x = torch.ones([1, 3, 256, 256]).cuda()
    flops, params = profile(net, inputs=(x,x,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" % (flops))
    print("params: %s" % (params))