import torch
import torch.nn as nn
import torch.nn.functional as F

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

class decoder_pred(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(decoder_pred, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x1,x2], dim=1)
        out = self.conv(x)
        return out

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        model_cls = '../../result/idea4/resnet50_cls_dsc/LEVIR-CD-256/checkpoints/best_model_epoch161_precision0.9224.pth'
        # model_cls = 'result/idea4/resnet50_cls_dsc/LEVIR-CD-256/checkpoints/best_model_epoch161_precision0.9224.pth'
        self.resnet = torch.load(model_cls).cuda()
        for name, param in self.resnet.named_parameters():
            param.requires_grad = False

        self.decoder1 = decoder(512, 256)
        self.decoder2 = decoder(256, 128)
        self.decoder3 = decoder(128, 64)
        self.decoder4 = decoder(64, 32)

        self.decoder1_pred = decoder_pred(512+256, 256)
        self.decoder2_pred = decoder_pred(256+128, 128)
        self.decoder3_pred = decoder_pred(128+64, 64)

        self.head = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, t1, t2):
        x1_e1, x2_e1, x3_e1, x4_e1, x5_e1, x1_e2, x2_e2, x3_e2, x4_e2, x5_e2, out_cls = self.resnet(t1, t2)
        out1_d1 = self.decoder1(x5_e1, x4_e1)
        out2_d1 = self.decoder2(out1_d1, x3_e1)
        out3_d1 = self.decoder3(out2_d1, x2_e1)
        out4_d1 = self.decoder4(out3_d1, x1_e1)

        out1_d2 = self.decoder1(x5_e2, x4_e2)
        out2_d2 = self.decoder2(out1_d2, x3_e2)
        out3_d2 = self.decoder3(out2_d2, x2_e2)
        out4_d2 = self.decoder4(out3_d2, x1_e2)

        y1 = torch.cat([out1_d1, out1_d2], dim=1)
        y2 = torch.cat([out2_d1, out2_d2], dim=1)
        y3 = torch.cat([out3_d1, out3_d2], dim=1)
        y4 = torch.cat([out4_d1, out4_d2], dim=1)

        out1 = self.decoder1_pred(y1, y2)
        out2 = self.decoder2_pred(out1, y3)
        out3 = self.decoder3_pred(out2, y4)

        out = self.head(out3)

        return out4_d1, out4_d2, out, out_cls


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
    x = torch.ones([4, 3, 256, 256]).cuda()
    y = net(x,x)
    for i in y:
        print(i.shape)