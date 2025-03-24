import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from model.idea4.resnet import resnet50
from thop import profile, clever_format

class decoder(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super(decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.maxpooling = nn.MaxPool2d(2)
        self.downsample = downsample

    def forward(self, x1, x2):
        if self.downsample:
            x1 = self.maxpooling(x1)
        x = torch.cat([x1, x2], dim=1)
        out = self.conv(x)
        return out

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.resnet = resnet50(pretrained=True)

        self.mjs2 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1, bias=False),
            # nn.BatchNorm2d(2),
            # nn.ReLU(inplace=True)
        )
        self.mjs3 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1, bias=False),
            # nn.BatchNorm2d(2),
            # nn.ReLU(inplace=True)
        )
        self.mjs4 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1, bias=False),
            # nn.BatchNorm2d(2),
            # nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # def forward(self, t1, t2):
    def forward_origin(self, t):
        t1 = t[0]
        t2 = t[1]
        x1_e1, x2_e1, x3_e1, x4_e1 = self.resnet(t1)
        x1_e2, x2_e2, x3_e2, x4_e2 = self.resnet(t2)

        f2 = torch.abs(x2_e1 - x2_e2)
        f3 = torch.abs(x3_e1 - x3_e2)
        f4 = torch.abs(x4_e1 - x4_e2)

        # print(f2.shape)
        # print(f3.shape)
        # print(f4.shape)

        cam2 = self.mjs2(f2)
        cam3 = self.mjs3(f3)
        cam4 = self.mjs4(f4)

        cls2 = self.avgpool(cam2).squeeze(3).squeeze(2)
        cls3 = self.avgpool(cam3).squeeze(3).squeeze(2)
        cls4 = self.avgpool(cam4).squeeze(3).squeeze(2)

        cam2 = F.softmax(cam2, dim=1)
        cam3 = F.softmax(cam3, dim=1)
        cam4 = F.softmax(cam4, dim=1)

        return cam2, cam3, cam4, cls2, cls3, cls4

    def forward_affine(self, t):
        t1 = t[0]
        t2 = t[1]
        trf = transforms.Compose([
            transforms.Resize(int(t.shape[-1]/2))
        ])
        t1 = trf(t1)
        t2 = trf(t2)

        x1_e1, x2_e1, x3_e1, x4_e1 = self.resnet(t1)
        x1_e2, x2_e2, x3_e2, x4_e2 = self.resnet(t2)

        f2 = torch.abs(x2_e1 - x2_e2)
        f3 = torch.abs(x3_e1 - x3_e2)
        f4 = torch.abs(x4_e1 - x4_e2)

        # print(f2.shape)
        # print(f3.shape)
        # print(f4.shape)

        cam2 = self.mjs2(f2)
        cam3 = self.mjs3(f3)
        cam4 = self.mjs4(f4)

        cls2 = self.avgpool(cam2).squeeze(3).squeeze(2)
        cls3 = self.avgpool(cam3).squeeze(3).squeeze(2)
        cls4 = self.avgpool(cam4).squeeze(3).squeeze(2)

        cam2 = F.softmax(cam2, dim=1)
        cam3 = F.softmax(cam3, dim=1)
        cam4 = F.softmax(cam4, dim=1)

        cam2 = F.interpolate(cam2, scale_factor=2, mode='bilinear', align_corners=True)
        cam3 = F.interpolate(cam3, scale_factor=2, mode='bilinear', align_corners=True)
        cam4 = F.interpolate(cam4, scale_factor=2, mode='bilinear', align_corners=True)

        return cam2, cam3, cam4, cls2, cls3, cls4

    def forward(self, t):
        cam2, cam3, cam4, cls2, cls3, cls4 = self.forward_origin(t)
        cam2_, cam3_, cam4_, cls2_, cls3_, cls4_ = self.forward_affine(t)
        return cam2, cam3, cam4, cam2_, cam3_, cam4_, cls2, cls3, cls4, cls2_, cls3_, cls4_
        # return out

if __name__ == '__main__':
    net = Network().cuda()
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
    x = torch.rand([2, 4, 3, 256, 256]).cuda()
    y = net(x)
    for i in y:
        print(i.shape)
    # print(y.shape)

    x = torch.rand([2, 1, 3, 256, 256]).cuda()
    flops, params = profile(net, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" % (flops))
    print("params: %s" % (params))