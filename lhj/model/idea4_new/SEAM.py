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

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4096, 2048, 1, 1, 0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = decoder(256+512, 512, downsample=True)
        self.decoder2 = decoder(512+1024, 1024, downsample=True)
        self.decoder3 = decoder(1024+2048, 2048, downsample=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(2048, 256, 1, bias=False)
        self.fc2 = nn.Conv2d(256, 2, 1, bias=False)

        self.conv_down_3 = nn.Conv2d(1024, 64, 1, bias=False)
        self.conv_down_4 = nn.Conv2d(2048, 128, 1, bias=False)
        self.conv_down_all = nn.Conv2d(64+128+3+3, 64+128, 1, bias=False)

    def PCM(self, cam, f):
        n, c, h, w = f.size()
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        # f = self.f9(f)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)

        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)

        return cam_rv

    # def forward(self, t1, t2):
    def forward_origin(self, t):
        t1 = t[0]
        t2 = t[1]
        x1_e1, x2_e1, x3_e1, x4_e1 = self.resnet(t1)
        x1_e2, x2_e2, x3_e2, x4_e2 = self.resnet(t2)

        y1 = torch.cat([x1_e1, x1_e2], dim=1)
        y2 = torch.cat([x2_e1, x2_e2], dim=1)
        y3 = torch.cat([x3_e1, x3_e2], dim=1)
        y4 = torch.cat([x4_e1, x4_e2], dim=1)

        y1 = self.conv1(y1)
        y2 = self.conv2(y2)
        y3 = self.conv3(y3)
        y4 = self.conv4(y4)

        # print(y1.shape)
        # print(y2.shape)
        # print(y3.shape)
        # print(y4.shape)

        out1 = self.decoder1(y1, y2)
        out2 = self.decoder2(out1, y3)
        out3 = self.decoder3(out2, y4)

        cam3 = F.relu(self.fc1(out3))
        cam3 = self.fc2(cam3)

        # n,c,h,w = cam3.size()
        # with torch.no_grad():
        #     cam_d = F.relu(cam3.detach())
        #     cam_d_max = torch.max(cam_d.view(n,c,-1), dim=-1)[0].view(n,c,1,1)+1e-5
        #     cam_d_norm = F.relu(cam_d-1e-5)/cam_d_max
        #     cam_d_norm[:,0,:,:] = 1-torch.max(cam_d_norm[:,1:,:,:], dim=1)[0]
        #     cam_max = torch.max(cam_d_norm[:,1:,:,:], dim=1, keepdim=True)[0]
        #     cam_d_norm[:,1:,:,:][cam_d_norm[:,1:,:,:] < cam_max] = 0
        #
        # f3 = F.relu(self.conv_down_3(y3.detach()), inplace=True)
        # f4 = F.relu(self.conv_down_4(y4.detach()), inplace=True)
        # t1_s = F.interpolate(t1, size=f4.shape[-1], mode='bilinear', align_corners=True)
        # t2_s = F.interpolate(t2, size=f4.shape[-1], mode='bilinear', align_corners=True)
        # f = torch.cat([f3, f4, t1_s, t2_s], dim=1)
        # f = self.conv_down_all(f)
        # # print(f.shape)

        # cam3_p = self.PCM(cam_d_norm, f)

        cls3 = self.avgpool(cam3).squeeze(3).squeeze(2)

        cam3 = F.softmax(cam3, dim=1)
        # cam3_p = F.softmax(cam3_p, dim=1)

        cam3_p = cam3

        return cam3, cam3_p, cls3

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

        y1 = torch.cat([x1_e1, x1_e2], dim=1)
        y2 = torch.cat([x2_e1, x2_e2], dim=1)
        y3 = torch.cat([x3_e1, x3_e2], dim=1)
        y4 = torch.cat([x4_e1, x4_e2], dim=1)

        y1 = self.conv1(y1)
        y2 = self.conv2(y2)
        y3 = self.conv3(y3)
        y4 = self.conv4(y4)

        # print(y1.shape)
        # print(y2.shape)
        # print(y3.shape)
        # print(y4.shape)

        out1 = self.decoder1(y1, y2)
        out2 = self.decoder2(out1, y3)
        out3 = self.decoder3(out2, y4)

        cam3 = F.relu(self.fc1(out3))
        cam3 = self.fc2(cam3)

        # n,c,h,w = cam3.size()
        # with torch.no_grad():
        #     cam_d = F.relu(cam3.detach())
        #     cam_d_max = torch.max(cam_d.view(n,c,-1), dim=-1)[0].view(n,c,1,1)+1e-5
        #     cam_d_norm = F.relu(cam_d-1e-5)/cam_d_max
        #     cam_d_norm[:,0,:,:] = 1-torch.max(cam_d_norm[:,1:,:,:], dim=1)[0]
        #     cam_max = torch.max(cam_d_norm[:,1:,:,:], dim=1, keepdim=True)[0]
        #     cam_d_norm[:,1:,:,:][cam_d_norm[:,1:,:,:] < cam_max] = 0
        #
        # f3 = F.relu(self.conv_down_3(y3.detach()), inplace=True)
        # f4 = F.relu(self.conv_down_4(y4.detach()), inplace=True)
        # t1_s = F.interpolate(t1, size=f4.shape[-1], mode='bilinear', align_corners=True)
        # t2_s = F.interpolate(t2, size=f4.shape[-1], mode='bilinear', align_corners=True)
        # f = torch.cat([f3, f4, t1_s, t2_s], dim=1)
        # f = self.conv_down_all(f)
        # # print(f.shape)
        #
        # cam3_p = self.PCM(cam_d_norm, f)

        cls3 = self.avgpool(cam3).squeeze(3).squeeze(2)

        cam3 = F.softmax(cam3, dim=1)
        # cam3_p = F.softmax(cam3_p, dim=1)
        cam3 = F.interpolate(cam3, scale_factor=2, mode='bilinear', align_corners=True)
        # cam3_p = F.interpolate(cam3_p, scale_factor=2, mode='bilinear', align_corners=True)

        cam3_p = cam3

        return cam3, cam3_p, cls3

    def forward(self, t):
        cam, cam_p, cls = self.forward_origin(t)
        cam_, cam_p_, cls_ = self.forward_affine(t)
        return cam, cam_p, cam_, cam_p_, cls, cls_

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
    # for i in y:
    #     print(i.shape)
    # print(y.shape)

    x = torch.rand([2, 1, 3, 256, 256]).cuda()
    flops, params = profile(net, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" % (flops))
    print("params: %s" % (params))