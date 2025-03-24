import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from model.idea4.resnet import resnet50

class decoder(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=True):
        super(decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.upsample = upsample

    def forward(self, x1, x2):
        if self.upsample:
            x2 = F.interpolate(x2, scale_factor=x1.shape[-1]//x2.shape[-1], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2], dim=1)
        out = self.conv(x)
        return out

class Network(nn.Module):
    def __init__(self, affine_type='scale'):
        super(Network, self).__init__()

        self.resnet = resnet50(pretrained=True)
        self.affine_type = affine_type

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4096, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = decoder(256, 256, upsample=True)
        self.decoder2 = decoder(256, 256, upsample=True)
        self.decoder3 = decoder(256, 256, upsample=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1_1 = nn.Conv2d(256, 256, 1, bias=False)
        self.fc1_2 = nn.Conv2d(256, 2, 1, bias=False)
        self.fc2_1 = nn.Conv2d(256, 256, 1, bias=False)
        self.fc2_2 = nn.Conv2d(256, 2, 1, bias=False)
        self.fc3_1 = nn.Conv2d(256, 256, 1, bias=False)
        self.fc3_2 = nn.Conv2d(256, 2, 1, bias=False)

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

        cam1 = F.relu(self.fc1_1(out1))
        cam1 = self.fc1_2(cam1)
        cls1 = self.avgpool(cam1).squeeze(3).squeeze(2)

        cam2 = F.relu(self.fc2_1(out2))
        cam2 = self.fc2_2(cam2)
        cls2 = self.avgpool(cam2).squeeze(3).squeeze(2)

        cam3 = F.relu(self.fc3_1(out3))
        cam3 = self.fc3_2(cam3)
        cls3 = self.avgpool(cam3).squeeze(3).squeeze(2)

        cam1 = F.softmax(cam1, dim=1)
        cam2 = F.softmax(cam2, dim=1)
        cam3 = F.softmax(cam3, dim=1)

        return cam1, cam2, cam3,\
            out1, out2, out3,\
            x1_e1, x2_e1, x3_e1, x4_e1, x1_e2, x2_e2, x3_e2, x4_e2,\
            cls1, cls2, cls3

    def forward_affine(self, t):
        t1 = t[0]
        t2 = t[1]
        if self.affine_type == 'scale':
            t1 = F.interpolate(t1, scale_factor=0.5, mode='bilinear', align_corners=True)
            t2 = F.interpolate(t2, scale_factor=0.5, mode='bilinear', align_corners=True)
        elif self.affine_type == 'flip':
            t1 = transforms.RandomHorizontalFlip(p=1)(t1)
            t2 = transforms.RandomHorizontalFlip(p=1)(t2)
        elif self.affine_type == 'rotation':
            t1 = transforms.RandomRotation((180, 180))(t1)
            t2 = transforms.RandomRotation((180, 180))(t2)
        else:
            raise NotImplementedError()

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

        cam1 = F.relu(self.fc1_1(out1))
        cam1 = self.fc1_2(cam1)
        cls1 = self.avgpool(cam1).squeeze(3).squeeze(2)

        cam2 = F.relu(self.fc2_1(out2))
        cam2 = self.fc2_2(cam2)
        cls2 = self.avgpool(cam2).squeeze(3).squeeze(2)

        cam3 = F.relu(self.fc3_1(out3))
        cam3 = self.fc3_2(cam3)
        cls3 = self.avgpool(cam3).squeeze(3).squeeze(2)

        cam1 = F.softmax(cam1, dim=1)
        cam2 = F.softmax(cam2, dim=1)
        cam3 = F.softmax(cam3, dim=1)

        if self.affine_type == 'scale':
            out1 = F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=True)
            out2 = F.interpolate(out2, scale_factor=2, mode='bilinear', align_corners=True)
            out3 = F.interpolate(out3, scale_factor=2, mode='bilinear', align_corners=True)
        elif self.affine_type == 'flip':
            out1 = transforms.RandomHorizontalFlip(p=1)(out1)
            out2 = transforms.RandomHorizontalFlip(p=1)(out2)
            out3 = transforms.RandomHorizontalFlip(p=1)(out3)
        elif self.affine_type == 'rotation':
            out1 = transforms.RandomRotation((180, 180))(out1)
            out2 = transforms.RandomRotation((180, 180))(out2)
            out3 = transforms.RandomRotation((180, 180))(out3)

        return cam1, cam2, cam3,\
            out1, out2, out3,\
            cls1, cls2, cls3

    def forward(self, t):
        cam1, cam2, cam3, out1, out2, out3, x1_e1, x2_e1, x3_e1, x4_e1, x1_e2, x2_e2, x3_e2, x4_e2, cls1, cls2, cls3 = self.forward_origin(t)
        cam1_, cam2_, cam3_, out1_, out2_, out3_, cls1_, cls2_, cls3_ = self.forward_affine(t)
        return cam1, cam2, cam3, cam1_, cam2_, cam3_,\
            out1, out2, out3, out1_, out2_, out3_,\
            x1_e1, x2_e1, x3_e1, x4_e1, x1_e2, x2_e2, x3_e2, x4_e2,\
            cls1, cls2, cls3, cls1_, cls2_, cls3_

if __name__ == '__main__':
    # net = Network(affine_type='scale').cuda()
    net = Network(affine_type='flip').cuda()
    # net = Network(affine_type='rotation').cuda()
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