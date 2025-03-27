import torch
import torch.nn as nn
import torch.nn.functional as F


class Basicblock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super(Basicblock, self).__init__()

        self.down_stride = 2 if downsample else 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=self.down_stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=self.down_stride, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.down_stride:
            res = self.downsample(res)
        out = res + x
        out = self.relu(out)
        return out


class FR_block(nn.Module):
    def __init__(self, h, w):
        super(FR_block, self).__init__()
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(1, w), stride=1)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(h, 1), stride=1)
        self.avg_max_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        batch, c, h, w = x.size()

        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)
        spatial_context = self.avg_max_conv(pool_out).squeeze(1)
        w_attention = F.softmax(spatial_context.clone(), dim=2)
        h_attention = F.softmax(torch.transpose(spatial_context, 1, 2), dim=2)
        f_cxhx1 = self.avg_pool1(x)
        f_hxc = f_cxhx1.flatten(2).transpose(1, 2)
        f_cx1xw = self.avg_pool2(x)
        f_wxc = f_cx1xw.flatten(2).transpose(1, 2)
        fh_attn = torch.bmm(w_attention, f_wxc).transpose(1, 2).unsqueeze(-1).contiguous()
        fw_attn = torch.bmm(h_attention, f_hxc).transpose(1, 2).unsqueeze(-2).contiguous()

        fh_attn = fh_attn.repeat(1, 1, 1, w)
        fh_out = fh_attn.permute(0, 2, 1, 3)

        fw_attn = fw_attn.repeat(1, 1, h, 1)
        fw_out = fw_attn.permute(0, 3, 1, 2)

        out = torch.cat((fh_out, fw_out), 1)

        return out


class CrossAtt(nn.Module):
    def __init__(self, in_channels, H, W):
        super().__init__()
        # self.in_channels = in_channels
        #
        self.query1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.key1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        #
        self.query2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.key2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        self.FR_q1 = FR_block(H, W)
        self.FR_q2 = FR_block(H, W)
        self.FR_k1 = FR_block(H, W)
        self.FR_k2 = FR_block(H, W)
        self.FR_v1 = FR_block(H, W)
        self.FR_v2 = FR_block(H, W)
        self.FR_in1 = FR_block(H, W)
        self.FR_in2 = FR_block(H, W)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(in_channels),
                                      nn.ReLU())  # conv_f

    def forward(self, input1, input2):
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1)
        k1 = self.key1(input1)
        v1 = self.value1(input1)

        q2 = self.query2(input2)
        k2 = self.key2(input2)
        v2 = self.value2(input2)

        # q = torch.cat((q1, q2), 1)
        query_2 = self.FR_q2(q2)
        key_1 = self.FR_k1(k1).permute(0, 1, 3, 2)
        value_1 = self.FR_v1(v1)
        attn_matrix1 = torch.matmul(query_2, key_1)
        attn_matrix1 = self.softmax(attn_matrix1)
        out1 = torch.matmul(attn_matrix1, value_1)
        out1 = out1.permute(0, 2, 1, 3)
        out1_list = torch.chunk(out1, 2, dim=2)
        out1_h, out1_w = out1_list
        out1 = out1_h + out1_w
        out1 = self.gamma * out1 + input1

        query_1 = self.FR_q1(q1)
        key_2 = self.FR_k2(k2).permute(0, 1, 3, 2)
        value_2 = self.FR_v2(v2)
        attn_matrix2 = torch.matmul(query_1, key_2)
        attn_matrix2 = self.softmax(attn_matrix2)
        out2 = torch.matmul(attn_matrix2, value_2)
        out2 = out2.permute(0, 2, 1, 3)
        out2_list = torch.chunk(out2, 2, dim=2)
        out2_h, out2_w = out2_list
        out2 = out2_h + out2_w
        out2 = self.beta * out2 + input2

        # feat_sum = torch.abs(out1 - out2)
        feat_sum = self.conv_cat(torch.cat((out1, out2), 1))
        return feat_sum
        # return feat_sum, out1, out2


class decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(decoder, self).__init__()
        self.conv = Basicblock(in_ch, out_ch, downsample=False)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.conv(x1)
        out = x1 + x2
        return out


class Resnet_basic(nn.Module):
    def __init__(self):
        super(Resnet_basic, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.res1 = Bottleneck(64, 128)
        # self.res2 = Bottleneck(128, 256)
        # # self.res3 = Bottleneck(256, 512)
        # self.res3 = nn.Sequential(
        #     Bottleneck(256, 512),
        #     Bottleneck(512, 512, downsample=False),
        #     Bottleneck(512, 512, downsample=False)
        # )
        # self.res4 = Bottleneck(512, 512)

        self.layer1 = nn.Sequential(
            Basicblock(64, 64, downsample=False),
            Basicblock(64, 64, downsample=False),
            Basicblock(64, 64, downsample=False)
        )
        self.layer2 = nn.Sequential(
            Basicblock(64, 128),
            Basicblock(128, 128, downsample=False),
            Basicblock(128, 128, downsample=False),
            Basicblock(128, 128, downsample=False)
        )
        # self.res3 = Bottleneck(256, 512)
        self.layer3 = nn.Sequential(
            Basicblock(128, 256),
            Basicblock(256, 256, downsample=False),
            Basicblock(256, 256, downsample=False),
            Basicblock(256, 256, downsample=False),
            Basicblock(256, 256, downsample=False),
            Basicblock(256, 256, downsample=False)
        )
        self.layer4 = nn.Sequential(
            Basicblock(256, 512),
            Basicblock(512, 512, downsample=False),
            Basicblock(512, 512, downsample=False),
        )
        self.crossattn1 = CrossAtt(64, 64, 64)
        self.crossattn2 = CrossAtt(64, 64, 64)
        self.crossattn3 = CrossAtt(128, 32, 32)
        self.crossattn4 = CrossAtt(256, 16, 16)
        self.crossattn5 = CrossAtt(512, 8, 8)

        self.decoder1 = decoder(512, 256)
        self.decoder2 = decoder(256, 128)
        self.decoder3 = decoder(128, 64)
        self.decoder4 = decoder(64, 64)

        self.head = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)

        self.kaiming_init()

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, t1, t2):

        x1_e1 = self.maxpool(self.stem(t1))
        x2_e1 = self.layer1(x1_e1)
        x3_e1 = self.layer2(x2_e1)
        x4_e1 = self.layer3(x3_e1)
        x5_e1 = self.layer4(x4_e1)
        x1_e2 = self.maxpool(self.stem(t2))
        x2_e2 = self.layer1(x1_e2)
        x3_e2 = self.layer2(x2_e2)
        x4_e2 = self.layer3(x3_e2)
        x5_e2 = self.layer4(x4_e2)

        # return x3_e1, x4_e1, x5_e1, x3_e2, x4_e2, x5_e2

        # y1 = torch.cat([x1_e1, x1_e2], dim=1)
        # y2 = torch.cat([x2_e1, x2_e2], dim=1)
        # y3 = torch.cat([x3_e1, x3_e2], dim=1)
        # y4 = torch.cat([x4_e1, x4_e2], dim=1)
        # y5 = torch.cat([x5_e1, x5_e2], dim=1)
        y1 = self.crossattn1(x1_e1, x1_e2)
        y2 = self.crossattn2(x2_e1, x2_e2)
        y3 = self.crossattn3(x3_e1, x3_e2)
        y4 = self.crossattn4(x4_e1, x4_e2)
        y5 = self.crossattn5(x5_e1, x5_e2)

        # return y2, y3, y4, y5

        out1 = self.decoder1(y5, y4)
        out2 = self.decoder2(out1, y3)
        out3 = self.decoder3(out2, y2)
        out4 = out3 + y1
        out4 = F.interpolate(out4, scale_factor=4, mode='bilinear', align_corners=True)
        out = self.head(out4)
        #
        return out


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A = torch.randn(4, 3, 256, 256).to(device)
    B = torch.randn(4, 3, 256, 256).to(device)
    model = Resnet_basic().to(device)
    c1 = model(A, B)
    # model = CrossAtt(3, 256, 256).to(device)
    # c2 = model(A, B)
