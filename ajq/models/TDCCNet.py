import torch
import torch.nn as nn
import torch.nn.functional as F
import models.model_utils.functional as LF


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class SBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_func=nn.ReLU(inplace=False)):
        super(SBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.conv1(x)
        out = self.conv2(res)

        return self.relu(out + res)


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

        feat_sum = self.conv_cat(torch.cat((out1, out2), 1))
        return feat_sum


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class Multisize_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Multisize_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        self.conv1 = BasicConv2d(out_channels, out_channels // 4, 3, 1, 1)
        self.conv2 = BasicConv2d(out_channels, out_channels // 4, 3, 2, 2)
        self.conv3 = BasicConv2d(out_channels, out_channels // 4, 3, 4, 4)
        self.conv4 = BasicConv2d(out_channels, out_channels // 4, 3, 6, 6)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // 8),
            nn.ReLU(),
            nn.Linear(out_channels // 8, out_channels))

        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        bs, c, h, w = x.shape

        x_0 = self.conv_in(x)
        x1 = self.conv1(x_0)
        x2 = self.conv2(x_0)
        x3 = self.conv3(x_0)
        x4 = self.conv4(x_0)

        x_fuse = torch.cat((x1, x2, x3, x4), 1)
        x_fuse_gap = self.gap(x_fuse).view(bs, self.out_channels)
        x_fuse_fc = self.fc(x_fuse_gap).view(bs, self.out_channels, 1, 1)
        x_attn = torch.sigmoid(x_fuse_fc)
        x_out = self.conv_out(x_fuse * x_attn + x_fuse)
        out = x_out + self.conv_res(x)

        out = self.relu(out)

        return out


class EdgeGuidance(nn.Module):
    def __init__(self):
        super(EdgeGuidance, self).__init__()
        self.up3 = up_conv(256, 128)
        self.up2 = up_conv(128, 64)
        self.up1 = up_conv(64, 32)
        self.out_conv = BasicConv2d(32, 32, 3, 1, 1)

    def forward(self, input4, input3, input2, input1):
        x3 = self.up3(input4) + input3
        x2 = self.up2(x3) + input2
        x1 = self.up1(x2) + input1
        out = self.out_conv(x1)
        return out


class Difference_enhance_module(nn.Module):
    def __init__(self, pool_scale, channels):
        super(Difference_enhance_module, self).__init__()
        self.pool_scale = pool_scale
        self.channels = channels

        self.pool_branches = nn.ModuleList()
        for idx in range(len(self.pool_scale)):
            self.pool_branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(self.pool_scale[idx]),
                nn.Conv2d(self.channels, self.channels // self.pool_scale[idx] ** 2, kernel_size=1, stride=1,
                          padding=0, bias=False)
            ))

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 16, False),
            nn.ReLU(),
            nn.Linear(channels // 16, channels, False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.abs(x1 - x2)
        b, c, h, w = x.size()

        outs = []
        for _, pool_layer in enumerate(self.pool_branches):
            out = pool_layer(x)
            reshape_out = out.flatten(1)
            outs.append(reshape_out)
        center = (outs[0] + outs[1] + outs[2])
        context = self.fc(center)

        attn = self.sigmoid(context).view([b, c, 1, 1])
        x1_out = attn * x1 + x1
        x2_out = attn * x2 + x2
        out = x1_out - x2_out
        return out, x1_out, x2_out


class TDCC(nn.Module):
    def __init__(self):
        super().__init__()

        nb_filter = [32, 64, 128, 256]
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0 = nn.Conv2d(6, nb_filter[0], kernel_size=3, stride=1, padding=1)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.bn0 = nn.BatchNorm2d(64)
        self.acf = nn.ReLU(inplace=True)

        self.pool_scale = [1, 2, 4]
        self.de1 = Difference_enhance_module(self.pool_scale, 32)
        self.de2 = Difference_enhance_module(self.pool_scale, 64)
        self.de3 = Difference_enhance_module(self.pool_scale, 128)
        self.de4 = Difference_enhance_module(self.pool_scale, 256)

        self.crossattn1 = CrossAtt(32, 256, 256)
        self.crossattn2 = CrossAtt(64, 128, 128)
        self.crossattn3 = CrossAtt(128, 64, 64)
        self.crossattn4 = CrossAtt(256, 32, 32)

        self.msblock0 = Multisize_block(6, 32)
        self.msblock1 = Multisize_block(32, 64)
        self.msblock2 = Multisize_block(64, 128)
        self.msblock3 = Multisize_block(128, 256)

        self.bn_xy1 = nn.BatchNorm2d(32)
        self.bn_xy2 = nn.BatchNorm2d(64)
        self.bn_xy3 = nn.BatchNorm2d(128)
        self.bn_xy4 = nn.BatchNorm2d(256)

        self.conv1 = SBlock(3, 32)
        self.conv2 = SBlock(32, 64)
        self.conv3 = SBlock(64, 128)
        self.conv4 = SBlock(128, 256)

        self.c1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.c2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.c3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.c4 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        self.edge_guidance = EdgeGuidance()


        self.Up_conv5 = conv_block(256, 256)
        self.Up4 = up_conv(nb_filter[3], nb_filter[2])
        self.Up_conv4 = conv_block(256, 128)
        self.Up3 = up_conv(nb_filter[2], nb_filter[1])
        self.Up_conv3 = conv_block(128, 64)
        self.Up2 = up_conv(nb_filter[1], nb_filter[0])
        self.Up_conv2 = conv_block(64, 32)

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.edge_out = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0)
        self.out = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0)

    def encoder(self, x, y):
        x1 = self.conv1(x)  # x1:(32,256,256)
        y1 = self.conv1(y)  # y1:(32,256,256)
        # d1_xy = torch.abs(x1 - y1)
        d1_xy, x1, y1 = self.de1(x1, y1)
        d1_xy = self.crossattn1(x1, y1)
        input = torch.cat((x, y), 1)  # input:(6,256,256)

        # --------------------------------------------------------
        # xy1 = self.conv_cat(input)
        xy1 = self.msblock0(input)
        xy1_out = torch.cat((d1_xy, xy1), 1)  # x:(64,256,256)
        xy1_out = self.c1(xy1_out)
        xy1_out = self.bn_xy1(xy1_out)
        xy1_out = self.acf(xy1_out)
        # --------------------------------------------------------

        x1 = self.pool(x1)
        y1 = self.pool(y1)
        x2 = self.conv2(x1)  # x2:(64,128,128)
        y2 = self.conv2(y1)  # y2:(64,128,128)
        # d2_xy = torch.abs(x2 - y2)  # d2_xy:(64,128,128)
        d2_xy, x2, y2 = self.de2(x2, y2)
        d2_xy = self.crossattn2(x2, y2)
        # --------------------------------------------------------
        xy2 = self.pool(self.msblock1(xy1))
        xy2_out = torch.cat((d2_xy, xy2), 1)
        xy2_out = self.c2(xy2_out)
        xy2_out = self.bn_xy2(xy2_out)
        xy2_out = self.acf(xy2_out)
        # --------------------------------------------------------

        x2 = self.pool(x2)
        y2 = self.pool(y2)
        x3 = self.conv3(x2)  # x3:(128,64,64)
        y3 = self.conv3(y2)  # y3:(128,64,64)
        # d3_xy = torch.abs(x3 - y3)  # d3_xy:(128,64,64)
        d3_xy, x3, y3 = self.de3(x3, y3)
        d3_xy = self.crossattn3(x3, y3)
        # --------------------------------------------------------
        xy3 = self.pool(self.msblock2(xy2))
        xy3_out = torch.cat((d3_xy, xy3), 1)
        xy3_out = self.c3(xy3_out)
        xy3_out = self.bn_xy3(xy3_out)
        xy3_out = self.acf(xy3_out)
        # --------------------------------------------------------

        x3 = self.pool(x3)
        y3 = self.pool(y3)
        x4 = self.conv4(x3)  # x4:(256,32,32)
        y4 = self.conv4(y3)  # y4:(256,32,32)
        # d4_xy = torch.abs(x4 - y4)  # d4_xy:(256,32,32)
        d4_xy, x4, y4 = self.de4(x4, y4)
        d4_xy = self.crossattn4(x4, y4)
        # --------------------------------------------------------
        xy4 = self.pool(self.msblock3(xy3))
        xy4_out = torch.cat((d4_xy, xy4), 1)
        xy4_out = self.c4(xy4_out)
        xy4_out = self.bn_xy4(xy4_out)
        xy4_out = self.acf(xy4_out)
        # --------------------------------------------------------

        edge_branch = self.edge_guidance(xy4, xy3, xy2, xy1)
        # edge_branch = self.sso(xy4, xy3, xy2, xy1)

        # return xy1_out, xy2_out, xy3_out, xy4_out
        # return d1_xy, d2_xy, d3_xy, d4_xy
        return xy1_out, xy2_out, xy3_out, xy4_out, edge_branch

    def forward(self, x1, x2):
        # x1, x2, x3, x4 = self.encoder(x1, x2)
        x1, x2, x3, x4, edge_branch = self.encoder(x1, x2)

        x4 = self.Up_conv5(x4)
        x4 = self.Up4(x4)
        x3 = torch.cat((x3, x4), 1)

        x3 = self.Up_conv4(x3)
        x3 = self.Up3(x3)
        x2 = torch.cat((x2, x3), 1)

        x2 = self.Up_conv3(x2)
        x2 = self.Up2(x2)
        x1 = torch.cat((x1, x2), 1)

        x1 = self.Up_conv2(x1)

        # ------------------------------------
        # out = self.out(x1)
        # return out
        # ------------------------------------
        x = self.out_conv(x1 + edge_branch)
        out = self.out(x)

        edge = self.edge_out(edge_branch)
        return out, edge


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A = torch.randn(2, 3, 256, 256).to(device)
    B = torch.randn(2, 3, 256, 256).to(device)
    model = TDCC().to(device)
    c1 = model(A, B)
