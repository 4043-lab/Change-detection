import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models import trunc_normal_
from timm.models.layers.weight_init import trunc_normal_
from einops import rearrange, repeat
# import torch.distributed as dist

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
    def __init__(self):
        super(Network, self).__init__()

        # proto parameters
        self.gamma = 0.999
        self.num_prototype = 3
        self.use_prototype = True
        self.update_prototype = True
        # self.pretrain_prototype = False
        self.num_classes = 2
        self.proto_channels = 128
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, self.proto_channels), requires_grad=True)

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.res1 = nn.Sequential(
            Bottleneck(64, 128),
            Bottleneck(128, 128, downsample=False)
        )
        self.res2 = nn.Sequential(
            Bottleneck(128, 256),
            Bottleneck(256, 256, downsample=False)
        )
        self.res3 = nn.Sequential(
            Bottleneck(256, 512),
            Bottleneck(512, 512, downsample=False)
        )
        self.res4 = nn.Sequential(
            Bottleneck(512, 512),
            Bottleneck(512, 512, downsample=False)
        )

        self.decoder1 = decoder(1024, 512)
        self.decoder2 = decoder(512, 256)
        self.decoder3 = decoder(256, 128)
        self.decoder4 = decoder(128, 128)

        # self.head = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(128, self.proto_channels, kernel_size=1, stride=1, padding=0)
        self.feat_norm = nn.LayerNorm(self.proto_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        trunc_normal_(self.prototypes, std=0.02)

        self.kaiming_init()

    def distributed_sinkhorn(self, out, sinkhorn_iterations=3, epsilon=0.05):
        L = torch.exp(out / epsilon).t()  # K x B
        B = L.shape[1]
        K = L.shape[0]

        # make the matrix sums to 1
        sum_L = torch.sum(L)
        L /= sum_L

        for _ in range(sinkhorn_iterations):
            L /= torch.sum(L, dim=1, keepdim=True)
            L /= K

            L /= torch.sum(L, dim=0, keepdim=True)
            L /= B

        L *= B
        L = L.t()

        indexs = torch.argmax(L, dim=1)
        # L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()
        L = F.gumbel_softmax(L, tau=0.5, hard=True)

        return L, indexs

    def momentum_update(self, old_value, new_value, momentum, debug=False):
        update = momentum * old_value + (1 - momentum) * new_value
        if debug:
            print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
                momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
                torch.norm(update, p=2)))
        return update

    def l2_normalize(self, c):
        return F.normalize(c, p=2, dim=-1)

    # _c:原始输出  out_seg:2通道输出  gt_seg:label  masks:输出与原型相似性度量矩阵
    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        # print(out_seg.shape)
        pred_seg = torch.max(out_seg, 1)[1]
        # print(pred_seg.shape)
        mask = (gt_seg == pred_seg.view(-1))  # 预测正确mask

        # print(_c.shape)
        # print((self.prototypes.view(-1, self.prototypes.shape[-1]).t()).shape)
        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())  # t() 转置

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = self.distributed_sinkhorn(init_q)

            m_k = mask[gt_seg == k]

            c_k = _c[gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)

                new_value = self.momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :], momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)

        self.prototypes = nn.Parameter(self.l2_normalize(protos), requires_grad=False)

        # if dist.is_available() and dist.is_initialized():
        #     protos = self.prototypes.data.clone()
        #     dist.all_reduce(protos.div_(dist.get_world_size()))
        #     self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, t, gt_semantic_seg=None, pretrain_prototype=False):
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

        y1 = torch.cat([x1_e1, x1_e2], dim=1)
        y2 = torch.cat([x2_e1, x2_e2], dim=1)
        y3 = torch.cat([x3_e1, x3_e2], dim=1)
        y4 = torch.cat([x4_e1, x4_e2], dim=1)
        y5 = torch.cat([x5_e1, x5_e2], dim=1)

        out1 = self.decoder1(y5, y4)
        out2 = self.decoder2(out1, y3)
        out3 = self.decoder3(out2, y2)
        out4 = self.decoder4(out3, y1)

        c = self.head(out4)
        _c = rearrange(c, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c)
        _c = self.l2_normalize(_c)

        self.prototypes.data.copy_(self.l2_normalize(self.prototypes)) #原型进行l2标准化

        # n: h*w, k: num_class, m: num_prototype
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)  #计算和每个原型的相似度
        # print(masks.shape)

        out_seg = torch.amax(masks, dim=1)  #每个类别中选择最相似的那个原型
        out_seg = self.mask_norm(out_seg)
        # print(out_seg.shape)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=out4.shape[0], h=out4.shape[2])

        if pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            # print('update proto.')
            # gt_seg = F.interpolate(gt_semantic_seg.float(), size=out4.size()[2:], mode='nearest').view(-1)
            gt_seg = gt_semantic_seg.float().view(-1)
            contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg, masks)
            return {'seg': out_seg, 'logits': contrast_logits, 'target': contrast_target}

        return out_seg


if __name__ == '__main__':
    net = Network()
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
    x = torch.ones([2, 8, 3, 256, 256])
    label = torch.randint(0, 2, (8, 256, 256))
    y = net(x, label)
    print(y['seg'].shape)
    print(y['logits'].shape)
    print(y['target'].shape)