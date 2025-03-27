import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from ..builder import LOSSES

#self_supervised_consistency_regularization_loss
@LOSSES.register_module()
class SscrLoss(nn.Module):
    def __init__(self,cri_weight=1.0,crf_weight=1.0,loss_weight=1.0):
        super(SscrLoss, self).__init__()
        self.loss_weight=loss_weight
        self.cri_weigth=cri_weight
        self.crf_weight=crf_weight

    def affine_transform(self,img_torch):
        # 缩小 0.5
        # theta = torch.tensor([
        #     [2, 0, 0],
        #     [0, 2, 0]
        # ], dtype=torch.float).unsqueeze(dim=0)
        # # N, C, H, W = img_torch.size()
        # C, H, W = img_torch.size()
        # # theta = theta.repeat(N, 1, 1)
        # theta = theta.repeat(C, 1)
        # grid = F.affine_grid(theta, img_torch.size()).to(img_torch.device)
        # output = F.grid_sample(img_torch, grid)

        #最后一维度翻转
        trans = transforms.RandomHorizontalFlip(p=1)
        dim = -1
        if isinstance(img_torch, tuple):
            output = list(img_torch)
            for i in range(len(img_torch)):
                output[i] = trans(img_torch[i]).contiguous()
        else:
            dim = -1
            indices = [slice(None)] * img_torch.dim()
            indices[dim] = torch.arange(img_torch.size(dim) - 1, -1, -1,
                                        dtype=torch.long, device=img_torch.device)
            output = img_torch[tuple(indices)]

        # output = F.interpolate(img_torch, scale_factor=0.5, mode='bilinear')
        return output

    def forward(self,fpn_feature,_fpn_feature,if_cri=False):
        loss_func=nn.L1Loss()
        if self.loss_weight==0.0:
            return torch.tensor(0.0).cuda()
        # print('sscr__'*5,len(fpn_feature))
        # print(fpn_feature[0].size())

        cri_loss=0
        # feature = self.affine_transform(fpn_feature)
        # _feature=_fpn_feature.detach()
        # _feature = _fpn_feature
        # cri_loss += loss_func(feature, _feature)
        # print(type(fpn_feature),len(fpn_feature),fpn_feature[0].size())
        for i in range(len(fpn_feature)):
            # feature=F.interpolate(fpn_feature[i],scale_factor=0.5,mode='bilinear')
            feature=self.affine_transform(fpn_feature[i])
            # feature = self.affine_transform(fpn_feature[i]).detach()
            # _feature=_fpn_feature[i].detach()
            _feature = _fpn_feature[i]
            cri_loss+=loss_func(feature,_feature)

        crf_loss=0
        if if_cri:
            for i in range(len(fpn_feature)):
                for j in range(len(fpn_feature)):
                    if i!=j:
                        # N,C,H,W=fpn_feature[j].size()
                        # fi=F.interpolate(fpn_feature[i],size=[H,W],mode='bilinear')
                        fi = fpn_feature[i]
                        fj=fpn_feature[j]
                        # fj = fpn_feature[j].detach()
                        crf_loss+=loss_func(fi,fj)

        loss=self.crf_weight*crf_loss+self.cri_weigth*cri_loss
        loss=self.loss_weight*loss
        return loss

