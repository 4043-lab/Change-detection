import torch
import torch.nn as nn
from torch.nn import functional as F
from ..builder import LOSSES

#self_supervised_consistency_regularization_loss
@LOSSES.register_module()
class BoxLevelSetLoss(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(BoxLevelSetLoss, self).__init__()
        self.loss_weight=loss_weight


    def forward(self,levelset_feature,box_mask):
        # print('levelset__'*5)
        # print(len(levelset_feature))
        # print(levelset_feature.size())
        # print(levelset_feature)
        # print(box_mask.size())
        loss=0
        for f in levelset_feature:
            result=torch.abs(f)*box_mask
            loss+=result.mean()
        loss=loss*self.loss_weight

        return loss

