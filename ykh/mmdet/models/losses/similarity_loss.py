import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from ..builder import LOSSES



class BCELoss(nn.Module):
    def __init__(self, size_average=True):
        super(BCELoss, self).__init__()
        self.size_average = size_average

    def forward(self, mask, target):
        if mask.dim() > 2:
            mask = mask.view(mask.size(0), mask.size(1), -1)  # N,C,H,W => N,C,H*W
            mask = mask.transpose(1, 2)  # N,C,H*W => N,H*W,C
            mask = mask.contiguous().view(-1, mask.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1).type(torch.int64)

        logpt = F.log_softmax(mask, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

#self_supervised_consistency_regularization_loss
@LOSSES.register_module()
class Similarity_Loss(nn.Module):
    def __init__(self,loss_weight=[1.0,1.0,1.0]):
        super(Similarity_Loss, self).__init__()
        self.loss_weight=loss_weight
        self.bce=BCELoss()

    def forward(self,x1,x2,score_map1,score_map2,mask_list):

        if self.loss_weight[0]==0.0 and self.loss_weight[1]==0.0 and self.loss_weight[2]==0.0:
            loss=torch.tensor(0.0).cuda()
            return loss,loss,loss

        _,_,sh,sw =score_map1.size()
        # print(score_map1.size())
        loss1 = 0
        loss2 = 0
        loss3 = 0
        MSELoss = nn.MSELoss(reduction='sum')
        Bceloss = self.bce

        bool_tensor = mask_list.unsqueeze(1).cuda()
        bool_tensor = bool_tensor.float()

        stage_list=[3]
        for i in stage_list:
            t1 = x1[i]
            t2 = x2[i]
            # 归一化
            # t1 = t1[i].unsqueeze(0)
            # t2 = t2[i].unsqueeze(0)
            # x = torch.cat([t1, t2], dim=0)
            # x = F.softmax(x, dim=0)
            # t1 = x[0]
            # t2 = x[1]

            B, C, H, W = x1[i].size()
            loss1_mask = nn.MaxPool2d(kernel_size=int(256 / H))(bool_tensor)
            # loss1_mask=F.interpolate(bool_tensor, scale_factor=256 / H, mode='nearest')

            #t1和t2的背景应该是相似的
            t1 = t1 * (1 - loss1_mask)
            t2 = t2 * (1 - loss1_mask)
            b,c,h,w=t2.size()
            loss1+=MSELoss(t1,t2)/(b*c*h*w-c*loss1_mask.sum()+1e-6)

            t1 = x1[i]
            t2 = x2[i]
            # print(t1.sum(), t2.sum())
            t1 = t1 * loss1_mask
            t2 = t2 * loss1_mask
            # loss2+=1-MSELoss(t1,t2)/(c*loss1_mask.sum()+1e-6)
            loss2=F.cosine_similarity(t1,t2,dim=1).sum()/(loss1_mask.sum()+1e-6)
            # loss2 = F.sigmoid(1 - torch.cosine_similarity(t1, t2, dim=1)).sum() / (loss1_mask.sum() + 1e-6)
            # print(self.loss_weight[1],t1.sum(),t2.sum(),loss1_mask.sum())

        bool_tensor = bool_tensor.squeeze(0)
        score_mask=nn.MaxPool2d(kernel_size=int(256 / sh))(bool_tensor).cuda()
        # print(score_mask.size())
        s1=score_map1
        s2=score_map2
        # print(s1.device,score_mask.device)
        diff=torch.mean(torch.abs(s1-s2),dim=1).unsqueeze(1)
        diff = torch.sigmoid(diff)
        diff=torch.cat([diff,1-diff],dim=1)
        loss3+=Bceloss(diff, score_mask)

        loss1 = self.loss_weight[0] * loss1
        loss2 = self.loss_weight[1] * loss2
        loss3 = self.loss_weight[2] * loss3
        # loss=loss1+loss2+loss3

        return loss1,loss2,loss3

    #只处理一个阶段
    # def forward(self,x1,x2,score_map1,score_map2,mask_list):
    #
    #     _,_,sh,sw =score_map1.size()
    #     # print(score_map1.size())
    #     loss1 = 0
    #     loss2 = 0
    #     loss3 = 0
    #     MSELoss = nn.MSELoss(reduction='sum')
    #     Bceloss = self.bce
    #
    #     # 归一化
    #     x1 = x1.unsqueeze(0)
    #     x2 = x2.unsqueeze(0)
    #     x = torch.cat([x1, x2], dim=0)
    #     x = F.softmax(x, dim=0)
    #     x1 = x[0]
    #     x2 = x[1]
    #
    #     for i in range(B):
    #         bool_tensor = mask_list[i].unsqueeze(0).unsqueeze(0).cuda()
    #         bool_tensor = bool_tensor.float()
    #
    #         loss1_mask = nn.MaxPool2d(kernel_size=int(256 / H))(bool_tensor)
    #         # loss1_mask=F.interpolate(bool_tensor, scale_factor=256 / H, mode='nearest')
    #
    #         t1 = x1[i] * (1 - loss1_mask)
    #         t2 = x2[i] * (1 - loss1_mask)
    #         # t1 = F.softmax(t1, dim=1)
    #         # t2 = F.softmax(t2, dim=1)
    #         b,c,h,w=t2.size()
    #         #归一化
    #
    #         loss1+=MSELoss(t1,t2)/(b*c*h*w-b*c*loss1_mask.sum()+1e-6)
    #
    #         t1 = x1[i] * loss1_mask
    #         t2 = x2[i] * loss1_mask
    #         # t1 = F.softmax(t1, dim=1)
    #         # t2 = F.softmax(t2, dim=1)
    #         # loss2+=1-MSELoss(t1,t2)/(b*c*loss1_mask.sum()+1e-6)
    #         loss2=F.cosine_similarity(t1,t2).sum()/(b*loss1_mask.sum()+1e-6)
    #
    #         bool_tensor = bool_tensor.squeeze(0)
    #         score_mask=nn.MaxPool2d(kernel_size=int(256 / sh))(bool_tensor).cuda()
    #         # print(score_mask.size())
    #         s1=score_map1[i].unsqueeze(0)
    #         s2=score_map2[i].unsqueeze(0)
    #         # print(s1.device,score_mask.device)
    #         diff=torch.mean(torch.abs(s1-s2),dim=1).unsqueeze(1)
    #         diff = torch.sigmoid(diff)
    #         diff=torch.cat([diff,1-diff],dim=1)
    #         loss3+=Bceloss(diff, score_mask)
    #
    #     loss1 = self.loss_weight[0] * loss1
    #     loss2 = self.loss_weight[1] * loss2
    #     loss3 = self.loss_weight[2] * loss3
    #     # loss=loss1+loss2+loss3
    #
    #     return loss1,loss2,loss3

