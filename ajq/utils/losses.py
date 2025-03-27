import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class BCELoss(nn.Module):
    def __init__(self, size_average=True):
        super(BCELoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, p=1):
        super().__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, pred, tar):
        pred = torch.argmax(pred, 1, keepdim=True)
        pred, tar = pred.flatten(1), tar.flatten(1)
        prob = F.sigmoid(pred)

        num = 2 * (prob * tar).sum(1) + self.smooth
        den = (prob.pow(self.p) + tar.pow(self.p)).sum(1) + self.smooth

        loss = 1 - num / den

        return loss.mean()


class BCLLoss(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """

    def __init__(self, margin=2.0):
        super(BCLLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label = label.float()
        label[label == 255] = 1
        label[label == 1] = -1
        label[label == 0] = 1
        mask = (label != 255).float()
        distance = distance * mask
        pos_num = torch.sum((label == 1).float()) + 0.0001
        neg_num = torch.sum((label == -1).float()) + 0.0001

        loss_1 = torch.sum((1+label) / 2 * torch.pow(distance, 2)) / pos_num
        loss_2 = torch.sum((1-label) / 2 * mask * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)) / neg_num
        loss = loss_1 + loss_2
        return loss

