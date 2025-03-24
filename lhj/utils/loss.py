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
        target = target.view(-1, 1).type(torch.int64)
        # print('input', input)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target):
        # print(input)
        # print(target)
        input = F.softmax(input, dim=1)
        loss = self.criterion(input, target)
        return loss

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


class L1Loss_edge(nn.Module):
    def __init__(self):
        super(L1Loss_edge,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, input, target):
        input = self.pool(input)-input
        target = self.pool(target)-target
        # print(input)
        # print(target)
        target0 = 1-target
        input_edge = F.softmax(input, dim=1)
        target_edge = torch.cat([target0,target], dim=1).float()
        loss = nn.L1Loss()(input_edge,target_edge)
        return loss

# class FocalLoss(nn.Module):
#
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
#         if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         # print(input.shape)
#         # print(target.shape)
#         if input.dim() > 2:
#             input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
#         target = target.view(-1, 1)
#
#         # print(input.shape)
#         if input.shape[1] == 1:
#             # input = torch.sigmoid(input)
#             # input = torch.cat([1 - input, input], dim=1)
#             input = torch.cat([2 - input, input], dim=1)
#         logpt = F.log_softmax(input, dim=1)
#
#         logpt = logpt.gather(1, target)
#         # print(logpt)
#         logpt = logpt.view(-1)
#         pt = logpt.exp()
#
#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * at
#
#         loss = -1 * (1 - pt) ** self.gamma * logpt
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()
#
class BCELoss_2(nn.Module):
    def __init__(self):
        super(BCELoss_2, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, input, target):
        # print(input.shape)
        # print(target.shape)
        if(len(target.shape)==3):
            target = target.unsqueeze(1)
        input = torch.sigmoid(input)
        one = torch.ones([target.shape[0],1,target.shape[2],target.shape[3]]).type(torch.FloatTensor).cuda()
        target_0 = torch.sub(one, target)
        target_new = torch.cat([target_0,target],dim=1)
        # print(target_new.shape)
        loss = self.bceloss(input, target_new)
        # print(loss)
        return loss

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(self, input, target):
        # print(input.shape)
        # print(target.shape)
        # if input.shape[1] == 2:
        #     input = F.softmax(input, dim=1)
        #     input, _ = torch.max(input, dim=1, keepdim=True)
        # elif input.shape[1] == 1:
        #     input = torch.sigmoid(input)

        # if input.shape[1] == 2:
        #     input, _ = torch.max(input, dim=1, keepdim=True)
        # input = torch.sigmoid(input)

        loss = self.mseloss(input, target)
        return loss

#
# class BCELoss2(nn.Module):
#     def __init__(self):
#         super(BCELoss2, self).__init__()
#         self.bceloss = nn.BCELoss()
#
#     def forward(self, input, target):
#         # print(input.shape)
#         # print(target.shape)
#         if(len(target.shape)==3):
#             target = target.unsqueeze(1)
#         # input = torch.sigmoid(input)
#         input = F.softmax(input, dim=1)
#         one = torch.ones([target.shape[0],1,target.shape[2],target.shape[3]]).type(torch.FloatTensor).cuda()
#         target_0 = torch.sub(one, target)
#         target_new = torch.cat([target_0,target],dim=1)
#         # print(target_new.shape)
#         loss = self.bceloss(input, target_new)
#         # print(loss)
#         return loss

if __name__ == '__main__':
    predict = torch.rand(4, 5)
    label = torch.tensor([4, 3, 3, 2])
    print(predict)
    print(label)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(F.softmax(predict, dim=1), label)
    print(loss)

    criterion = CELoss()
    print(criterion(predict, label))