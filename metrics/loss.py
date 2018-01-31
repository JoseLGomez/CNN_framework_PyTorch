import torch.nn.functional as F
import torch.nn as nn


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=False, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):

        n, c, h, w = inputs.size()

        log_p = F.log_softmax(inputs,dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()                      # I dont understand this part...
        log_p = log_p[targets.view(n, h, w, 1).repeat(1, 1, 1, c) != self.ignore_index] # rm invalid index
        log_p = log_p.view(-1, c)

        mask = (targets != self.ignore_index)
        targets = targets[mask]

        targets = targets.view(-1)

        loss = F.nll_loss(log_p, targets, weight=None, size_average=False)

        # if self.size_average:
        loss /= mask.data.sum()

        return loss.mean()


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)