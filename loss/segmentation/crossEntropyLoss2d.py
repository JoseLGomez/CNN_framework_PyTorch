import torch.nn.functional as F
from semantic_loss import Semantic_Loss


class CrossEntropyLoss2d(Semantic_Loss):
    def __init__(self, cf, weight=None, size_average=False, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__(cf, weight, size_average, ignore_index)

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

        if self.size_average:
            loss /= mask.data.sum()

        return loss.mean()