import torch.nn.functional as F
import torch.nn as nn
from classification_loss import Classification_Loss


class CrossEntropyLoss(Classification_Loss):
    def __init__(self, cf, weight=None, size_average=False, ignore_index=255):
        super(CrossEntropyLoss, self).__init__(cf, weight, size_average, ignore_index)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        # print targets.size()
        # exit(-1)
        # targets.view(-1)
        return self.criterion(inputs,targets.view(-1))
        # n, c = inputs.size()
        #
        # log_p = F.log_softmax(inputs,dim=1)
        # log_p = log_p.view(-1, c)
        #
        # targets = targets.view(-1)
        #
        # #mask = (targets != self.ignore_index)
        # #targets = targets[mask]
        #
        # loss = F.nll_loss(log_p, targets, weight=None, size_average=False)
        # # if self.size_average:
        # loss /= n
        #
        # return loss.mean()
