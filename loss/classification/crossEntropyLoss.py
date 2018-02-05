import torch.nn.functional as F
import torch.nn as nn
from classification_loss import Classification_Loss


class CrossEntropyLoss(Classification_Loss):
    def __init__(self, cf, weight=None, size_average=False, ignore_index=255):
        super(CrossEntropyLoss, self).__init__(cf, weight, size_average, ignore_index)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):

        return self.criterion(inputs,targets.view(-1))
