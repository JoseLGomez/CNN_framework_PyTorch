import torch.nn.functional as F
from semantic_loss import Semantic_Loss

class FocalLoss2d(Semantic_Loss):
    def __init__(self, cf):
        super(FocalLoss2d, self).__init__(cf)
        self.gamma = cf.gamma

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)