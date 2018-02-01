import torch.nn as nn

class Semantic_Loss(nn.Module):
    def __init__(self, cf, weight=None, size_average=False, ignore_index=255):
        super(Semantic_Loss, self).__init__()
        self.cf = cf
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.loss_manager = None

    def forward(self, inputs, targets):
        return self.loss_manager.forward(inputs, targets)