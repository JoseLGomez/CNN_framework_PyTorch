import torch.nn.functional as F

from segmentation.crossEntropyLoss2d import CrossEntropyLoss2d
from segmentation.focal_loss2d import FocalLoss2d

class Loss_Builder():
    def __init__(self, cf):
        self.cf = cf
        self.size_average = False
        self.loss_manager = None

    def build(self):
        if self.cf.loss_type.lower() == 'cross_entropy_segmentation':
            self.loss_manager = CrossEntropyLoss2d(self.cf,size_average=self.size_average, ignore_index=self.cf.void_class)
        elif self.cf.loss_type.lower() == 'focal_segmentation':
            self.loss_manager = FocalLoss2d(self.cf,self.cf.gamma, size_average=self.size_average, ignore_index=self.cf.void_class)
        else:
            raise ValueError('Unknown loss type')
        return self.loss_manager