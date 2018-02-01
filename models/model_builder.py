import os

import torch

from models.segmentation.FCN8 import FCN8
from models.segmentation.FCdenseNetTorch import FCDenseNet


class Model_builder():
    def __init__(self, cf):
        self.cf = cf
        self.train_mLoss = float('inf')
        self.valid_mLoss = float('inf')
        self.mIoU_valid = 0
        self.mAcc_valid = 0
        
    def build(self):
        if self.cf.pretrained_model and not self.cf.load_weight_only:
            self.net = self.restore_model()
            return net
        if self.cf.model_type == 'DenseNetFCN':
            self.net = FCDenseNet(nb_layers_per_block=self.cf.model_layers,
                                growth_rate=self.cf.model_growth,
                                nb_dense_block=self.cf.model_blocks, 
                                n_channel_start=48,
                                n_classes=self.cf.num_classes,
                                drop_rate=0, bottle_neck=False).cuda()
        elif self.cf.model_type == 'FCN8':
            self.net = FCN8(num_classes=self.cf.num_classes, pretrained=self.cf.basic_pretrained_model).cuda()
        else:
            raise ValueError('Unknown model')
        if self.cf.pretrained_model and self.cf.load_weight_only:
            self.net = self.restore_weights(self.net)

    def restore_weights(self, net):
        print('\t Restoring model from ' + self.cf.model_path + self.cf.model_name)
        net.load_state_dict(torch.load(os.path.join(self.cf.model_path, self.cf.model_name + '.pth')))           
        return net

    def restore_model(self):
        print('\t Restoring weight from ' + self.cf.model_path + self.cf.model_name)
        net = torch.load(os.path.join(self.cf.model_path, self.cf.model_name + '.pth'))
        return net

    def save_model(self, net):
        if self.cf.save_weight_only:
            torch.save(net.state_dict(), os.path.join(self.cf.exp_folder, 
                self.cf.model_name + '.pth'))
            #torch.save(optimizer.state_dict(), os.path.join(self.cf.exp_folder, 
                #self.cf.model_name, 'opt.pth'))
        else:
            torch.save(net, os.path.join(self.cf.exp_folder, self.cf.model_name + '.pth'))

    def save(self, net, train_mLoss, valid_mLoss=None, mIoU_valid=None, mAcc_valid=None):
        if self.cf.save_condition == 'always':
            self.save_model(net)
        elif self.cf.save_condition == 'train_loss':
            if train_mLoss < self.train_mLoss:
                self.train_mLoss = train_mLoss
                self.save_model(net)
        elif self.cf.save_condition == 'valid_loss':
            if valid_mLoss < self.valid_mLoss:
                self.valid_mLoss = valid_mLoss
                self.save_model(net)
        elif self.cf.save_condition == 'valid_mIoU':
            if mIoU_valid > self.mIoU_valid:
                self.mIoU_valid = mIoU_valid
                self.save_model(net)
        elif self.cf.save_condition == 'valid_mAcc':
            if mAcc_valid > self.mAcc_valid:
                self.mAcc_valid = mAcc_valid
                self.save_model(net)
        


                   