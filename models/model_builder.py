import os

from models.segmentation.FCN8 import FCN8
from models.segmentation.FCdenseNetTorch import FCDenseNet

from models.classification.VGG16 import VGG16

class Model_builder():
    def __init__(self, cf):
        self.cf = cf
        
    def build(self):
        if self.cf.pretrained_model.lower() == 'custom' and not self.cf.load_weight_only:
            self.net = self.restore_model()
            return self.net

        if self.cf.model_type.lower() == 'densenetfcn':
            self.net = FCDenseNet(self.cf, nb_layers_per_block=self.cf.model_layers,
                                growth_rate=self.cf.model_growth,
                                nb_dense_block=self.cf.model_blocks, 
                                n_channel_start=48,
                                n_classes=self.cf.num_classes,
                                drop_rate=0, bottle_neck=False).cuda()
        elif self.cf.model_type.lower() == 'fcn8':
            self.net = FCN8(self.cf, num_classes=self.cf.num_classes, pretrained=self.cf.basic_pretrained_model).cuda()
        elif self.cf.model_type.lower() == 'vgg16':
            self.net = VGG16(self.cf, num_classes=self.cf.num_classes, pretrained=self.cf.basic_pretrained_model).cuda()
        else:
            raise ValueError('Unknown model')

        if self.cf.pretrained_model.lower() == 'custom' and self.cf.load_weight_only:
            self.net.restore_weights(os.path.join(self.cf.input_model_path))

        


                   