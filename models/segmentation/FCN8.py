import sys
import torch
import numpy as np
from torch import nn

from FCN16 import FCN16
sys.path.append('../')
from models.model import Model

class FCN8(Model):


    '''@classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU',
            path=cls.pretrained_model,
            md5='dbd9bbb3829a3184913bccc74373afbb',
        )
        '''

    def __init__(self, num_classes=21, pretrained=None):
        super(FCN8, self).__init__()
        self.pretrained_model = './pretrained_models/fcn16s_from_caffe.pth'

        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            num_classes, num_classes, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, bias=False)

        self._initialize_weights()

        self.copy_params_from_fcn16s()

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
            9:9 + upscore_pool4.size()[2],
            9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h


    def copy_params_from_fcn16s(self):
        fcn16s = FCN16()
        fcn16s.load_state_dict(torch.load(self.pretrained_model))

        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            if l1.weight.size() == l2.weight.size():
                l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                if l1.bias.size() == l2.bias.size():
                    l2.bias.data.copy_(l1.bias.data)



# # This is implemented in full accordance with the original one (https://github.com/shelhamer/fcn.berkeleyvision.org)
# class FCN8(nn.Module):
#     def __init__(self, num_classes, pretrained=None):
#         super(FCN8, self).__init__()
#         vgg = models.vgg16()
#         if pretrained is not None:
#             vgg.load_state_dict(torch.load(pretrained))
#         features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
#         print features
#
#         '''
#         100 padding for 2 reasons:
#             1) support very small input size
#             2) allow cropping in order to match size of different layers' feature maps
#         Note that the cropped part corresponds to a part of the 100 padding
#         Spatial information of different layers' feature maps cannot be align exactly because of cropping, which is bad
#         '''
#         features[0].padding = (100, 100)
#
#         for f in features:
#             if 'MaxPool' in f.__class__.__name__:
#                 f.ceil_mode = True
#             elif 'ReLU' in f.__class__.__name__:
#                 f.inplace = True
#
#         self.features3 = nn.Sequential(*features[: 17])
#         self.features4 = nn.Sequential(*features[17: 24])
#         self.features5 = nn.Sequential(*features[24:])
#
#         self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
#         self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
#         self.score_pool3.weight.data.zero_()
#         self.score_pool3.bias.data.zero_()
#         self.score_pool4.weight.data.zero_()
#         self.score_pool4.bias.data.zero_()
#
#         fc6 = nn.Conv2d(512, 4096, kernel_size=7)
#         fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
#         fc6.bias.data.copy_(classifier[0].bias.data)
#         fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
#         fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
#         fc7.bias.data.copy_(classifier[3].bias.data)
#         score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
#         score_fr.weight.data.zero_()
#         score_fr.bias.data.zero_()
#         self.score_fr = nn.Sequential(
#             fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
#         )
#
#         self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
#         self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
#         self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
#         self.upscore2.weight.data.copy_(self.get_upsampling_weight(num_classes, num_classes, 4))
#         self.upscore_pool4.weight.data.copy_(self.get_upsampling_weight(num_classes, num_classes, 4))
#         self.upscore8.weight.data.copy_(self.get_upsampling_weight(num_classes, num_classes, 16))
#
#     def forward(self, x):
#         x_size = x.size()
#         pool3 = self.features3(x)
#         print pool3
#         pool4 = self.features4(pool3)
#         pool5 = self.features5(pool4)
#
#         score_fr = self.score_fr(pool5)
#         upscore2 = self.upscore2(score_fr)
#
#         score_pool4 = self.score_pool4(pool4)
#         upscore_pool4 = self.upscore_pool4(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
#                                            + upscore2)
#
#         score_pool3 = self.score_pool3(pool3)
#         upscore8 = self.upscore8(score_pool3[:, :, 9: (9 + upscore_pool4.size()[2]), 9: (9 + upscore_pool4.size()[3])]
#                                  + upscore_pool4)
#         return upscore8[:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous()
#
#     def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
#         factor = (kernel_size + 1) // 2
#         if kernel_size % 2 == 1:
#             center = factor - 1
#         else:
#             center = factor - 0.5
#         og = np.ogrid[:kernel_size, :kernel_size]
#         filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
#         weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
#         weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
#         return torch.from_numpy(weight).float()