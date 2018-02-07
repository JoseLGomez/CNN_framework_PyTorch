import sys
from torch import nn
import torchvision.models.vgg as models
sys.path.append('../')
from models.model import Model

class VGG16(Model):

    def __init__(self, num_classes=21, pretrained=False, basic_model_path='./pretrained_model/', net_name='vgg16'):
        super(VGG16, self).__init__()

        self.model = models.vgg16(pretrained=False, num_classes=num_classes)

        '''self.model.classfier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )'''
        if pretrained:
            self.load_basic_weights(basic_model_path, net_name)

    def forward(self, x):

        return self.model.forward(x)
