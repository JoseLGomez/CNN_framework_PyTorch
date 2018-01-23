import numpy as np
import random
import torch

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class ToTensor(object):
    def __call__(self, image):
        return torch.from_numpy(np.flip(image.transpose((2, 0, 1)),axis=0).copy())

class rescale(object):
    def __init__(self,rescale):
        self.rescale = rescale
    def __call__(self, image):
        return torch.mul(image, self.rescale)

class mean_norm(object):
    def __init__(self, mean):
        self.mean = torch.FloatTensor(mean)
    def __call__(self, image):
        return torch.add(image, torch.neg(self.mean))
        
class std_norm(object):
    def __init__(self, std):
        self.std = torch.FloatTensor(std)
    def __call__(self, image):
        return torch.div(image,(self.std + 1e-7))
        
class preproces_input(object):
    def __init__(self, cf):
        self.cf = cf
        #self.rescale = rescale(self.cf.rescale)
        self.mean = mean_norm(self.cf.mean)
        self.std = std_norm(self.cf.std)
    def __call__(self, image):
        '''if cf.rescale is not None:
            image = self.rescale(image)'''
        if self.cf.mean is not None:
            image = self.mean(image)
        if self.cf.std is not None:
            image = self.std(image)
        return image

class applyCrop(object):
    def __init__(self, cf):
        self.cf = cf
    def __call__(self, img, mask):
        w, h = self.cf.size_image_train
        th, tw = self.cf.crop_train
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            img = img.resize((tw, th), Image.BILINEAR)
            mask = mask.resize((tw, th), Image.NEAREST)
        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            img = img.crop((x1, y1, x1 + tw, y1 + th))
            mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        return img, mask

class RandomHorizontalFlip(object):
    def __call__(self, img, gt):
        if random.random() < 0.5:
            return np.fliplr(img), np.fliplr(gt)
        return img, gt
