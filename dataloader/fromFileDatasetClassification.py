import torch
import numpy as np
from dataloader import Data_loader

class fromFileDatasetClassification(Data_loader):

    def __init__(self, cf, image_txt, gt_txt, num_images, resize=None,
                 preprocess=None, transform=None, valid=False):
        super(fromFileDatasetClassification, self).__init__()
        self.cf = cf
        self.resize = resize
        self.transform = transform
        self.preprocess = preprocess
        self.num_images = num_images
        print ("\t Images from: " + image_txt)
        with open(image_txt) as f:
            image_names = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        lines = [x.strip() for x in image_names]
        self.image_names = lines

        print ("\t gt from: " + gt_txt)
        with open(gt_txt) as f:
            gt = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        lines = [x.strip() for x in gt]
        self.gt = [int(line) for line in lines]

        if len(self.gt) != len(self.image_names):
            raise ValueError('number of images != number GT images')
        print ("\t Images found: " + str(len(self.image_names)))
        if len(self.image_names) < self.num_images or self.num_images == -1:
            self.num_images = len(self.image_names)
        self.img_indexes = np.arange(len(self.image_names))
        self.update_indexes(valid=valid)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_path = self.image_names[self.indexes[idx]]
        img = self.load_img(img_path, self.resize, self.cf.grayscale, order=1)
        gt = [self.gt[self.indexes[idx]]]
        if self.transform is not None:
            img, _ = self.transform(img, None)
        if self.preprocess is not None:
            img = self.preprocess(img)
        gt = torch.from_numpy(np.array(gt, dtype=np.int32)).long()
        return img, gt

    def update_indexes(self, num_images=None, valid=False):
        if self.cf.shuffle and not valid:
            np.random.shuffle(self.img_indexes)
        if num_images is not None:
            if len(self.image_names) < self.num_images or num_images == -1:
                self.num_images = len(self.image_names)
            else:
                self.num_images = num_images
        self.indexes = self.img_indexes[:self.num_images]