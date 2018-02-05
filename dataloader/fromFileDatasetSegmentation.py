import torch
import numpy as np

from dataloader import Data_loader

class fromFileDatasetSegmentation(Data_loader):

    def __init__(self, cf, image_txt, gt_txt, num_images, resize=None,
                        preprocess=None, transform=None, valid=False):
        super(fromFileDatasetSegmentation, self).__init__()
        self.cf = cf
        self.resize = resize
        self.transform = transform
        self.preprocess = preprocess
        self.num_images = num_images
        print ("\t Images from: " + image_txt)
        with open(image_txt) as f:
            image_names = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        self.image_names = [x.strip() for x in image_names]
        print ("\t Gt from: " + gt_txt)
        with open(gt_txt) as f:
            gt_names = f.readlines()
        self.gt_names = [x.strip() for x in gt_names]
        if len(self.gt_names) != len(self.image_names):
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
        gt_path = self.gt_names[self.indexes[idx]]
        img = self.load_img(img_path, self.resize, self.cf.grayscale, order=1)
        gt = self.load_img(gt_path, self.resize, grayscale=True, order=0)
        if self.transform is not None:
            img, gt = self.transform(img, gt)
        #img = Image.fromarray(img.astype(np.uint8))
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