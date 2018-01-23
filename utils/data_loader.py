import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import skimage.io as io
from PIL import Image
import ntpath
import os

# List the subdirectories in a directory
def list_subdirs(directory):
    subdirs = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            subdirs.append(subdir)
    return subdirs

# Get file names
def Get_filenames(directory):
    subdirs = list_subdirs(directory)
    subdirs.append(directory)
    file_names = []
    file_paths = []
    for subdir in subdirs:
        subpath = os.path.join(directory, subdir)
        for fname in os.listdir(subpath):
            if has_valid_extension(fname):
                file_paths.append(os.path.join(directory, subdir, fname))
                file_names.append(fname)
    return file_paths, file_names

# Load image
def Load_image(image_path, resize, grayscale, order = 1):
    img = Image.open(image_path)
    # Resize
    if resize is not None:
        img = img.resize(resize, resample=Image.BILINEAR)
    # Color conversion
    if len(img.size) == 2 and not grayscale:
        img = img.convert('RGB')
    elif len(img.size) > 2 and img.size[2] == 3 and grayscale:
        img = img.convert('LA')
    return img

# Checks if a file is an image
def has_valid_extension(fname, white_list_formats={'png', 'jpg', 'jpeg',
                        'bmp', 'tif'}):
    for extension in white_list_formats:
        if fname.lower().endswith('.' + extension):
            return True
    return False

class fromFileDataset(Dataset):

    def __init__(self, cf, image_txt, gt_txt, num_images, resize, 
                        preprocess=None, transform=None, valid=False):

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
        print ("\t Gt from: " + image_txt)
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
        img = Load_image(img_path, self.resize, self.cf.grayscale, order=1)
        gt = Load_image(gt_path, self.resize, grayscale=True, order=0)
        if self.transform is not None:
            img, gt = self.transform(img, gt)
        if self.preprocess is not None:
            img = self.preprocess(img)
        gt = torch.from_numpy(np.array(gt, dtype=np.int32)).long()
        return img, gt

    def update_indexes(self, num_images=None, valid=False):
        if self.cf.shuffle and not valid:
            np.random.shuffle(self.img_indexes)
        if num_images is not None:
            self.num_images = num_images
        self.indexes = self.img_indexes[:self.num_images]

class fromFileDatasetToPredict(Dataset):

    def __init__(self, cf, image_txt, num_images, resize, 
                        preprocess=None):
        self.cf = cf
        self.resize = resize
        self.preprocess = preprocess
        self.num_images = num_images
        with open(image_txt) as f:
            image_names = f.readlines()
        self.image_names = [x.strip() for x in image_names] 
        if len(self.image_names) < self.num_images or self.num_images == -1:
            self.num_images = len(self.image_names)
        self.img_indexes = np.arange(len(self.image_names))
        self.indexes = self.img_indexes[:self.num_images]

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_path = self.image_names[self.indexes[idx]]
        img_path_comp = img_path.split("/")
        img_name = img_path_comp[-1]
        img = Load_image(img_path, self.resize, self.cf.grayscale, order=1)
        if self.preprocess is not None:
            img = self.preprocess(img)
        return img, img_name
