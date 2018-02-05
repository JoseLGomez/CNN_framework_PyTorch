from torch.utils.data import Dataset, DataLoader
import os
import skimage.io as io
from skimage.color import rgb2gray, gray2rgb
import skimage.transform
from PIL import Image

class Data_loader(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass

    # List the subdirectories in a directory
    def list_subdirs(self, directory):
        subdirs = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                subdirs.append(subdir)
        return subdirs

    # Get file names
    def Get_filenames(self, directory):
        subdirs = self.list_subdirs(directory)
        subdirs.append(directory)
        file_names = []
        file_paths = []
        for subdir in subdirs:
            subpath = os.path.join(directory, subdir)
            for fname in os.listdir(subpath):
                if self.has_valid_extension(fname):
                    file_paths.append(os.path.join(directory, subdir, fname))
                    file_names.append(fname)
        return file_paths, file_names

    # Load image using PIL
    def Load_image(self, image_path, resize, grayscale, order = 1):
        img = Image.open(image_path)
        # Resize
        if resize is not None:
            img = img.resize(resize, resample=Image.BILINEAR)
        # Color conversion
        if img.mode != 'RGB' and not grayscale:
            img = img.convert('RGB')
        elif img.mode == 'RGB' and grayscale:
            img = img.convert('LA')
        return img

    # Load image using Skimage
    def load_img(self, path, resize=None, grayscale=False, order=1):
        # Load image
        img = io.imread(path)
        if resize is not None:
            img = skimage.transform.resize(img, resize, order=order,
                                           preserve_range=True, mode='reflect')

        # Color conversion
        if len(img.shape) == 2 and not grayscale:
            img = gray2rgb(img)
        elif len(img.shape) > 2 and img.shape[2] == 3 and grayscale:
            img = rgb2gray(img)
        # Return image
        return img

    # Checks if a file is an image
    def has_valid_extension(self, fname, white_list_formats={'png', 'jpg', 'jpeg',
                            'bmp', 'tif'}):
        for extension in white_list_formats:
            if fname.lower().endswith('.' + extension):
                return True
        return False