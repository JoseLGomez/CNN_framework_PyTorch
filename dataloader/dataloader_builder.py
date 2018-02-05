from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as standard_transforms
import sys

sys.path.append('../')
import utils.preprocessAndTransformations as preprocess
from fromFileDatasetSegmentation import fromFileDatasetSegmentation
from fromFileDatasetClassification import fromFileDatasetClassification
from fromFileDatasetToPredict import fromFileDatasetToPredict

class Dataloader_Builder(object):
    def __init__(self, cf):
        self.cf = cf
        # Compose preprocesing function for dataloaders
        self.img_preprocessing = standard_transforms.Compose([preprocess.preproces_input(cf), preprocess.ToTensor()])
        self.train_transformation = preprocess.Compose([preprocess.applyCrop(cf),
                                                   preprocess.RandomHorizontalFlip(cf)])

    def build_train(self):
        if self.cf.problem_type =='segmentation':
            self.train_set = fromFileDatasetSegmentation(self.cf, self.cf.train_images_txt, self.cf.train_gt_txt,
                                        self.cf.train_samples, self.cf.resize_image_train,
                                preprocess=self.img_preprocessing, transform=self.train_transformation)
        elif self.cf.problem_type =='classification':
            self.train_set = fromFileDatasetClassification(self.cf, self.cf.train_images_txt, self.cf.train_gt_txt,
                                                      self.cf.train_samples, self.cf.resize_image_train,
                                        preprocess=self.img_preprocessing, transform=self.train_transformation)
        self.train_loader = DataLoader(self.train_set, batch_size=self.cf.train_batch_size, num_workers=8)

    def build_valid(self, valid_samples, images_txt, gt_txt, resize_image, batch_size):
        if self.cf.problem_type == 'segmentation':
            self.loader_set = fromFileDatasetSegmentation(self.cf, images_txt, gt_txt,
                                                          valid_samples, resize_image,
                                                          preprocess=self.img_preprocessing, transform=None,
                                                          valid=True)
        elif self.cf.problem_type == 'classification':
            self.loader_set = fromFileDatasetClassification(self.cf, images_txt, gt_txt,
                                                            valid_samples, resize_image,
                                                            preprocess=self.img_preprocessing, transform=None,
                                                            valid=True)
        self.loader = DataLoader(self.loader_set, batch_size=batch_size, num_workers=8)

    def build_predict(self):
        self.predict_set = fromFileDatasetToPredict(self.cf, self.cf.test_images_txt,
                                               self.cf.test_samples, self.cf.resize_image_test,
                                               preprocess=self.img_preprocessing)
        self.predict_loader = DataLoader(self.predict_set, batch_size=1, num_workers=8)