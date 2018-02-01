import argparse
import time
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

import utils.preprocessAndTransformations as preprocess
from tasks.semanticSegmentator_manager import SemanticSegmentation_Manager
from tasks.classification_manager import Classification_Manager
from config.configuration import Configuration
from loss.loss_builder import Loss_Builder
from models.model_builder import Model_builder
from utils.data_loader import fromFileDataset, fromFileDatasetToPredict
from utils.optimizer_builder import Optimizer_builder
from utils.scheduler_builder import scheduler_builder


def main():
    start_time = time.time()
    # Input arguments
    parser = argparse.ArgumentParser(description="TensorFlow framework for Semantic Segmentation")
    parser.add_argument("--config_file",
                        type=str,
                        default='config/configFile.py',
                        help="configuration file path")

    parser.add_argument("--exp_name",
                        type=str,
                        default='Sample',
                        help="Experiment name")

    parser.add_argument("--exp_folder",
                        type=str,
                        default='/home/jlgomez/Experiments/',
                        help="Experiment folder path")

    args = parser.parse_args()

    # Prepare configutation
    print ('Loading configuration ...')
    config = Configuration(args.config_file, args.exp_name, args.exp_folder)
    cf = config.Load()

    # Enable log file
    #logger_debug = Logger(cf.log_file_debug)
    #logger_stats = Logger(cf.log_file_stats)

    print ('\n ---------- Init experiment: ' + cf.exp_name + ' ---------- \n')

    # Model building
    print ('- Building model: ' + cf.model_name + ' <--- ')
    model = Model_builder(cf)
    model.build()
    model.net.train() # enable dropout modules and others

    # Problem type
    if cf.problem_type == 'segmentation':
        problem_manager = SemanticSegmentation_Manager(cf, model)
    elif cf.problem_type == 'classification':
        problem_manager = Classification_Manager(cf, model)
    else:
        raise ValueError('Unknown problem type')

    # Compose preprocesing function for dataloaders
    '''img_preprocessing = standard_transforms.Compose([standard_transforms.ToTensor(),
                                            standard_transforms.Normalize(cf.mean,cf.std)])'''
    img_preprocessing = standard_transforms.Compose([preprocess.preproces_input(cf), preprocess.ToTensor()])
    # ,preprocess.PrintInput()])
    train_transformation = preprocess.Compose([preprocess.applyCrop(cf),
                                               preprocess.RandomHorizontalFlip(cf)])

    # Loss definition
    #criterion = CrossEntropyLoss2d(size_average=False, ignore_index=cf.void_class).cuda()
    criterion = Loss_Builder(cf).build().cuda()

    # Optimizer definition
    optimizer = Optimizer_builder().build(cf, model.net)

    # Learning rate scheduler
    scheduler = scheduler_builder().build(cf, optimizer)

    if cf.train:
        train_time = time.time()
        # Dataloaders
        print ('\n- Reading Train dataset: ')
        train_set = fromFileDataset(cf, cf.train_images_txt, cf.train_gt_txt,
                            cf.train_samples, cf.resize_image_train,
                            preprocess=img_preprocessing, transform=train_transformation)
        train_loader = DataLoader(train_set, batch_size=cf.train_batch_size, num_workers=8)
        if cf.valid_images_txt is not None and cf.valid_gt_txt is not None and cf.valid_samples_epoch > 0:
            print ('\n- Reading Validation dataset: ')
            valid_set = fromFileDataset(cf, cf.valid_images_txt, cf.valid_gt_txt,
                                cf.valid_samples_epoch, cf.resize_image_valid,
                                preprocess=img_preprocessing, transform=None, valid=True)
            valid_loader = DataLoader(valid_set, batch_size=cf.valid_batch_size, num_workers=8)
            problem_manager.train(criterion, optimizer, train_loader, train_set, valid_set, valid_loader, scheduler)
        else:
            # Train without validation inside epoch
            problem_manager.train(criterion, optimizer, train_loader, train_set, scheduler=scheduler)
        train_time = time.time() - train_time
        print('\t Train step finished: %ds ' % (train_time))

    if cf.validation:
        valid_time = time.time()
        if not cf.train:
            print ('- Reading Validation dataset: ')
            valid_set = fromFileDataset(cf, cf.valid_images_txt, cf.valid_gt_txt,
                            cf.valid_samples, cf.resize_image_valid,
                            preprocess=img_preprocessing, transform=None, valid=True)
            valid_loader = DataLoader(valid_set, batch_size=cf.valid_batch_size, num_workers=8)
        else:
        #If the Dataloader for validation was used on train, only update the total number of images to take
            valid_set.update_indexes(cf.valid_samples, valid=True) #valid=True avoids shuffle for validation
        print ('\n- Starting validation <---')
        problem_manager.validation(criterion, valid_set, valid_loader)
        valid_time = time.time() - valid_time
        print('\t Validation step finished: %ds ' % (valid_time))

    if cf.test:
        test_time = time.time()
        print ('\n- Reading Test dataset: ')
        test_set = fromFileDataset(cf, cf.test_images_txt, cf.test_gt_txt,
                        cf.test_samples, cf.resize_image_test,
                        preprocess=img_preprocessing, transform=None, valid=True)
        test_loader = DataLoader(test_set, batch_size=cf.test_batch_size, num_workers=8)
        print ('\n - Starting test <---')
        problem_manager.validation(criterion, test_set, test_loader)
        test_time = time.time() - test_time
        print('\t Test step finished: %ds ' % (test_time))

    if cf.predict_test:
        pred_time = time.time()
        print ('\n- Reading Prediction dataset: ')
        predict_set = fromFileDatasetToPredict(cf, cf.test_images_txt,
                        cf.test_samples, cf.resize_image_test,
                        preprocess=img_preprocessing)
        predict_loader = DataLoader(predict_set, batch_size=1, num_workers=8)
        print ('\n - Generating predictions <---')
        problem_manager.predict(predict_loader)
        pred_time = time.time() - pred_time
        print('\t Prediction step finished: %ds ' % (pred_time))

    total_time = time.time() - start_time
    print('\n- Experiment finished: %ds ' % (total_time))
    print('\n')

# Entry point of the script
if __name__ == "__main__":
    main()