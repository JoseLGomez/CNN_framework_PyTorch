import time
import numpy as np
import math
import argparse
import torchvision.transforms as standard_transforms
import torch
import os
import sys
import scipy.misc
import cv2 as cv
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from config.configuration import Configuration
import utils.preprocessAndTransformations as preprocess
from utils.data_loader import fromFileDataset, fromFileDatasetToPredict
from models.model_builder import Model_builder
from metrics.loss import CrossEntropyLoss2d
from utils.optimizer_builder import Optimizer_builder
from utils.utils import AverageMeter, Early_Stopping
from utils.ProgressBar import ProgressBar
from metrics.metrics import evaluate
from utils.logger import Logger

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
                        default='/home/jlgomez/Experiments/DenseNetFCN/',
                        help="Experiment folder path")

	args = parser.parse_args()

    # Prepare configutation
	print ('Loading configuration ...')
	config = Configuration(args.config_file, args.exp_name, args.exp_folder)
	cf = config.Load()

	# Enable log file
	sys.stdout = Logger(cf.log_file)

	print ('\n ---------- Init experiment: ' + cf.exp_name + ' ---------- \n')

	# Model building
	print ('- Building model: ' + cf.model_name + ' <--- ')
	model = Model_builder(cf)
	model.build()
	model.net.train() # enable dropout modules and others

	# Compose preprocesing function for dataloaders
	img_preprocessing = standard_transforms.Compose([standard_transforms.ToTensor(),
    										standard_transforms.Normalize(cf.mean,cf.std)])
	train_transformation = preprocess.Compose([preprocess.applyCrop(cf), 
    										preprocess.RandomHorizontalFlip()])


	predict_set = fromFileDatasetToPredict(cf, cf.test_images_txt, 
    					cf.test_samples, cf.resize_image_test, 
    					preprocess=img_preprocessing)
	predict_loader = DataLoader(predict_set, batch_size=1, num_workers=8)

	# Loss definition
	criterion = CrossEntropyLoss2d(size_average=False, ignore_index=cf.void_class).cuda()

	# Optimizer definition
	optimizer = Optimizer_builder().build(cf, model.net)

	# Learning rate scheduler
	scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, min_lr=1e-10, factor=0.1)

	if cf.train:
		train_time = time.time()
		# Dataloaders
		print ('\n- Reading Train dataset: ' + cf.model_name + ' <---')
		train_set = fromFileDataset(cf, cf.train_images_txt, cf.train_gt_txt, 
	    					cf.train_samples, cf.resize_image_train, 
	    					preprocess=img_preprocessing, transform=train_transformation)
		train_loader = DataLoader(train_set, batch_size=cf.train_batch_size, num_workers=8)

		print ('\n- Reading Validation dataset: ' + cf.model_name + ' <---')
		valid_set = fromFileDataset(cf, cf.valid_images_txt, cf.valid_gt_txt, 
	    					cf.valid_samples_epoch, cf.resize_image_valid, 
	    					preprocess=img_preprocessing, transform=None, valid=True)
		valid_loader = DataLoader(valid_set, batch_size=cf.valid_batch_size, num_workers=8)

 		print ('\n- Starting train <---')
 		num_batches = math.ceil(train_set.num_images/float(cf.train_batch_size))
 		train(train_loader, train_set, model, criterion, optimizer, 
 			cf, num_batches, valid_set, valid_loader)
 		train_time = time.time() - train_time
 		print('\t Train step finished: %ds ' % (train_time))    

 	if cf.validation:
 		valid_time = time.time()
 		if not cf.train:
 			print ('- Reading Validation dataset: ' + cf.model_name + ' <---')
 			valid_set = fromFileDataset(cf, cf.valid_images_txt, cf.valid_gt_txt, 
	    					cf.valid_samples, cf.resize_image_valid, 
	    					preprocess=img_preprocessing, transform=None, valid=True)
 		else:
 			valid_set.update_indexes(cf.valid_samples, valid=True) #valid=True avoids shuffle for validation
		valid_loader = DataLoader(valid_set, batch_size=cf.valid_batch_size, num_workers=8)
 		print ('\n- Starting validation <---')
 		validate(valid_loader, model, criterion, cf)
 		valid_time = time.time() - valid_time
		print('\t Validation step finished: %ds ' % (valid_time))  

 	if cf.test:
 		test_time = time.time()
 		print ('\n- Reading Test dataset: ' + cf.model_name + ' <---')
 		test_set = fromFileDataset(cf, cf.test_images_txt, cf.test_gt_txt, 
    					cf.test_samples, cf.resize_image_test, 
    					preprocess=img_preprocessing, transform=None, valid=True)
		test_loader = DataLoader(test_set, batch_size=cf.test_batch_size, num_workers=8)

 		print ('\n - Starting test <---')
 		validate(test_loader, model, criterion, cf)
 		test_time = time.time() - test_time
 		print('\t Test step finished: %ds ' % (test_time))  

 	if cf.predict_test:
 		pred_time = time.time()
 		print ('\n - Generating predictions <---')
 		predict(predict_loader, model, criterion, cf)
 		pred_time = time.time() - pred_time
 		print('\t Prediction step finished: %ds ' % (pred_time))  

	total_time = time.time() - start_time    
	print('\n- Experiment finished: %ds ' % (total_time))
	print('\n')

def train(train_loader, train_set, model, criterion, optimizer, cf, num_batches, 
		valid_set=None, valid_loader=None, scheduler=None):
	curr_epoch = cf.initial_epoch
 	# Define early stopping control
 	if cf.early_stopping:
 		early_Stopping = Early_Stopping(cf)
	#Train process
	for epoch in range(curr_epoch, cf.epochs + 1):
		print ('\t ------ Epoch: ' + str(epoch) + ' ------ \n')
		#Progress bar
		prog_bar = ProgressBar(num_batches)
		
		train_loss = AverageMeter()
		curr_iter = (epoch - 1) * len(train_loader)
		for i, data in enumerate(train_loader):
			inputs, labels = data
			assert inputs.size()[2:] == labels.size()[1:]
			N = inputs.size(0)
			inputs = Variable(inputs).cuda()
			labels = Variable(labels).cuda()
			optimizer.zero_grad()
			outputs = model.net(inputs)
			assert outputs.size()[2:] == labels.size()[1:]
			assert outputs.size()[1] == cf.num_classes

			loss = criterion(outputs, labels) / N
			loss.backward()
			optimizer.step()

			train_loss.update(loss.data[0], N)
			#prog_bar.update(loss=train_loss.avg)
			curr_iter += 1
			#writer.add_scalar('train_loss', train_loss.avg, curr_iter)
			# Display progress
			if (i + 1) % math.ceil(num_batches/20.) == 0:
				print('[Global iteration %d], [iter %d / %d], [train loss %.5f]' % (
					curr_iter, i + 1, len(train_loader), train_loss.avg))
		# validate epoch
		if valid_set is not None:
			val_loss, acc_cls, mean_IoU = validate(valid_loader, model, criterion, cf, optimizer, epoch)
			# Early stopping checking
			if cf.early_stopping:
				early_Stopping.check(train_loss.avg, val_loss, mean_IoU, acc_cls)
				if early_Stopping.stop == True:
					print (' Early Stopping Interruption [Epoch: ' + str(epoch) + ' ] \n')
					return
			if scheduler is not None:
				scheduler.step(val_loss)
			# Shuffle validation data
			valid_set.update_indexes()
			model.net.train() 
		# Saving model if needed
		model.save(model.net, train_loss.avg, val_loss, mean_IoU, acc_cls)
		# Shuffle train data
		train_set.update_indexes()
		

def validate(dataloader, model, criterion, cf, optimizer=None, epoch=None):
    model.net.eval()

    val_loss = AverageMeter()
    gts_all, predictions_all = [], []
    acc = np.zeros(len(dataloader))
    acc_cls = np.zeros(len(dataloader))
    mean_IoU = np.zeros(len(dataloader))
    fwavacc = np.zeros(len(dataloader))

    for vi, data in enumerate(dataloader):
        inputs, gts = data
        N = inputs.size(0)
        inputs = Variable(inputs, volatile=True).cuda()
        gts = Variable(gts, volatile=True).cuda()

        outputs = model.net(inputs)
        predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()

        val_loss.update(criterion(outputs, gts).data[0] / N, N)

        gts_all.append(gts.data.cpu().numpy())
        predictions_all.append(predictions)

        metric = evaluate(predictions_all, gts_all, cf.num_classes)
        acc[vi] = metric[0]
        acc_cls[vi] = metric[1]
        mean_IoU[vi] = metric[2]
        fwavacc[vi] = metric[3]

    acc = np.mean(acc)
    acc_cls = np.mean(acc_cls)
    mean_IoU = np.mean(mean_IoU)
    fwavacc = np.mean(fwavacc)
	
    if epoch is not None:
	    print('----------------- Epoch scores summary -------------------------')
	    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_IoU %.5f], [fwavacc %.5f]' % (
	        epoch, val_loss.avg, acc, acc_cls, mean_IoU, fwavacc))
	    print('---------------------------------------------------------------- \n')
    else:
	    print('----------------- Scores summary --------------------')
	    print('[val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_IoU %.5f], [fwavacc %.5f]' % (
	        val_loss.avg, acc, acc_cls, mean_IoU, fwavacc))
	    print('---------------------------------------------------------------- \n')
    '''writer.add_scalar('val_loss', val_loss.avg, epoch)
    writer.add_scalar('acc', acc, epoch)
    writer.add_scalar('acc_cls', acc_cls, epoch)
    writer.add_scalar('mean_IoU', mean_IoU, epoch)
    writer.add_scalar('fwavacc', fwavacc, epoch)
    writer.add_scalar('lr', optimizer.param_groups[1]['lr'], epoch)'''
    return val_loss.avg, acc_cls, mean_IoU

def predict(dataloader, model, criterion, cf):
    model.net.eval()

    for vi, data in enumerate(dataloader):
        inputs, img_name = data

        inputs = Variable(inputs, volatile=True).cuda()

        outputs = model.net(inputs)
        predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        path = os.path.join(cf.predict_path_output, img_name[0])
        #scipy.misc.imsave(path,predictions)
        cv.imwrite(path, predictions)
        print('%d / %d' % (vi + 1, len(dataloader)))


# Entry point of the script
if __name__ == "__main__":
	main()