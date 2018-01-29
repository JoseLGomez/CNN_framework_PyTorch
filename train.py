import math
import time
import numpy as np
import cv2 as cv
import os
from torch.autograd import Variable
from PIL import Image

from utils.utils import AverageMeter, Early_Stopping
from utils.ProgressBar import ProgressBar
from metrics.metrics import evaluate

class Train():
    def __init__(self, cf, train_loader, train_set, valid_set, valid_loader):
        self.cf = cf
        self.train_loader = train_loader
        self.train_set = train_set
        self.validator = Validation(self.cf, valid_set, valid_loader, self.cf.valid_batch_size)
        self.num_batches = math.ceil(train_set.num_images/float(cf.train_batch_size))

    def start(self, model, criterion, optimizer, scheduler=None):
        curr_epoch = self.cf.initial_epoch
        # Define early stopping control
        if self.cf.early_stopping:
            early_Stopping = Early_Stopping(self.cf)
        #Train process
        for epoch in range(curr_epoch, self.cf.epochs + 1):
            epoch_time = time.time()
            print ('\t ------ Epoch: ' + str(epoch) + ' ------ \n')
            #Progress bar
            prog_bar = ProgressBar(self.num_batches)
            
            train_loss = AverageMeter()
            curr_iter = (epoch - 1) * len(self.train_loader)
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                assert inputs.size()[2:] == labels.size()[1:]
                N = inputs.size(0)
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()
                optimizer.zero_grad()
                outputs = model.net(inputs)
                assert outputs.size()[2:] == labels.size()[1:]
                assert outputs.size()[1] == self.cf.num_classes

                loss = criterion(outputs, labels) / N
                loss.backward()
                optimizer.step()

                train_loss.update(loss.data[0], N)
                #prog_bar.update(loss=train_loss.avg)
                curr_iter += 1
                #writer.add_scalar('train_loss', train_loss.avg, curr_iter)
                # Display progress
                if (i + 1) % math.ceil(self.num_batches/20.) == 0:
                    print('[Global iteration %d], [iter %d / %d], [train loss %.5f]' % (
                        curr_iter, i + 1, len(self.train_loader), train_loss.avg))
            # validate epoch
            val_loss, acc_cls, mean_IoU = self.validator.start(model, criterion, epoch)
            # Early stopping checking
            if self.cf.early_stopping:
                early_Stopping.check(train_loss.avg, val_loss, mean_IoU, acc_cls)
                if early_Stopping.stop == True:
                    print (' Early Stopping Interruption [Epoch: ' + str(epoch) + ' ] \n')
                    return
            if scheduler is not None:
                scheduler.step(val_loss)
            # Shuffle validation data
            self.validator.valid_set.update_indexes()
            model.net.train() 
            # Saving model if needed
            model.save(model.net, train_loss.avg, val_loss, mean_IoU, acc_cls)
            # Shuffle train data
            self.train_set.update_indexes()
            epoch_time = time.time() - epoch_time
            print('\t Epoch step finished: %ds ' % (epoch_time)) 

class Validation():
    def __init__(self, cf, valid_set, valid_loader, batch_size):
        self.cf = cf
        self.valid_set = valid_set
        self.valid_loader = valid_loader
        self.num_batches = math.ceil(valid_set.num_images/float(batch_size))

    def start(self, model, criterion, epoch=None):
        model.net.eval()

        val_loss = AverageMeter()
        gts_all, predictions_all = [], []
        acc = np.zeros(len(self.valid_loader))
        acc_cls = np.zeros(len(self.valid_loader))
        mean_IoU = np.zeros(len(self.valid_loader))
        fwavacc = np.zeros(len(self.valid_loader))

        for vi, data in enumerate(self.valid_loader):
            inputs, gts = data
            N = inputs.size(0)
            inputs = Variable(inputs, volatile=True).cuda()
            gts = Variable(gts, volatile=True).cuda()

            outputs = model.net(inputs)
            predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()

            val_loss.update(criterion(outputs, gts).data[0] / N, N)

            gts_all.append(gts.data.cpu().numpy())
            predictions_all.append(predictions)

            metric = evaluate(predictions_all, gts_all, self.cf.num_classes)
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

class Predict():
    def __init__(self, cf, dataloader):
        self.dataloader = dataloader
        self.cf = cf

    def start(self, model, criterion):
        model.net.eval()

        for vi, data in enumerate(self.dataloader):
            inputs, img_name = data

            inputs = Variable(inputs, volatile=True).cuda()

            outputs = model.net(inputs)
            predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

            path = os.path.join(self.cf.predict_path_output, img_name[0])
            #scipy.misc.imsave(path,predictions)
            predictions = Image.fromarray(predictions.astype(np.uint8))
            predictions = predictions.resize((self.cf.original_size[1],
                                    self.cf.original_size[0]), resample=Image.BILINEAR)
            predictions = np.array(predictions)
            cv.imwrite(path, predictions)
            print('%d / %d' % (vi + 1, len(self.dataloader)))