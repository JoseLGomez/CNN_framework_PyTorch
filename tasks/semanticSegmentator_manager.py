import math
import sys
import time
import numpy as np
import cv2 as cv
import os
from torch.autograd import Variable
from PIL import Image
import operator

sys.path.append('../')
from utils.tools import AverageMeter, Early_Stopping
from utils.ProgressBar import ProgressBar
from utils.logger import Logger
from metrics.metrics import compute_stats, compute_mIoU

class SemanticSegmentation_Manager():
    def __init__(self, cf, model):
        self.cf = cf
        self.model = model
        self.logger_stats = Logger(cf.log_file_stats)

    def train(self, criterion, optimizer, train_loader, train_set, valid_set=None, valid_loader=None, scheduler=None):
        self.logger_stats.write('\n- Starting train <---')
        train_num_batches = math.ceil(train_set.num_images / float(self.cf.train_batch_size))
        curr_epoch = self.cf.initial_epoch
        # Define early stopping control
        if self.cf.early_stopping:
            early_Stopping = Early_Stopping(self.cf)

        print '\nTotal estimated training time...'
        global_bar = ProgressBar((self.cf.epochs+1-curr_epoch)*train_num_batches, lenBar=20)


        # Train process
        for epoch in range(curr_epoch, self.cf.epochs + 1):
            epoch_time = time.time()
            self.logger_stats.write('\t ------ Epoch: ' + str(epoch) + ' ------ \n')
            # Progress bar

            accum_str = '\n\nEpoch %d/%d estimated time...\n' % (epoch, self.cf.epochs + 1 - curr_epoch)
            epoch_bar = ProgressBar(train_num_batches, lenBar=20)
            epoch_bar.update(show=False)

            train_loss = AverageMeter()
            curr_iter = (epoch - 1) * len(train_loader)
            for i, data in enumerate(train_loader):
                inputs, labels = data
                assert inputs.size()[2:] == labels.size()[1:]
                N = inputs.size(0)
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()
                optimizer.zero_grad()
                outputs = self.model.net(inputs)
                assert outputs.size()[2:] == labels.size()[1:]
                assert outputs.size()[1] == self.cf.num_classes

                loss = criterion(outputs, labels) / N
                loss.backward()
                optimizer.step()

                train_loss.update(loss.data[0], N)

                epoch_bar.set_msg('loss = ' + str(train_loss.avg))
                last_str = epoch_bar.get_message(step=True)
                global_bar.set_msg(accum_str + last_str)
                global_bar.update()

                curr_iter += 1
                # writer.add_scalar('train_loss', train_loss.avg, curr_iter)
                # Display progress
                if (i + 1) % math.ceil(train_num_batches / 20.) == 0:
                    self.logger_stats.write('[Global iteration %d], [iter %d / %d], [train loss %.5f]' % (
                        curr_iter, i + 1, len(train_loader), train_loss.avg))
            # validate epoch
            val_loss, acc_cls, mean_IoU = 0, 0, 0
            if valid_set is not None and valid_loader is not None:
                val_loss, acc_cls, mean_IoU = self.validation(criterion, valid_set, valid_loader,  epoch)
                # Early stopping checking
                if self.cf.early_stopping:
                    early_Stopping.check(train_loss.avg, val_loss, mean_IoU, acc_cls)
                    if early_Stopping.stop == True:
                        self.logger_stats.write(' Early Stopping Interruption [Epoch: ' + str(epoch) + ' ] \n')
                        return
                if scheduler is not None:
                    scheduler.step(val_loss)
                # Shuffle validation data
                valid_set.update_indexes()
            self.model.net.train()
            # Saving model if needed
            self.model.save(self.model.net, train_loss.avg, val_loss, mean_IoU, acc_cls)
            # Shuffle train data
            train_set.update_indexes()
            epoch_time = time.time() - epoch_time
            self.logger_stats.write('\t Epoch step finished: %ds ' % (epoch_time))

    def validation(self, criterion, valid_set, valid_loader, epoch=None):
        self.model.net.eval()

        valid_num_batches = math.ceil(valid_set.num_images / float(self.cf.valid_batch_size))
        # Progress bar
        prog_bar = ProgressBar(valid_num_batches)

        val_loss = AverageMeter()
        # gts_all, predictions_all = [], []
        acc = np.zeros(len(valid_loader))
        acc_cls = np.zeros(len(valid_loader))
        mean_IoU = np.zeros(len(valid_loader))
        fwavacc = np.zeros(len(valid_loader))

        TP_list, FP_list, FN_list = np.zeros(self.cf.num_classes), np.zeros(self.cf.num_classes), np.zeros(
            self.cf.num_classes)

        for vi, data in enumerate(valid_loader):
            inputs, gts = data
            N = inputs.size(0)
            inputs = Variable(inputs, volatile=True).cuda()
            gts = Variable(gts, volatile=True).cuda()

            outputs = self.model.net(inputs)
            predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()

            val_loss.update(criterion(outputs, gts).data[0] / N, N)

            TP, FP, FN = compute_stats(predictions, gts.cpu().data.numpy(), self.cf.num_classes, self.cf.void_class)
            TP_list = map(operator.add, TP_list, TP)
            FN_list = map(operator.add, FN_list, FN)
            FP_list = map(operator.add, FP_list, FP)

            # gts_all.append(gts.data.cpu().numpy())
            # predictions_all.append(predictions)
            #
            # metric = evaluate(predictions_all, gts_all, self.cf.num_classes)
            # acc[vi] = metric[0]
            # acc_cls[vi] = metric[1]
            # mean_IoU[vi] = metric[2]
            # fwavacc[vi] = metric[3]

        mean_IoU = compute_mIoU(TP_list, FP_list, FN_list)
        acc = 0  # np.mean(acc)
        acc_cls = 0  # np.mean(acc_cls)
        mean_IoU = np.mean(mean_IoU)
        fwavacc = 0  # np.mean(fwavacc)

        if epoch is not None:
            self.logger_stats.write('----------------- Epoch scores summary -------------------------')
            self.logger_stats.write('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_IoU %.5f], [fwavacc %.5f]' % (
                epoch, val_loss.avg, acc, acc_cls, mean_IoU, fwavacc))
            self.logger_stats.write('---------------------------------------------------------------- \n')
        else:
            self.logger_stats.write('----------------- Scores summary --------------------')
            self.logger_stats.write('[val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_IoU %.5f], [fwavacc %.5f]' % (
                val_loss.avg, acc, acc_cls, mean_IoU, fwavacc))
            self.logger_stats.write('---------------------------------------------------------------- \n')
        return val_loss.avg, acc_cls, mean_IoU

    def predict(self, dataloader):
        self.model.net.eval()

        for vi, data in enumerate(dataloader):
            inputs, img_name = data

            inputs = Variable(inputs, volatile=True).cuda()

            outputs = self.model.net(inputs)
            predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

            path = os.path.join(self.cf.predict_path_output, img_name[0])
            # scipy.misc.imsave(path,predictions)
            predictions = Image.fromarray(predictions.astype(np.uint8))
            if self.cf.resize is not None:
                predictions = predictions.resize((self.cf.original_size[1],
                                                  self.cf.original_size[0]), resample=Image.BILINEAR)
            predictions = np.array(predictions)
            cv.imwrite(path, predictions)
            self.logger_stats.write('%d / %d' % (vi + 1, len(dataloader)))