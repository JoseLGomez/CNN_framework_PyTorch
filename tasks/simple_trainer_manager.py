import math
import sys
import time
import numpy as np
from torch.autograd import Variable
import operator

sys.path.append('../')
from utils.tools import AverageMeter, Early_Stopping
from utils.ProgressBar import ProgressBar
from utils.logger import Logger
from utils.statistics import Statistics
from metrics.metrics import compute_stats, compute_accuracy, compute_confusion_matrix

class SimpleTrainer(object):
    def __init__(self, cf, model):
        self.cf = cf
        self.model = model
        self.logger_stats = Logger(cf.log_file_stats)
        self.stats = Statistics()

        self.validator = self.validation(self.logger_stats, self.model, cf, self.stats)
        self.trainer = self.train(self.logger_stats, self.model, cf, self.validator, self.stats)
        self.predictor = self.predict(self.logger_stats, self.model, cf)

    class train(object):
        def __init__(self, logger_stats, model, cf, validator, stats):
            # Initialize training variables
            self.logger_stats = logger_stats
            self.model = model
            self.cf = cf
            self.validator = validator
            self.logger_stats.write('\n- Starting train <---')
            self.curr_epoch = self.cf.initial_epoch
            self.msg_stats_last = ''
            self.msg_stats_best = ''
            self.stop = False
            self.stats = stats
            self.best_acc = 0

        def start(self, criterion, optimizer, train_loader, train_set, valid_set=None, valid_loader=None, scheduler=None):
            train_num_batches = math.ceil(train_set.num_images / float(self.cf.train_batch_size))
            # Define early stopping control
            if self.cf.early_stopping:
                early_Stopping = Early_Stopping(self.cf)
            else:
                early_Stopping = None

            prev_msg = '\nTotal estimated training time...\n'
            global_bar = ProgressBar((self.cf.epochs+1-self.curr_epoch)*train_num_batches, lenBar=20)
            global_bar.set_prev_msg(prev_msg)


            # Train process
            for epoch in range(self.curr_epoch, self.cf.epochs + 1):
                # Shuffle train data
                train_set.update_indexes()

                # Initialize logger
                epoch_time = time.time()
                self.logger_stats.write('\t ------ Epoch: ' + str(epoch) + ' ------ \n')

                # Initialize epoch progress bar
                accum_str = '\n\nEpoch %d/%d estimated time...\n' % (epoch, self.cf.epochs + 1 - self.curr_epoch)
                epoch_bar = ProgressBar(train_num_batches, lenBar=20)
                epoch_bar.update(show=False)

                # Initialize stats
                train_loss = AverageMeter()

                # Train epoch
                for i, data in enumerate(train_loader):
                    # Read Data
                    inputs, labels = data

                    N = inputs.size(0)
                    inputs = Variable(inputs).cuda()
                    labels = Variable(labels).cuda()

                    # Predict model
                    optimizer.zero_grad()
                    outputs = self.model.net(inputs)

                    # Compute gradients
                    loss = criterion(outputs, labels) / N
                    loss.backward()
                    optimizer.step()

                    # Update loss
                    train_loss.update(loss.data[0], N)
                    self.stats.train.loss = train_loss.avg

                    # Update epoch messages
                    self.update_epoch_messages(epoch_bar, global_bar, accum_str, train_num_batches,epoch, i)

                # Validate epoch
                self.validate_epoch(valid_set, valid_loader, criterion, early_Stopping, epoch)

                # Update scheduler
                if scheduler is not None:
                    scheduler.step(self.stats.val.loss)

                # Saving model if needed
                self.model.save(self.model.net, self.stats)

                # Update display values
                self.update_messages(epoch, epoch_time)

                if self.stop:
                    return

        def validate_epoch(self,valid_set, valid_loader, criterion, early_Stopping, epoch):

            if valid_set is not None and valid_loader is not None:
                # Set model in validation mode
                self.model.net.eval()

                self.validator.start(criterion, valid_set, valid_loader, epoch)

                # Early stopping checking
                if self.cf.early_stopping:
                    early_Stopping.check(self.stats.train.loss, self.stats.val.loss, self.stats.val.mIoU, self.stats.val.acc)
                    if early_Stopping.stop == True:
                        self.stop=True

                # Set model in training mode
                self.model.net.train()


        def update_messages(self, epoch, epoch_time):
            # Update logger
            epoch_time = time.time() - epoch_time
            self.logger_stats.write('\t Epoch step finished: %ds ' % (epoch_time))

            # Compute best stats
            self.msg_stats_last = '\nLast epoch: acc = %.2f, loss = %.5f\n' % (100 * self.stats.val.mIoU, self.stats.val.loss)
            if self.best_acc < self.stats.val.mIoU:
                self.msg_stats_best = 'Best case: epoch = %d, acc = %.2f, loss = %.5f\n' % (
                epoch, 100 * self.stats.val.acc, self.stats.val.loss)

                msg_confm = self.stats.val.get_confm_str()
                self.msg_stats_best = self.msg_stats_best + msg_confm

                self.best_acc = self.stats.val.acc

        def update_epoch_messages(self, epoch_bar, global_bar, accum_str, train_num_batches,epoch, batch):
            # Update progress bar
            epoch_bar.set_msg('loss = %.5f' % self.stats.train.loss)
            last_str = epoch_bar.get_message(step=True)
            global_bar.set_msg(accum_str + last_str + self.msg_stats_last + self.msg_stats_best)
            global_bar.update()

            # writer.add_scalar('train_loss', train_loss.avg, curr_iter)

            # Display progress
            curr_iter = (epoch - 1) * train_num_batches + batch
            if (batch + 1) % math.ceil(train_num_batches / 20.) == 0:
                self.logger_stats.write('[Global iteration %d], [iter %d / %d], [train loss %.5f]' % (
                    curr_iter, batch + 1, train_num_batches, self.stats.train.loss))

    class validation(object):
        def __init__(self, logger_stats, model, cf, stats):
            # Initialize validation variables
            self.logger_stats = logger_stats
            self.model = model
            self.cf = cf
            self.stats = stats

        def start(self, criterion, valid_set, valid_loader, epoch=None):
            TP_list, TN_list, FP_list, FN_list = np.zeros(self.cf.num_classes), np.zeros(self.cf.num_classes), \
                                                 np.zeros(self.cf.num_classes), np.zeros(self.cf.num_classes)
            confm_list = np.zeros((self.cf.num_classes,self.cf.num_classes))

            val_loss = AverageMeter()

            # Validate model
            for vi, data in enumerate(valid_loader):
                # Read data
                inputs, gts = data
                n_images = inputs.size(0)
                inputs = Variable(inputs, volatile=True).cuda()
                gts = Variable(gts, volatile=True).cuda()

                # Predict model
                outputs = self.model.net(inputs)
                predictions = outputs.data.max(1)[1].cpu().numpy()

                # Compute batch stats
                val_loss.update(criterion(outputs, gts).data[0] / n_images, n_images)
                TP, TN, FP, FN = compute_stats(predictions, gts.cpu().data.numpy(), self.cf.num_classes, self.cf.void_class)
                TP_list = map(operator.add, TP_list, TP)
                TN_list = map(operator.add, TN_list, TN)
                FN_list = map(operator.add, FN_list, FN)
                FP_list = map(operator.add, FP_list, FP)
                confm = compute_confusion_matrix(predictions,gts.cpu().data.numpy(),self.cf.num_classes,self.cf.void_class)
                confm_list = map(operator.add, confm_list, confm)

            # Compute stats
            self.compute_stats(TP_list, TN_list, FP_list, FN_list,val_loss)
            # confm_list = map(list, zip(*confm_list))
            confm_list = [[0 if np.sum(row)==0 else el/np.sum(row) for el in row] for row in confm_list]

            # Save stats
            self.save_stats(epoch)
            self.stats.val.conf_m = confm_list

        def compute_stats(self, TP_list, TN_list, FP_list, FN_list, val_loss):
            mean_accuracy = compute_accuracy(TP_list, TN_list, FP_list, FN_list)
            self.stats.val.acc = np.mean(mean_accuracy)
            self.stats.val.loss = val_loss.avg

        def save_stats(self, epoch):
            # Save logger
            if epoch is not None:
                self.logger_stats.write('----------------- Epoch scores summary -------------------------')
                self.logger_stats.write('[epoch %d], [val loss %.5f], [acc %.2f],' % (
                    epoch, self.stats.val.loss, 100*self.stats.val.acc))
                self.logger_stats.write('---------------------------------------------------------------- \n')
            else:
                self.logger_stats.write('----------------- Scores summary --------------------')
                self.logger_stats.write('[val loss %.5f], [acc %.2f]' % (
                    self.stats.val.loss, 100 * self.stats.val.acc))
                self.logger_stats.write('---------------------------------------------------------------- \n')

    class predict(object):
        def __init__(self, logger_stats, model, cf):
            self.logger_stats = logger_stats
            self.model = model
            self.cf = cf

        def start(self, dataloader):
            self.model.net.eval()

            for vi, data in enumerate(dataloader):
                inputs, img_name = data

                inputs = Variable(inputs, volatile=True).cuda()

                outputs = self.model.net(inputs)
                predictions = outputs.data.max(1)[1].cpu().numpy()

                self.write_results(predictions,img_name)

                self.logger_stats.write('%d / %d' % (vi + 1, len(dataloader)))

        def write_results(self,predictions, img_name):
                pass