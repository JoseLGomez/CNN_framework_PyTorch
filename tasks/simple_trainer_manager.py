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
from utils.messages import Messages
from metrics.metrics import compute_accuracy, compute_confusion_matrix, extract_stats_from_confm,compute_mIoU

class SimpleTrainer(object):
    def __init__(self, cf, model, writer):
        self.cf = cf
        self.model = model
        self.logger_stats = Logger(cf.log_file_stats)
        self.stats = Statistics()
        self.msg = Messages()
        self.writer = writer

        self.validator = self.validation(self.logger_stats, self.model, cf, self.stats, self.msg, self.writer)
        self.trainer = self.train(self.logger_stats, self.model, cf, self.validator, self.stats, self.msg, self.writer)
        self.predictor = self.predict(self.logger_stats, self.model, cf, self.writer)

    class train(object):
        def __init__(self, logger_stats, model, cf, validator, stats, msg, writer):
            # Initialize training variables
            self.logger_stats = logger_stats
            self.model = model
            self.cf = cf
            self.validator = validator
            self.logger_stats.write('\n- Starting train <--- \n')
            self.curr_epoch = self.cf.initial_epoch
            self.stop = False
            self.stats = stats
            self.best_acc = 0
            self.msg = msg
            self.writer = writer

        def start(self, criterion, optimizer, train_loader, train_set, valid_set=None, valid_loader=None, scheduler=None):
            train_num_batches = math.ceil(train_set.num_images / float(self.cf.train_batch_size))
            val_num_batches = 0 if valid_set is None else math.ceil(valid_set.num_images / float(self.cf.valid_batch_size))
            # Define early stopping control
            if self.cf.early_stopping:
                early_Stopping = Early_Stopping(self.cf)
            else:
                early_Stopping = None

            prev_msg = '\nTotal estimated training time...\n'
            global_bar = ProgressBar((self.cf.epochs+1-self.curr_epoch)*(train_num_batches+val_num_batches), lenBar=20)
            global_bar.set_prev_msg(prev_msg)


            # Train process
            for epoch in range(self.curr_epoch, self.cf.epochs + 1):
                # Shuffle train data
                train_set.update_indexes()

                # Initialize logger
                epoch_time = time.time()
                self.logger_stats.write('\t ------ Epoch: ' + str(epoch) + ' ------ \n')

                # Initialize epoch progress bar
                self.msg.accum_str = '\n\nEpoch %d/%d estimated time...\n' % (epoch, self.cf.epochs + 1 - self.curr_epoch)
                epoch_bar = ProgressBar(train_num_batches, lenBar=20)
                epoch_bar.update(show=False)

                # Initialize stats
                train_loss = AverageMeter()

                # Train epoch
                for i, data in enumerate(train_loader):
                    # Read Data
                    inputs, labels = data

                    N,w,h,c = inputs.size()
                    inputs = Variable(inputs).cuda()
                    labels = Variable(labels).cuda()

                    # Predict model
                    optimizer.zero_grad()
                    outputs = self.model.net(inputs)

                    # Compute gradients
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # Update loss
                    train_loss.update(loss.data[0], N)
                    self.stats.train.loss = train_loss.avg / (w*h*c)
                    # tensorboard loss
                    curr_iter = (epoch - 1) * train_num_batches + i
                    self.writer.add_scalar('train_loss_iter', self.stats.train.loss, curr_iter)

                    # Update epoch messages
                    self.update_epoch_messages(epoch_bar, global_bar, train_num_batches,epoch, i)

                # Epoch loss tensorboard
                self.writer.add_scalar('train_loss_epoch', self.stats.train.loss, epoch)

                # Validate epoch
                self.validate_epoch(valid_set, valid_loader, criterion, early_Stopping, epoch, global_bar)

                # Update scheduler
                if scheduler is not None:
                    scheduler.step(self.stats.val.loss)

                # Saving model if needed
                self.model.save(self.model.net, self.stats)

                # Update display values
                self.update_messages(epoch, epoch_time)

                if self.stop:
                    return

            # Save model without training
            if self.cf.epochs == 0:
                self.model.save_model(self.model.net)

        def validate_epoch(self,valid_set, valid_loader, criterion, early_Stopping, epoch, global_bar):

            if valid_set is not None and valid_loader is not None:
                # Set model in validation mode
                self.model.net.eval()

                self.validator.start(criterion, valid_set, valid_loader, epoch, global_bar=global_bar)

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
            self.logger_stats.write('\t Epoch step finished: %ds \n' % (epoch_time))

            # Compute best stats
            self.msg.msg_stats_last = '\nLast epoch: acc = %.2f, loss = %.5f\n' % (100 * self.stats.val.mIoU, self.stats.val.loss)
            if self.best_acc < self.stats.val.mIoU:
                self.msg.msg_stats_best = 'Best case: epoch = %d, acc = %.2f, loss = %.5f\n' % (
                epoch, 100 * self.stats.val.acc, self.stats.val.loss)

                msg_confm = self.stats.val.get_confm_str()
                self.logger_stats.write(msg_confm)
                self.msg.msg_stats_best = self.msg.msg_stats_best + msg_confm

                self.best_acc = self.stats.val.acc

        def update_epoch_messages(self, epoch_bar, global_bar, train_num_batches, epoch, batch):
            # Update progress bar
            epoch_bar.set_msg('loss = %.5f' % self.stats.train.loss)
            self.msg.last_str = epoch_bar.get_message(step=True)
            global_bar.set_msg(self.msg.accum_str + self.msg.last_str + self.msg.msg_stats_last + self.msg.msg_stats_best)
            global_bar.update()

            # writer.add_scalar('train_loss', train_loss.avg, curr_iter)

            # Display progress
            curr_iter = (epoch - 1) * train_num_batches + batch + 1
            if (batch + 1) % math.ceil(train_num_batches / 20.) == 0:
                self.logger_stats.write('[Global iteration %d], [iter %d / %d], [train loss %.5f] \n' % (
                    curr_iter, batch + 1, train_num_batches, self.stats.train.loss))

    class validation(object):
        def __init__(self, logger_stats, model, cf, stats, msg, writer):
            # Initialize validation variables
            self.logger_stats = logger_stats
            self.model = model
            self.cf = cf
            self.stats = stats
            self.msg = msg
            self.writer = writer

        def start(self, criterion, valid_set, valid_loader, epoch=None, global_bar=None):
            confm_list = np.zeros((self.cf.num_classes,self.cf.num_classes))
            mIoU_list = []

            val_loss = AverageMeter()

            # Initialize epoch progress bar
            val_num_batches = math.ceil(valid_set.num_images / float(self.cf.valid_batch_size))
            prev_msg = '\nValidation estimated time...\n'
            bar = ProgressBar(val_num_batches, lenBar=20)
            bar.set_prev_msg(prev_msg)
            bar.update(show=False)

            # Validate model
            for vi, data in enumerate(valid_loader):
                # Read data
                inputs, gts = data
                n_images,w,h,c = inputs.size()
                inputs = Variable(inputs, volatile=True).cuda()
                gts = Variable(gts, volatile=True).cuda()

                # Predict model
                outputs = self.model.net(inputs)
                predictions = outputs.data.max(1)[1].cpu().numpy()

                # Compute batch stats
                val_loss.update(criterion(outputs, gts).data[0] / n_images, n_images)
                confm = compute_confusion_matrix(predictions,gts.cpu().data.numpy(),self.cf.num_classes,self.cf.void_class)
                confm_list = map(operator.add, confm_list, confm)

                # Save epoch stats
                self.stats.val.conf_m = [[0 if np.sum(row) == 0 else el / np.sum(row) for el in row] for row in confm_list]
                self.stats.val.loss = val_loss.avg / (w * h * c)

                # Update messages
                self.update_msg(bar, global_bar)

            # Compute stats
            confm_list = [[0 if np.sum(row) == 0 else el / np.sum(row) for el in row] for row in confm_list]
            self.stats.val.conf_m = confm_list

            TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(np.asarray(confm_list))
            self.compute_stats(TP_list, TN_list, FP_list, FN_list,val_loss)

            # Save stats
            self.save_stats(epoch)

        def update_msg(self, bar, global_bar):
            if global_bar==None:
                # Update progress bar
                bar.update()
            else:
                self.msg.eval_str = '\n' + bar.get_message(step=True)
                global_bar.set_msg(self.msg.accum_str + self.msg.last_str + self.msg.msg_stats_last + self.msg.msg_stats_best + self.msg.eval_str)
                global_bar.update()

        def compute_stats(self, confm_list, val_loss):
            TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(confm_list)
            mean_accuracy = compute_accuracy(TP_list, TN_list, FP_list, FN_list)
            self.stats.val.acc = np.nanmean(mean_accuracy)
            self.stats.val.loss = val_loss.avg

        def save_stats(self, epoch):
            # Save logger
            if epoch is not None:
                self.logger_stats.write('----------------- Epoch scores summary ------------------------- \n')
                self.logger_stats.write('[epoch %d], [val loss %.5f], [acc %.2f] \n' % (
                    epoch, self.stats.val.loss, 100*self.stats.val.acc))
                self.logger_stats.write('---------------------------------------------------------------- \n')
            else:
                self.logger_stats.write('----------------- Scores summary -------------------- \n')
                self.logger_stats.write('[val loss %.5f], [acc %.2f] \n' % (
                    self.stats.val.loss, 100 * self.stats.val.acc))
                self.logger_stats.write('---------------------------------------------------------------- \n')

    class predict(object):
        def __init__(self, logger_stats, model, cf, writer):
            self.logger_stats = logger_stats
            self.model = model
            self.cf = cf
            self.writer = writer

        def start(self, dataloader):
            self.model.net.eval()

            for vi, data in enumerate(dataloader):
                inputs, img_name = data

                inputs = Variable(inputs, volatile=True).cuda()

                outputs = self.model.net(inputs)
                predictions = outputs.data.max(1)[1].cpu().numpy()

                self.write_results(predictions,img_name)

                self.logger_stats.write('%d / %d \n' % (vi + 1, len(dataloader)))

        def write_results(self,predictions, img_name):
                pass