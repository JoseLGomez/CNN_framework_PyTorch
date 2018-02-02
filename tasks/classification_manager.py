import sys
import time
import numpy as np
import os

sys.path.append('../')
from metrics.metrics import compute_stats, compute_precision, compute_recall, compute_f1score, compute_accuracy, compute_confusion_matrix
from simple_trainer_manager import SimpleTrainer

class Classification_Manager(SimpleTrainer):
    def __init__(self, cf, model):
        super(Classification_Manager, self).__init__(cf, model)

    class train(SimpleTrainer.train):
        def __init__(self, logger_stats, model, cf, validator, stats):
            super(Classification_Manager.train, self).__init__(logger_stats, model, cf, validator, stats)
            self.best_f1score = -1

        def validate_epoch(self, valid_set, valid_loader, criterion, early_Stopping, epoch):

            if valid_set is not None and valid_loader is not None:
                # Set model in validation mode
                self.model.net.eval()

                self.validator.start(criterion, valid_set, valid_loader, epoch)

                # Early stopping checking
                if self.cf.early_stopping:
                    early_Stopping.check(self.stats.train.loss, self.stats.val.loss, self.stats.val.mIoU,
                                         self.stats.val.acc)
                    if early_Stopping.stop == True:
                        self.stop = True

                # Set model in training mode
                self.model.net.train()

        def update_messages(self, epoch, epoch_time):
            # Update logger
            epoch_time = time.time() - epoch_time
            self.logger_stats.write('\t Epoch step finished: %ds ' % (epoch_time))

            # Compute best stats
            self.msg_stats_last = '\nLast epoch: acc= %.2f, precision= %.2f, recall= %.2f, f1score= %.2f, loss = %.5f\n' % (
                100 * self.stats.val.acc, 100 * self.stats.val.precision, 100 * self.stats.val.recall, 100 * self.stats.val.f1score, self.stats.val.loss)
            if self.best_f1score < self.stats.val.f1score:
                self.msg_stats_best = 'Best case: epoch = %d, acc= %.2f, precision= %.2f, recall= %.2f, f1score= %.2f, loss = %.5f\n' % (
                    epoch, 100 * self.stats.val.acc, 100 * self.stats.val.precision, 100 * self.stats.val.recall, 100 * self.stats.val.f1score, self.stats.val.loss)
                self.best_f1score = self.stats.val.f1score

                msg_confm = self.stats.val.get_confm_str()
                self.msg_stats_best = self.msg_stats_best + '\nConfusion matrix:\n' + msg_confm

    class validation(SimpleTrainer.validation):
        def __init__(self, logger_stats, model, cf, stats):
            super(Classification_Manager.validation, self).__init__(logger_stats, model, cf, stats)

        def compute_stats(self, TP_list, TN_list, FP_list, FN_list, val_loss):
            mean_accuracy = compute_accuracy(TP_list, TN_list, FP_list, FN_list)
            mean_precision = compute_precision(TP_list,FP_list)
            mean_recall = compute_recall(TP_list,FN_list)
            mean_f1score = compute_f1score(TP_list,FP_list,FN_list)
            self.stats.val.acc = np.mean(mean_accuracy)
            self.stats.val.recall= np.mean(mean_recall)
            self.stats.val.precision = np.mean(mean_precision)
            self.stats.val.f1score = np.mean(mean_f1score)
            self.stats.val.loss = val_loss.avg

        def save_stats(self, epoch):
            # Save logger
            if epoch is not None:
                self.logger_stats.write('----------------- Epoch scores summary -------------------------')
                self.logger_stats.write(
                    '[epoch %d], [val loss %.5f], [acc %.2f], [precision %.2f], [recall %.2f], [f1score %.2f],' % (
                        epoch, self.stats.val.loss, 100 * self.stats.val.acc, 100 * self.stats.val.precision,
                        100 * self.stats.val.recall, 100 * self.stats.val.f1score))
                self.logger_stats.write('---------------------------------------------------------------- \n')
            else:
                self.logger_stats.write('----------------- Scores summary --------------------')
                self.logger_stats.write(
                    '[val loss %.5f], [acc %.2f], [precision %.2f], [recall %.2f], [f1score %.2f]' % (
                        self.stats.val.loss, 100 * self.stats.val.acc, 100 * self.stats.val.precision,
                        100 * self.stats.val.recall, 100 * self.stats.val.f1score))
                self.logger_stats.write('---------------------------------------------------------------- \n')

    class predict(SimpleTrainer.predict):
        def __init__(self, logger_stats, model, cf):
            super(Classification_Manager.predict, self).__init__(logger_stats, model, cf)
            self.filename = os.path.join(self.cf.predict_path_output, 'predictions.txt')
            self.f = open(self.filename,'w')

        def write_results(self,predictions, img_name):
                msg = img_name[0] + ' ' + str(predictions[0]) + '\n'
                self.f.writelines(msg)