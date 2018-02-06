import sys
import time
import numpy as np
import os

sys.path.append('../')
from metrics.metrics import compute_precision, compute_recall, compute_f1score, compute_accuracy, compute_confusion_matrix, extract_stats_from_confm
from simple_trainer_manager import SimpleTrainer
from utils.tools import confm_metrics2image

class Classification_Manager(SimpleTrainer):
    def __init__(self, cf, model):
        super(Classification_Manager, self).__init__(cf, model)

    class train(SimpleTrainer.train):
        def __init__(self, logger_stats, model, cf, validator, stats, msg):
            super(Classification_Manager.train, self).__init__(logger_stats, model, cf, validator, stats, msg)
            self.best_f1score = -1

        def validate_epoch(self, valid_set, valid_loader, criterion, early_Stopping, epoch, global_bar):

            if valid_set is not None and valid_loader is not None:
                # Set model in validation mode
                self.model.net.eval()

                self.validator.start(criterion, valid_set, valid_loader, epoch, global_bar=global_bar)

                # Early stopping checking
                if self.cf.early_stopping:
                    early_Stopping.check(self.stats.train.loss, self.stats.val.loss, self.stats.val.mIoU,
                                         self.stats.val.acc)
                    if early_Stopping.stop == True:
                        self.stop = True

                # Set model in training mode
                self.model.net.train()

        def compute_stats(self, confm_list, train_loss):
            TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(confm_list)
            mean_accuracy = compute_accuracy(TP_list, TN_list, FP_list, FN_list)
            mean_precision = compute_precision(TP_list,FP_list)
            mean_recall = compute_recall(TP_list,FN_list)
            mean_f1score = compute_f1score(TP_list,FP_list,FN_list)
            self.stats.train.acc = np.nanmean(mean_accuracy)
            self.stats.train.recall= np.nanmean(mean_recall)
            self.stats.train.precision = np.nanmean(mean_precision)
            self.stats.train.f1score = np.nanmean(mean_f1score)
            if train_loss is not None:
                self.stats.train.loss = train_loss.avg

        def update_messages(self, epoch, epoch_time):
            # Update logger
            epoch_time = time.time() - epoch_time
            self.logger_stats.write('\t Epoch step finished: %ds \n' % (epoch_time))

            # Compute best stats
            self.msg.msg_stats_last = '\nLast epoch: acc= %.2f, precision= %.2f, recall= %.2f, f1score= %.2f, loss = %.5f\n' % (
                100 * self.stats.val.acc, 100 * self.stats.val.precision, 100 * self.stats.val.recall, 100 * self.stats.val.f1score, self.stats.val.loss)
            if self.best_f1score < self.stats.val.f1score:
                self.msg.msg_stats_best = 'Best case: epoch = %d, acc= %.2f, precision= %.2f, recall= %.2f, f1score= %.2f, loss = %.5f\n' % (
                    epoch, 100 * self.stats.val.acc, 100 * self.stats.val.precision, 100 * self.stats.val.recall, 100 * self.stats.val.f1score, self.stats.val.loss)
                self.best_f1score = self.stats.val.f1score

                msg_confm = self.stats.val.get_confm_str()
                self.msg.msg_stats_best = self.msg.msg_stats_best + '\nConfusion matrix:\n' + msg_confm

        def save_stats_epoch(self, epoch):
            # Save logger
            if epoch is not None:
                # Epoch loss tensorboard
                self.writer.add_scalar('losses/epoch', self.stats.train.loss, epoch)
                self.writer.add_scalar('metrics/accuracy', 100.*self.stats.train.acc, epoch)
                self.writer.add_scalar('metrics/precision', 100.*self.stats.train.precision, epoch)
                self.writer.add_scalar('metrics/recall', 100.*self.stats.train.recall, epoch)
                self.writer.add_scalar('metrics/f1score', 100.*self.stats.train.f1score, epoch)
                conf_mat_img = confm_metrics2image(self.stats.train.get_confm_norm(), self.cf.labels)
                self.writer.add_image('metrics/conf_matrix', conf_mat_img, epoch)


    class validation(SimpleTrainer.validation):
        def __init__(self, logger_stats, model, cf, stats, msg):
            super(Classification_Manager.validation, self).__init__(logger_stats, model, cf, stats, msg)

        def compute_stats(self, confm_list, val_loss):
            TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(confm_list)
            mean_accuracy = compute_accuracy(TP_list, TN_list, FP_list, FN_list)
            mean_precision = compute_precision(TP_list,FP_list)
            mean_recall = compute_recall(TP_list,FN_list)
            mean_f1score = compute_f1score(TP_list,FP_list,FN_list)
            self.stats.val.acc = np.nanmean(mean_accuracy)
            self.stats.val.recall= np.nanmean(mean_recall)
            self.stats.val.precision = np.nanmean(mean_precision)
            self.stats.val.f1score = np.nanmean(mean_f1score)
            if val_loss is not None:
                self.stats.val.loss = val_loss.avg

        def save_stats(self, epoch):
            # Save logger
            if epoch is not None:
                # add scores to log
                self.logger_stats.write('----------------- Epoch scores summary -------------------------\n')
                self.logger_stats.write(
                    '[epoch %d], [val loss %.5f], [acc %.2f], [precision %.2f], [recall %.2f], [f1score %.2f]\n' % (
                        epoch, self.stats.val.loss, 100 * self.stats.val.acc, 100 * self.stats.val.precision,
                        100 * self.stats.val.recall, 100 * self.stats.val.f1score))
                self.logger_stats.write('---------------------------------------------------------------- \n')
                self.logger_stats.save_json(self.stats.val, epoch)

                # add scores to tensorboard
                self.writer.add_scalar('losses/epoch', self.stats.val.loss, epoch)
                self.writer.add_scalar('metrics/accuracy', 100.*self.stats.val.acc, epoch)
                self.writer.add_scalar('metrics/precision', 100.*self.stats.val.precision, epoch)
                self.writer.add_scalar('metrics/recall', 100.*self.stats.val.recall, epoch)
                self.writer.add_scalar('metrics/f1score', 100.*self.stats.val.f1score, epoch)
                conf_mat_img = confm_metrics2image(self.stats.val.get_confm_norm(), self.cf.labels)
                self.writer.add_image('metrics/conf_matrix', conf_mat_img, epoch)
            else:
                self.logger_stats.write('----------------- Scores summary --------------------\n')
                self.logger_stats.write(
                    '[val loss %.5f], [acc %.2f], [precision %.2f], [recall %.2f], [f1score %.2f]\n' % (
                        self.stats.val.loss, 100 * self.stats.val.acc, 100 * self.stats.val.precision,
                        100 * self.stats.val.recall, 100 * self.stats.val.f1score))
                self.logger_stats.write('---------------------------------------------------------------- \n')

        def update_msg(self, bar, global_bar):

            self.compute_stats(np.asarray(self.stats.val.conf_m), None)
            bar.set_msg(', acc: %.02f, precision: %.02f, recall: %.02f f1score: %.02f' % (100.*self.stats.val.acc,
                                                            100.*self.stats.val.precision, 100.*self.stats.val.recall,
                                                            100.*self.stats.val.f1score))

            if global_bar==None:
                # Update progress bar
                bar.update()
            else:
                self.msg.eval_str = '\n' + bar.get_message(step=True)
                global_bar.set_msg(self.msg.accum_str + self.msg.last_str + self.msg.msg_stats_last + self.msg.msg_stats_best + self.msg.eval_str)
                global_bar.update()

    class predict(SimpleTrainer.predict):
        def __init__(self, logger_stats, model, cf):
            super(Classification_Manager.predict, self).__init__(logger_stats, model, cf)
            self.filename = os.path.join(self.cf.predict_path_output, 'predictions.txt')
            self.f = open(self.filename,'w')

        def write_results(self,predictions, img_name):
                msg = img_name[0] + ' ' + str(predictions[0]) + '\n'
                self.f.writelines(msg)