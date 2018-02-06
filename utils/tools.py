import numpy as np
import cv2 as cv
import matplotlib
import StringIO
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def confm_metrics2image(conf_matrix,names=None):
    nLabels = np.shape(conf_matrix)[0]

    if names==None:
        plt_names = range(nLabels)
    else:
        plt_names = names

    conf_matrix = np.asarray(conf_matrix, dtype=np.float32)

    for i in range(nLabels):
        sum_row = sum(conf_matrix[i][:])
        for j in range(nLabels):
            if sum_row == 0:
                conf_matrix[i][j] = 0
            else:
                conf_matrix[i][j] = (conf_matrix[i][j]) / float(sum_row)

    img = StringIO.StringIO()
    plt.ioff()
    plt.cla()
    plt.clf()
    plt.imshow(conf_matrix,
               interpolation='nearest',
               cmap=plt.cm.Blues,
               vmin=0.0,
               vmax=1.0)
    plt.colorbar()
    plt.title('Confusion Matrix')

    plt.xticks(range(nLabels),plt_names, rotation=90)
    ystick = zip(plt_names, [conf_matrix[i][i] for i in range(nLabels)])
    ystick_str = [str(ystick[i][0]) + '(%.2f)' % ystick[i][1] for i in range(nLabels)]

    plt.yticks(range(nLabels), ystick_str)

    plt.xlabel('Prediction Label')
    plt.ylabel('True Label')

    plt.draw()
    plt.pause(0.1)
    plt.savefig(img, format='png')
    img.seek(0)

    data = np.asarray(bytearray(img.buf), dtype=np.uint8)
    img = cv.imdecode(data, cv.IMREAD_UNCHANGED)[:, :, 0:3]
    img = img[..., ::-1]

    return img

def save_prediction(output_path, predictions, names):
    for img in range(len(names)):
        output_file = output_path + names[img]
        cv.imwrite(output_file, np.squeeze(predictions[img], axis=2))

class Early_Stopping():
    def __init__(self, cf):
        self.cf = cf
        self.best_loss_metric = float('inf')
        self.best_metric = 0
        self.counter = 0
        self.patience = self.cf.patience
        self.stop = False

    def check(self, save_condition, train_mLoss, valid_mLoss=None, 
                mIoU_valid=None, mAcc_valid=None):
        if self.cf.stop_condition == 'train_loss':
            if train_mLoss < self.best_loss_metric:
                self.best_loss_metric = train_mLoss
                self.counter = 0
            else:
                self.counter += 1
        elif self.cf.stop_condition == 'valid_loss':
            if valid_mLoss < self.best_loss_metric:
                self.best_loss_metric = valid_mLoss
                self.counter = 0
            else:
                self.counter += 1
        elif self.cf.stop_condition == 'valid_mIoU':
            if mIoU_valid > self.best_metric:
                self.best_metric = mIoU_valid
                self.counter = 0
            else:
                self.counter += 1
        elif self.cf.stop_condition == 'valid_mAcc':
            if mAcc_valid > self.best_metric:
                self.best_metric = mAcc_valid
                self.counter = 0
            else:
                self.counter += 1
        if self.counter == self.patience:
            self.logger_stats.write(' Early Stopping Interruption\n')
            return True
        else:
            return False

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count