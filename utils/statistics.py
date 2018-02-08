import numpy as np

class TaskStats(object):
    def __init__(self):
        self.loss = float('inf')
        self.mIoU = 0
        self.acc = 0
        self.precision = 0
        self.recall = 0
        self.f1score = 0
        self.conf_m = []
        self.mIoU_perclass = []
        self.acc_perclass = []
        self.precision_perclass = []
        self.recall_perclass = []
        self.f1score_perclass = []

    def get_confm_str(self):
        msg = ''
        conf_m = [[0 if np.sum(row) == 0 else el / np.sum(row) for el in row] for row in self.conf_m]
        for row in conf_m:
            row_s = ['{:6.2f}'.format(100.*float(el)) for el in row]
            for el_s in row_s:
                msg = msg + el_s + ' '
            msg = msg + '\n'
        return msg

    def get_confm_norm(self):
        conf_m = [[0 if np.sum(row) == 0 else el / np.sum(row) for el in row] for row in self.conf_m]
        return np.asarray(conf_m)

class Statistics(object):
    def __init__(self):
        self.val = TaskStats()
        self.test = TaskStats()
        self.train = TaskStats()