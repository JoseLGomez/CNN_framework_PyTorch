class TaskStats(object):
    def __init__(self):
        self.loss = 0
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
        for row in self.conf_m:
            row_s = ['{:6.2f}'.format(100.*float(el)) for el in row]
            for el_s in row_s:
                msg = msg + el_s + ' '
            msg = msg + '\n'
        return msg

class Statistics(object):
    def __init__(self):
        self.val = TaskStats()
        self.test = TaskStats()
        self.train = TaskStats()