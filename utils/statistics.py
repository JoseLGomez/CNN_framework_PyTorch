class TaskStats(object):
    def __init__(self):
        self.loss = 0
        self.mIoU = 0
        self.acc = 0
        self.f1score = 0

class Statistics(object):
    def __init__(self):
        self.val = TaskStats()
        self.test = TaskStats()
        self.train = TaskStats()