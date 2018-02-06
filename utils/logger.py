import json

# Save the printf to a log file
class Logger(object):
    def __init__(self, log_file):
        self.log = open(log_file, "w")

    def write(self, message):
        self.log.write(message)

    def create_json(self, json_file):
        self.json_file = open(json_file, "w")

    def save_json(self, stats, epoch=0):
        mIoU_class_list = stats.mIoU_perclass.tolist()
        stats_dic = {'epoch': epoch, 'loss': stats.loss, 'mIoU': stats.mIoU, 'acc': stats.acc,
                     'precision': stats.precision,'recall': stats.recall, 'f1score': stats.f1score,
                     'conf_m': stats.conf_m,'mIoU_perclass': mIoU_class_list}
        json.dump(stats_dic, self.json_file)