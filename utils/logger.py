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
        mIoU_class_list = stats.mIoU_perclass if stats.mIoU_perclass==[] else stats.mIoU_perclass.tolist()
        acc_class_list = stats.acc_perclass if stats.acc_perclass == [] else stats.acc_perclass.tolist()
        precision_class_list = stats.precision_perclass if stats.precision_perclass == [] else stats.precision_perclass.tolist()
        recall_class_list = stats.recall_perclass if stats.recall_perclass == [] else stats.recall_perclass.tolist()
        f1score_class_list = stats.f1score_perclass if stats.f1score_perclass == [] else stats.f1score_perclass.tolist()
        stats_dic = {'epoch': epoch, 'loss': stats.loss, 'mIoU': stats.mIoU, 'acc': stats.acc,
                     'precision': stats.precision,'recall': stats.recall, 'f1score': stats.f1score,
                     'conf_m': stats.conf_m,'mIoU_perclass': mIoU_class_list,'accuracy_perclass': acc_class_list,
                     'precision_perclass': precision_class_list,'recall_perclass': recall_class_list,
                     'f1score_perclass': f1score_class_list}
        json.dump(stats_dic, self.json_file)