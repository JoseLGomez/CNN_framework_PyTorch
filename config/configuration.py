import imp
import os
import shutil
import numpy as np

class Configuration():
    def __init__(self, config_path, exp_name, exp_folder):
        self.config_path = config_path
        self.exp_name = exp_name
        self.exp_folder = os.path.join(exp_folder, exp_name)
        if not os.path.exists(self.exp_folder):
            os.makedirs(self.exp_folder)

    def Load(self):
        cf = imp.load_source('config', self.config_path)
        cf.config_path = self.config_path
        cf.exp_name = self.exp_name
        cf.exp_folder = self.exp_folder
        cf.tensorboard_path = os.path.join(self.exp_folder,'tensorboard/')
        cf.log_file = os.path.join(cf.exp_folder, "logfile.log")
        cf.log_file_stats = os.path.join(cf.exp_folder, "logfile_stats.log")
        cf.log_file_debug = os.path.join(cf.exp_folder, "logfile_debug.log")
        cf.json_file = os.path.join(cf.exp_folder, "stats.json")
        # Copy config file
        shutil.copyfile(cf.config_path, os.path.join(cf.exp_folder, "config.py"))

        if cf.predict_path_output is None:
            cf.predict_path_output = os.path.join(self.exp_folder,'predictions/')
            if not os.path.exists(cf.predict_path_output):
                os.makedirs(cf.predict_path_output)
        cf.original_size = cf.size_image_test

        if cf.input_model_path is None:
            cf.input_model_path = cf.exp_folder + cf.model_name + '.pth'
        if cf.output_model_path is None:
            cf.output_model_path = cf.exp_folder
        else:
            if not os.path.exists(cf.output_model_path):
                os.makedirs(cf.output_model_path)
        if cf.map_labels is not None:
            cf.map_labels = np.asarray(cf.map_labels,dtype=np.uint16)
        if cf.pretrained_model is None:
            cf.pretrained_model = 'None'
        if not cf.pretrained_model.lower() in ('none', 'basic', 'custom'):
            raise ValueError('Unknown pretrained_model definition')
        if cf.pretrained_model == 'basic':
            cf.basic_pretrained_model = True
        else:
            cf.basic_pretrained_model = False
        if cf.basic_models_path is None:
            cf.basic_models_path = './pretrained_model/',
        return cf