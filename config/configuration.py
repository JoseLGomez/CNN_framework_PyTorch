import imp
import os
import shutil

class Configuration():
    def __init__(self, config_path, exp_name, exp_folder):
        self.config_path = config_path
        self.exp_name = exp_name
        self.exp_folder = exp_folder + exp_name + '/'
        if not os.path.exists(self.exp_folder):
    		os.makedirs(self.exp_folder)

    def Load(self):
        cf = imp.load_source('config', self.config_path)
        cf.config_path = self.config_path
        cf.exp_name = self.exp_name
        cf.exp_folder = self.exp_folder
        cf.log_file = os.path.join(cf.exp_folder, "logfile.log")
        # Copy config file
        shutil.copyfile(cf.config_path, os.path.join(cf.exp_folder, "config.py"))

        if cf.predict_path_output is None:
            cf.predict_path_output = self.exp_folder + 'predictions/'
            if not os.path.exists(cf.predict_path_output):
                os.makedirs(cf.predict_path_output)
        cf.original_size = cf.size_image_test

        if cf.model_path is None:
            cf.model_path = cf.exp_folder      
        return cf