# SemanticSegmentation_PyTorch

The following repository contains a functional framework to perform semantic segmentation using Convolutional Neural Networks in PyTorch.

## Actually in development

### Functionalities implemented:
- Train, validation and test in one run using a proper configuration.
- Early stopping with model saving control via different metrics
- Test as a prediction system
- Configuration file to run experiments easily
- Metrics: mean accuracy, mean IoU
- Models: DenseNetFCN (tiramisu)
- Dataloader on GPU
- And more

### How to run it
- Configure the configuration file in config/ConfigFile.py
- Run code using: CUDA_VISIBLE_DEVICES=[gpu_number] python main.py --exp_name [experiment_name] 
  --exp_folder [path_to_save_experiment] --config_file [path_to_config_file]
  You can define default values to this input arguments in main.py
  
 ### Requirements
 - Python 2.7
 - PyTorch 0.3.0
 - Scipy 1.0.0
 - Numpy 1.13.3
 - OpenCV 3.0
  
### Actual limitations
- Datasets are defined in txt files, one for images and gt files for each instance of training validation and test. The txt file must contain the path to the image and have the same line order per image respect to the gt file. This files path must be defined in the ConfigFile.py
- Mult-GPU training is not supported yet.
