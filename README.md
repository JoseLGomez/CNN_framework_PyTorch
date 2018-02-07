# CNN framework for PyTorch

The following repository contains a functional framework to perform different deep learning tasks like Classification, Semantic segmentation and Domain adaptation.

## Actually in development
- Classification (functional)
    - Model: VGG16
- Semantic segmentation (functional)
    - Model: FCN8, DenseNetFCN (Tiramisú)
- Domain Adaptation (not implemented yet)

### Functionalities implemented:
- Train, validation and test in one run using a proper configuration.
- Early stopping with model saving control via different metrics
- TensorBoardX support (with confusion matrix display)
- Configuration file to run experiments easily [ConfigFile](https://github.com/gvillalonga89/CNN_framework_PyTorch_private/tree/master/config/README.md)
- Metrics: mean accuracy, mean IoU
- Dataloader on GPU
- Training scheduler
- Train Progress bar
- Auto downloadable pretrained models from ImageNet

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
 - wget
 - TensorboardX (https://github.com/lanpa/tensorboard-pytorch)
  
### Actual limitations
- Datasets are defined in txt files, one for images and gt files for each instance of training validation and test. The txt file must contain the path to the image and have the same line order per image respect to the gt file. This files path must be defined in the ConfigFile.py
- Multi-GPU is not supported yet.
