## ConfigFile parameters

#### Problem type
- problem_type: define the type of problem to solve. Actual options values: ['segmentation','classification']
#### Model
- model_type: define the network to use for the problem. Actual options values: ['DenseNetFCN', 'FCN8', 'VGG16'], they can we write in lowercase or uppercase
##### DenseNetFCN options ####
This options are defined only to control the DenseNetFCN arquitecture definition:
- model_blocks                : Number of blocks
- model_layers                : Number of layers per block
- model_growth                : Growth rate per block (k)
- model_dropout               : Dropout rate (not really necessary due to batchnormalization layers)
- model_compression           : Compression rate
For more information about this network check the paper [https://arxiv.org/pdf/1611.09326.pdf]
##### load options
- pretrained_model            : Defines which type of pretrained model we want. Option are the follwoing: 'None': model from scratch without pretrained model, 'basic': pretrained from imagenet, 'custom': personal model that it will need specified in _input_model_path_
- input_model_path            : Path and pretrained file to load, option _None_ uses experiment path and model name by default
- load_weight_only            : Loads only weights and parameters. We recommend have this option always true to use only the weights, but with False an entire model can be loaded without a previous definition
- basic_models_path           : Path to store or find the basic models (ImageNet weights), if this path don't exist will be created and pretrained models downloaded there. By default we recommend './pretrained_models/'
##### Save options
- save_weight_only            : If True stores only weights and parameters of the network, if False stores all the network arquitecture plus weights
- model_name                  : Defines the name that will have the model stored on disk, in the experiment folder or especified in output_model_path
- output_model_path           : Path to store the model named in model_name. None uses the default experiment path especified on the command execution

#### Loss type
- loss_type                   : Defines the loss type to use in the problem. Actual options available: ['cross_entropy_segmentation','focal_segmentation']

#### General parameters

- train_samples               : Number of train images to use per epoch, -1 uses all the dataset images available
- valid_samples               : Number of validation images [valid_samples_epoch: on the training defines the number of validation images to use in a epoch], -1 uses all the dataset images available
- test_samples                : Number of test images to use , -1 uses all the dataset images available
- train_batch_size            : batch size for train
- valid_batch_size            : batch size for validation
- test_batch_size             : batch size for test
- train                       : Enables/disable the training step
- validation                  : Enables/disable the validation after finish the training step
- test                        : Enables/disable the test after finish the training step
- predict_test                : Enables/disable the prediction step, when you want to generate predictions from test, does not need gt
- predict_path_output         : Defines the output folder for the predictions. None uses the default output in the experiment folder /predictions

#### Image properties
The following parameters defines the input images and resize dimensions, can be put to None, 
- size_image_train            
- size_image_valid             
- size_image_test              
- resize_image_train          
- resize_image_valid         
- resize_image_test           
- crop_train                  : defines an image crop size for the training step 
- grayscale                   : Enable/Disable the conversion to grauscales of the input images

#### Dataset properties
The following parameters are used to specify the txt file and path where the images paths are contained
- train_images_txt            
- train_gt_txt                
- valid_images_txt            
- valid_gt_txt                
- test_images_txt             
- test_gt_txt                 

- labels                      : list to define the classes names used in the problem. This classes names will be displayed in the confusion matrix on TensorBoard
- map_labels                  : List of the indexes to map the previous classes defined with the ground Truth. Useful to remap ground truth not preprocessed like cityscapes
- num_classes                 : Define the number of classes of the problem
- shuffle                     : Enable/Disable random images selection on train
- void_class                  : Specifies the index value of the class void. Use map_labels to assign different classes to void

#### Training
- epochs                      : Maximum number of epochs of train, use 0 to save directly a model, useful to make conversions.
- initial_epoch               : Define the epoch number to start, normally 1, but can be set freely to display purposes
- valid_samples_epoch         : Number of validation images used to validate an epoch during training

##### Optimizer
- optimizer                   : Define the optimizer to use in the training step. Options available ['SGD','Adam','RMSProp']
- momentum1                   : principal momentum parameter for the different optimizers, beta1 for Adam
- momentum2                   : beta2 parameter for Adam
- learning_rate               
- learning_rate_bias          : learning rate for the bias
- weight_decay                
##### Scheduler
- scheduler                   : Defines the scheduler approach to use. The option available are ['ReduceLROnPlateau','Step','MultiStep','Exponential', None]
- decay                       : Learning rate decay to apply (lr*decay)
- sched_patience              : ReduceLROnPlateau option: epoch patience without loss change until a lr decrement
- step_size                   : Step option: epoch counter to decrease lr
- milestone                   : MultiStep option: define different milestones (epochs) to decrease lr
##### Save criteria
- save_condition              : Defines the metric to take into account to store the best model obtained in validation per epoch. Options available ['always','(x)_loss','(x)_mAcc','(x)_mIoU'] x = valid or train_loss
##### Early Stopping
- early_stopping              : Enable/Disable early stopping control to finish the experiment when not improvement is obtained after X epochs
- stop_condition              : Defines the metric to take into account to perform the early stopping. Options available [(x)_loss','(x)_mAcc','(x)_mIoU'] x = valid or train_loss
- patience                    : define the number of epochs to wait before stop

#### Image preprocess
- rescale                     : Rescaling factor of the image values
- mean                        : define the dataset mean
- std                         : define the dataset standard deviation 

#### Data augmentation
- hflips                      : Enable/Disable random horitzontals flips in the images on training