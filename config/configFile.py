# Model
model_type                  = 'DenseNetFCN'
	### DenseNetFCN options ####
model_blocks                = 5               # Number of block densenetFCN_Custom only
model_layers                = 4               # Number of layers per block densenetFCN_Custom only
model_growth                = 12              # Growth rate per block (k) densenetFCN_Custom only
model_upsampling            = 'deconv'        # upsampling types available: 'upsampling' , 'subpixel', 'deconv'
model_dropout               = 0.0             # Dropout rate densenetFCN_Custom only
model_compression           = 0.0             # Compression rate for DenseNet densenetFCN_Custom only
	### load/store options
pretrained_model			= True			  # True to use a pretrained model or restore experiment
load_weight_only			= True 			  # Recomended true, loads only weights and parameters
save_weight_only			= True 			  # Recomended true, stores only weights and parameters
model_name                  = 'densenetFCN_Custom'
model_path                  = '/home/jlgomez/Experiments/DenseNetFCN/' # None uses experiment path by default if pretrained_model is True


# General parameters

train_samples               = 50 #-1 uses all the data available inside the dataset files
valid_samples				= 10 #-1 uses all the data available inside the dataset files
test_samples				= 10 #-1 uses all the data available inside the dataset files
train_batch_size            = 4
valid_batch_size            = 1
test_batch_size             = 1
train                       = True
validation                  = True
test                        = True # Calculate metrics on test giving the gt
predict_test				= True	# True when you want to generate predictions from test, doesn't need gt
predict_path_output			= None # None uses the default output in the experiment folder /predictions

# Image properties
size_image_train			= (1280, 960)#(960, 1280)
size_image_valid			= (1280, 960)#(960, 1280)
size_image_test				= (1280, 960)#(960, 1280)
resize_image_train          = (640, 480)#(480, 640)
resize_image_valid          = (640, 480)#(480, 640)
resize_image_test           = (640, 480)#(480, 640)
crop_train					= (320, 320)
image_channels              = 3
grayscale                   = False

# Dataset properties
train_images_txt			= '/home/jlgomez/Datasets/DatasetX/images.txt'
train_gt_txt				= '/home/jlgomez/Datasets/DatasetX/gt.txt'
valid_images_txt			= '/home/jlgomez/Datasets/DatasetX/images.txt'
valid_gt_txt				= '/home/jlgomez/Datasets/DatasetX/gt.txt'
test_images_txt				= '/home/jlgomez/Datasets/DatasetX/images.txt'
test_gt_txt				= '/home/jlgomez/Datasets/DatasetX/gt.txt'

labels						= ['person', 'car', 'truck', 'drivable', 'nondrivable', 'blocker', 
															'info', 'sky', 'buildings', 'nature'] 
num_classes                 = 10
shuffle                     = True
void_class                  = 10

#Training
epochs                      = 2
initial_epoch				= 1 #Defines the starting epoch number 
valid_samples_epoch			= 5
is_training                 = True
optimizer                   = 'Adam'
momentum1					= 0.9
momentum2					= 0.99
learning_rate               = 0.0001
weight_decay				= 0.01
save_condition				= 'valid_mIoU'		  # ['always','(x)_loss','(x)_mAcc','(x)_mIoU'] x = valid or train_loss
early_stopping 				= True
stop_condition				= 'valid_mIoU'		  # [(x)_loss','(x)_mAcc','(x)_mIoU'] x = valid or train_loss
patience					= 5

# Image preprocess
rescale                     = 1/255.
mean                        = [0.37296272, 0.37296272, 0.37296272]
std                         = [0.21090189, 0.21090189, 0.21090189]
