import tensorflow as tf
from skimage import io
import sys
sys.path.append('create_models')
import create_models as cm
import os
import numpy as np

## Set the which GPU to run
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  

## trained model folder
model_folder = './trained_models/'

## Load testing images
image_dir = './images'								# image directory: save each input image as a RGB image [HxWx3], where each channel 
													# contain the same phase contrast image; normalize the values into pixel values of range: [0,255], dtype: np.uint8
image_fns = os.listdir(image_dir)
images = [io.imread(image_dir+'/{}'.format(img_fn) for img_fn in image_fns]

## Image preprocessing
print('Preprocessing ...')
backbone = 'efficientnetb3'
preprocess_input = cm.get_preprocessing(backbone) 	## preprocessing function
images = preprocess_input(images); 					# will scale pixels between 0 and 1 and then will normalize each channel with respect to the ImageNet dataset
print('Preprocessing done !')

## Load the trained model
model=tf.keras.models.load_model(model_folder+'/ready_model.h5')

## Label map prediction
pr_masks = model.predict(images, batch_size=1); 	## probability maps [N(num of images) x H x W x C(class)] for 0: live, 1: intermediate, 2: dead, 3: background
pr_maps = np.argmax(pr_masks,axis=-1)   			# predicted label map