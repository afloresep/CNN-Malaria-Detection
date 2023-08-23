import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

import pandas as pd 
import numpy as np
import os
import glob
import shutil
from tools import plot_sample_images, split_folder_to_train_test_valid, dir_nohidden
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight as cw

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

def get_data(batch_size, target_size, class_mode, train_dir, testing_dir, color_mode):
    print('Preprocessing and Generating Data Batches...\n')


    train_batch_size = batch_size
    test_batch_size = batch_size

    train_shuffle = True
    val_shuffle = True
    test_shuffle = False 

    training_data_generator = ImageDataGenerator(
        rescale=1.0/255,
        hodizontal_flip = True, 
        vertical_flip = True, 
        rotation_angle=45, 
        width_shift_range=0.05, 
        height_shift_range=0.05, 
        validation_split = 0.25
    )

    '''
Modify training_data_generator so that it will also automatically perform pixel normalization. 
Rescale keyword argument equal to 1.0/255.
Set zoom_range to be 0.2. This will randomly increase or decrease the size of the image by up to 20%.
Set rotation_range to be 15. This randomly rotates the image between [-15,15] degrees.
Set width_shift_range equal to 0.05. This shift the image along its width by up to +/- 5%
Set height_shift_range to be 0.05. This shifts the image along its height by up to +/- 5%
    '''

    training_iterator = training_data_generator.flow_from_directory(
        train_dir, 
        class_mode=class_mode, # 'categorical' / 
        color_mode=color_mode, # 'rgb' / 'greyscale'
        target_size=target_size, 
        batch_size=batch_size, 
        shuffle=train_shuffle, 
        seed=42, 
        subset='training'
    ) 

    validation_data_generator = training_data_generator.flow_from_directory(
        train_dir, 
        target_size=target_size, 
        class_mode=class_mode, 
        color_mode=color_mode, 
        batch_size=256, # Not sure what to use?
        shuffle= val_shuffle, 
        seed=42, 
        subset='validation'
    )

    test_data_generator = ImageDataGenerator(
        rescale=1.0/255
    )

    test_generator = None
        
    
    class_weights = get_weight(training_data_generator.classes)
    
    steps_per_epoch = len(training_data_generator)
    validation_steps = len(validation_data_generator)

    print("\n Preprocessing and Data Batch Generation Completed.\n")

    return training_data_generator, validation_data_generator, test_generator, class_weights, steps_per_epoch, validation_steps

# Calculate Class Weights
def get_weight(y):
    class_weight_current =  cw.compute_class_weight('balanced', np.unique(y), y)
    return class_weight_current