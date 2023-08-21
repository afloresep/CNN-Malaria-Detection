import pandas as pd 
import numpy as np
import os
import glob
from tools import plot_sample_images
from tools import split_folder_to_train_test_valid
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
tf.get_logger().setLevel('INFO') # Avoid info messages from TF 


# Visualazing Images and Corresponding labels from the dataset
DIRECTORY = "/data/cell_images"

# plot_sample_images(DIRECTORY)

# Image data loading
# # Generates dataset from images files in a directory to make sure all files are fetched

dataset= image_dataset_from_directory(
    DIRECTORY,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)
# Found 27558 files belonging to 2 classes.


# Data Augmentation using Keras library. 

training_data_generator = ImageDataGenerator(
    rescale=1.0/255, 
    zoom_range=0.2, 
    rotation_range=15, 
    width_shift_range=0.05, 
    height_shift_range=0.05)

validation_data_generator = ImageDataGenerator()


'''
Modify training_data_generator so that it will also automatically perform pixel normalization. 
Rescale keyword argument equal to 1.0/255.

Set zoom_range to be 0.2. This will randomly increase or decrease the size of the image by up to 20%.

Set rotation_range to be 15. This randomly rotates the image between [-15,15] degrees.

Set width_shift_range equal to 0.05. This shift the image along its width by up to +/- 5%

Set height_shift_range to be 0.05. This shifts the image along its height by up to +/- 5%
'''

# print(training_data_generator.__dict__)


# Parameters: 
CLASS_MODE="categorical"
'''
Categorical loss function will be used, which will exepct labels to be in a a OHE format. 
[0,1] for Infected and [1,0] for not infected
'''

COLOR_MODE="rgb"

TARGET_SIZE = (256,256)

BATCH_SIZE = 32

# Creates a DirectoryIterator object using the above parameters: 
training_iterator = training_data_generator.flow_from_directory(
    DIRECTORY,
    class_mode=CLASS_MODE,
    color_mode=COLOR_MODE, 
    target_size=TARGET_SIZE, 
    batch_size=BATCH_SIZE
)

training_iterator = validation_data_generator.flow_from_directory(
    DIRECTORY,
    class_mode=CLASS_MODE,
    color_mode=COLOR_MODE, 
    target_size=TARGET_SIZE, 
    batch_size=BATCH_SIZE
)


'''
data/cell_images is the folder containing cell images organized in 2 folders
Parasitized: RBC infected with Plasmodium
Uninfected: RBC not infected
flow_from_directory will automatically label the images according to their subfolder
'''

# next() is used to fetch the next batch of data from the iterator. Returns a tuple containing the input batch (images) and corresponding label batch. 
sample_batch_input, sample_batch_labels = training_iterator.next()
print("\nLoading validation data...")

class_mapping = training_iterator.class_indices

print(sample_batch_input.shape,sample_batch_labels.shape)
# (32, 256, 256, 3) (32, 2)

print(training_data_generator.__dict__)

print(class_mapping)
# {'Parasitized': 0, 'Uninfected': 1}


