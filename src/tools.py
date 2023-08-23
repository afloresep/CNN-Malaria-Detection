# System
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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout


import sys
import os
from sklearn.model_selection import train_test_split
from shutil import move
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
import argparse

# Time
import time
import datetime

# Numerical Data
import random
import numpy as np 
import pandas as pd

# Tools
import shutil
from glob import glob


# NLP
import re

# Preprocessing
from sklearn import preprocessing
from sklearn.utils import class_weight as cw
from sklearn.utils import shuffle

# Model Selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Machine Learning Models
from sklearn import svm
from sklearn.svm import LinearSVC, SVC

# Evaluation Metrics
from sklearn import metrics 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score


# Deep Learning - Keras -  Preprocessing
from keras.preprocessing.image import ImageDataGenerator

# Deep Learning - Keras - Model
import keras
from keras import models
from keras.models import Model
from keras.models import Sequential

# Deep Learning - Keras - Layers
from keras.layers import Convolution1D, concatenate, SpatialDropout1D, GlobalMaxPool1D, GlobalAvgPool1D, Embedding, \
    Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, \
    Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D
from keras.layers.pooling import _GlobalPooling1D

# Deep Learning - Keras - Pretrained Models
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetMobile, NASNetLarge

from keras.applications.nasnet import preprocess_input

# Deep Learning - Keras - Model Parameters and Evaluation Metrics
from keras import optimizers
from keras.optimizers import Adam, SGD , RMSprop
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy

# Deep Learning - Keras - Visualisation
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
# from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

# Deep Learning - TensorFlow
import tensorflow as tf

# Graph/ Visualization
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.image as mpimg
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix

# Image
import cv2
from PIL import Image
from IPython.display import display


def dir_nohidden(directory):
    ''''
    list directories not hidden (to avoid .DS_store and such when listing dir)
    '''
    dir = []
    for f in os.listdir(directory):
         if not f.startswith('.'):
            dir.append(f)
    return dir



def split_folder_to_train_test_valid(data_directory):
    """
    Splits data from the original directory into train, test, and validation directories for each class.
    
    Args:
        data_directory (str): Path to the original data directory containing subdirectories for each class.
    """
    train_dir = os.path.join(data_directory, 'train')
    test_dir = os.path.join(data_directory, 'test')
    validation_dir = os.path.join(data_directory, 'validation')

    # Get a list of class subdirectories in the original data directory
    class_subdirectories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]

    for class_subdir in class_subdirectories: # for 'Class' in ['Class1', 'Class2']
        class_path = os.path.join(data_directory, class_subdir) # ../data/cell_images/Parasitized and ../data/cell_images/Uninfected
        class_images = [img for img in os.listdir(class_path)] # ['C13NThinF_IMG_20150614_131318_cell_179.png'... all images for each class

        train_images, test_validation_images = train_test_split(class_images, test_size=0.3, random_state=42)
        test_images, validation_images = train_test_split(test_validation_images, test_size=0.5, random_state=42)

        print(f'Total images for class {class_subdir}: ', len(class_images)) # 13780 (all images for class_subdir) or 100%
        print(f'Total train images: {len(train_images)} or {(len(train_images)/len(class_images))*100}% ') # 9646 (70% of all images from class_images)
        print(f'Total test images: {len(test_images)} or {len(test_images)/len(class_images)*100}% ') # 2067 class 1 15%
        print(f'Total validation images: {len(validation_images)} or {len(validation_images)/len(class_images)*100}% ')
        print('\n')


        # Create Class 1 and Class 2 subdirectories inside new folders
        train_class_dir = os.path.join(train_dir, class_subdir)
        os.makedirs(train_class_dir, exist_ok=True)

        test_class_dir = os.path.join(test_dir, class_subdir)
        os.makedirs(test_class_dir, exist_ok=True)

        validation_class_dir = os.path.join(validation_dir, class_subdir)
        os.makedirs(validation_class_dir, exist_ok=True)

        # Move images from each class folder to new subdirectories
        for img in train_images:
            src_path =os.path.join(class_path, img)
            dst_path =os.path.join(train_class_dir,img)
            shutil.copy(src_path, dst_path)


        for img in test_images:
            src_path =os.path.join(class_path, img)
            dst_path =os.path.join(test_class_dir,img)
            shutil.copy(src_path, dst_path)


        for img in validation_images:
            src_path =os.path.join(class_path, img)
            dst_path =os.path.join(validation_class_dir,img)
            shutil.copy(src_path, dst_path)

    folder_dir = []
    for x in ['train', 'test', 'validation']: folder_dir.append((os.path.join('../data/cell_images/', x)))

    for i in folder_dir:
        for name in os.listdir(i):
            print(i, name ,len(os.listdir(os.path.join(i,name))))
        # Print new sizes for each new folder
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Split data from original directory into train, test, and validation directories.")
        parser.add_argument("data_directory", type=str, help="Path to the original data directory")
        args = parser.parse_args()
        
        split_folder_to_train_test_valid(args.data_directory)
        

# Visualization

def plot_sample_images(directory, num_samples=5):
            classes = sorted(os.listdir(directory)[1:])
            fig, axes = plt.subplots(len(classes), num_samples, figsize=(15, 10))


            for i, cls in enumerate(classes):
                class_dir = os.path.join(directory, cls)
                class_images = os.listdir(class_dir)[:num_samples]
                for j, image_name in enumerate(class_images):
                    image_path = os.path.join(class_dir, image_name)
                    img = load_img(image_path, target_size=(224, 224))
                    axes[i, j].imshow(img)
                    axes[i, j].axis('off')
                    if j == 0:
                        axes[i, j].set_title(cls)
            plt.show()


def plot_image(file, directory=None, sub=False, aspect=None):
    path = directory + file
    
    img = plt.imread(path)
    
    plt.imshow(img, aspect=aspect)
#     plt.title(file)
    plt.xticks([])
    plt.yticks([])
    
    if sub:
        plt.show()

def plot_img_dir(directory, count):
     selected_files = random.sample(os.listdir(directory), count)

     ncols = 5
     nrows = count//ncols if count%ncols==0 else count//ncols+1

     figsize=(20, ncols*nrows)

     ticksize = 14
     titlesize = ticksize + 8
     labelsize= ticksize + 5

     params = {'figure.figsize' : figsize,
              'axes.labelsize' : labelsize,
              'axes.titlesize' : titlesize,
              'xtick.labelsize': ticksize,
              'ytick.labelsize': ticksize}
     
     plt.rcParams.update(params)

    
     for file in selected_files:        
        plt.subplot(nrows, ncols, i+1)
        path = directory + file
        plot_image(file, directory, aspect=None)

        i=i+1
    
     plt.tight_layout()
     plt.show()


def plot_img_dir_main(directory, count):
    labels = os.listdir(directory)
    for label in labels:
        print(label)
        plot_img_dir(directory=directory+"/"+label+"/", count=count)