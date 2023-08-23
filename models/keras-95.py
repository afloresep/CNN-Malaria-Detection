
# System
import sys
import os
import argparse

import random

# Time
import time
import datetime

import numpy as np
import matplotlib.pyplot as plt
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight as cw
from sklearn.utils.class_weight import compute_class_weight


import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras import models
from keras.models import Model 
from keras.models import Sequential
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten

# Deep Learning - Keras - Pretrained Models
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetMobile, NASNetLarge
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

# from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

def date_time(x):
    if x==1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==2:    
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==3:  
        return 'Date now: %s' % datetime.datetime.now()
    if x==4:  
        return 'Date today: %s' % datetime.date.today()
    

# Input

input_directory = '../data/cell_images'
output_directory = '../output/'

training_dir = input_directory
testing_dir = os.path.join(input_directory, 'test')

if not os.path.exists(output_directory): 
    os.mkdir(output_directory)

figure_directory = '../output/figures'
if not os.path.exists(figure_directory):
    os.mkdir(figure_directory)

file_name_pred_batch = figure_directory+r"/result"
file_name_pred_sample = figure_directory+r"/sample"

'''
file_name_pred_batch = figure_directory+r"/result"
file_name_pred_sample = figure_directory+r"/sample"

?????????????
'''


# Visualization 
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
    labelsize = ticksize + 5


    params = {'figure.figsize' : figsize,
              'axes.labelsize' : labelsize,
              'axes.titlesize' : titlesize,
              'xtick.labelsize': ticksize,
              'ytick.labelsize': ticksize}

    plt.rcParams.update(params)
    
    i=0
    
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
        


def plot_sample_images(directory, num_samples):
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



def plot_performance(history=None, figure_directory=None):
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    ylim_pad = [0.005, 0.005]
#     ylim_pad = [0, 0]


    plt.figure(figsize=(20, 5))

    # Plot training & validation Accuracy values

    y1 = history.history['acc']
    y2 = history.history['val_acc']

    min_y = min(min(y1), min(y2))-ylim_pad[0]
    max_y = max(max(y1), max(y2))+ylim_pad[0]


    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Accuracy\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()


    # Plot training & validation loss values

    y1 = history.history['loss']
    y2 = history.history['val_loss']

    min_y = min(min(y1), min(y2))-ylim_pad[1]
    max_y = max(max(y1), max(y2))+ylim_pad[1]


    plt.subplot(122)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Loss\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()
    if figure_directory:
        plt.savefig(figure_directory+"/history")

    plt.show()

#    plot_img_dir_main(directory=training_dir, count=5)




def dir_nohidden(directory):
    ''''
    list directories not hidden (to avoid .DS_store and such when listing dir)
    '''
    dir = []
    for f in os.listdir(directory):
         if not f.startswith('.'):
            dir.append(f)
    return dir

if len(dir_nohidden(training_dir)) == 2: 
    plot_img_dir_main(directory=training_dir, count=5)

# Preprocess


def get_data(batch_size, target_size, class_mode, train_dir, testing_dir, color_mode):
    print('Preprocessing and Generating Data Batches...\n')


    train_batch_size = batch_size
    test_batch_size = batch_size

    train_shuffle = True
    val_shuffle = True
    test_shuffle = False 

    training_data_generator = ImageDataGenerator(
        rescale=1.0/255,
        horizontal_flip = True, 
        vertical_flip = True, 
        rotation_range=15, 
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

    validation_data_iterator = training_data_generator.flow_from_directory(
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
        
    
    # class_weights = get_weight(training_data_generator.classes)
    class_weights = 'lol'
    
    steps_per_epoch = training_iterator.samples/batch_size
    validation_steps = training_iterator.samples/batch_size

    print("\n Preprocessing and Data Batch Generation Completed.\n")

    return training_data_generator, training_iterator, validation_data_iterator, test_generator, class_weights, steps_per_epoch, validation_steps

# Calculate Class Weights
def get_weight(x):
        class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(x),
                                        y = x
                                        )
        return class_weights


# Model Functions: 

# Pre-trained model
def get_model(model_name, input_shape, num_class):
    inputs = Input(input_shape)
    if model_name == "Xception":
        base_model = Xception(include_top=False, input_shape=input_shape)
    elif model_name == "ResNet50":
        base_model = ResNet50(include_top=False, input_shape=input_shape)
    elif model_name == "InceptionV3":
        # base_model = InceptionV3(include_top=False, input_shape=input_shape)
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, input_shape=input_shape)
    if model_name == "DenseNet201":
        base_model = DenseNet201(include_top=False, input_shape=input_shape)
    if model_name == "NASNetMobile":
        base_model = NASNetMobile(include_top=False, input_shape=input_shape)
    if model_name == "NASNetLarge":
        base_model = NASNetLarge(include_top=False, input_shape=input_shape)


    x = base_model(inputs)
    
    output1 = GlobalMaxPooling2D()(x)
    output2 = GlobalAveragePooling2D()(x)
    output3 = Flatten()(x)
    
    outputs = Concatenate(axis=-1)([output1, output2, output3])
    
    outputs = Dropout(0.5)(outputs)
    outputs = BatchNormalization()(outputs)
    
    if num_class>1:
        outputs = Dense(num_class, activation="softmax")(outputs)
    else:
        outputs = Dense(1, activation="sigmoid")(outputs)
        
    model = Model(inputs, outputs)
    
    model.summary()
    
    
    return model
    

# Keras model
def keras_model(num_class=2, input_shape=(3,150,150)):
    model = Sequential()
    
    model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=input_shape))
    model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Flatten())
    
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Dense(num_class , activation='softmax'))

    print(model.summary())
    
    return model

# Sumit Kumar et al. Model 
def design_model():
    '''
    Model as defined by Sumit Kumar et al. 
    Ref: https://arxiv.org/pdf/2303.03397.pdf
    '''
    input_shape = (256, 256, 3)
    
    model = Sequential()
    
    model.add(tf.keras.Input(shape=input_shape))
              
    # First convolution layer
    model.add(Conv2D(
        32, (3, 3), 
        strides=(1, 1), 
        padding='valid', 
        activation='relu', 
        input_shape=input_shape))
    
    # Max pooling layer
    model.add(MaxPooling2D(
        pool_size=(2, 2), 
        strides=(2,2)))
    
    # Batch normalization
    model.add(BatchNormalization())
    
    # Dropout layer
    model.add(Dropout(0.2))
    
    # Second convolution layer
    model.add(Conv2D(
        32, (3, 3), 
        strides=(1, 1), 
        padding='valid', 
        activation='relu'))
    
    # Max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Batch normalization
    model.add(BatchNormalization())
    
    # Dropout layer
    model.add(Dropout(0.2))
    
    # Flatten layer to prepare for fully connected layers
    model.add(Flatten())

    # Feed-Forward Network
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Output Layer
    model.add(Dense(2,activation="softmax")) # 2 Classes


    # compile model with Adam optimizer
    # loss function is categorical crossentropy
    # metrics are categorical accuracy and AUC
    print("\nCompiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
        loss=tf.keras.losses.CategoricalCrossentropy(), 
        metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()],)
    
    # summarize model
    print(model.summary())

    return model



# Output
main_model_dir = output_directory+r'models/'
main_log_dir = output_directory+r'logs/'

try: 
    os.mkdir(main_model_dir)
except:
    print('Could not create main model directory')

try: 
    os.mkdir(main_log_dir)
except:
    print('Could not create main log directory')


# create a unique directory path for each run of the model.
model_dir = main_model_dir + time.strftime('%Y-%m-%d %H-%M-%S') + "/"
log_dir = main_log_dir + time.strftime('%Y-%m-%d %H-%M-%S')


try:
    os.mkdir(model_dir)
except:
    print("Could not create model directory")
    
try:
    os.mkdir(log_dir)
except:
    print("Could not create log directory")


# variable is constructing a file path for saving model checkpoints. 
# It is composed of the model_dir, an epoch number ({epoch:02d}) with zero-padding, 
# validation accuracy ({val_acc:.2f}) with two decimal places, 
# validation loss ({val_loss:.2f}) with two decimal places, and the extension
    
model_file = model_dir + "{epoch:02d}-val_acc-{val_acc:.2f}-val_loss-{val_loss:.2f}.hdf5"


# Call Back

print('Setting Callbacks')

checkpoint = ModelCheckpoint(
    model_file, 
    monitor='val_acc', 
    dave_best_only=True
)

early_stopping = EarlyStopping(
    monitor ='val_loss',
    patience=2, 
    verbose=1, 
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.6, 
    patience=1, 
    verbose=1
)

callbacks = [reduce_lr, early_stopping, checkpoint]

print("Set Callbacks at ", date_time(1))


# Model

print('Building Model', date_time(1))

input_shape = (96, 96, 3)

# input_shape = (224, 224, 3)

#input_shape = (64, 64, 3)


# input_shape = (256, 256, 3)


num_class = 2

#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context
# model = get_model(model_name="NASNetMobile", input_shape=input_shape, num_class=num_class)

#model = design_model()

model = keras_model(input_shape=input_shape)

print("Loaded Base Model", date_time(1))


loss = 'categorical_crossentropy'
# loss = 'binary_crossentropy'
metrics = ['acc']
# metrics = [auroc]



training_data_generator, training_iterator, validation_data_iterator, test_generator, class_weights, steps_per_epoch, validation_steps = get_data(32, (256, 256), 'categorical', training_dir, None, 'rgb')

#print(training_data_generator, training_iterator, validation_data_iterator, test_generator, class_weights, steps_per_epoch, validation_steps)


print('Training model...\n')
history=model.fit(
    training_iterator, 
    steps_per_epoch=steps_per_epoch, 
    epochs = 10, 
    verbose = 1, 
    callbacks = callbacks, 
    validation_data=validation_data_iterator, 
    validation_steps = validation_steps
)

start_time = time.time()
elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

print("\nElapsed Time: " + elapsed_time)
print("Completed Model Trainning", date_time(1))