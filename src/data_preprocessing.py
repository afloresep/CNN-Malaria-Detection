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


# Visualazing Images and Corresponding labels from the dataset
DIRECTORY = "../data/cell_images"
train_dir = os.path.join(DIRECTORY, 'train')
test_dir = os.path.join(DIRECTORY, 'test')
valid_dir = os.path.join(DIRECTORY, 'validation')
#plot_sample_images(DIRECTORY)

# Generates dataset from images files in a directory to make sure all files are fetched
'''
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
'''

# Split the data into train, validation and test folders from original folder with 2 classes each

def dir_nohidden(path):
    dir = []
    for f in os.listdir(path):
         if not f.startswith('.'):
            dir.append(f)
    return dir


if len(dir_nohidden(DIRECTORY)) <= 2: # Make sure we're not splitting data more than once. 
    split_folder_to_train_test_valid(DIRECTORY)

test_data_generator = ImageDataGenerator()

validation_data_generator = ImageDataGenerator()

# Data Augmentation using Keras library. 
training_data_generator = ImageDataGenerator(
    rescale=1.0/255, 
    zoom_range=0.2, 
    rotation_range=15, 
    width_shift_range=0.05, 
    height_shift_range=0.05)

'''
Modify training_data_generator so that it will also automatically perform pixel normalization. 
Rescale keyword argument equal to 1.0/255.

Set zoom_range to be 0.2. This will randomly increase or decrease the size of the image by up to 20%.

Set rotation_range to be 15. This randomly rotates the image between [-15,15] degrees.

Set width_shift_range equal to 0.05. This shift the image along its width by up to +/- 5%

Set height_shift_range to be 0.05. This shifts the image along its height by up to +/- 5%
'''

# print(training_data_generator.__dict__)


'''
Categorical loss function will be used, which will expect labels to be in a a OHE format. 
[0,1] for Infected and [1,0] for not infected
'''

# Parameters
CLASS_MODE="categorical"
COLOR_MODE="rgb"
TARGET_SIZE = (64,64)
BATCH_SIZE = 64


# Creates a DirectoryIterator object using the above parameters: 
print('TRAINING FOLDER')
training_iterator = training_data_generator.flow_from_directory(
    train_dir,
    class_mode=CLASS_MODE,
    color_mode=COLOR_MODE, 
    target_size=TARGET_SIZE, 
    batch_size=BATCH_SIZE
)# subset='training'

sample_batch_input, sample_batch_labels = training_iterator.next()

# next() is used to fetch the next batch of data from the iterator. 
# Returns a tuple containing the input batch (images) and corresponding label batch. 

print("\nLoading validation data...")
'''
data/cell_images is the folder containing cell images organized in 2 folders
Parasitized: RBC infected with Plasmodium
Uninfected: RBC not infected
flow_from_directory will automatically label the images according to their subfolder
'''

print('VALIDATION FOLDER: ')
validation_iterator = validation_data_generator.flow_from_directory(
    valid_dir,
    class_mode=CLASS_MODE,
    color_mode=COLOR_MODE, 
    target_size=TARGET_SIZE, 
    batch_size=BATCH_SIZE
) # subset='validation

print('TEST FOLDER: ')
test_iterator = test_data_generator.flow_from_directory(
    test_dir,
    class_mode=CLASS_MODE, 
    color_mode=COLOR_MODE,
    target_size=TARGET_SIZE, 
    batch_size=BATCH_SIZE
) # subset='test


class_mapping = training_iterator.class_indices
print(class_mapping)
# {'Parasitized': 0, 'Uninfected': 1}

print(sample_batch_input.shape,sample_batch_labels.shape)
# (32, 256, 256, 3) (32, 2)

# print(training_data_generator.__dict__)


print("\nBuilding model...")

def design_model(training_data):
    '''
    Model as defined by Sumit Kumar et al. 
    Ref: https://arxiv.org/pdf/2303.03397.pdf
    '''
    input_shape = (64, 64, 3)
    
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
    model.summary()
    return model
    
model = design_model(training_iterator)
# model with input shape (64, 64, 3)


# model with input shape (256, 256, 3)
def design_model_2(training_data):
    '''
    Model as defined by Sumit Kumar et al. but changing input shape to (256,256,3)
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
    model.summary()
    return model

# model_2 = design_model_2(training_iterator)


# early stopping implementation
es = EarlyStopping(monitor='val_auc', mode='min', verbose=1, patience=20)

print("\nTraining model...")
# fit the model with 10 ephochs and early stopping


history= model.fit(
    training_iterator, 
    steps_per_epoch=training_iterator.samples/BATCH_SIZE, epochs=50, 
    validation_data=validation_iterator,
    validation_steps=validation_iterator.samples/BATCH_SIZE, 
    callbacks=[es]
)

# Evaluate the model on the test data
test_metrics = model.evaluate(test_iterator, steps=test_iterator.samples / BATCH_SIZE)

# Print the test metrics
print("Test Loss:", test_metrics[0])
print("Test Accuracy:", test_metrics[1])


# plotting categorical and validation accuracy over epochs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')
plt.show()


test_steps_per_epoch = np.math.ceil(validation_iterator.samples / validation_iterator.batch_size)
predictions = model.predict(validation_iterator, steps=test_steps_per_epoch)
test_steps_per_epoch = np.math.ceil(validation_iterator.samples / validation_iterator.batch_size)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_iterator.classes
class_labels = list(validation_iterator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)   


cm=confusion_matrix(true_classes,predicted_classes)
print(cm)

