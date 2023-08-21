import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Model with input_shape (64, 64, 3)


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
