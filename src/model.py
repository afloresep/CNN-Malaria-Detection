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


def get_model(model_name, input_shape=(96, 96, 3), num_class=2):
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
            
#     for layer in base_model.layers:
#         layer.trainable = False
        
#     for layer in model.layers[:249]:
#         layer.trainable = False
#     for layer in model.layers[249:]:
#         layer.trainable = True
    
#     x = base_model(inputs)
#     x = GlobalAveragePooling2D()(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(x)
#     out = Dense(2, activation="softmax")(x)
#     model = Model(inputs, out)

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

# Custom Convolutional Neural Network 
def get_conv_model(num_class=2, input_shape=(3,150,150)):
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
