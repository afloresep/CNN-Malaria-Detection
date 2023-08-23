# First-model: binary-classification.py

Compiling model...
Model: "sequential_5"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_10 (Conv2D)          (None, 62, 62, 32)        896       
                                                                 
 max_pooling2d_10 (MaxPooli  (None, 31, 31, 32)        0         
 ng2D)                                                           
                                                                 
 batch_normalization_20 (Ba  (None, 31, 31, 32)        128       
 tchNormalization)                                               
                                                                 
 dropout_20 (Dropout)        (None, 31, 31, 32)        0         
                                                                 
 conv2d_11 (Conv2D)          (None, 29, 29, 32)        9248      
                                                                 
 max_pooling2d_11 (MaxPooli  (None, 14, 14, 32)        0         
 ng2D)                                                           
                                                                 
 batch_normalization_21 (Ba  (None, 14, 14, 32)        128       
 tchNormalization)                                               
                                                                 
 dropout_21 (Dropout)        (None, 14, 14, 32)        0         
                                                                 
 flatten_5 (Flatten)         (None, 6272)              0         
                                                                 
 dense_15 (Dense)            (None, 512)               3211776   
                                                                 
 batch_normalization_22 (Ba  (None, 512)               2048      
 tchNormalization)                                               
                                                                 
 dropout_22 (Dropout)        (None, 512)               0         
                                                                 
 dense_16 (Dense)            (None, 256)               131328    
                                                                 
 batch_normalization_23 (Ba  (None, 256)               1024      
 tchNormalization)                                               
                                                                 
 dropout_23 (Dropout)        (None, 256)               0         
                                                                 
 dense_17 (Dense)            (None, 2)                 514       
                                                                 
=================================================================
Total params: 3357090 (12.81 MB)
Trainable params: 3355426 (12.80 MB)
Non-trainable params: 1664 (6.50 KB)
_________________________________________________________________


```602/602 [==============================] - 94s 156ms/step - loss: 0.1273 - categorical_accuracy: 0.9581 - auc_5: 0.9888 - val_loss: 73255.8984 - val_categorical_accuracy: 0.5000 - val_auc_5: 0.5000 ```

It looks like the model is performing well on the training data based on the loss, accuracy, and AUC values. However, there appears to be a serious problem on the validation data, as the loss and metrics there suggest that the model isn't learning correctly or there might be an issue with the data...

I used splitted folders for this. I just realized keras does this for you...




# Second-model: keras-model-classification.py
Let's try different approach. This time using keras project reference. Not splitting the data folder...