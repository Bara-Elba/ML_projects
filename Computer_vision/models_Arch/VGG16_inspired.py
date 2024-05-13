import numpy as np 
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from keras.layers import Conv2D, MaxPooling2D, Flatten , Dense, Activation,Dropout



# model architecture for KerasClassifier
# the puprpos for this function is to use grid search at the end to choose the best params. 
def mlp_model(input_shape=(512,512,3), output_shape=10 ,dropout_rate=0, dropout_cnn=False, activation='relu'):

    # Inspired from the VGG16 model 
    model = keras.Sequential()
    model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation=activation))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation=activation))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=activation))
    # model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=activation))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    if dropout_rate !=0 and dropout_cnn:
        model.add(Dropout(dropout_rate))

    # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=activation))
    # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=activation))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=activation))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    if dropout_rate !=0 and dropout_cnn:
        model.add(Dropout(dropout_rate))

    # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=activation))
    # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=activation))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=activation))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    if dropout_rate !=0 and dropout_cnn:
        model.add(Dropout(dropout_rate))
    # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=activation))
    # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=activation))
    # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=activation))
    # model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
    model.add(Flatten())
    model.add(Dense(units=1024,activation=activation))
    if dropout_rate !=0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(units=1024,activation=activation))
    if dropout_rate !=0 and dropout_cnn:
        model.add(Dropout(dropout_rate))
    model.add(Dense(units=output_shape, activation="softmax"))




    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        )
    return model


input_shape=(32,32,3)
