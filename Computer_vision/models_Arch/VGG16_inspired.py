# import numpy as np 
# import os
import time
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import models, layers
from keras.layers import Conv2D, MaxPooling2D, Flatten , Dense, Activation,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns



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

# define function to display the results of the grid search
def display_cv_results(search_results):
    print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))
    means = search_results.cv_results_['mean_test_score']
    stds = search_results.cv_results_['std_test_score']
    params = search_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))






if __name__ == '__main__':


    # start= time.Time()
    # print(start)

    Cifar10=keras.datasets.cifar10 # Loading the dataset

    (xtrain,ytrain),(xtest,ytest)= Cifar10.load_data()
    class_names =['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    print(class_names)

    # Pixel value of the image falls between 0 to 255.

    xtrain = xtrain/255 # So, we are scale the value between 0 to 1 before by deviding each value by 255
    print(xtrain.shape)

    xtest = xtest/255 # So, we are scale the value between 0 to 1 before by deviding each value by 255
    print(xtest.shape)

    ytrain=to_categorical(ytrain, num_classes=10)
    ytest=to_categorical(ytest, num_classes=10)



    input_shape=(32,32,3)
    # create model
    model = KerasClassifier(build_fn=mlp_model, verbose=1)
    # define parameters and values for grid search
    param_grid = {
        'batch_size': [16, 32],
        'epochs': [1],
        'dropout_rate': [0.0, 0.50],
    }
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=n_cv)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(xtrain, ytrain)  # fit the full dataset as we are using cross validation

    # print out results
    print('time for grid search = {:.0f} sec'.format(time()-start))
    # display_cv_results(grid_result)

    # reload best model
    mlp = grid_result.best_estimator_

    # retrain best model on the full training set
    history = mlp.fit(
        xtrain,
        ytrain,
        validation_data = (xtest, ytest),
        epochs = 1
    )
    
    # get prediction on validation dataset
    y_pred = mlp.predict(xtest)
    print('Accuracy on validation data = {:.4f}'.format(accuracy_score(ytest, y_pred)))

    # plot accuracy on training and validation data
    df_history = pd.DataFrame(history.history)
    swarm_plot = sns.lineplot(data=df_history[['accuracy','val_accuracy']], palette="tab10", linewidth=2.5);
    fig = swarm_plot.get_figure()
    fig.savefig("out.png") 












