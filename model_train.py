"""
In this script, we use KFold Cross Validation to train the model.
We save all of the K best models.
"""

from __future__ import print_function
import numpy as np
import random
import pandas as pd
from scipy import stats
import sys, os
import logging
import tensorflow as tf
from keras import layers
from keras import regularizers
from keras.models import Model
from keras.models import Sequential
from keras.layers import *
from keras.layers import GlobalAveragePooling1D
from keras.regularizers import l1,l2, L1L2
from sklearn.metrics.pairwise import cosine_similarity
import keras
import keras.utils.np_utils as kutils
from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping,Callback,ModelCheckpoint,ReduceLROnPlateau
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import datasets, linear_model
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

random.seed(123)
np.random.seed(123)

def model_creator():
    """
    Define the model, compile it and return the compiled model.
    """
    model = Sequential()
    model.add(Conv1D(16, 5, input_shape=(6, 21731), padding='same', activation='relu', 
                       kernel_initializer='LecunNormal', kernel_regularizer='L1L2'))
    model.add(Conv1D(32, 5, padding='same', activation='relu', 
                       kernel_initializer='LecunNormal', kernel_regularizer='L1L2'))
    model.add(Conv1D(64, 3, padding='same',activation='relu', 
                       kernel_initializer='LecunNormal', kernel_regularizer='L1L2'))
    model.add(Conv1D(128, 3, padding='same',activation='relu', 
                       kernel_initializer='LecunNormal', kernel_regularizer='L1L2'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.6))
    model.add(Activation('linear'))
    model.add(Dense(1))
    
    opt = Adam(learning_rate=0.0003)
    model.compile(loss='mae', optimizer=opt)

    return model

if __name__ == '__main__':
    exper_name = 'sv'
    os.makedirs(os.path.join('results/', exper_name), exist_ok=True)
    os.makedirs(os.path.join('figures/', exper_name), exist_ok=True)
    os.makedirs(os.path.join('model_weights/', exper_name), exist_ok=True)
    
    X = np.load(os.path.join('data/', exper_name, 'X_train_sv.npy')).transpose(0, 2, 1)
    Y = np.load(os.path.join('data/', exper_name, 'y_train_sv.npy'))
    X_test = np.load(os.path.join('data/', exper_name, 'X_test_sv.npy')).transpose(0, 2, 1)
    y_test = np.load(os.path.join('data/', exper_name, 'y_test_sv.npy'))

    print('X Train shape is: ', X.shape)
    print('Y Train shape is: ', Y.shape)
    print('X Test shape is: ', X_test.shape)
    print('Y Test shape is: ', y_test.shape)
    print('-' * 50)
    print()
    
    kf = KFold(n_splits=3)
    fold_index = 1
    for train_index, val_index in kf.split(X, Y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = Y[train_index], Y[val_index]

        model = model_creator()
        checkpoint = ModelCheckpoint(os.path.join('model_weights', exper_name, f'model-{fold_index:2d}.h5'), 
                                     verbose=1, 
                                     save_best_only=True,
                                     monitor='val_loss', 
                                     mode='auto')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

        history = model.fit(X_train, y_train, 
                            validation_data=(X_val, y_val), 
                            epochs=150, 
                            batch_size=64, 
                            callbacks=[checkpoint, es])
        with open(os.path.join('results/', exper_name, 'results.txt'), 'a') as writer: 
            writer.write(f'Evaluation Results for fold {fold_index:2d}')
            writer.write('\n')
            writer.write(' '.join([str(item) for item in [model.evaluate(x=X_test, y=y_test)]]))
            writer.write('\n')
            writer.write('*' * 30)
            writer.write('\n')

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.savefig(os.path.join('figures/', exper_name, f'losses_{fold_index:2d}.png'))
        
        # Clear the session. 
        tf.keras.backend.clear_session()
        
        fold_index += 1
        print('*' * 50)