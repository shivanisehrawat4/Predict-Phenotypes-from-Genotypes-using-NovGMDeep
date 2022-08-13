"""
In this script, We test the best models saved in the training phase.
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
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import datasets, linear_model
import matplotlib
import matplotlib.pyplot as plt
import sklearn

def model_creator():
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

    return model

if __name__ == '__main__': 
    exper_name = 'sv'
    os.makedirs(os.path.join('figures/', exper_name), exist_ok=True)
    
    X_test = np.load(os.path.join('data', exper_name, 'X_test_sv.npy')).transpose(0,2,1)
    y_test = np.load(os.path.join('data', exper_name, 'y_test_sv.npy'))

    model = model_creator()
    model.load_weights(os.path.join('model_weights/', exper_name, 'model- 1.h5'))
    pred = model.predict(X_test)
    pred.shape = (pred.shape[0],)
    corr = pearsonr(pred, y_test)[0]

    print('Correlation Value is :', corr)
    
    # Visualize Correlation. 
    plt.plot(y_test, pred, 'o')
    m, b = np.polyfit(y_test, pred, 1)
    plt.xlabel('Ground Truth Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.axis('equal')
    plt.plot(y_test, m*y_test + b)
    plt.savefig(os.path.join('figures/', exper_name, 'sv_pcc_fold1.png'))
    
    err = np.abs(y_test - pred)
    std = np.std(err)
    print('Standard Deviation Value is :', std)
