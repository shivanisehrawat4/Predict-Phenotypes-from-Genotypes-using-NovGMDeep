"""
Splitting the data into train and test subsets.
"""

# import needed libraries.
import os
import random
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

random.seed(123)
np.random.seed(123)

# Define output directories for saving split data.
out_dir = 'sv'
os.makedirs(os.path.join('data/', out_dir), exist_ok=True)

# Load the whole preprocessed dataset.
X = np.load(os.path.join('data/', out_dir, 'sv_X.npy'))
Y = np.load(os.path.join('data/', out_dir, 'sv_Y.npy'))

# Print out the shape of the original data.
print('X shape is: ', X.shape)
print('Y shape is: ', Y.shape)

# Split the data into train, valid, and test subsets. 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Save the split data.
np.save(os.path.join('data', out_dir, 'X_train_sv.npy'), X_train)
np.save(os.path.join('data', out_dir, 'X_test_sv.npy'),  X_test)
np.save(os.path.join('data', out_dir, 'y_train_sv.npy'), y_train)
np.save(os.path.join('data', out_dir, 'y_test_sv.npy'),  y_test)

