# coding: utf-8

# In[1]:


import scipy.io as sio
import os
import h5py
import numpy as np
from  model import *
#from resnet_model import *


class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
            self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X

class StandardNormalization(object):

    def __init__(self):
        pass

    def fit(self, X):
        self._std = X.std()
        self._mean = X.mean()
        print("mean:", self._mean, "std:", self._std)

    def transform(self, X):
        X = 1. * (X - self._mean) / self._std
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = X * self._std +  self._mean
        return X


def load_data(path, var_name):
    print("Loading raw data from ", path, "...", "variable_name:", var_name)
    with h5py.File(path, 'r') as file:
        raw_data = list(np.array(file[var_name]).T)
        print("Raw data has been loaded.")
        print("Raw data shape:", len(raw_data))
        file.close()
    return raw_data



def preprocess_data(X, Y, split_ratio):
    print("Normalization for dataset...")
    tot_size = len(X)
    train_size = int(tot_size * split_ratio)

    n_split = int(tot_size * split_ratio)

    X = np.asarray(X)
    Y = np.asarray(Y)

    scalar_Y = MinMaxNormalization()
    Y = scalar_Y.fit_transform(Y)

    scalar_X = MinMaxNormalization()
    X = scalar_X.fit_transform(X)

    print("Normalization Done")
    return list(X), list(Y), scalar_Y
def create_st_dataset(X, Y, predict_day, split_ratio):
    tot_size = len(X) #总天数
    train_X = []
    extra_train_X = []
    train_Y = []

    test_X = []
    extra_test_X = []
    test_Y = []

    train_size = int(tot_size*split_ratio)

    X_shift = [0]#,1,2,60,120,180,480,600,720]
    for i in range(0, train_size):
        tmp_x = []
        extra_tmp_x = []
        for day in X_shift:
            tmp_x.append(X[i-day])
            extra_tmp_x.append(Y[i-day])

        tmp_x = np.asarray(tmp_x)
        tmp_x = np.transpose(tmp_x, (1,2,0))#经、纬、时间

        extra_tmp_x = np.asarray(extra_tmp_x)
        train_X.append(tmp_x)
        extra_train_X.append(extra_tmp_x)
        train_Y.append(Y[i+predict_day])

    for i in range(train_size,tot_size-predict_day):
        tmp_x = []
        extra_tmp_x = []
        for day in X_shift:
            tmp_x.append(X[i-day])
                extra_tmp_x.append(Y[i-day])

        tmp_x = np.asarray(tmp_x)
        tmp_x = np.transpose(tmp_x, (1,2,0))
        extra_tmp_x = np.asarray(extra_tmp_x)
        test_X.append(tmp_x)
        extra_test_X.append(extra_tmp_x)
        test_Y.append(Y[i+predict_day])

    train_X = np.asarray(train_X)
    extra_train_X = np.asarray(extra_train_X)
    train_Y = np.asarray(train_Y)

    test_X = np.asarray(test_X)
    extra_test_X = np.asarray(extra_test_X)
    test_Y = np.asarray(test_Y)

    n_example = extra_train_X.shape[0]
    extra_train_X = np.reshape(extra_train_X, (n_example, -1))
    train_Y = np.reshape(train_Y, (n_example, -1))

    n_example = extra_test_X.shape[0]
    extra_test_X = np.reshape(extra_test_X,(n_example,-1))
    test_Y = np.reshape(test_Y, (n_example,-1))

    return train_X, extra_train_X, train_Y, test_X, extra_test_X,test_Y