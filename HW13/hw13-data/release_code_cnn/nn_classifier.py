import glob
import os
import random
import sys
import time

import IPython
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform
from  sklearn.neighbors import KNeighborsClassifier


class NN():
    def __init__(self, train_data, val_data, n_neighbors=5):
        self.train_data = train_data
        self.val_data = val_data

        self.sample_size = 400

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def preprocess(self, data):
        X = [d['features'] for d in data]
        y = np.array([d['label'] for d in data])

        X = np.reshape(X, (len(data), -1))
        y = np.argmax(y, axis=1)
        return X, y

    def train_model(self):
        '''
        Train Nearest Neighbors model
        '''
        X, y = self.preprocess(self.train_data)
        self.model.fit(X, y)

    def get_error(self, data):
        X, y = self.preprocess(data)
        indicies = np.array(np.random.choice(len(data), self.sample_size, replace=False), dtype=int)
        X = X[indicies]
        y = y[indicies]
        yhat = self.model.predict(X)

        return np.count_nonzero(y == yhat) / self.sample_size

    def get_validation_error(self):
        '''
        Compute validation error. Please only compute the error on the sample_size number
        over randomly selected data points. To save computation.
        '''
        return self.get_error(self.val_data)

    def get_train_error(self):
        '''
        Compute train error. Please only compute the error on the sample_size number
        over randomly selected data points. To save computation.
        '''
        return self.get_error(self.train_data)