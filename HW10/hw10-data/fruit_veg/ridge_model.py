from numpy.random import uniform
import random
import time

import numpy as np
import numpy.linalg as LA

import sys

from sklearn.linear_model import Ridge

from utils import create_one_hot_label


class Ridge_Model():

    def __init__(self, class_labels):
        self.lmbda = 1.0
        self.NUM_CLASSES = len(class_labels)

    def train_model(self, X, Y):
        Y_one_hot = create_one_hot_label(Y, self.NUM_CLASSES)

        self.ridge = Ridge(alpha=self.lmbda)

        self.ridge.fit(X, Y_one_hot)

    def eval(self, x):
        x = np.array(x).reshape(-1, 1).T
        prediction = self.ridge.predict(x)

        return np.argmax(prediction)
