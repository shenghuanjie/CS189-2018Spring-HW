import random
import time

import glob
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys
from numpy.linalg import inv
from numpy.linalg import det
from sklearn.svm import LinearSVC
from projection import Project2D, Projections


class LDA_Model():

    def __init__(self, class_labels):
        ###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
        self.reg_cov = 0.001
        self.NUM_CLASSES = len(class_labels)
        self.mus = []
        self.sigma = []
        self.yclasses = []
        self.d = -1
        self.inv_sigma = []

    def train_model(self, X, Y):
        ''''
        FILL IN CODE TO TRAIN MODEL
        MAKE SURE TO ADD HYPERPARAMTER TO MODEL

        '''
        X = np.array(X, dtype=float)
        self.d = X.shape[1]
        self.yclasses = np.unique(Y)
        self.mus = np.zeros((self.yclasses.size, X.shape[1]))
        self.sigma = np.zeros((X.shape[1], X.shape[1]), dtype=float)
        for iclass, classid in enumerate(self.yclasses):
            classX = X[Y == classid, :]
            self.mus[iclass] = np.mean(classX, axis=0).real
            classX -= self.mus[iclass]
            self.sigma += classX.T.dot(classX)
        self.sigma /= X.shape[0]
        # Add the hyperparameter
        self.sigma += self.reg_cov * np.eye(self.d)
        self.inv_sigma = np.linalg.inv(self.sigma)

    def eval(self, x):
        ''''
        Fill in code to evaluate model and return a prediction
        Prediction should be an integer specifying a class
        '''
        # x is a row vector here
        predclass = -1
        lossValue = np.inf
        x = np.array(x).reshape(-1, 1).T.real

        for iclass, classid in enumerate(self.yclasses):
            tempValue = (x - self.mus[iclass]).dot(self.inv_sigma).\
                dot((x - self.mus[iclass]).T)
            tempValue = np.real(tempValue)

            if tempValue < lossValue:
                predclass = classid
                lossValue = tempValue

        return predclass
