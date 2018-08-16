import random
import time

import numpy as np
import numpy.linalg as LA

from numpy.linalg import inv
from numpy.linalg import det

from projection import Project2D, Projections


class QDA_Model():

    def __init__(self, class_labels):
        ###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
        self.reg_cov = 0.01
        self.NUM_CLASSES = len(class_labels)
        self.mus = []
        self.sigma = []
        self.yclasses = []
        self.d = -1
        self.inv_sigma = []
        self.log_det_sigma = []

    def train_model(self, X, Y):

        ''''
        FILL IN CODE TO TRAIN MODEL
        MAKE SURE TO ADD HYPERPARAMTER TO MODEL

        '''
        X = np.array(X, dtype=float)
        self.d = X.shape[1]
        self.yclasses = np.unique(Y)
        self.mus = np.zeros((self.yclasses.size, X.shape[1]))
        self.sigma = np.zeros((self.yclasses.size, X.shape[1], X.shape[1]), dtype=float)
        self.inv_sigma = np.zeros((self.yclasses.size, X.shape[1], X.shape[1]), dtype=float)
        self.log_det_sigma = np.zeros(self.yclasses.size, dtype=float)
        for iclass, classid in enumerate(self.yclasses):
            classX = X[Y == classid, :]
            self.mus[iclass] = np.mean(classX, axis=0)
            classX -= self.mus[iclass]
            self.sigma[iclass] = classX.T.dot(classX) / classX.shape[0]
            self.sigma[iclass] += self.reg_cov * np.eye(self.d)
            self.inv_sigma[iclass] = np.linalg.inv(self.sigma[iclass])
            # slogdet is suggested by numpy.linalg.det for large matrix
            _, self.log_det_sigma[iclass] = np.linalg.slogdet(self.sigma[iclass])


    def eval(self, x):
        ''''
        Fill in code to evaluate model and return a prediction
        Prediction should be an integer specifying a class
        '''
        # print('x.shape'+str(x.shape))
        # print('mus[0].shape'+str(self.mus[0].shape))
        # x is a row vector here
        predclass = -1
        lossValue = np.inf
        x = np.array(x).reshape(-1, 1).T
        for iclass, classid in enumerate(self.yclasses):
            tempValue = (x - self.mus[iclass]).dot(self.inv_sigma[iclass]). \
                dot((x - self.mus[iclass]).T) + self.log_det_sigma[iclass]
            if tempValue < lossValue:
                predclass = classid
                lossValue = tempValue

        return predclass
