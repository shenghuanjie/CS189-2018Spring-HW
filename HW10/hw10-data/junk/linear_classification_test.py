from numpy.random import uniform
import random
import time
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys

from projection import Project2D, Projections
from confusion_mat import getConfusionMatrixPlot

from ridge_model import Ridge_Model
from qda_model import QDA_Model
from lda_model import LDA_Model
from qda_model_li import QDA_Model as QDA_Model_Li
from lda_model_li import LDA_Model as LDA_Model_Li
from svm_model import SVM_Model
from logistic_model import Logistic_Model

CLASS_LABELS = ['apple', 'banana', 'eggplant']

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class Model():
    """ Generic wrapper for specific model instance. """

    def __init__(self, model):
        """ Store specific pre-initialized model instance. """

        self.model = model

    def train_model(self, X, Y):
        """ Train using specific model's training function. """

        self.model.train_model(X, Y)

    def test_model(self, X, Y):
        """ Test using specific model's eval function. """
        if hasattr(self.model, "evals"):
            labels = np.array(Y)
            p_labels = self.model.evals(X)

        else:
            labels = []  # List of actual labels
            p_labels = []  # List of model's predictions
            success = 0  # Number of correct predictions
            total_count = 0  # Number of images

            for i in range(len(X)):

                x = X[i]  # Test input
                y = Y[i]  # Actual label

                y_ = self.model.eval(x)  # Model's prediction
                labels.append(y)
                p_labels.append(y_)

                if y == y_:
                    success += 1
                total_count += 1

        print("Computing Confusion Matrix \\\\")
        # Compute Confusion Matrix
        print('\\[')
        plt = getConfusionMatrixPlot(labels, p_labels, CLASS_LABELS)
        print('\\]')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Model: '+self.model.__class__.__name__)
        return plt


if __name__ == "__main__":
    # Load Training Data and Labels
    X = list(np.load('little_x_train.npy'))
    Y = list(np.load('little_y_train.npy'))

    # Load Validation Data and Labels
    X_val = list(np.load('little_x_val.npy'))
    Y_val = list(np.load('little_y_val.npy'))

    CLASS_LABELS = ['apple', 'banana', 'eggplant']

    # Project Data to 2 Dimensions using CCA
    feat_dim = max(X[0].shape)
    projections = Projections(feat_dim, CLASS_LABELS)
    cca_proj, white_cov = projections.cca_projection(X, Y, k=2)

    X = projections.project(cca_proj, white_cov, X)
    X_val = projections.project(cca_proj, white_cov, X_val)

    ####RUN LDA REGRESSION#####

    lda_m = LDA_Model(CLASS_LABELS)
    model = Model(lda_m)

    print('\n'+model.model.__class__.__name__.replace('_', '\\_')+'\\\\')
    model.train_model(X, Y)
    print(model.model.sigma)
    print('training data: ')
    plt = model.test_model(X, Y)
    plt.title('Model: ' + model.model.__class__.__name__+' (training)')
    #plt.savefig('Figure_3e-training.png')
    #plt.close()
    print('test data: ')
    plt = model.test_model(X_val, Y_val)
    plt.title('Model: '+model.model.__class__.__name__+' (test)')
    #plt.savefig('Figure_3e-test.png')
    #plt.close()

    ####RUN QDA REGRESSION#####
    '''
    qda_m = QDA_Model(CLASS_LABELS)
    model = Model(qda_m)

    print('\n'+model.model.__class__.__name__.replace('_', '\\_')+'\\\\')
    model.train_model(X, Y)
    print('training data: ')
    plt = model.test_model(X, Y)
    plt.title('Model: '+model.model.__class__.__name__+' (training)')
    #plt.savefig('Figure_3f-training.png')
    #plt.close()
    print('test data: ')
    plt = model.test_model(X_val, Y_val)
    plt.title('Model: '+model.model.__class__.__name__+' (test)')
    #plt.savefig('Figure_3f-test.png')
    #plt.close()
    '''

    ####RUN LDA REGRESSION#####

    lda_m = LDA_Model_Li(CLASS_LABELS)
    model = Model(lda_m)

    print('\n'+model.model.__class__.__name__.replace('_', '\\_')+'\\\\')
    model.train_model(X, Y)
    print(model.model.cov)
    print('training data: ')
    plt = model.test_model(X, Y)
    plt.title('Model: ' + model.model.__class__.__name__+' (training)')
    #plt.savefig('Figure_3e-training.png')
    #plt.close()
    print('test data: ')
    plt = model.test_model(X_val, Y_val)
    plt.title('Model: '+model.model.__class__.__name__+' (test)')
    #plt.savefig('Figure_3e-test.png')
    #plt.close()



    '''
    matplotlib.pyplot.figure()
    colors = ['r', 'g', 'b', 'k']
    names = ['apple', 'banana', 'eggplant']
    X = np.array(X)
    Y = np.array(Y)
    this_mu = model.model.mus
    print(this_mu)
    print(model.model.sigma)
    yclasses = np.arange(1, model.model.NUM_CLASSES + 1)
    for iclass, classid in enumerate(yclasses):
        classX_true = X[np.bitwise_and(Y == classid, p_labels == classid), :]
        classX_false = X[np.bitwise_and(Y == classid, p_labels != classid), :]
        matplotlib.pyplot.scatter(classX_true[:, 0], classX_true[:, 1], c=colors[iclass])
        matplotlib.pyplot.scatter(classX_false[:, 0], classX_false[:, 1], c=colors[iclass], edgecolors='k')
        matplotlib.pyplot.scatter(this_mu[iclass, 0], this_mu[iclass, 1], 200, c=colors[iclass], marker='x', label=names[iclass])
    matplotlib.pyplot.legend()
    #matplotlib.pyplot.show()
    '''
