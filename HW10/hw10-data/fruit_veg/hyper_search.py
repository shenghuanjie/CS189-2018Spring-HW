import cv2
import IPython
from numpy.random import uniform
import random
import time

import glob
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys

from projection import Project2D, Projections

from confusion_mat import getConfusionMatrix
from confusion_mat import plotConfusionMatrix

from ridge_model import Ridge_Model
from qda_model_test import QDA_Model
from lda_model import LDA_Model
from svm_model import SVM_Model

from utils import bmatrix

CLASS_LABELS = ['apple', 'banana', 'nectarine', 'plum', 'peach', 'watermelon', 'pear', 'mango', 'grape', 'orange',
                'strawberry', 'pineapple',
                'radish', 'carrot', 'potato', 'tomato', 'bellpepper', 'broccoli', 'cabbage', 'cauliflower', 'celery',
                'eggplant', 'garlic', 'spinach', 'ginger']


def eval_model(X, Y, k, model_key, proj):
    # PROJECT DATA
    cca_proj, white_cov = proj.cca_projection(X, Y, k=k)

    X_p = proj.project(cca_proj, white_cov, X)
    X_val_p = proj.project(cca_proj, white_cov, X_val)

    # TRAIN MODEL
    model = models[model_key]

    model.train_model(X_p, Y)
    acc, cm = model.test_model(X_val_p, Y_val)

    return acc, cm


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
            success = np.sum(labels == p_labels)
            total_count = len(X)

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

        return 1.0 * success / total_count, getConfusionMatrix(labels, p_labels)


if __name__ == "__main__":

    orig_stdout = sys.stdout
    f = open('Results_3k.txt', 'a')
    sys.stdout = f

    # Load Training Data and Labels
    X = list(np.load('big_x_train.npy'))
    Y = list(np.load('big_y_train.npy'))

    # Load Validation Data and Labels
    X_val = list(np.load('big_x_val.npy'))
    Y_val = list(np.load('big_y_val.npy'))

    # Project Data to 200 Dimensions using CCA
    feat_dim = max(X[0].shape)
    projections = Projections(feat_dim, CLASS_LABELS)

    models = {}  # Dictionary of key: model names, value: model instance

    #########MODELS TO EVALUATE############
    qda_m = QDA_Model(CLASS_LABELS)
    models['qda'] = Model(qda_m)

    lda_m = LDA_Model(CLASS_LABELS)
    models['lda'] = Model(lda_m)

    ridge_m = Ridge_Model(CLASS_LABELS)
    models['ridge'] = Model(ridge_m)

    ridge_m_10 = Ridge_Model(CLASS_LABELS)
    ridge_m_10.lmbda = 10.0
    models['ridge_lmda_10'] = Model(ridge_m_10)

    ridge_m_01 = Ridge_Model(CLASS_LABELS)
    ridge_m_01.lmbda = 0.1
    models['ridge_lmda_01'] = Model(ridge_m_01)

    svm_m = SVM_Model(CLASS_LABELS)
    models['svm'] = Model(svm_m)

    svm_m_10 = SVM_Model(CLASS_LABELS)
    svm_m_10.C = 10.0
    models['svm_C_10'] = Model(svm_m_10)

    svm_m_01 = SVM_Model(CLASS_LABELS)
    svm_m_01.C = 0.1
    models['svm_C_01'] = Model(svm_m_01)

    #########GRID SEARCH OVER MODELS############
    highest_accuracy = 0  # Highest validation accuracy
    best_model_name = None  # Best model name
    best_model = None  # Best model instance
    best_k = None
    K = [50, 200, 600, 800]  # List of dimensions
    for model_key in models.keys():
        print(model_key + '\\\\')

        val_acc = []  # List of model's accuracies for each dimension
        for k in K:
            print("k =", k, '\\\\')

            # Evaluate specific model's validation accuracy on specific dimension
            acc, c_m = eval_model(X, Y, k, model_key, projections)
            #acc = np.random.rand()
            #c_m = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)))

            val_acc.append(acc)

            if acc > highest_accuracy:
                highest_accuracy = acc
                best_model_name = model_key
                best_cm = c_m
                best_k = k
        # Plot specific model's accuracies across validation error
        plt.plot(K, val_acc, label=model_key)

    # Display aggregate plot of models across validation error
    plt.legend()
    plt.xlabel('Dimension')
    plt.ylabel('Accuracy')
    # plt.show()
    plt.title('Models\' validation')
    plt.savefig('Figure_3k-validation.png')

    # Plot best model's confusion matrix
    print("Computing Confusion Matrix \\\\")
    # Compute Confusion Matrix
    print('\\[')
    print(bmatrix(best_cm))
    print('\\]')
    plt = plotConfusionMatrix(best_cm, CLASS_LABELS)

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Model: ' + best_model_name + ' dim: ' + str(best_k))
    plt.savefig('Figure_3k-bestmodel.png')

    sys.stdout = orig_stdout
    f.close()
