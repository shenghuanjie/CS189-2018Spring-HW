from numpy.random import uniform
import random
import time

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys

from projection import Project2D, Projections
from confusion_mat import getConfusionMatrixPlot

from ridge_model import Ridge_Model
from qda_model import QDA_Model
from lda_model import LDA_Model
from svm_model import SVM_Model
from logistic_model import Logistic_Model

import matplotlib.pyplot as plt

CLASS_LABELS = ['apple', 'banana']


def compute_tp_fp(thres, scores, labels):
    scores = np.array(scores)
    prediction = (scores > thres)
    tp = np.sum(prediction * labels)
    tpr = 1.0 * tp / np.sum(labels)

    fp = np.sum(prediction * (1 - labels))
    fpr = 1.0 * fp / np.sum(1 - labels)
    return tpr, fpr


def plot_ROC(tps, fps):
    # plot
    plt.plot(fps, tps)
    plt.ylabel("True Positive Rates")
    plt.xlabel("False Positive Rates")


def ROC(scores, labels):
    thresholds = sorted(np.unique(scores))
    thresholds = [-float("Inf")] + thresholds + [float("Inf")]
    tps = []
    fps = []

    # student code start here
    # TODO: Your code
    # student code end here
    n = len(labels)
    tps = np.zeros(len(thresholds))
    fps = np.zeros(len(thresholds))
    for ithold, thold in enumerate(thresholds):
        #tp = np.count_nonzero((scores > thold) & (labels == 1))
        #fp = np.count_nonzero((scores > thold) & (labels == 0))
        #tn = np.count_nonzero((scores <= thold) & (labels == 0))
        #fn = np.count_nonzero((scores <= thold) & (labels == 1))
        #tps[ithold] = tp / (tp + fn)
        #fps[ithold] = fp / (fp + tn)
        tps[ithold], fps[ithold] = compute_tp_fp(thold, scores, labels)
    return tps, fps


def eval_with_ROC(method, train_X, train_Y, val_X, val_Y, C):
    m = method(CLASS_LABELS)
    m.C = C
    m.train_model(train_X, train_Y)
    scores = m.scores(val_X)

    # change the scores here
    #scores = 10.0 * np.array(scores)

    tps, fps = ROC(scores, val_Y)
    plot_ROC(tps, fps)


def trim_data(X, Y):
    # throw away the 3rd class data
    X = np.array(X)
    Y = np.array(Y)
    retain = (Y < 2)
    return X[retain, :], Y[retain]


if __name__ == "__main__":
    # Load Training Data and Labels
    X = list(np.load('little_x_train.npy'))
    Y = list(np.load('little_y_train.npy'))
    X, Y = trim_data(X, Y)

    # Load Validation Data and Labels
    X_val = list(np.load('little_x_val.npy'))
    Y_val = list(np.load('little_y_val.npy'))
    X_val, Y_val = trim_data(X_val, Y_val)

    CLASS_LABELS = ['apple', 'banana']

    # Project Data to 200 Dimensions using CCA
    feat_dim = max(X[0].shape)
    projections = Projections(feat_dim, CLASS_LABELS)
    cca_proj, white_cov = projections.cca_projection(X, Y, k=2)

    X = projections.project(cca_proj, white_cov, X)
    X_val = projections.project(cca_proj, white_cov, X_val)

    ####RUN SVM REGRESSION#####
    eval_with_ROC(SVM_Model, X, Y, X_val, Y_val, 1.0)
    eval_with_ROC(SVM_Model, X, Y, X_val, Y_val, 100000.0)
    plt.legend(["C=1.0", "C=100000.0"])
    #plt.show()
    plt.savefig('Figure_3i.png')
