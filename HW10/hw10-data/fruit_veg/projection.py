from numpy.random import uniform
from numpy.random import randn
import random
import time

import matplotlib.pyplot as plt

from scipy.linalg import eig
from scipy.linalg import sqrtm
from numpy.linalg import inv
from numpy.linalg import svd

from utils import create_one_hot_label
from utils import subtract_mean_from_data
from utils import compute_covariance_matrix

import numpy as np
import numpy.linalg as LA

import sys
from numpy.linalg import svd
import IPython


class Project2D():
    '''
    Class to draw projection on 2D scatter space
    '''

    def __init__(self, projection, clss_labels):

        self.proj = projection
        self.clss_labels = clss_labels

    def project_data(self, X, Y, white=None):

        '''
        Takes list of state space and class labels
        State space should be 2D
        Labels shoud be int
        '''

        p_a = []
        p_b = []
        p_c = []

        ###PROJECT ALL DATA###
        proj = np.matmul(self.proj, white)

        X_P = np.matmul(proj, np.array(X).T)

        for i in range(len(Y)):

            if Y[i] == 0:
                p_a.append(X_P[:, i])
            elif Y[i] == 1:
                p_b.append(X_P[:, i])
            else:
                p_c.append(X_P[:, i])

        p_a = np.array(p_a)
        p_b = np.array(p_b)
        p_c = np.array(p_c)

        plt.figure()
        plt.scatter(p_a[:, 0], p_a[:, 1], label='apple')
        plt.scatter(p_b[:, 0], p_b[:, 1], label='banana')
        plt.scatter(p_c[:, 0], p_c[:, 1], label='eggplant')

        plt.legend()
        #plt.show()
        return plt


class Projections():

    def __init__(self, dim_x, classes):
        '''
        dim_x: the dimension of the state space x
        classes: The list of class labels
        '''

        self.d_x = dim_x
        self.NUM_CLASSES = len(classes)

    def get_random_proj(self):
        '''
        Return A which is size 2 by 729
        '''

        return randn(2, self.d_x)

    def pca_projection(self, X, Y):
        '''
        Return U_2^T
        '''
        Y = create_one_hot_label(Y, self.NUM_CLASSES)
        X, Y = subtract_mean_from_data(X, Y)

        C_XX = compute_covariance_matrix(X, X)

        u, s, d = svd(C_XX)

        return u[:, 0:2].T

    def cca_projection(self, X, Y, k=2):
        '''
        Return U_K^T, \Simgma_{XX}^{-1/2}
        '''

        Y = create_one_hot_label(Y, self.NUM_CLASSES)
        X, Y = subtract_mean_from_data(X, Y)

        C_XY = compute_covariance_matrix(X, Y)
        C_XX = compute_covariance_matrix(X, X)
        C_YY = compute_covariance_matrix(Y, Y)

        dim_x = C_XX.shape[0]
        dim_y = C_YY.shape[0]

        A = inv(sqrtm(C_XX + 1e-5 * np.eye(dim_x)))
        B = inv(sqrtm(C_YY + 1e-5 * np.eye(dim_y)))

        C = np.matmul(A, np.matmul(C_XY, B))

        u, s, d = svd(C)

        return u[:, 0:k].T, A

    def project(self, proj, white, X):
        '''
        proj, numpy matrix to perform projection
        whit, numpy matrix to perform whitenting
        X, list of states
        '''

        proj = np.matmul(proj, white)

        X_P = np.matmul(proj, np.array(X).T)

        return list(X_P.T)


if __name__ == "__main__":
    X = list(np.load('little_x_train.npy'))
    Y = list(np.load('little_y_train.npy'))

    CLASS_LABELS = ['apple', 'banana', 'eggplant']

    feat_dim = max(X[0].shape)
    projections = Projections(feat_dim, CLASS_LABELS)

    # Random Projection
    rand_proj = projections.get_random_proj()
    # #Show Random 2D Projection
    proj2D_viz = Project2D(rand_proj, CLASS_LABELS)
    plt = proj2D_viz.project_data(X, Y, white=np.eye(feat_dim))
    plt.xlabel('Random PC 1')
    plt.ylabel('Random PC 2')
    plt.title('Random Projection')
    plt.savefig('Figure_3a')
    plt.close()

    # PCA Projection
    pca_proj = projections.pca_projection(X, Y)
    # Show PCA 2D Projection
    proj2D_viz = Project2D(pca_proj, CLASS_LABELS)
    plt = proj2D_viz.project_data(X, Y, white=np.eye(feat_dim))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('Principle Component Analysis')
    plt.savefig('Figure_3b')
    plt.close()

    # CCA Projection
    cca_proj, white_cov = projections.cca_projection(X, Y)
    # Show CCA 2D Projection
    proj2D_viz = Project2D(cca_proj, CLASS_LABELS)
    plt = proj2D_viz.project_data(X, Y, white=white_cov)
    plt.xlabel('CCA 1')
    plt.ylabel('CCA 2')
    plt.title('Canonical Correlation Analysis')
    plt.savefig('Figure_3c')
    plt.close()
