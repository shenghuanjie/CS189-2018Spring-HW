import os
import numpy as np
import cv2
import copy
import glob

import sys

from numpy.random import uniform

import pickle
from scipy.linalg import eig
from scipy.linalg import sqrtm
from numpy.linalg import inv
from numpy.linalg import svd
import numpy.linalg as LA
import matplotlib.pyplot as plt
import IPython
from sklearn.preprocessing import StandardScaler


def standardized(v):
    return (v / 255.0) * 2.0 - 1.0


def flatten_and_standardize(data):
    result = []
    for d in data:
        d = d.flatten()
        d = standardized(d)
        result.append(d)
    return result


class Mooney(object):

    def __init__(self):
        self.lmbda = 1e-5

    def load_data(self):
        self.x_train = pickle.load(open('x_train.p', 'rb'))
        self.y_train = pickle.load(open('y_train.p', 'rb'))
        self.x_test = pickle.load(open('x_test.p', 'rb'))
        self.y_test = pickle.load(open('y_test.p', 'rb'))

    def compute_covariance_matrices(self):
        # USE STANDARD SCALAR TO DO MEAN SUBTRACTION
        ss_x = StandardScaler(with_std=False)
        ss_y = StandardScaler(with_std=False)

        num_data = len(self.x_train)

        x = self.x_train[0]
        y = self.y_train[0]

        x_f = x.flatten()
        y_f = y.flatten()

        x_f_dim = x_f.shape[0]
        y_f_dim = y_f.shape[0]

        self.x_dim = x_f_dim
        self.y_dim = y_f_dim

        self.C_xx = np.zeros([x_f_dim, x_f_dim])
        self.C_yy = np.zeros([y_f_dim, y_f_dim])
        self.C_xy = np.zeros([x_f_dim, y_f_dim])

        x_data = []
        y_data = []

        for i in range(num_data):
            x_image = self.x_train[i]
            y_image = self.y_train[i]

            # FLATTEN DATA
            x_f = x_image.flatten()
            y_f = y_image.flatten()

            # STANDARDIZE DATA
            x_f = standardized(x_f)
            y_f = standardized(y_f)

            x_data.append(x_f)
            y_data.append(y_f)

        # SUBTRACT MEAN
        ss_x.fit(x_data)
        x_data = ss_x.transform(x_data)

        ss_y.fit(y_data)
        y_data = ss_y.transform(y_data)

        for i in range(num_data):
            x_f = np.array([x_data[i]])
            y_f = np.array([y_data[i]])
            # TODO: COMPUTE COVARIANCE MATRICES
            # BE CAREFUL: x_f is a row vector here
            self.C_xx += x_f.T.dot(x_f)
            self.C_yy += y_f.T.dot(y_f)
            self.C_xy += x_f.T.dot(y_f)


        # DIVIDE BY THE NUMBER OF DATA POINTS
        self.C_xx = 1.0 / float(num_data) * self.C_xx
        self.C_yy = 1.0 / float(num_data) * self.C_yy
        self.C_xy = 1.0 / float(num_data) * self.C_xy

    def compute_projected_data_matrix(self, X_proj):
        Y = []
        X = []

        Y_test = []
        X_test = []

        # LOAD TRAINING DATA
        for x in self.x_train:
            x_f = np.array([x.flatten()])
            # STANDARDIZE DATA
            x_f = standardized(x_f)
            # TODO: PROJECT DATA
            # x_f is a row vector and X_proj is a column vector
            X.append(np.reshape(X_proj.T.dot(x_f.T), (X_proj.shape[1])))
            #X.append(np.zeros((X_proj.shape[0])))

        Y = flatten_and_standardize(self.y_train)

        #print('X.shape'+str(np.array(X).shape))
        #print('Y.shape'+str(np.array(Y).shape))

        for x in self.x_test:
            x_f = np.array([x.flatten()])
            # STANDARDIZE DATA
            x_f = standardized(x_f)
            # TODO: PROJECT DATA
            X_test.append(np.reshape(X_proj.T.dot(x_f.T), (X_proj.shape[1])))
            #X_test.append(np.zeros((X_proj.shape[0])))

        Y_test = flatten_and_standardize(self.y_test)
        #print('np.zeros((X_proj.shape[0]))'+str(np.zeros((X_proj.shape[0])).shape))
        #print('X_test.shape'+str(np.array(X_test).shape))
        #print('Y_test.shape'+str(np.array(Y_test).shape))

        # CONVERT TO MATRIX
        self.X_ridge = np.vstack(X)
        self.Y_ridge = np.vstack(Y)

        self.X_test_ridge = np.vstack(X_test)
        self.Y_test_ridge = np.vstack(Y_test)

    def compute_data_matrix(self):
        X = flatten_and_standardize(self.x_train)
        Y = flatten_and_standardize(self.y_train)
        X_test = flatten_and_standardize(self.x_test)
        Y_test = flatten_and_standardize(self.y_test)

        # CONVERT TO MATRIX
        self.X_ridge = np.vstack(X)
        self.Y_ridge = np.vstack(Y)

        self.X_test_ridge = np.vstack(X_test)
        self.Y_test_ridge = np.vstack(Y_test)

    def solve_for_variance(self):
        eigen_values = np.zeros((675,))
        eigen_vectors = np.zeros((675, 675))
        # TODO: COMPUTE CORRELATION MATRIX
        corr_matrix = inv(sqrtm(self.C_xx + self.lmbda * np.eye(self.C_xx.shape[0]))).\
            dot(self.C_xy).\
            dot(inv(sqrtm(self.C_yy + self.lmbda * np.eye(self.C_yy.shape[0]))))
        eigen_vectors, eigen_values, _ = svd(corr_matrix)
        return eigen_values, eigen_vectors

    def project_data(self, eig_val, eig_vec, proj=150):
        # TODO: COMPUTE PROJECTION SINGULAR VECTORS
        return eig_vec[:, 0:proj]

    def ridge_regression(self):
        w_ridge = []

        for i in range(self.y_dim):
            # TODO: IMPLEMENT RIDGE REGRESSION
            w_i = inv(self.X_ridge.T.dot(self.X_ridge)
                      + self.lmbda * np.eye(self.X_ridge.shape[1])).\
                dot(self.X_ridge.T).dot(self.Y_ridge[:, i])
            w_ridge.append(np.reshape(w_i, (self.X_ridge.shape[1],)))
            #w_ridge.append(np.zeros((self.X_ridge.shape[1],)))

        self.w_ridge = np.vstack(w_ridge)
        #print('self.y_dim'+str(self.y_dim))
        #print('self.X_ridge'+str(np.array(self.X_ridge).shape))
        #print('self.Y_ridge'+str(np.array(self.Y_ridge).shape))
        #print('self.w_ridge'+str(np.array(self.w_ridge).shape))

    def plot_image(self, vector):
        vector = ((vector + 1.0) / 2.0) * 255.0
        vector = np.reshape(vector, (15, 15, 3))
        p = vector.astype("uint8")
        p = cv2.resize(p, (100, 100))
        count = 0

        cv2.imwrite('a_face_' + str(count) + '.png', p)

    def measure_error(self, X_ridge, Y_ridge):
        #print('X_ridge.T.shape'+str(X_ridge.T.shape))
        #print('self.w_ridge.shape'+str(self.w_ridge.shape))
        #print('Y_ridge.T.shape'+str(Y_ridge.T.shape))

        prediction = np.matmul(self.w_ridge, X_ridge.T)

        evaluation = Y_ridge.T - prediction

        print(evaluation)

        dim, num_data = evaluation.shape

        error = []

        for i in range(num_data):
            # COMPUTE L2 NORM for each vector then square
            error.append(LA.norm(evaluation[:, i]) ** 2)

        # Return average error
        return np.mean(error)

    def draw_images(self, subfix=''):
        for count, x in enumerate(self.X_test_ridge):
            prediction = np.matmul(self.w_ridge, x)
            prediction = ((prediction + 1.0) / 2.0) * 255.0
            prediction = np.reshape(prediction, (15, 15, 3))
            p = prediction.astype("uint8")
            p = cv2.resize(p, (100, 100))
            cv2.imwrite('face_' + str(count) + '_' + subfix + '.png', p)

        for count, x in enumerate(self.x_test):
            x = x.astype("uint8")
            x = cv2.resize(x, (100, 100))
            cv2.imwrite('og_face_' + str(count) + '_' + subfix + '.png', x)

        for count, x in enumerate(self.y_test):
            x = x.astype("uint8")
            x = cv2.resize(x, (100, 100))
            cv2.imwrite('gt_face_' + str(count) + '_' + subfix + '.png', x)


if __name__ == '__main__':

    mooney = Mooney()

    mooney.load_data()
    mooney.compute_covariance_matrices()
    eig_val, eig_vec = mooney.solve_for_variance()

    # Plot eigenvalues
    plt.figure()
    plt.plot(eig_val)
    plt.title('Eigenvalues of the correlation matrix')
    plt.xlabel('Rank Order of Eigenvalues')
    plt.ylabel('Eigenvalues')
    #plt.show()

    # Show face u0
    ss_x = StandardScaler(with_std=False)
    x_data = flatten_and_standardize(mooney.x_train)
    ss_x.fit(x_data)
    mooney.plot_image(ss_x.inverse_transform(eig_vec[:, 0]))

    proj = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 650]
    error_test = []
    for p in proj:
        X_proj = mooney.project_data(eig_val, eig_vec, proj=p)

        # COMPUTE REGRESSION
        mooney.compute_projected_data_matrix(X_proj)
        mooney.ridge_regression()
        training_error = mooney.measure_error(mooney.X_ridge, mooney.Y_ridge)
        test_error = mooney.measure_error(mooney.X_test_ridge, mooney.Y_test_ridge)
        ##mooney.draw_images()

        error_test.append(test_error)

    plt.figure()
    plt.plot(proj, error_test)
    plt.title('Test error of CCA Ridge Regression')
    plt.xlabel('Projection dimension k')
    plt.ylabel('Test error')
    #plt.show()

    # COMPUTER REGRESSION WITH PROJECT AND OPTIMAL proj=k
    opt_proj = proj[error_test.index(min(error_test))]
    print('optimal degree of projection: '+str(opt_proj))
    X_proj = mooney.project_data(eig_val, eig_vec, proj=opt_proj)
    mooney.compute_projected_data_matrix(X_proj)
    mooney.ridge_regression()
    mooney.draw_images('proj')

    # COMPUTE REGRESSION NO PROJECT
    mooney.compute_data_matrix()
    mooney.ridge_regression()
    mooney.draw_images()
    training_error = mooney.measure_error(mooney.X_ridge, mooney.Y_ridge)
    test_error = mooney.measure_error(mooney.X_test_ridge, mooney.Y_test_ridge)


