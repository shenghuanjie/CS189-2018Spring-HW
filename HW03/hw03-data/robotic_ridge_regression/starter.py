import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys


class HW3_Sol(object):

    def __init__(self):
        pass

    def load_data(self):
        self.x_train = pickle.load(open('x_train.p', 'rb'), encoding='latin1')
        self.y_train = pickle.load(open('y_train.p', 'rb'), encoding='latin1')
        self.x_test = pickle.load(open('x_test.p', 'rb'), encoding='latin1')
        self.y_test = pickle.load(open('y_test.p', 'rb'), encoding='latin1')

        self.x_train_float_flatten = np.asarray(self.x_train, dtype=float).reshape(
            (self.x_train.shape[0], int(np.round(self.x_train.size / self.x_train.shape[0]))))
        self.y_train_float = np.asarray(self.y_train, dtype=float)
        self.x_test_float_flatten = np.asarray(self.x_test, dtype=float).reshape(
            (self.x_test.shape[0], int(np.round(self.x_train.size / self.x_train.shape[0]))))
        self.y_test_float = np.asarray(self.y_test, dtype=float)

    def plot_image(self):
        plt.figure(figsize=(7, 10))
        plt.subplot(311)
        plt.title('0th image')
        plt.imshow(self.x_train[0, :, :, :], aspect='equal', interpolation='none')
        plt.axis('off')

        plt.subplot(312)
        plt.title('10th image')
        plt.imshow(self.x_train[10, :, :, :], aspect='equal', interpolation='none')
        plt.axis('off')

        plt.subplot(313)
        plt.title('20th image')
        plt.imshow(self.x_train[20, :, :, :], aspect='equal', interpolation='none')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def print_control(self):
        print("The 0th control vector is: ", self.y_train[0, :])
        print("The 10th control vector is: ", self.y_train[10, :])
        print("The 20th control vector is: ", self.y_train[20, :])

    def try_ols(self):
        pi = np.linalg.lstsq(self.x_train_float_flatten, self.y_train_float)[0]
        print(pi)

    def ridge(self, A, b, lambda_):
        # Make sure data are centralized
        # return np.linalg.inv(A.T.dot(A) + lambda_ * np.eye(A.shape[1])).dot(A.T.dot(b))
        return np.linalg.solve(A.T.dot(A) + lambda_ * np.eye(A.shape[1]), A.T.dot(b))

    def get_euclidean_error(self, X, w, y):
        return np.sum((X.dot(w) - y) ** 2) / y.shape[0]

    def get_singular_values(self, A):
        _, eig, _ = np.linalg.svd(A)
        return eig

    LAMBDAs = [0.1, 1.0, 10, 100, 1000]

    def try_ridge_5c(self):
        errors = np.zeros(len(self.LAMBDAs))
        x_train = self.x_train_float_flatten
        y_train = self.y_train_float
        for i in range(len(self.LAMBDAs)):
            pi = self.ridge(x_train, y_train, self.LAMBDAs[i])
            errors[i] = self.get_euclidean_error(x_train, pi, y_train)
            print("Training error for lambda=%.1f is %s" % (self.LAMBDAs[i], errors[i]))

    def try_ridge_5d(self):
        errors = np.zeros(len(self.LAMBDAs))
        x_train_normalized = self.x_train_float_flatten / 255 * 2 - 1
        y_train = self.y_train_float
        for i in range(len(self.LAMBDAs)):
            pi = self.ridge(x_train_normalized, y_train, self.LAMBDAs[i])
            errors[i] = self.get_euclidean_error(x_train_normalized, pi, y_train)
            print("Training error for lambda=%.1f is %s" % (self.LAMBDAs[i], errors[i]))

    def test_ridge_5e(self):
        errors = np.zeros(len(self.LAMBDAs))
        errors_xnormalized = np.zeros(len(self.LAMBDAs))
        x_train = self.x_train_float_flatten
        x_train_normalized = x_train / 255 * 2 - 1
        x_test = self.x_test_float_flatten
        x_test_normalized = x_test / 255 * 2 - 1
        y_train = self.y_train_float
        y_test = self.y_test_float
        for i in range(len(self.LAMBDAs)):
            pi = self.ridge(x_train, y_train, self.LAMBDAs[i])
            pi_normalized = self.ridge(x_train_normalized, y_train, self.LAMBDAs[i])
            errors[i] = self.get_euclidean_error(x_test, pi, y_test)
            errors_xnormalized[i] = self.get_euclidean_error(x_test_normalized, pi_normalized, y_test)

        print("Errors without standardization: ")
        for i in range(len(self.LAMBDAs)):
            print("Training error for lambda=%.1f is %s" % (self.LAMBDAs[i], errors[i]))

        print("Errors with standardization: ")
        for i in range(len(self.LAMBDAs)):
            print("Training error for lambda=%.1f is %s" % (self.LAMBDAs[i], errors_xnormalized[i]))

    def get_condition_number(self):
        x_train = self.x_train_float_flatten
        x_train_normalized = x_train / 255 * 2 - 1
        lmbda = 100
        n_feature = x_train.shape[1]
        result = self.get_singular_values(x_train.T.dot(x_train) + lmbda * np.eye(n_feature))
        print("Condition number without standardization: ", result[0] / result[-1])
        result = self.get_singular_values(x_train_normalized.T.dot(x_train_normalized) + lmbda * np.eye(n_feature))
        print("Condition number with standardization: ", result[0] / result[-1])


if __name__ == '__main__':
    hw3_sol = HW3_Sol()

    hw3_sol.load_data()

    # Your solution goes here

    # plot 0th, 10th, 20th image
    hw3_sol.plot_image()
    hw3_sol.print_control()

    # try OLS
    print("---------5b---------")
    hw3_sol.try_ols()

    # try Ridge
    print("---------5c---------")
    hw3_sol.try_ridge_5c()

    # try Ridge after X normalization
    print("---------5d---------")
    hw3_sol.try_ridge_5d()

    # try Ridge on test data
    print("---------5e---------")
    hw3_sol.test_ridge_5e()

    # get condition number
    print("---------5f---------")
    hw3_sol.get_condition_number()
