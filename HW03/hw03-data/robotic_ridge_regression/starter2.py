import pickle
import matplotlib.pyplot as plt
import numpy as np


class HW3_Sol(object):

    def __init__(self):
        pass

    def load_data(self):
        self.x_train = pickle.load(open('x_train.p', 'rb'), encoding='latin1')
        self.y_train = pickle.load(open('y_train.p', 'rb'), encoding='latin1')
        self.x_test = pickle.load(open('x_test.p', 'rb'), encoding='latin1')
        self.y_test = pickle.load(open('y_test.p', 'rb'), encoding='latin1')

    # plot the ith image
    def plot_img(self, i):
        plt.imshow(self.x_train[i])
        plt.show()

    def get_control_vec(self, i):
        print("(a) control vector for %d th image is" % i)
        print(self.y_train[i])

    def flatten_x(self):
        X = self.x_train.reshape(self.x_train.shape[0],
                                 self.x_train.shape[1] * self.x_train.shape[2] * self.x_train.shape[3])
        X = X * 1.0
        return X

    def flatten_x_test(self):
        X_test = self.x_test.reshape(self.x_test.shape[0],
                                     self.x_test.shape[1] * self.x_test.shape[2] * self.x_test.shape[3])
        X_test = X_test * 1.0
        return X_test


def OLS(x, y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


def ridge(x, y, lam):
    return np.linalg.inv(x.T.dot(x) + lam * (np.eye(x.T.shape[0]))).dot(x.T).dot(y)


def average_euclidean_dist(X, pi, U):
    error_vec = X.dot(pi) - U
    norm_squares = []
    for i in range(U.shape[0]):
        norm_squares.append(np.linalg.norm(error_vec[i]) ** 2)
    return np.mean(norm_squares)


def standardize_X(x):
    standardizer = lambda t: (t / 255) * 2 - 1
    func = np.vectorize(standardizer)
    return func(x)


if __name__ == '__main__':

    hw3_sol = HW3_Sol()

    hw3_sol.load_data()

    # a. plot 0th, 10th, 20th images
    # to_plt = [0,10,20]
    # for i in to_plt:
    #     hw3_sol.plot_img(i)
    #     hw3_sol.get_control_vec(i)

    # b. OLS
    print("---------Part B---------")
    X = hw3_sol.flatten_x()
    U = hw3_sol.y_train
    # pi = OLS(X, U)
    # print("The predicted pi from OLS is:")
    # print(pi)

    # c. ridge regression
    print("---------Part C---------")
    lambda_list = [0.1, 1.0, 10, 100, 1000]
    for l in lambda_list:
        pi = ridge(X, U, l)
        err = average_euclidean_dist(X, pi, U)
        print("Training error for lambda=%f is %s" % (l, err))

    # d. strandardizing states
    print("---------Part D---------")
    # X_stand = standardize_X(X)
    # for l in lambda_list:
    #     pi = ridge(X_stand, U, l)
    #     err = average_euclidean_dist(X_stand, pi, U)
    #     print("Training error for lambda=%f is %s" %(l, err))

    # e. Evaluate policies
    print("---------Part E---------")
    print("Test error no standardization: ")
    # X_test = hw3_sol.flatten_x_test()
    # U_test = hw3_sol.y_test
    # for l in lambda_list:
    #    pi = ridge(X, U, l)
    #    err = average_euclidean_dist(X_test, pi, U_test)
    #    print("lambda=%s, test_err=%s" %(l, err))
    # print("Test error with standardization: ")
    # X_test_stand = standardize_X(X_test)
    # for l in lambda_list:
    #    pi = ridge(X_stand, U, l)
    #    err = average_euclidean_dist(X_test_stand, pi, U_test)
    #    print("lambda=%s, test_err=%s" %(l, err))
