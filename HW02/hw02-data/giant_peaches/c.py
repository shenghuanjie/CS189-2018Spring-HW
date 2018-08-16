#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio


# There is numpy.linalg.lstsq, whicn you should use outside of this classs
def lstsq(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)


def main():
    data = spio.loadmat('1D_poly.mat', squeeze_me=True)
    x_train = np.array(data['x_train'])
    y_train = np.array(data['y_train']).T
    y_fresh = np.array(data['y_fresh']).T

    n = 20  # max degree
    err_train = np.zeros(n - 1)
    err_fresh = np.zeros(n - 1)
    # fill in err_fresh and err_train
    # YOUR CODE HERE

    x_poly = np.ones((1, n))

    for i in range(1, n):
        x_poly = np.vstack((x_poly, np.power(x_train, i)))
        w = lstsq(x_poly.T, y_train)
        err_train[i - 1] = np.mean((x_poly.T.dot(w) - y_train) ** 2)
        err_fresh[i - 1] = np.mean((x_poly.T.dot(w) - y_fresh) ** 2)

    plt.figure()
    plt.ylim([0, 6])
    plt.plot(np.linspace(1, n-1, n-1), err_train, label='train')
    plt.plot(np.linspace(1, n-1, n-1), err_fresh, label='fresh')
    plt.legend()
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Error')
    plt.xticks(np.linspace(1, n-1, n-1))
    plt.show()


if __name__ == "__main__":
    main()

