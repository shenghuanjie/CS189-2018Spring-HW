#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# data = np.load(’circle.npz’)
# data = np.load(’heart.npz’)
data = np.load('asymmetric.npz')

SPLIT = 0.8
X = data["x"]
y = data["y"]
X /= np.max(X)  # normalize the data

n_train = int(X.shape[0] * SPLIT)
X_train = X[:n_train:, :]
X_valid = X[n_train:, :]
y_train = y[:n_train]
y_valid = y[n_train:]

LAMBDA = 0.001


def lstsq(A, b, lambda_=0):
    return np.linalg.solve(A.T @ A + lambda_ * np.eye(A.shape[1]), A.T @ b)

def heatmap(f, clip=5):
    # example: heatmap(lambda x, y: x * x + y * y)
    # clip: clip the function range to [-clip, clip] to generate a clean plot
    # set it to zero to disable this function

    xx0 = xx1 = np.linspace(np.min(X), np.max(X), 72)
    x0, x1 = np.meshgrid(xx0, xx1)
    x0, x1 = x0.ravel(), x1.ravel()
    z0 = f(x0, x1)

    if clip:
        z0[z0 > clip] = clip
        z0[z0 < -clip] = -clip

    plt.figure()
    plt.hexbin(x0, x1, C=z0, gridsize=50, cmap=cm.jet, bins=None)
    plt.colorbar()
    cs = plt.contour(
        xx0, xx1, z0.reshape(xx0.size, xx1.size), [-2, -1, -0.5, 0, 0.5, 1, 2], cmap=cm.jet)
    plt.clabel(cs, inline=1, fontsize=10)

    pos = y[:] == +1.0
    neg = y[:] == -1.0
    plt.scatter(X[pos, 0], X[pos, 1], c='red', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], c='blue', marker='v')
    # plt.show()
    return plt

def assemble_feature(x, D):
    from scipy.special import binom
    xs = []
    for d0 in range(D + 1):
        for d1 in range(D - d0 + 1):
            # non-kernel polynomial feature
            #xs.append((x[:, 0] ** d0) * (x[:, 1] ** d1))
    # # kernel polynomial feature
            xs.append((x[:, 0]**d0) * (x[:, 1]**d1) * np.sqrt(binom(D, d0) * binom(D - d0, d1)))
    return np.column_stack(xs)


def main():
    for D in range(1, 17):
        Xd_train = assemble_feature(X_train, D)
        Xd_valid = assemble_feature(X_valid, D)
        w = lstsq(Xd_train, y_train, LAMBDA)
        error_train = np.average(np.square(y_train - Xd_train @ w))
        error_valid = np.average(np.square(y_valid - Xd_valid @ w))
        print("D = {:2d} train_error = {:10.6f} validation_error = {:10.6f} cond = {:14.6f}".
              format(D, error_train, error_valid, np.linalg.cond(Xd_valid.T @ Xd_valid + np.eye(Xd_valid.shape[1]))))
        # if D in [2, 4, 6, 8, 10, 12]:
        # fname = "asym%02d.pdf" % D
        heatplt = heatmap(lambda x, y: assemble_feature(np.vstack([x, y]).T, D) @ w)
        heatplt.title('asymmetric_D' + str(D))
        heatplt.savefig('Figure_5b_kernel_heatmap_asymmetric_D' + str(D) + '.png')
        heatplt.close()

if __name__ == "__main__":
    main()
