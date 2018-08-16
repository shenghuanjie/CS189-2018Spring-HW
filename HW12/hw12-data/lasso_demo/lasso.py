from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# from simulator import *
# matplotlib inline


def randMasks(numMasks, numPixels):
    randNormalMat = np.random.normal(0, 1, (numMasks, numPixels))
    # make the columns zero mean and normalize
    for k in range(numPixels):
        # make zero mean
        randNormalMat[:, k] = randNormalMat[:, k] - np.mean(randNormalMat[:, k])
        # normalize to unit norm
        randNormalMat[:, k] = randNormalMat[:, k] / np.linalg.norm(randNormalMat[:, k])
    A = randNormalMat.copy()
    Mask = randNormalMat - np.min(randNormalMat)
    return Mask, A


def simulate():
    # read the image in grayscale
    I = np.load('helper.npy')
    sp = np.sum(I)
    numMeasurements = 6500
    numPixels = I.size
    Mask, A = randMasks(numMeasurements, numPixels)
    full_signal = I.reshape((numPixels, 1))
    measurements = np.dot(Mask, full_signal)
    measurements = measurements - np.mean(measurements)
    return measurements, A, I


measurements, X, I = simulate()

# THE SETTINGS FOR THE IMAGE - PLEASE DO NOT CHANGE
height = 91
width = 120
imDims = (height, width)
sparsity = 476
numPixels = len(X[0])

plt.imshow(I, cmap=plt.cm.gray, interpolation='nearest')
plt.title('Original Sparse Image')
plt.savefig('Figure_2f-original.png')
plt.close()

chosenMaskToDisplay = 0
M0 = X[chosenMaskToDisplay].reshape((height, width))
plt.title('Matrix X')
plt.imshow(M0, cmap=plt.cm.gray, interpolation='nearest')
plt.savefig('Figure_2f-mask.png')
plt.close()

# measurements
plt.title('measurement vector (y)')
plt.plot(measurements)
plt.xlabel('measurement index')
# plt.show()
plt.savefig('Figure_2f-measurement.png')
plt.close()


def LASSO(imDims, measurements, X, _lambda):
    clf = linear_model.Lasso(alpha=_lambda)
    clf.fit(X, measurements)
    Ihat = clf.coef_.reshape(imDims)
    return Ihat


def getError(I, Ihat):
    return np.sum((I - Ihat) ** 2)


Ihat = []
best_lambda = -1
err = float("inf")
all_errs = []
all_lambdas = np.linspace(-8, -6, 20)
all_lambdas = np.power(10, all_lambdas)

for _lambda in all_lambdas:
    Ihat_lambda = LASSO(imDims, measurements, X, _lambda)
    all_errs.append(getError(Ihat_lambda, I))
    if all_errs[-1] < err:
        err = all_errs[-1]
        Ihat = Ihat_lambda
        best_lambda = _lambda

plt.title('estimated image with lambda=' + str(best_lambda))
plt.imshow(Ihat, cmap=plt.cm.gray, interpolation='nearest')
plt.savefig('Figure_2f-reconstructed.png')
plt.close()

plt.loglog(all_lambdas, all_errs)
plt.title('semilogy plot of lambda Vs. MSE')
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.savefig('Figure_2f-lambdaVsMSE.png')
plt.close()
