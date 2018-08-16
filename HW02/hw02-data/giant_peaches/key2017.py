import numpy as np
import scipy.io as spio
import time

data = spio.loadmat('polynomial_regression_samples.mat', squeeze_me=True)
# data = spio.loadmat('matlab.mat', squeeze_me=True)
data_x = data['x']
data_y = data['y']
Kc = 4  # 4-fold cross validation
KD = 6  # max D = 6
LAMBDA = [0.05, 0.1, 0.15, 0.2]

Ntotal = data_x.shape[0]
Ns = int(data_x.shape[0] * (Kc - 1) / Kc)
n = data_x.shape[1]
Nv = int(Ns / (Kc - 1))

np.set_printoptions(precision=11)
Etrain = np.zeros((KD, len(LAMBDA)))
Evalid = np.zeros((KD, len(LAMBDA)))
Ecv = np.zeros((KD, 1))
Ecvt = np.zeros((KD, 1))
a = LAMBDA

D = 3
for l in range(len(LAMBDA)):
    lmbda = a[l]
    print("D=3 lambda=%f" % lmbda)
    for c in range(4):
        xv = data_x[c * Nv:(c + 1) * Nv]
        yv = data_y[c * Nv:(c + 1) * Nv]
        x = np.delete(data_x, list(range(c * Nv, (c + 1) * Nv)), 0)
        y = np.delete(data_y, list(range(c * Nv, (c + 1) * Nv)))

        X = np.concatenate([x, np.ones((Ns, 1))], axis=1)
        Nf = 56
        Xf = np.zeros((Ns, Nf))
        for i in range(Ns):
            k = 0
            for i1 in range(n + 1):
                for i2 in range(i1, n + 1):
                    for i3 in range(i2, n + 1):
                        Xf[i][k] = X[i][i1] * X[i][i2] * X[i][i3]
                        k += 1

        Xv = np.concatenate([xv, np.ones((Nv, 1))], axis=1)
        Nf = 56
        Xfv = np.zeros((Nv, Nf))
        for i in range(Nv):
            k = 0
            for i1 in range(n + 1):
                for i2 in range(i1, n + 1):
                    for i3 in range(i2, n + 1):
                        Xfv[i][k] = Xv[i][i1] * Xv[i][i2] * Xv[i][i3]
                        k += 1

        W = np.linalg.solve(Xf.transpose().dot(Xf) + lmbda * np.eye(Nf), Xf.transpose().dot(y))
        y_predicted = Xfv.dot(W)
        Ecv[c] = np.mean((yv - y_predicted) ** 2)
        y_predicted = Xf.dot(W)
        Ecvt[c] = np.mean((y - y_predicted) ** 2)

    Evalid[D - 1, l] = np.mean(Ecv)
    Etrain[D - 1, l] = np.mean(Ecvt)

print("D=3 done")
D = 4
for l in range(len(LAMBDA)):
    lmbda = a[l]
    print("D=4 lambda=%f" % lmbda)
    for c in range(4):
        xv = data_x[c * Nv:(c + 1) * Nv]
        yv = data_y[c * Nv:(c + 1) * Nv]
        x = np.delete(data_x, list(range(c * Nv, (c + 1) * Nv)), 0)
        y = np.delete(data_y, list(range(c * Nv, (c + 1) * Nv)))

        X = np.concatenate([x, np.ones((Ns, 1))], axis=1)
        Nf = 126
        Xf = np.zeros((Ns, Nf))
        for i in range(Ns):
            k = 0
            for i1 in range(n + 1):
                for i2 in range(i1, n + 1):
                    for i3 in range(i2, n + 1):
                        for i4 in range(i3, n + 1):
                            Xf[i][k] = X[i][i1] * X[i][i2] * X[i][i3] * X[i][i4]
                            k += 1

        Xv = np.concatenate([xv, np.ones((Nv, 1))], axis=1)
        Nf = 126
        Xfv = np.zeros((Nv, Nf))
        for i in range(Nv):
            k = 0
            for i1 in range(n + 1):
                for i2 in range(i1, n + 1):
                    for i3 in range(i2, n + 1):
                        for i4 in range(i3, n + 1):
                            Xfv[i][k] = Xv[i][i1] * Xv[i][i2] * Xv[i][i3] * Xv[i][i4]
                            k += 1

        W = np.linalg.solve(Xf.transpose().dot(Xf) + lmbda * np.eye(Nf), Xf.transpose().dot(y))
        y_predicted = Xfv.dot(W)
        Ecv[c] = np.mean((yv - y_predicted) ** 2)
        y_predicted = Xf.dot(W)
        Ecvt[c] = np.mean((y - y_predicted) ** 2)

    Evalid[D - 1, l] = np.mean(Ecv)
    Etrain[D - 1, l] = np.mean(Ecvt)

# YOUR CODE to find best D and i
minTestError = float('inf')
minD = 0
minLambda = 0
for D in range(2, 4):
    for i in range(len(LAMBDA)):
        if Evalid[D, i] < minTestError:
            minD = D + 1
            minLambda = LAMBDA[i]
            minTestError = Evalid[D, i]
print('Best degree of polynomial:', minD, sep='\n')
print('Best lambda value:', minLambda, sep='\n')
print(Evalid)