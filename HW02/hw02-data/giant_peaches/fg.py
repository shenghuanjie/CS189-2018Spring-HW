import time

import numpy as np
import scipy.io as spio

data = spio.loadmat('polynomial_regression_samples.mat', squeeze_me=True)
# data = spio.loadmat('matlab.mat', squeeze_me=True)
data_x = data['x']
data_y = data['y']
Kc = 4  # 4-fold cross validation
KD = 6  # max D = 6
LAMBDA = [0, 0.05, 0.1, 0.15, 0.2]

# my global variables
sample_size = data_y.size / Kc
elongated_data_x = np.vstack([data_x, data_x])
elongated_data_y = np.hstack([data_y, data_y])


def ridge(A, b, lambda_):
    # Make sure data are centralized
    return np.linalg.solve(A.T.dot(A) + lambda_ * np.eye(A.shape[1]), A.T.dot(b))


def get_error(X, w, y):
    return np.mean((X.dot(w) - y) ** 2)


def fit(D, lambda_):
    # YOUR CODE TO COMPUTE THE AVERAGE ERROR PER SAMPLE
    errors = np.asarray([0.0, 0.0])
    for iv in range(0, Kc):
        validation_x = elongated_data_x[int(iv * sample_size): int((iv + 1) * sample_size), :]
        train_x = elongated_data_x[int((iv + 1) * sample_size):
        int((iv + Kc) * sample_size), :]
        validation_y = elongated_data_y[int(iv * sample_size): int((iv + 1) * sample_size)]
        train_y = elongated_data_y[int((iv + 1) * sample_size):
        int((iv + Kc) * sample_size)]
        poly_train_x = polynomial(train_x, D)
        w = ridge(poly_train_x, train_y, lambda_)
        errors[0] += get_error(poly_train_x, w, train_y)
        errors[1] += get_error(polynomial(validation_x, D), w, validation_y)
    errors = errors / Kc
    return errors


def polynomial(poly_data, degree):
    if poly_data.ndim != 2:
        pass
    nvar = poly_data.shape[1]
    npoints = poly_data.shape[0]
    results = np.ones((npoints, comb(degree + nvar, degree)))
    if degree == 0:
        return results
    start = 0
    counts = np.zeros(nvar, dtype=int)
    cur_index = 1
    cur_len = 1
    for d in range(1, degree + 1):
        last_len = cur_len
        cur_start = 0
        cur_len = 0
        for i in range(0, nvar):
            for j in range(start + cur_start, start + last_len):
                results[:, cur_index] = poly_data[:, i] * results[:, j]
                cur_index += 1
            temp = counts[i]
            counts[i] = last_len - cur_start
            cur_start += temp
            cur_len += counts[i]
        start = start + last_len

    return results


def comb(n, k):
    result = 1
    for i in range(k):
        result *= (n - i)
    for i in range(k):
        result /= (k - i)
    return int(np.round(result))


def main():
    np.set_printoptions(precision=11)
    Etrain = np.zeros((KD, len(LAMBDA)))
    Evalid = np.zeros((KD, len(LAMBDA)))
    for D in range(KD):
        print(D)
        for i in range(len(LAMBDA)):
            Etrain[D, i], Evalid[D, i] = fit(D + 1, LAMBDA[i])

    print('Average train error:', Etrain, sep='\n')
    print('Average valid error:', Evalid, sep='\n')

    # YOUR CODE to find best D and i
    minTestError = float('inf')
    minD = 0
    minLambda = 0
    for D in range(KD):
        for i in range(len(LAMBDA)):
            if Evalid[D, i] < minTestError:
                minD = D + 1
                minLambda = LAMBDA[i]
                minTestError = Evalid[D, i]
    print('Best degree of polynomial:', minD, sep='\n')
    print('Best lambda value:', minLambda, sep='\n')


if __name__ == "__main__":
    main()

