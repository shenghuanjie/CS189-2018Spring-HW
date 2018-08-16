import math
import matplotlib.pyplot as plt
import numpy as np

SPLIT = 0.15
KD = [5, 6]  # max D = 16
LAMBDA = [0.0001, 0.001, 0.01]
np.random.seed(0)
data_names = ('asymmetric',)


def bmatrix(a):
    """Returns a LaTeX bmatrix
    Retrieved from https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']
    return '\n'.join(rv)


def ridge_none_kernel(A, b, lambda_):
    # Make sure data are centralized
    return np.linalg.solve(A.T.dot(A) + lambda_ * np.eye(A.shape[1]), A.T.dot(b))


def get_error_none_kernel(X, w, y):
    return np.mean((X.dot(w) - y) ** 2)


def fit_none_kernel(D, lambda_, train_x, train_y, validation_x, validation_y):
    # YOUR CODE TO COMPUTE THE AVERAGE ERROR PER SAMPLE
    errors = np.asarray([0.0, 0.0])
    poly_train_x = polynomial(train_x, D)
    w = ridge_none_kernel(poly_train_x, train_y, lambda_)
    errors[0] = get_error_none_kernel(poly_train_x, w, train_y)
    errors[1] = get_error_none_kernel(polynomial(validation_x, D), w, validation_y)
    return errors, w


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


def kernel_ridge(gram_matrix, b, lambda_):
    # Make sure data are centralized
    return np.linalg.solve(gram_matrix + lambda_ * np.eye(gram_matrix.shape[1]), b)


def get_error_kernel(X, alpha, y):
    return np.mean((X.dot(alpha) - y) ** 2)


def fit_kernel(D, lambda_, train_x, train_y, validation_x, validation_y):
    # YOUR CODE TO COMPUTE THE AVERAGE ERROR PER SAMPLE
    errors = np.asarray([0.0, 0.0])
    alpha = kernel_ridge(kernel(train_x, train_x, D), train_y, lambda_)
    # no need to calculate training error for this question
    # errors[0] = get_error(kernel(train_x, train_x, D), alpha, train_y)
    errors[1] = get_error_kernel(kernel(validation_x, train_x, D), alpha, validation_y)
    return errors, alpha


def kernel(phi_x, phi_z, degree):
    nx = phi_x.shape[0]
    nz = phi_z.shape[0]
    gram_matrix = np.zeros((nx, nz))
    for i in range(nx):
        xi = phi_x[i, :]
        for j in range(nz):
            xj = phi_z[j, :]
            gram_matrix[i, j] = (1 + xi.T.dot(xj)) ** degree
    return gram_matrix


def main():
    np.set_printoptions(precision=11)

    for i_database, name_database in enumerate(data_names):
        # choose the data you want to load
        data = np.load(name_database + '.npz')
        X = data["x"]
        y = data["y"]
        X /= np.max(X)  # normalize the data
        n_train = int(X.shape[0] * SPLIT)
        # X_train_all = X[:n_train:, :]
        X_valid = X[n_train:, :]
        # y_train_all = y[:n_train]
        y_valid = y[n_train:]

        replicates = 10
        log_base = 3
        n_train_10fold = int(math.log(n_train, log_base))
        x_spec = np.zeros(n_train_10fold)
        Evalid_kernel = np.zeros((n_train_10fold, len(KD), len(LAMBDA)))
        Evalid_none_kernel = np.zeros((n_train_10fold, len(KD), len(LAMBDA)))

        for iSample in range(n_train_10fold):
            print(iSample + 1, ' of ', n_train_10fold)
            this_sample_size = log_base ** (iSample + 1)
            x_spec[iSample] = this_sample_size
            # Etrain = np.zeros((len(KD),len(LAMBDA)))
            for iReplicate in range(replicates):
                indexes = np.asarray(np.random.choice(n_train, this_sample_size), dtype=int)
                X_train = X[indexes, :]
                y_train = y[indexes]
                for iD, D in enumerate(KD):
                    # print(D + 1)
                    for ilambda, lmbda in enumerate(LAMBDA):
                        [_, temp], _ = fit_kernel(D, lmbda, X_train, y_train, X_valid, y_valid)
                        Evalid_kernel[iSample, iD, ilambda] += temp
                        [_, temp], _ = fit_none_kernel(D, lmbda, X_train, y_train, X_valid, y_valid)
                        Evalid_none_kernel[iSample, iD, ilambda] += temp

        Evalid_kernel /= replicates
        Evalid_none_kernel /= replicates
        # plt.plot(np.linspace(1, KD, KD), Etrain, label='Average training error')
        plt.figure()
        for iD, D in enumerate(KD):
            for ilambda, lmbda in enumerate(LAMBDA):
                plt.semilogx(x_spec, Evalid_kernel[:, iD, ilambda], label='D=' + str(D) + ', \lambda=' + str(lmbda))
        plt.xlabel('Number of training data')
        plt.ylabel('Average validation error')
        plt.legend()
        plt.savefig('Figure_5d_kernel_semilogx_' + name_database + '.png')
        plt.close()

        plt.figure()
        for iD, D in enumerate(KD):
            for ilambda, lmbda in enumerate(LAMBDA):
                plt.loglog(x_spec, Evalid_kernel[:, iD, ilambda], label='D=' + str(D) + ', \lambda=' + str(lmbda))
        plt.xlabel('Number of training data')
        plt.ylabel('Average validation error')
        plt.legend()
        plt.savefig('Figure_5d_kernel_loglog_' + name_database + '.png')
        plt.close()

        plt.figure()
        for iD, D in enumerate(KD):
            for ilambda, lmbda in enumerate(LAMBDA):
                plt.semilogx(x_spec, Evalid_none_kernel[:, iD, ilambda], label='D=' + str(D) + ', \lambda=' + str(lmbda))
        plt.xlabel('Number of training data')
        plt.ylabel('Average validation error')
        plt.legend()
        plt.savefig('Figure_5d_nokernel_semilogx_' + name_database + '.png')
        plt.close()

        plt.figure()
        for iD, D in enumerate(KD):
            for ilambda, lmbda in enumerate(LAMBDA):
                plt.loglog(x_spec, Evalid_none_kernel[:, iD, ilambda], label='D=' + str(D) + ', \lambda=' + str(lmbda))
        plt.xlabel('Number of training data')
        plt.ylabel('Average validation error')
        plt.legend()
        plt.savefig('Figure_5d_nokernel_loglog_' + name_database + '.png')
        plt.close()


if __name__ == "__main__":
    main()
