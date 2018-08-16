import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.io as spio

SPLIT = 0.8
KD = 16  # max D = 16
LAMBDA = 0.001

data_names = ('circle', 'heart', 'asymmetric')

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

def ridge(A, b, lambda_):
    # Make sure data are centralized
    return np.linalg.solve(A.T.dot(A) + lambda_ * np.eye(A.shape[1]), A.T.dot(b))


def get_error(X, w, y):
    return np.mean((X.dot(w) - y) ** 2)


def fit(D, lambda_, train_x, train_y, validation_x, validation_y):
    # YOUR CODE TO COMPUTE THE AVERAGE ERROR PER SAMPLE
    errors = np.asarray([0.0, 0.0])
    poly_train_x = polynomial(train_x, D)
    w = ridge(poly_train_x, train_y, lambda_)
    errors[0] = get_error(poly_train_x, w, train_y)
    errors[1] = get_error(polynomial(validation_x, D), w, validation_y)
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


def get_polyvalue(x0, y0, coefficient, degree):
    N = len(y0)
    result = np.sum(polynomial(np.column_stack([x0, y0]), degree) * np.vstack([coefficient] * N), axis=1)
    return result

def comb(n, k):
    result = 1
    for i in range(k):
        result *= (n - i)
    for i in range(k):
        result /= (k - i)
    return int(np.round(result))


def main():
    np.set_printoptions(precision=11)

    for i_database, name_database in enumerate(data_names):
        # choose the data you want to load
        data = np.load(name_database + '.npz')
        X = data["x"]
        y = data["y"]
        X /= np.max(X)  # normalize the data

        n_train = int(X.shape[0] * SPLIT)
        X_train = X[:n_train:, :]
        X_valid = X[n_train:, :]
        y_train = y[:n_train]
        y_valid = y[n_train:]
        Etrain = np.zeros(KD)
        Evalid = np.zeros(KD)
        print(r'\textbf{' + name_database + r'}\\')
        for D in range(KD):
            # print(D + 1)
            [Etrain[D], Evalid[D]], weights = fit(D + 1, LAMBDA, X_train, y_train, X_valid, y_valid)
            heatplt = heatmap(X, y, weights, D + 1, clip=5)
            heatplt.title(name_database + '_D' + str(D+1))
            heatplt.savefig('Figure_5b_heatmap_' + name_database + '_D' + str(D+1) + '.png')

        print('Average training error:')
        print('\[')
        print(bmatrix(Etrain))
        print('\]')
        print('Average validation error:')
        print('\[')
        print(bmatrix(Evalid))
        print('\]')
        print('\includegraphics[scale=0.9]{Figure_5b_mse_'+ name_database + r'}\\')

        plt.figure()
        plt.plot(np.linspace(1, KD, KD), Etrain, label='Average training error')
        plt.plot(np.linspace(1, KD, KD), Evalid, label='Average validation error')
        plt.xlabel('Degree of polynomial')
        plt.ylabel('Average error')
        plt.title(name_database)
        plt.legend()
        plt.savefig('Figure_5b_mse_'+ name_database +'.png')

def heatmap(X, y, coef, degree, clip=5):
    # example: heatmap(lambda x, y: x * x + y * y)
    # clip: clip the function range to [-clip, clip] to generate a clean plot
    #   set it to zero to disable this function

    xx = yy = np.linspace(np.min(X), np.max(X), 72)
    x0, y0 = np.meshgrid(xx, yy)
    x0, y0 = x0.ravel(), y0.ravel()
    z0 = get_polyvalue(x0, y0, coef, degree)
    # print(z0)

    if clip:
        z0[z0 > clip] = clip
        z0[z0 < -clip] = -clip

    plt.figure()
    plt.hexbin(x0, y0, C=z0, gridsize=50, cmap=cm.jet, bins=None)
    plt.colorbar()
    cs = plt.contour(
        xx, yy, z0.reshape(xx.size, yy.size), [-2, -1, -0.5, 0, 0.5, 1, 2], cmap=cm.jet)
    plt.clabel(cs, inline=1, fontsize=10)

    pos = y[:] == +1.0
    neg = y[:] == -1.0
    plt.scatter(X[pos, 0], X[pos, 1], c='red', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], c='blue', marker='v')
    # plt.show()
    return plt

if __name__ == "__main__":
    main()
