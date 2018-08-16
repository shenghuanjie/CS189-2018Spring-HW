import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.io as spio

SPLIT = 0.8
LAMBDA = 0.001
SIGMA = [10, 3, 1, 0.3, 0.1, 0.03]

data_names = ('heart', )

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

def kernel_ridge(gram_matrix, b, lambda_):
    # Make sure data are centralized
    return np.linalg.solve(gram_matrix + lambda_ * np.eye(gram_matrix.shape[1]), b)


def get_error(X, alpha, y):
    return np.mean((X.dot(alpha) - y) ** 2)


def fit(sigma, lambda_, train_x, train_y, validation_x, validation_y):
    # YOUR CODE TO COMPUTE THE AVERAGE ERROR PER SAMPLE
    errors = np.asarray([0.0, 0.0])
    alpha = kernel_ridge(gaussian_kernel(train_x, train_x, sigma), train_y, lambda_)
    errors[0] = get_error(gaussian_kernel(train_x, train_x, sigma), alpha, train_y)
    errors[1] = get_error(gaussian_kernel(validation_x, train_x, sigma), alpha, validation_y)
    return errors, alpha


def gaussian_kernel(phi_x, phi_z, sigma):
    nx = phi_x.shape[0]
    nz = phi_z.shape[0]
    gram_matrix = np.zeros((nx, nz))
    for i in range(nx):
        xi = phi_x[i, :]
        for j in range(nz):
            xj = phi_z[j, :]
            gram_matrix[i, j] = np.sum((xi - xj) ** 2)
    gram_matrix = np.exp(- gram_matrix/(2 * sigma * sigma))
    return gram_matrix


def get_kernelvalue(x0, y0, X_train, coefficient, degree):
    N = len(y0)
    result = np.sum(gaussian_kernel(np.column_stack([x0, y0]), X_train, degree) * np.vstack([coefficient] * N), axis=1)
    return result


def main():
    np.set_printoptions(precision=11)
    len_sigma = len(SIGMA)
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
        Etrain = np.zeros(len_sigma)
        Evalid = np.zeros(len_sigma)
        print(r'\textbf{' + name_database + r'}\\')
        for iSigma, sigma in enumerate(SIGMA):
            # print(D + 1)
            [Etrain[iSigma], Evalid[iSigma]], weights = fit(sigma, LAMBDA, X_train, y_train, X_valid, y_valid)
            heatplt = heatmap(X, y, X_train, weights, sigma, clip=5)
            heatplt.title(name_database + '_sigma=' + str(sigma))
            heatplt.savefig('Figure_5e_heatmap_' + name_database + '_sigma=' + str(sigma) + '.png')
            heatplt = heatmap(X, y, X_train, weights, sigma, clip=0)
            heatplt.title(name_database + '_sigma=' + str(sigma))
            heatplt.savefig('Figure_5e_heatmap_noclip_' + name_database + '_sigma=' + str(sigma) + '.png')

        print('Average training error:')
        print('\[')
        print(bmatrix(Etrain))
        print('\]')
        print('Average validation error:')
        print('\[')
        print(bmatrix(Evalid))
        print('\]')
        print('\includegraphics[scale=0.9]{Figure_5e_mse_'+ name_database + r'}\\')
        plt.figure()
        plt.plot(SIGMA, Etrain, label='Average training error')
        plt.plot(SIGMA, Evalid, label='Average validation error')
        plt.xlabel('Degree of polynomial')
        plt.ylabel('Average error')
        plt.title(name_database)
        plt.legend()
        plt.savefig('Figure_5e_mse_'+ name_database +'.png')


def heatmap(X, y, X_train, coef, degree, clip=5):
    # example: heatmap(lambda x, y: x * x + y * y)
    # clip: clip the function range to [-clip, clip] to generate a clean plot
    #   set it to zero to disable this function

    xx = yy = np.linspace(np.min(X), np.max(X), 72)
    x0, y0 = np.meshgrid(xx, yy)
    x0, y0 = x0.ravel(), y0.ravel()
    z0 = get_kernelvalue(x0, y0, X_train, coef, degree)
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
