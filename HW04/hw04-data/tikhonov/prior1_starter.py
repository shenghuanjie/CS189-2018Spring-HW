import matplotlib
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def generate_data(n):
    """
    This function generates data of size n.
    """
    # TODO implement this
    sigma_x = 5
    sigma_z = 1
    X = np.random.normal(0, np.sqrt(sigma_x), (n, 2))
    Z = np.random.normal(0, np.sqrt(sigma_z), (n, 1))
    y = X.dot(np.ones((2, 1))) + Z
    return (X, y)


def tikhonov_regression(X, Y, Sigma):
    """
    This function computes w based on the formula of tikhonov_regression.
    """
    # TODO implement this
    w = np.linalg.inv(X.T.dot(X) + (np.linalg.inv(Sigma))).dot(X.T).dot(Y)
    return w


def compute_mean_var(X, y, Sigma):
    """
    This function computes the mean and variance of the posterior
    """
    # TODO implement this
    sigma = np.linalg.inv(X.T.dot(X) + (np.linalg.inv(Sigma)))
    mu = sigma.dot(X.T).dot(Y)
    mux = mu[0, 0]
    muy = mu[1, 0]
    sigmax = np.sqrt(sigma[0, 0])
    sigmay = np.sqrt(sigma[1, 1])
    sigmaxy = (sigma[0, 1] + sigma[1, 0]) / 2
    return mux, muy, sigmax, sigmay, sigmaxy


Sigmas = [np.array([[1, 0], [0, 1]]), np.array([[1, 0.25], [0.25, 1]]),
          np.array([[1, 0.9], [0.9, 1]]), np.array([[1, -0.25], [-0.25, 1]]),
          np.array([[1, -0.9], [-0.9, 1]]), np.array([[0.1, 0], [0, 0.1]])]
names = [str(i) for i in range(1, 6 + 1)]

for num_data in [5, 50, 500]:
    X, Y = generate_data(num_data)
    for i, Sigma in enumerate(Sigmas):
        mux, muy, sigmax, sigmay, sigmaxy = compute_mean_var(X, Y,
                                                             Sigma)  # TODO compute the mean and covariance of posterior.

        x = np.arange(0.5, 1.5, 0.01)
        y = np.arange(0.5, 1.5, 0.01)
        X_grid, Y_grid = np.meshgrid(x, y)

        # TODO Generate the function values of bivariate normal.
        Z = matplotlib.mlab.bivariate_normal(X_grid, Y_grid, sigmax, sigmay, mux, muy, sigmaxy)

        # plot
        plt.figure(figsize=(10, 10))
        CS = plt.contour(X_grid, Y_grid, Z,
                         levels=np.concatenate([np.arange(0, 0.05, 0.01), np.arange(0.05, 1, 0.05)]))
        plt.clabel(CS, inline=1, fontsize=10)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Sigma' + names[i] + ' with num_data = {}'.format(num_data))
        plt.savefig('Figure_3d_Sigma' + names[i] + '_num_data_{}.png'.format(num_data))
