# imports
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

np.random.seed(0)

def generate_data(n):
    """
    This function generates data of size n.
    """
    # TODO implement this
    sigma_x = np.sqrt(5)
    sigma_z = 1
    X = np.random.normal(0, sigma_x, (n, 2))
    Z = np.random.normal(0, sigma_z, (n, 1))
    Y = X.dot(np.ones((2, 1))) + Z
    return X, Y


def tikhonov_regression(X, Y, Sigma):
    """
    This function computes w based on the formula of tikhonov_regression.
    """
    # TODO implement this
    w = np.linalg.inv(X.T.dot(X) + (np.linalg.inv(Sigma))).dot(X.T).dot(Y)
    return w


def compute_mean_var(X, Y, Sigma):
    """
    This function computes the mean and variance of the posterior
    """
    # TODO implement this
    sigma = np.linalg.inv(X.T.dot(X) + (np.linalg.inv(Sigma)))
    mu = sigma.dot(X.T).dot(Y)
    mu_w1 = mu[0, 0]
    mu_w2 = mu[1, 0]
    sigma_w1 = np.sqrt(sigma[0, 0])
    sigma_w2 = np.sqrt(sigma[1, 1])
    sigma_w1w2 = (sigma[0, 1] + sigma[1, 0]) / 2
    return mu_w1, mu_w2, sigma_w1, sigma_w2, sigma_w1w2


# Define the sigmas and number of samples to use
Sigmas = [np.array([[1, 0], [0, 1]]), np.array([[1, 0.25], [0.25, 1]]),
          np.array([[1, 0.9], [0.9, 1]]), np.array([[1, -0.25], [-0.25, 1]]),
          np.array([[1, -0.9], [-0.9, 1]]), np.array([[0.1, 0], [0, 0.1]])]
Num_Sample_Range = [5, 500]


##############################################################

def gen_plot():
    """
    This function refreshes the interactive plot.
    """
    plt.sca(ax)
    plt.cla()
    CS = plt.contour(w1_grid, w2_grid, w_posterior, levels=
    np.concatenate([np.arange(0, 0.05, 0.01), np.arange(0.05, 1, 0.05)]))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Sigma' + names[i] + ' with num_data = {}'.format(num_data))


names = [str(i) for i in range(1, len(Sigmas) + 1)]

fig = plt.figure(figsize=(7.5, 7.5))
ax = plt.axes()
plt.subplots_adjust(left=0.15, bottom=0.3)

# define the interactive sliders
sigma_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
sample_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
sigma_slider = Slider(sigma_ax, 'Sigma', valmin=0, valmax=len(Sigmas) - 1e-5,
                      valinit=0, valfmt="%d")
num_data_slider = Slider(sample_ax, 'Num Samples', valmin=Num_Sample_Range[0],
                         valmax=Num_Sample_Range[1], valinit=Num_Sample_Range[0],
                         valfmt="%d")
sigma_slider.valtext.set_visible(False)
num_data_slider.valtext.set_visible(False)

# initial settings for plot
num_data = Num_Sample_Range[0]
Sigma = Sigmas[0]
i = 0
w1 = np.arange(0.5, 1.5, 0.01)
w2 = np.arange(0.5, 1.5, 0.01)
w1_grid, w2_grid = np.meshgrid(w1, w2)
# Generate the function values of bivariate normal.
X, Y = generate_data(num_data)
mu_w1, mu_w2, sigma_w1, sigma_w2, sigma_w1w2 = compute_mean_var(X, Y, Sigma)
w_posterior = matplotlib.mlab.bivariate_normal(w1_grid, w2_grid, sigma_w1, sigma_w2, mu_w1, mu_w2, sigma_w1w2)


def sigma_update(val):
    """
    This function is called in response to interaction with the Sigma sliding bar.
    """
    global w_posterior, i
    if val != -1:
        i = int(val)
    Sigma = Sigmas[i]
    mu_w1, mu_w2, sigma_w1, sigma_w2, sigma_w1w2 = compute_mean_var(X, Y, Sigma)
    w_posterior = matplotlib.mlab.bivariate_normal(w1_grid, w2_grid, sigma_w1, sigma_w2, mu_w1, mu_w2, sigma_w1w2)
    gen_plot()


def num_sample_update(val):
    """
    This function is called in response to interaction with the number of samples sliding bar.
    """
    global X, Y, num_data
    max_val = Num_Sample_Range[1]
    min_val = Num_Sample_Range[0]
    r = max_val - min_val
    num_data_ = int(((val - min_val) / r) ** 2 * r + min_val)
    if num_data == num_data_:
        return
    num_data = num_data_
    X, Y = generate_data(num_data)
    sigma_update(-1)


sigma_slider.on_changed(sigma_update)
num_data_slider.on_changed(num_sample_update)

gen_plot()
plt.show()
