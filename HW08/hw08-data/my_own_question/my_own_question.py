import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider


def loss_func(X, w, y):
    return np.mean((X.dot(w) - y) ** 2)


def sgd(X, y, w_actual, threshold, max_iterations, step_size, gd=False):
    if isinstance(step_size, float):
        step_size_func = lambda i: step_size
    else:
        step_size_func = step_size
    # run 10 gradient descent at the same time, for averaging purpose
    # w_guesses stands for the current iterates (for each run)
    n_runs = 1
    # w_guesses = [np.zeros((X.shape[1], 1)) for _ in range(n_runs)]
    # w_guesses = np.random.normal(scale = 10, size = (n_runs, 2, 1))
    w_guesses = - np.tile(w_true.reshape((1,) + w_true.shape), (n_runs, 1, 1))
    w_hist = np.tile(w_guesses[0, :, :], (max_iterations + 1, 1, 1))
    n = X.shape[0]
    error = []
    it = 0
    above_threshold = True
    previous_w = np.array(w_guesses)
    batch_size = 1  # as is pointed out on Piazza
    while it < max_iterations and above_threshold:
        it += 1
        curr_error = 0
        for j in range(len(w_guesses)):
            if gd:
                sample_gradient = 2 / n * (X.T.dot(X).dot(w_guesses[j]) - X.T.dot(y))
            else:
                sample_idxes = np.random.choice(X.shape[0], batch_size, replace=False)
                sampleX = X[sample_idxes, :]
                sampleY = y[sample_idxes, :]
                sample_gradient = 2 / batch_size * (sampleX.T.dot(sampleX).dot(w_guesses[j]) - sampleX.T.dot(sampleY))
            w_guesses[j] -= step_size_func(it) * sample_gradient
            curr_error += np.linalg.norm(w_guesses[j]-w_actual)
        error.append(curr_error / n_runs)
        w_hist[it, :, :] = np.mean(np.asarray(w_guesses), axis=0)
        diff = np.array(previous_w) - np.array(w_guesses)
        diff = np.mean(np.linalg.norm(diff, axis=1))
        above_threshold = (diff > threshold)
        previous_w = np.array(w_guesses)
    w_hist = w_hist[0: it, :, :]
    return w_hist, error


# set a seed
np.random.seed(0)

# X = np.array([[0, 0]])
X = np.random.normal(scale=20, size=(100, 2))

# Theoretical optimal solution
w_true = np.random.normal(scale=10, size=(2, 1))
# w_true = np.array([[-5.0, -5.0]]).T
y = X.dot(w_true)
y2 = y + np.random.normal(scale=5, size=y.shape)

limits = [-10.0, 10.0]
x1_step = np.arange(-10.0, 10.0, 0.1)
x2_step = np.arange(-10.0, 10.0, 0.1)
X1, X2 = np.meshgrid(x1_step, x2_step)
Z = np.array([[loss_func(X, np.array([[X1[i, j], X2[i, j]]]).T, y) for j in range(len(x1_step))] \
             for i in range(len(x2_step))])

plt.figure(figsize=(12.5, 7.5))
ax1 = plt.subplot2grid((8, 2), (0, 0), rowspan=6)
ax2 = plt.subplot2grid((8, 2), (0, 1), rowspan=6)
ax3 = plt.subplot2grid((8, 2), (7, 0), colspan=2)

step_slider = Slider(ax3, 'step', valmin=-6, valmax=-3,
                     valinit=-4.5, valfmt="%2f")
step_slider.valtext.set_text('%3f' % np.power(10.0, step_slider.val))
its = 1000


def slider_update(val):
    amp = np.power(10, val)
    step_slider.valtext.set_text('%3f' % amp)
    plt.sca(ax1)
    plt.cla()
    CS = plt.contour(X1, X2, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('With true w at (%.1f,%.1f)' % (w_true[0, 0], w_true[1, 0]))
    plt.xlabel('w1')
    plt.xlabel('w2')
    plt.xlim(limits)
    plt.ylim(limits)

    w_hist, error = sgd(X, y2, w_true, 1e-5, its, amp)

    plt.plot(w_hist[:, 0, 0], w_hist[:, 1, 0], 'k-')
    plt.plot(w_hist[:, 0, 0], w_hist[:, 1, 0], 'k>')
    plt.plot(w_true[0, 0], w_true[1, 0], 'r.', markersize=10)
    plt.plot(w_hist[0, 0, 0], w_hist[0, 1, 0], 'g.', markersize=10)

    plt.sca(ax2)
    plt.cla()
    plt.plot(error)
    plt.xlabel('iteration')
    plt.xlabel('error')
    plt.title('||w-w_true||_2^2')


# print(plt.axes())
CS = ax1.contour(X1, X2, Z)

ax1.clabel(CS, inline=1, fontsize=10)
ax1.set_title('With true w at (%.1f,%.1f)' % (w_true[0, 0], w_true[1, 0]))
ax1.set_xlabel('w1')
ax1.set_ylabel('w2')
ax1.set_xlim(limits)
ax1.set_ylim(limits)

w_hist, error = sgd(X, y2, w_true, 1e-5, its, np.power(10.0, step_slider.val))

ax1.plot(w_hist[:, 0, 0], w_hist[:, 1, 0], 'k-')
ax1.plot(w_hist[:, 0, 0], w_hist[:, 1, 0], 'k>')
ax1.plot(w_true[0, 0], w_true[1, 0], 'r.', markersize=10)
ax1.plot(w_hist[0, 0, 0], w_hist[0, 1, 0], 'g.', markersize=10)

ax2.plot(error)
ax2.set_xlabel('iteration')
ax2.set_ylabel('error')
ax2.set_title('||w-w_true||_2^2')


# step_slider.valtext.set_visible(True)
step_slider.on_changed(slider_update)

plt.show()
