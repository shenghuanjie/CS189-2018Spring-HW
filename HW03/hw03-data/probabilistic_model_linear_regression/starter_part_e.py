import numpy as np
import matplotlib.pyplot as plt
import os

sample_size = [5, 25, 125, 625]
max_sample_size = sample_size[-1] * 2  # make sure we have straight lines
w = 1
np.random.seed(0)
count = 1
for k in range(4):
    n = sample_size[k]

    # generate data
    # np.linspace, np.random.normal and np.random.uniform might be useful functions
    X = np.random.uniform(0, 1, n)
    Z = np.random.uniform(-0.5, 0.5, n)
    Y = X * w + Z
    # make sure W covers the boundaries
    W = np.hstack([np.linspace(0, np.max((Y - 0.5) / X), max_sample_size),
                   np.linspace(np.max((Y - 0.5) / X), np.min((Y + 0.5) / X), max_sample_size),
                   np.linspace(np.min((Y + 0.5) / X), 2, max_sample_size)])
    N = len(W)
    likelihood = np.ones(N)  # likelihood as a function of w

    for i1 in range(N):
        if np.count_nonzero(abs(Y - X * W[i1]) <= 0.5) == n:
            likelihood[i1] = 1
        else:
            likelihood[i1] = 0
        # compute likelihood
    likelihood /= sum(likelihood)  # normalize the likelihood

    plt.figure()
    # plotting likelihood for different n
    plt.plot(W, likelihood)
    plt.xlabel('w', fontsize=10)
    plt.title('w=' + str(w) + 'n=' + str(n), fontsize=14)
    plt.show(block=False)
    filename = 'Figure_2e' + str(count) + '.png'
    savepath = os.path.join('.', filename)
    plt.savefig(savepath)
    count += 1


