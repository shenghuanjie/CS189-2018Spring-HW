import numpy as np
import scipy.stats as scipystats
import matplotlib.pyplot as plt
import os

sample_size = [5, 25, 125]
Sigma = [1, 2]
WtrueAll = np.zeros((len(Sigma), 2))
Wmin = -4
Wmax = 4
count = 1
np.random.seed(0)
for iw in range(len(Sigma)):
    Wtrue = scipystats.truncnorm(
        Wmin / 2 / Sigma[iw], Wmax / 2 / Sigma[iw], loc=0, scale=Sigma[iw]).rvs(2)
    WtrueAll[iw, :] = Wtrue
    for k in range(len(sample_size)):
        n = sample_size[k]
        # generate data
        # np.linspace, np.random.normal and np.random.uniform might be useful functions
        X = np.random.normal(0, 1, (n, 2))
        Z = np.random.normal(0, 1, n)
        Y = X.dot(Wtrue) + Z

        # compute likelihood
        N = 1001
        W0s = np.linspace(Wmin, Wmax, N)
        W1s = np.linspace(Wmin, Wmax, N)

        likelihood = np.ones([N, N])  # likelihood as a function of w_1 and w_0

        for i1 in range(N):
            w_0 = W0s[i1]
            tempW0 = Y - X[:, 0] * w_0
            for i2 in range(N):
                w_1 = W1s[i2]
                temp = tempW0 - X[:, 1] * w_1
                # for i in range(n):
                # compute the likelihood here
                likelihood[i1, i2] = (1 / np.sqrt(2 * np.pi)) ** n \
                                     * np.exp(- np.linalg.norm(temp) / 2) * \
                                     (1 / np.sqrt(2 * np.pi) / Sigma[iw]) \
                                     * np.exp(- (w_0 ** 2 + w_1 ** 2) / 2 / (Sigma[iw] ** 2))

        # likelihood /= np.sum(likelihood)
        # plotting the likelihood
        plt.figure()
        # for 2D likelihood using imshow
        plt.imshow(likelihood, cmap='hot', aspect='auto', origin='lower', extent=[Wmin, Wmax, Wmin, Wmax])
        plt.title('W = [%.2f, %.2f] from N(0, %d) with %d samples' % (Wtrue[0], Wtrue[1], Sigma[iw] ** 2, n))
        plt.xlabel('w0')
        plt.ylabel('w1')
        plt.show(block=False)
        filename = 'Figure_2i' + str(count) + '.png'
        savepath = os.path.join('.', filename)
        plt.savefig(savepath)
        count += 1

print(WtrueAll)
