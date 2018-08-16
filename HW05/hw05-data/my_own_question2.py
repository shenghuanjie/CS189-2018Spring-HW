import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
np.random.seed(0)
n = 10
# create two clusters of data
# mu =[3, 3]
mu = [3, 6]
sigma = [[1, 0.5], [0.5, 1]]
samplesA = np.random.multivariate_normal(mu, sigma, size=n)


# mu =[-3, 3]
mu = [6, 3]
sigma = [[1, -0.5], [-0.5, 1]]
samplesB = np.random.multivariate_normal(mu, sigma, size=n)
samplesB = np.abs(samplesB)

samples = np.vstack((samplesA, samplesB))
_, Lambda, Vd = np.linalg.svd(samples)

model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(samples)
H = model.components_

plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(samplesA[:, 0], samplesA[:, 1], c='b', label='Sample A')
plt.scatter(samplesB[:, 0], samplesB[:, 1], c='r', label='Sample B')
plt.arrow(0, 0, -Lambda[0] * Vd[0, 0], -Lambda[0] * Vd[0, 1], head_width=1,
          head_length=1, fc='k', ec='k')
plt.arrow(0, 0, -Lambda[1] * Vd[1, 0], -Lambda[1] * Vd[1, 1], head_width=1,
          head_length=1, fc='k', ec='k', linestyle=':')
lgd = plt.legend(bbox_to_anchor=(0., 1.03, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(color='k', linestyle=':', linewidth=1)
plt.title('Latent Factor Analysis')
plt.xlim([-15, 15])
plt.ylim([-15, 15])

plt.subplot(1, 2, 2)
plt.scatter(samplesA[:, 0], samplesA[:, 1], c='b', label='Sample A')
plt.scatter(samplesB[:, 0], samplesB[:, 1], c='r', label='Sample B')
plt.arrow(0, 0, H[0, 0], H[1, 0], head_width=1,
          head_length=1, fc='k', ec='k')
plt.arrow(0, 0, H[0, 1], H[1, 1], head_width=1,
          head_length=1, fc='k', ec='k', linestyle=':')
lgd = plt.legend(bbox_to_anchor=(0., 1.03, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(color='k', linestyle=':', linewidth=1)
plt.title('Non-negative Matrix Factorization')
plt.xlim([-15, 15])
plt.ylim([-15, 15])
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show()
