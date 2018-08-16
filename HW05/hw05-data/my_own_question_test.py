import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
n = 10
# create two clusters of data
# mu =[3, 3]
muA = [0, 0]
sigmaA = [[1, 0.5], [0.5, 1]]
samplesA = np.random.multivariate_normal(muA, sigmaA, size=n)
samplesA = samplesA - np.ones((samplesA.shape[0], 1)).dot(
    np.reshape(np.mean(samplesA, axis=0), (1, samplesA.shape[1])))
samplesA /= np.ones((samplesA.shape[0], 1)).dot(
    np.reshape(np.std(samplesA, axis=0), (1, samplesA.shape[1])))

# mu =[-3, 3]
muB = [0, 0]
sigmaB = [[1, -0.5], [-0.5, 1]]
samplesB = np.random.multivariate_normal(muB, sigmaB, size=n)
samplesB = samplesB - np.ones((samplesB.shape[0], 1)).dot(
    np.reshape(np.mean(samplesB, axis=0), (1, samplesB.shape[1])))
samplesB /= np.ones((samplesB.shape[0], 1)).dot(
    np.reshape(np.std(samplesB, axis=0), (1, samplesB.shape[1])))

_, LambdaA, VdA = np.linalg.svd(samplesA)

_, LambdaB, VdB = np.linalg.svd(samplesB)

plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(samplesA[:, 0], samplesA[:, 1], c='b', label='Sample A')
plt.arrow(0, 0, -LambdaA[0] * VdA[0, 0], -LambdaA[0] * VdA[0, 1], head_width=1,
          head_length=1, fc='k', ec='k')
plt.arrow(0, 0, -LambdaA[1] * VdA[1, 0], -LambdaA[1] * VdA[1, 1], head_width=1,
          head_length=1, fc='k', ec='k', linestyle=':')
lgd = plt.legend(bbox_to_anchor=(0., 1.03, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(color='k', linestyle=':', linewidth=1)
plt.title(str(sigmaA))
plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.subplot(1, 2, 2)
plt.scatter(samplesB[:, 0], samplesB[:, 1], c='b', label='Sample B')
plt.arrow(0, 0, -LambdaB[0] * VdB[0, 0], -LambdaB[0] * VdB[0, 1], head_width=1,
          head_length=1, fc='k', ec='k')
plt.arrow(0, 0, -LambdaB[1] * VdB[1, 0], -LambdaB[1] * VdB[1, 1], head_width=1,
          head_length=1, fc='k', ec='k', linestyle=':')
lgd = plt.legend(bbox_to_anchor=(0., 1.03, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(color='k', linestyle=':', linewidth=1)
plt.title(str(sigmaB))
plt.xlim([-5, 5])
plt.ylim([-5, 5])
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show()
