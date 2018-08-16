from sklearn.mixture.gaussian_mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[-4, 2], [-2, 1], [-2, 3], [0, 2], [2, -1], [3, -1], [2, -2], [3, -2]])

estimator = GaussianMixture(n_components=2, max_iter=1000, random_state=0, init_params='random')


plt.figure()
plt.plot(X_train[:, 0], X_train[:, 1], 'k.', markersize=25)
plt.savefig('Figure_6-datapoints.png')
plt.close()

colors = ['r', 'b']


# initial means
estimator.means_init = np.array([[0, 0.5], [0.5, 0]])
# Train the other parameters using the EM algorithm.
estimator.fit(X_train)
classes = estimator.predict(X_train)

plt.figure()
for i in range(2):
    plt.plot(X_train[classes == i, 0], X_train[classes == i, 1], colors[i]+'.', markersize=25, label='class'+str(i+1))
    plt.plot(estimator.means_init[i, 0], estimator.means_init[i, 1], colors[i]+'*', markersize=20, label='mean_init')
    plt.plot(np.mean(X_train[classes == i, 0]), np.mean(X_train[classes == i, 1]), colors[i] + 'P', markersize=15, label='mean_init')
plt.title('mean_init='+str(estimator.means_init))
plt.legend()
plt.savefig('Figure_6-EM1.png')
plt.close()

# initial means
estimator.means_init = np.array([[0, 1], [0, -1]])
# Train the other parameters using the EM algorithm.
estimator.fit(X_train)
classes = estimator.predict(X_train)

plt.figure()
for i in range(2):
    plt.plot(X_train[classes == i, 0], X_train[classes == i, 1], colors[i]+'.', markersize=25, label='class'+str(i+1))
    plt.plot(estimator.means_init[i, 0], estimator.means_init[i, 1], colors[i]+'*', markersize=20, label='mean_init')
    plt.plot(np.mean(X_train[classes == i, 0]), np.mean(X_train[classes == i, 1]), colors[i] + 'P', markersize=15, label='mean_init')
plt.title('mean_init=' + str(estimator.means_init))
plt.legend()
plt.savefig('Figure_6-EM2.png')
plt.close()

# initial means
estimator.means_init = np.array([[-1, 1], [1.5, -1]])
# Train the other parameters using the EM algorithm.
estimator.fit(X_train)
classes = estimator.predict(X_train)

plt.figure()
for i in range(2):
    plt.plot(X_train[classes == i, 0], X_train[classes == i, 1], colors[i]+'.', markersize=25, label='class'+str(i+1))
    plt.plot(estimator.means_init[i, 0], estimator.means_init[i, 1], colors[i]+'*', markersize=20, label='mean_init')
    plt.plot(np.mean(X_train[classes == i, 0]), np.mean(X_train[classes == i, 1]), colors[i] + 'P', markersize=15, label='mean_init')
plt.title('mean_init=' + str(estimator.means_init))
plt.legend()
plt.savefig('Figure_6-EM3.png')
plt.close()

# initial means
estimator.means_init = np.array([[-2, 2], [2.5, -1.5]])
# Train the other parameters using the EM algorithm.
estimator.fit(X_train)
classes = estimator.predict(X_train)

plt.figure()
for i in range(2):
    plt.plot(X_train[classes == i, 0], X_train[classes == i, 1], colors[i]+'.', markersize=25, label='class'+str(i+1))
    plt.plot(estimator.means_init[i, 0], estimator.means_init[i, 1], colors[i]+'*', markersize=20, label='mean_init')
    plt.plot(np.mean(X_train[classes == i, 0]), np.mean(X_train[classes == i, 1]), colors[i] + 'P', markersize=15, label='mean_init')
plt.title('mean_init='+str(estimator.means_init))
plt.legend()
plt.savefig('Figure_6-EM4.png')
plt.close()
