import numpy as np
import matplotlib.pyplot as plt

# assign problem parameters
w1 = 1
w0 = 1
sample_size_increment = 1000
sample_sizes = np.arange(sample_size_increment, 15000, sample_size_increment)
degrees = np.arange(1, 15)
error = np.zeros((len(degrees), len(sample_sizes)))

for iN in range(len(sample_sizes)):
    # generate data
    # np.random might be useful
    N = sample_sizes[iN]
    alpha = np.random.uniform(-1, 1, N)
    Z = np.random.normal(0, 1, N)
    Y_true = alpha * w1 + w0
    Y = Y_true + Z
    # fit data with different models
    # np.polyfit and np.polyval might be useful
    for ideg in range(len(degrees)):
        D = degrees[ideg]
        w = np.polyfit(alpha, Y, D)
        y_predicted = np.polyval(w, alpha)
        error[ideg, iN] = np.mean((y_predicted - Y_true) ** 2)

# plotting figures
# sample code

plt.figure(figsize=(7, 10))
plt.subplot(311)
plt.semilogy(degrees, error[:, -1])
plt.xlabel('degree of polynomial')
plt.ylabel('log of error')

plt.subplot(312)
plt.semilogy(sample_sizes, error[-1, :])
plt.xlabel('number of samples')
plt.ylabel('log of error')

plt.subplot(313)
plt.imshow(error, cmap='hot', aspect='auto', interpolation='none', origin='lower',
           extent=[sample_sizes[0] - sample_size_increment / 2, sample_sizes[-1] + sample_size_increment / 2,
                   degrees[0] - 0.5, degrees[-1] + 0.5])
plt.colorbar()
plt.xlabel('number of samples')
plt.ylabel('degree of polynomial')

plt.show()
