import numpy as np
import matplotlib.pyplot as plt
import os


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


mu = [15, 5]
all_sigmas = np.asarray(([[20, 0], [0, 10]], [[20, 14], [14, 10]], [[20, -14], [-14, 10]]))
np.random.seed(0)
n = 100
vec_one = np.ones((n, 1))
for i in range(len(all_sigmas)):
    sigma = all_sigmas[i]
    samples = np.random.multivariate_normal(mu, sigma, size=n)

    e_mu = samples.T.dot(vec_one) / n
    e_sigma = (samples.T - e_mu.dot(vec_one.T)).dot(samples - vec_one.dot(e_mu.T)) / n

    print('\[')
    print('\Sigma_' + str(i + 1) + '=')
    print(bmatrix(sigma))
    print('\hat{\mu}=')
    print(bmatrix(e_mu))
    print('\hat{\Sigma}=')
    print(bmatrix(e_sigma))
    print('\]\n')
    plt.scatter(samples[:, 0], samples[:, 1], label='Sigma' + str(i + 1))

lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show(block=False)
filename = 'Figure_2c.png'
save_path = os.path.join('.', filename)
plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
