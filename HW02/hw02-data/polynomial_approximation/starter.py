import numpy as np
import matplotlib.pyplot as plt
import os
import math

plot_col = ['r', 'g', 'b', 'k', 'm']
plot_mark = ['o', '^', 'v', 'D', 'x', '+']

# Plots the rows in 'ymat' on the y-axis vs. 'xvec' on the x-axis
# with labels 'ylabels'
# and saves figure as pdf to 'dirname/filename' 
def plotmatnsave(ymat, xvec, ylabels, dirname, filename):
    no_lines = len(ymat)
    fig = plt.figure(0)

    if len(ylabels) > 1:
        for i in range(no_lines):
            xs = np.array(xvec)
            ys = np.array(ymat[i])
            plt.plot(xs, ys, color = plot_col[i % len(plot_col)], lw=1, label=ylabels[i])
        
        lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

    savepath = os.path.join(dirname, filename)
    plt.xlabel('$x$', labelpad=10)
    plt.ylabel('$f(x)$', labelpad=10)
    plt.savefig(savepath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

# Sets the labels
labels = ['$e^x$', '1st order', '2nd order', '3rd order', '4th order']

# TODO: Given x values in "x_vec", save the respective function values e^x,
# and its first to fourth degree Taylor approximations
# as rows in the matrix "y_mat"
x_vec = np.linspace(-4, 3)
y_mat = np.vstack([np.exp(x_vec), np.ones(x_vec.size) + x_vec,
                   np.ones(x_vec.size) + x_vec + x_vec * x_vec / 2,
                   np.ones(x_vec.size) + x_vec + x_vec * x_vec / 2 + x_vec * x_vec * x_vec / 6,
                   np.ones(x_vec.size) + x_vec + x_vec * x_vec / 2 + x_vec * x_vec * x_vec / 6 + x_vec * x_vec * x_vec * x_vec / 24])

# Define filename, invoke plotmatnsave
filename = 'approx_plot.pdf'
plotmatnsave(y_mat, x_vec, labels, '.', filename)

# and its first to fourth degree Taylor approximations
# as rows in the matrix "y_mat"
x_vec = np.linspace(-20, 8)
y_mat = np.vstack([np.exp(x_vec), np.ones(x_vec.size) + x_vec,
                   np.ones(x_vec.size) + x_vec + x_vec * x_vec / 2,
                   np.ones(x_vec.size) + x_vec + x_vec * x_vec / 2 + x_vec * x_vec * x_vec / 6,
                   np.ones(x_vec.size) + x_vec + x_vec * x_vec / 2 + x_vec * x_vec * x_vec / 6 + x_vec * x_vec * x_vec * x_vec / 24])

# Define filename, invoke plotmatnsave
filename = 'approx_plot2.pdf'
plotmatnsave(y_mat, x_vec, labels, '.', filename)
