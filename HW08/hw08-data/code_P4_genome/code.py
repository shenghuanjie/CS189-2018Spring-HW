import numpy as np
import scipy as sp
import scipy.stats as st
import scipy.interpolate
import scipy.linalg as la
import pylab as pl
import math
from scipy import stats
from scipy.stats import multivariate_normal, probplot
from sklearn import preprocessing
import statsmodels.api as sm
import pickle

import scipy as SP
import scipy.optimize as opt
import pylab
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

# import helper functions from starter code
from gwas import *


# generate 3 distributions
n_points =  2000
unif = np.random.uniform(0, 1, n_points)
skewed_left = np.random.exponential(scale=0.2, size=n_points)
skewed_right = 1-np.random.exponential(scale=0.3, size=n_points)

fig, axes = plt.subplots(nrows=3, ncols=1)
fig.set_figheight(10)
fig.set_figwidth(8)

# first subplot
axes[0].hist(unif)
axes[0].set_title('Uniform distribution')
axes[0].set_xlim(0, 1)

# second subplot
axes[1].hist(skewed_left)
axes[1].set_title('Left-skewed distribution')
axes[1].set_xlim(0, 1)

# third subplot
axes[2].hist(skewed_right)
axes[2].set_title('Right-skewed distribution')
axes[2].set_xlim(0, 1)

def qqplot(empirical_dist, legend = ""):
    """
    Generates two qq-plots one in the original scale of the data and
    one using a negative log transformation to make the difference
    between the expected and actual behavior more visible.

    If the observed line is outside the grey region denoted by the error bars,
    then it is quite unlikely that our data came from a Unif[0, 1] distribution.
    """
    x, y = probplot(empirical_dist, dist=stats.distributions.uniform(), fit=False)
    plt.figure()
    plt.scatter(x, y, c='r', alpha=1, )
    plt.xlabel("")
    plt.ylabel("")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Empirical Quantiles")
    plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), "--", label='x=y')
    plt.legend()
    # plt.show()
    fastlmm_qqplot(empirical_dist, legend=legend, xlim=(0,3.6), ylim=(0,3.6), fixaxes=False)
    return plt


# SOLUTION CELL
plt = qqplot(skewed_left)
plt.savefig('Figure_4a')

# SOLUTION CELL
plt = qqplot(skewed_right)
plt.savefig('Figure_4b')

# SOLUTION CELL
qqplot(unif)


# Part b

# import all the data
X = load_X()
y=load_y()

#Normalize the columns
X = preprocessing.normalize(X, norm='l2', axis = 0)
y = y.reshape(-1, 1)


# SOLUTION CELL
# Note that the function naive_model returns the empirical distribution of p-values computed by the naive model
# Green lines indicate error bars.
naive_pvals = naive_model(X,y)
plt.show()
qqplot(naive_pvals)

# If our data were iid we would expect to see a correlation matrix to look something like this
X_r = np.random.randn(X.shape[0], X.shape[1])

corr_matrix = np.corrcoef(X_r)
plt.grid(False)
plt.imshow(corr_matrix, "hot")
plt.colorbar()
plt.title("What we should see if our data were IID")

# Now let's see what we observe in our plot

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def get_projection(X,k):
    _, _, V = np.linalg.svd(X)
    V = V.T
    return V[:, :k]

def corr_matrix_data(X):
    # demean the data
    X_demeaned = X - np.mean(X, 0)

    # get the projection matrix
    proj_matrix = get_projection(X_demeaned, 3)

    # get projected data
    X_proj = X_demeaned @ proj_matrix

    # do some k-means clustering to identify which points are in which cluster
    km = KMeans(n_clusters = 3)
    clusters = km.fit_predict(X_proj)

    # sort data based on identified clusters
    t = pd.DataFrame(X)
    t['cluster'] = clusters
    t = t.sort_values("cluster")
    t = t.drop("cluster", 1)
    plt.imshow(np.corrcoef(t.as_matrix()), "hot")
    plt.colorbar()
    plt.grid(False)

corr_matrix_data(X)

# Let's project our data in 2 dimensions
X_demeaned = X - np.mean(X, 0)
proj_matrix_2d = get_projection(X_demeaned, 2)
X_proj2d = X_demeaned @ proj_matrix_2d
plt.scatter(X_proj2d[:, 0], X_proj2d[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")

# Now let's project on 3 dimensions
from mpl_toolkits.mplot3d import Axes3D

proj_matrix_3d = get_projection(X_demeaned, 3)
X_proj3d = X_demeaned @ proj_matrix_3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = X_proj3d[:, 0]
ys = X_proj3d[:, 1]
zs = X_proj3d[:, 2]

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

ax.scatter(xs, ys, zs)


#Part C


"""
remember that the function pca_corrected_model returns the 
empirical distribution of the p-values for the pca-corrected model
"""


pca_model_pvals = pca_corrected_model(X, y)
qqplot(pca_model_pvals)

# Now, let's see if there is any patterns in our correlation matrix after we have removed the
# 3 largest directions of variance identified by our PCA
X_new = X - X_proj3d  @ proj_matrix_3d.T
# change the scale a bit so that patterns are more clearly visible
corr_matrix = abs(np.corrcoef(X_new))
plt.imshow(np.log(corr_matrix+0.0001))

# SOLUTION CELL
lmm_pvals = lmm()
qqplot(lmm_pvals)
