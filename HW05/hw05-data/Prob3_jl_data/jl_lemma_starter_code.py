import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import sklearn.linear_model
from sklearn.model_selection import train_test_split


######## PROJECTION FUNCTIONS ##########

## Random Projections ##
def random_matrix(d, k):
    """
    d = original dimension
    k = projected dimension
    """
    return 1. / np.sqrt(k) * np.random.normal(0, 1, (d, k))


def random_proj(X, k):
    _, d = X.shape
    return X.dot(random_matrix(d, k))


## PCA and projections ##
def my_pca(X, k):
    """
    compute PCA components
    X = data matrix (each row as a sample)
    k = #principal components
    """
    n, d = X.shape
    assert (d >= k)
    _, _, Vh = np.linalg.svd(X)
    V = Vh.T
    return V[:, :k]


def pca_proj(X, k):
    """
    compute projection of matrix X
    along its first k principal components
    """
    P = my_pca(X, k)
    # P = P.dot(P.T)
    return X.dot(P)


######### LINEAR MODEL FITTING ############

def rand_proj_accuracy_split(X, y, k):
    """
    Fitting a k dimensional feature set obtained
    from random projection of X, versus y
    for binary classification for y in {-1, 1}
    """
    # test train split
    _, d = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # random projection
    J = np.random.normal(0., 1., (d, k))
    rand_proj_X = X_train.dot(J)

    # fit a linear model
    line = sklearn.linear_model.LinearRegression(fit_intercept=False)
    line.fit(rand_proj_X, y_train)

    # predict y
    y_pred = line.predict(X_test.dot(J))

    # return the test error
    return 1 - np.mean(np.sign(y_pred) != y_test)


def pca_proj_accuracy(X, y, k):
    """
    Fitting a k dimensional feature set obtained
    from PCA projection of X, versus y
    for binary classification for y in {-1, 1}
    """

    # test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # pca projection
    P = my_pca(X_train, k)
    P = P.dot(P.T)
    pca_proj_X = X_train.dot(P)

    # fit a linear model
    line = sklearn.linear_model.LinearRegression(fit_intercept=False)
    line.fit(pca_proj_X, y_train)

    # predict y
    y_pred = line.predict(X_test.dot(P))

    # return the test error
    return 1 - np.mean(np.sign(y_pred) != y_test)


######## LOADING THE DATASETS #########
np.random.seed(0) # seed the random number generator
n_dataset = 3  # the number of data set
n_trials = 10  # to average for accuracies over random projections
default_k = 2  # the default number of principle component to keep
for iData in range(1, n_dataset + 1):
    # to load the data:
    data = np.load('data' + str(iData) + '.npz')
    X = data['X']
    y = data['y']
    n, d = X.shape

    ######### YOUR CODE GOES HERE ##########

    # Using PCA and Random Projection for:
    # Visualizing the datasets
    pcaXk = pca_proj(X, default_k)
    randomXk = random_proj(X, default_k)
    one_ids = y == 1
    neg_one_ids = y == -1

    plt.figure()
    plt.scatter(pcaXk[one_ids, 0], pcaXk[one_ids, 1], c='r', marker='o', label='y=1')
    plt.scatter(pcaXk[neg_one_ids, 0], pcaXk[neg_one_ids, 1], c='b', marker='o', label='y=-1')
    plt.legend(loc='upper right')
    plt.title("Top " + str(default_k) + " component(s) of PCA in dataset" + str(iData))
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.savefig("Figure_3h_visual_pca_data" + str(iData) + ".png")
    plt.close()

    plt.figure()
    plt.scatter(randomXk[one_ids, 0], randomXk[one_ids, 1], c='r', marker='o', label='y=1')
    plt.scatter(randomXk[neg_one_ids, 0], randomXk[neg_one_ids, 1], c='b', marker='o', label='y=-1')
    plt.title("Top " + str(default_k) + " component(s) of random projection in dataset" + str(iData))
    plt.legend(loc='upper right')
    plt.xlabel("random projection 1")
    plt.ylabel("random projection 2")
    plt.savefig("Figure_3h_visual_random_data" + str(iData) + ".png")
    plt.close()

    # Computing the accuracies over different datasets.
    rand_accuracies = np.zeros(d)
    pca_accuracies = np.zeros(d)
    for k in range(d):
        for iTry in range(n_trials):
            rand_accuracies[k] += rand_proj_accuracy_split(X, y, k + 1)
            pca_accuracies[k] += pca_proj_accuracy(X, y, k + 1)

    rand_accuracies /= n_trials
    pca_accuracies /= n_trials

    plt.figure()
    plt.plot(np.linspace(1, d, d), pca_accuracies, c='r', marker='.', markersize=20, label='PCA')
    plt.plot(np.linspace(1, d, d), rand_accuracies, c='b', marker='.', markersize=20, label='Random Projection')
    plt.title("Number of component Vs Accuracy in dataset" + str(iData))
    plt.legend(loc='lower right')
    plt.xlabel("number of component")
    plt.ylabel("accuracy")
    plt.savefig("Figure_3i_numVsAccuracy_data" + str(iData) + ".png")
    plt.close()
    # Don't forget to average the accuracy for multiple
    # random projections to get a smooth curve.

    # And computing the SVD of the feature matrix
    Sigma = np.linalg.svd(X, compute_uv=False)
    plt.figure()
    plt.plot(np.linspace(1, len(Sigma), len(Sigma)), Sigma, c='k', marker='.', markersize=20, label='Singular Values')
    plt.title("Singular Values in dataset" + str(iData))
    plt.legend(loc='upper right')
    plt.xlabel("PCA components")
    plt.ylabel("Singular Values")
    plt.savefig("Figure_3j_SingularValues_data" + str(iData) + ".png")
    plt.close()

    ######## YOU CAN PLOT THE RESULTS HERE ########

    # plt.plot, plt.scatter would be useful for plotting
