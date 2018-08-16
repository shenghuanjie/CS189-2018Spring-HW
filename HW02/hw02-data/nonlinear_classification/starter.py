import numpy as np
import matplotlib.pyplot as plt

Xtrain = np.load("Xtrain.npy")
ytrain = np.load("ytrain.npy")
threshold_label = 0


def visualize_dataset(X, y):
    plt.scatter(X[y < 0.0, 0], X[y < 0.0, 1])
    plt.scatter(X[y > 0.0, 0], X[y > 0.0, 1])
    plt.show()


# visualize the dataset:
 visualize_dataset(Xtrain, ytrain)

# TODO: solve the linear regression on the training data
w = np.linalg.lstsq(Xtrain, ytrain)[0]

Xtest = np.load("Xtest.npy")
ytest = np.load("ytest.npy")


# TODO: report the classification accuracy on the test set
def getNcorrect(X, w, y, threshold):
    y_predicted = X.dot(w)
    ncorrect = np.count_nonzero(
        np.logical_or(
            (y_predicted > threshold) == (y > threshold),
            (y_predicted <= threshold) == (y <= threshold)
        )
    )
    return ncorrect


correct_point = getNcorrect(Xtest, w, ytest, threshold_label)
print("(6b) OLS: %d out of %d points (%.2f%%) are correctly classified" % (
    correct_point, ytest.size, correct_point / ytest.size * 100))

# TODO: Create a matrix Phi_train with polynomial features from the training data
# and solve the linear regression on the training data

Xtrain_poly = np.vstack([
    Xtrain[:, 0],
    Xtrain[:, 1],
    Xtrain[:, 0] ** 2,
    Xtrain[:, 0] * Xtrain[:, 1],
    Xtrain[:, 1] ** 2,
    np.ones(ytrain.size),
]).T

Xtest_poly = np.vstack([
    Xtest[:, 0],
    Xtest[:, 1],
    Xtest[:, 0] ** 2,
    Xtest[:, 0] * Xtest[:, 1],
    Xtest[:, 1] ** 2,
    np.ones(ytest.size),
]).T

w_poly = np.linalg.lstsq(Xtrain_poly, ytrain)[0]

# TODO: Create a matrix Phi_test with polynomial features from the test data
# and report the classification accuracy on the test set

correct_point = getNcorrect(Xtest_poly, w_poly, ytest, threshold_label)
print("(6c) PolyFit: %d out of %d points (%.2f%%) are correctly classified" % (
    correct_point, ytest.size, correct_point / ytest.size * 100))

a, b, c, d, e, f= np.squeeze(w_poly)
print("(6d) {:.3f} x + {:.3f} y + {:.3f} x^2 + {:.3f} xy + {:.3f} y^2 + {:.3f} = 0".format(a, b, c, d, e, f))
