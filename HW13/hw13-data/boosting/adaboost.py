import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from sklearn.datasets import make_sparse_coded_signal
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier

# globals
n_estimators = 200
DT1 = DecisionTreeClassifier(max_depth=1, min_samples_leaf=15)
DT2 = DecisionTreeClassifier(max_depth=2, min_samples_leaf=15)
DT4 = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15)
DT9 = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)

"""Loads the training data from the SPAM dataset used in HW12."""
def load_data():
    # load data
    data = loadmat("datasets/spam_data/spam_data.mat")
    # training data
    data_, labels_ = data["training_data"], np.squeeze(data["training_labels"])
    X_train, y_train = data_, labels_
    # test data
    y_test=[]
    with open("datasets/spam_data/spam_test_labels.txt","r") as f:
        for l in f.readlines():
            y_test.append(int(l.split(",")[1]))
    y_test = np.array(y_test)
    X_test = data['test_data']

    return X_train, y_train, X_test, y_test

"""Runs the maching pursuit algorithm."""
def mp(y, X, w_true, y_test, X_test):
    train_err = []
    test_err = []
    X_ = X
    y = np.copy(y); X = np.copy(X)
    curr = np.copy(y)
    w_est = np.zeros(len(X[0]))
    for j in range(len(X[0])):
        i = np.argmax(np.abs(np.dot(X.T, curr)))
        col = np.copy(X[:,i])
        # use each column only once
        X[:,i] = 0
        w_est[i] = np.dot(col, curr)
        curr = curr - col*w_est[i]
        # error defined here as ||y - D x_hat||_2
        train_err.append(np.linalg.norm(X_.dot(w_est) - y))
        test_err.append(np.linalg.norm(X_test.dot(w_est)-y_test))

    return w_est, train_err, test_err

if __name__ == "__main__":
    ###### CHANGE THESE VARIABLES TO RUN PROBLEM PARTS
    PART_G = True
    PART_H = True
    PART_J = True
    ######
    X_train, y_train, X_test, y_test = load_data()

    ### PART G
    if PART_G:
        fig = plt.figure(figsize=(8,7))
        ax = fig.add_subplot(111)
        styles=["k-", "k--", "k-."]
        depths=[1,2,4]
        j=0
        # for each weak classifier, train it and an AdaBoost instance based on it
        for w in [DT1, DT2,DT4]:
            # Weak classifier
            w.fit(X_train, y_train)
            err = 1.0 - w.score(X_train, y_train)
            ax.plot([1, n_estimators], [err] * 2, styles[j],
                label="Decision Tree, max depth %d (DT%d)" % (depths[j],depths[j]))
            # AdaBoost classifier
            ada = AdaBoostClassifier(base_estimator=w,
                                     n_estimators=n_estimators,
                                     random_state=0)
            
            ada_train_err = np.zeros((n_estimators,))
            ada.fit(X_train, y_train)
            for i, y_pred in enumerate(ada.staged_predict(X_train)):
                ada_train_err[i] = zero_one_loss(y_pred, y_train)

            smoothed = []
            # use moving average filter to smooth plots -- done to make easier
            # to see trends; you are encouraged to also plot 'ada_train_err' to
            # see the actual error plots!!
            for i in range(len(ada_train_err)):
                temp = 0.
                counter = 0.
                for k in range(i-5, i+1):
                    if k >= 0: 
                        temp += ada_train_err[k]
                        counter += 1.
                smoothed.append(temp/counter)

            ax.plot(np.arange(n_estimators) + 1, smoothed, styles[j],
                label="AdaBoost on DT%d" % depths[j],
                color="red")

            j += 1

        ax.set_ylim((0.1, 0.3))
        ax.set_yscale('log')
        ax.set_xlabel("Number of Classifiers (for AdaBoost)")
        ax.set_ylabel("Error [%], log scale")
        ax.set_title("Weak Classifiers and AdaBoost vs. Training Error")
        leg = ax.legend(loc='upper right', fancybox=True)
        leg.get_frame().set_alpha(0.7)
        #plt.show()
        plt.savefig('Figure_2g')
        plt.close()

    ### PART H
    if PART_H:
        fig = plt.figure(figsize=(8,7))
        ax = fig.add_subplot(111)
        # Basline classifier (a "deep" tree)
        DT9.fit(X_train, y_train)
        err = 1.0 - DT9.score(X_test, y_test)
        ax.plot([1, n_estimators], [err] * 2, "k-",
            label="Baseline Classifier -- Decision Tree, max depth 9")
        # AdaBoost
        styles=["k-", "k--", "k-."]
        depths=[1,2,4]
        j=0
        # for each weak classifier, train an AdaBoost instance based on it
        for w in [DT1, DT2, DT4]:

            # AdaBoost classifier
            ada = AdaBoostClassifier(base_estimator=w,
                                     n_estimators=n_estimators,
                                     random_state=0)
            ada_train_err = np.zeros((n_estimators,))

            ada.fit(X_train, y_train)
            for i, y_pred in enumerate(ada.staged_predict(X_test)):
                ada_train_err[i] = zero_one_loss(y_pred, y_test)

            smoothed = []
            # use moving average filter to smooth plots -- done to make easier
            # to see trends; you are encouraged to also plot 'ada_train_err' to 
            # see the actual error plots!!
            for i in range(len(ada_train_err)):
                temp = 0.
                counter = 0.
                for k in range(i-5, i+1):
                    if k >= 0: 
                        temp += ada_train_err[k]
                        counter += 1.
                smoothed.append(temp/counter)

            ax.plot(np.arange(n_estimators) + 1, smoothed, styles[j],
                label="AdaBoost on DT%d" % depths[j],
                color="red")

            j += 1

        ax.set_ylim((0.1, 0.3))
        ax.set_yscale('log')
        ax.set_xlabel("Number of Classifiers (for AdaBoost)")
        ax.set_ylabel("Error [%], log scale")
        ax.set_title("Decision Tree Classifier and AdaBoost vs. Test Error")
        leg = ax.legend(loc='lower right', fancybox=True)
        leg.get_frame().set_alpha(0.7)
        #plt.show()
        plt.savefig('Figure_2h')
        plt.close()

    ### PART J
    if PART_J:
        fig = plt.figure(figsize=(8,7))
        ax = fig.add_subplot(111)

        n_components = 100
        n_features = 30
        n_nonzero_coefs = 5
        # y = Xw; w is a sparse vector
        y_train, X_train, w = make_sparse_coded_signal(n_samples=1,
                                   n_components=n_components,
                                   n_features=n_features,
                                   n_nonzero_coefs=n_nonzero_coefs,
                                   random_state=0)
        # test set
        _, X_test, _ = make_sparse_coded_signal(n_samples=1,
                                   n_components=n_components,
                                   n_features=n_features,
                                   n_nonzero_coefs=n_nonzero_coefs,
                                   random_state=0)
        y_test = np.dot(X_test, w)

        np.random.seed(10)
        y_noised_train = y_train + 2e-1*np.random.randn(len(y_train))
        y_noised_test = y_test + 2e-1*np.random.randn(len(y_test))
        w_est, train_err, test_err = mp(y_noised_train, X_train, w,
                                        y_noised_test, X_test)
        
        ax.plot(np.arange(n_components), test_err, label="Maching Pursuit test error")
        ax.plot(np.arange(n_components), train_err, label="Maching Pursuit train error")
        ax.set_ylim((0., 2.0))
        ax.set_xlabel("Number of features used")
        ax.set_ylabel("Reconstruction error")
        ax.set_title("Maching Pursuit Train and Test Reconstruction Error")
        
        leg = ax.legend(loc='upper right', fancybox=True)
        leg.get_frame().set_alpha(0.7)
        
        #plt.show()
        plt.savefig('Figure_2j')
        plt.close()

