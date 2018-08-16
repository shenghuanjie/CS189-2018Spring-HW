# You may want to install "gprof2dot"
import io
from collections import Counter

import numpy as np
import scipy.io
import sklearn.model_selection
import sklearn.tree
from numpy import genfromtxt
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin

import pydot
import gprof2dot
import os
import builtins as __builtin__
import pandas as pd

eps = 1e-5  # a small number


def print(*args, **kwargs):
    """My custom print() function."""
    # Adding new arguments to the print function signature
    # is probably a bad idea.
    # Instead consider testing if custom argument keywords
    # are present in kwargs
    tempargs = list(args)
    for iarg, arg in enumerate(tempargs):
        if (type(arg).__module__ == np.__name__):
            tempargs[iarg] = bmatrix(arg)
        elif isinstance(arg, pd.DataFrame):
            tempargs[iarg] = btabu(arg)
        elif isinstance(arg, str):
            #if '\\' in arg:
            #    arg = arg.replace('\\', r' \textbackslash ')
            if '_' in arg:
                arg = arg.replace('_', r'\_')
            if '<' in arg:
                arg = arg.replace('<', r'\textless ')
            if '>' in arg:
                arg = arg.replace('>', r'\textgreater ')
            if '<=' in arg:
                arg = arg.replace('<=', r'\le ')
            if '>=' in arg:
                arg = arg.replace('>=', r'\ge ')
            tempargs[iarg] = arg
        else:
            tempargs[iarg] = str(arg).replace('_', r'\_')
    tempargs = tuple(tempargs)
    __builtin__.print(*tempargs, **kwargs, end='')
    __builtin__.print(r' \\')


def bmatrix(a):
    """Returns a LaTeX bmatrix
    Retrieved from https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    a = np.array(a)
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\[']
    rv += [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']
    rv += [r'\]']
    return '\n'.join(rv)


def btabu(a):
    nCol = len(a.columns)
    rv = [r'\begin{tabu} to 1.0\textwidth {  ' + '|X[c] ' * (nCol + 1) + '| }']
    rv += [r'\hline']
    currentRow = ' '
    for idx, column in enumerate(a.columns):
        currentRow += ' & ' + str(column)
    rv += [currentRow + '\\\\']
    for idx, row in a.iterrows():
        currentRow = str(idx) + ' '
        for _, column in enumerate(a.columns):
            currentRow += ' & ' + str(row[column])
        rv += [r'\hline']
        rv += [currentRow + '\\\\']
    rv += [r'\hline']
    rv += [r'\end{tabu}\\']
    return '\n'.join(rv)


class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO implement information gain function
        class_labels = np.unique(y)
        y0, y1 = y[X < thresh], y[X >= thresh]
        n0 = len(y0)
        n1 = len(y1)
        n = len(y)
        entropy = 0.0
        for label in class_labels:
            if np.count_nonzero(y == label) > 0:
                entropy -= np.count_nonzero(y == label) * np.log2(np.count_nonzero(y == label) / n)
                if np.count_nonzero(y0 == label) > 0:
                    entropy += np.count_nonzero(y0 == label) * np.log2(np.count_nonzero(y0 == label) / n0)
                if np.count_nonzero(y1 == label) > 0:
                    entropy += np.count_nonzero(y1 == label) * np.log2(np.count_nonzero(y1 == label) / n1)
        entropy /= n
        return entropy

    @staticmethod
    def gini_impurity(X, y, thresh):
        # TODO implement gini_impurity function
        class_labels = np.unique(y)
        y0, y1 = y[X < thresh], y[X >= thresh]
        n0 = len(y0)
        n1 = len(y1)
        n = len(y)
        gini = n0 + n1
        for label in class_labels:
            gini -= np.count_nonzero(y0 == label) ** 2 / n0
            gini -= np.count_nonzero(y1 == label) ** 2 / n1
        gini /= n
        return -gini

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([
                np.linspace(np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([self.information_gain(X[:, i], y, t) for t in thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat


class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            sklearn.tree.DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO implement function
        for i in range(self.n):
            idx = np.random.choice(len(y), len(y), replace=True)
            X_rand = X[idx]
            y_rand = y[idx]
            self.decision_trees[i].fit(X_rand, y_rand)
        return self

    def predict(self, X):
        yhat = []
        for i in range(self.n):
            yhat.append(self.decision_trees[i].predict(X))
        yhat = np.vstack(yhat)
        return np.round(np.mean(yhat, axis=0))

    def print_common_split(self):
        roots = []
        for _, t in enumerate(self.decision_trees):
            roots.append(str(t.tree_.feature[0]) + '-' + str(t.tree_.threshold[0]))
        counter = Counter(roots)
        first_splits = [(term[0], term[1]) for term in
                        counter.most_common()]
        print('The most common splits at the root node of the tree are: ')
        for i, this_split in enumerate(first_splits):
            name, thold = this_split[0].split('-', 1)
            name = features[int(name)]
            if isinstance(name, bytes):
                name = str(name, "utf-8")
            else:
                name = str(name)
            print(str(i + 1) + '. ("' + name + '")' + '<' + str(thold) + ' (' + str(this_split[1]) + ' trees)')


class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {'max_features': m}
        # TODO implement function
        super().__init__(params=params, n=n)


class BoostedRandomForest(RandomForest):
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        # TODO implement function
        for j in range(self.n):
            idx = np.random.choice(len(y), len(y), replace=True, p=self.w)
            X_rand = X[idx]
            y_rand = y[idx]
            self.decision_trees[j].fit(X_rand, y_rand)
            yhat = self.decision_trees[j].predict(X)
            Ihat = np.array(y != yhat, dtype=np.float)
            ej = sum(self.w * Ihat) / sum(self.w)
            if ej == 0:
                self.n = j + 1
                break
            self.a[j] = 1 / 2 * np.log((1 - ej) / ej)
            self.w[Ihat == 1] *= np.exp(self.a[j])
            self.w[Ihat != 1] *= np.exp(-self.a[j])
            self.w /= sum(self.w)
        return self

    def predict(self, X):
        # TODO implement function
        ypred = []
        yhat = []
        for j in range(self.n):
            yhat.append(self.decision_trees[j].predict(X))
        yhat = np.vstack(yhat)
        for i in range(X.shape[0]):
            class_labels = np.unique(yhat[:, i])
            yscores = []
            for label in class_labels:
                yscores.append(self.a.dot(yhat[:, i] == label))
            ypred.append(yhat[np.argmax(yscores, axis=0), i])
        return ypred


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(features[col] + b'-' + term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(np.float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    if onehot_cols:
        data = np.hstack([np.array(data, dtype=np.float), np.array(onehot_encoding)])
    else:
        data = np.array(data, dtype=np.float)

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode[0]
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", sklearn.model_selection.cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        # counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        # first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        # print("First splits", first_splits)
        roots = []
        for _, t in enumerate(clf.decision_trees):
            roots.append(str(t.tree_.feature[0]) + '-' + str(t.tree_.threshold[0]))
        counter = Counter(roots)
        first_splits = [(term[0], term[1]) for term in
                        counter.most_common()]
        print('The most common splits at the root node of the tree are: ')
        for i, this_split in enumerate(first_splits):
            if i > 0:
                break
            name, thold = this_split[0].split('-', 1)
            name = features[int(name)]
            if isinstance(name, bytes):
                name = str(name, "utf-8")
            else:
                name = str(name)
            print(str(i + 1) + '. ("' + name + '")' + '<' + str(thold) + ' (' + str(this_split[1]) + ' trees)')


if __name__ == "__main__":
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/bin/'
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    dataset='bio1b'
    # Load titanic data
    path_train = 'bio1b_grades.csv'
    data = genfromtxt(path_train, delimiter=',', dtype=None)
    path_test = 'bio1b_grades.csv'
    # test_data = genfromtxt(path_test, delimiter=',', dtype=None)
    y = data[1:, 0]  # label = survived
    class_names = ["Section313", "Section314"]
    features = list(data[0, 1:])

    labeled_idx = np.where(y != b'')[0]
    y = np.array(y[labeled_idx], dtype=np.int)
    X, onehot_features = preprocess(data[1:, 1:])
    X = X[labeled_idx, :]
    # Z, _ = preprocess(test_data[1:, :])
    # assert X.shape[1] == Z.shape[1]
    features = list(data[0, 1:]) + onehot_features

    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # Basic decision tree
    print("\n\nsimplified decision tree")
    dt = DecisionTree(max_depth=3, feature_labels=features)
    dt.fit(X, y)
    # print("Predictions", dt.predict(Z)[:100])
    print("Accuracy", 1 - np.sum(abs(dt.predict(X) - y)) / y.size)

    print("\n\nsklearn's decision tree")
    this_params = params
    this_params['max_depth'] = 3
    clf = sklearn.tree.DecisionTreeClassifier(random_state=0, **this_params)
    clf.fit(X, y)
    evaluate(clf)
    out = io.StringIO()
    sklearn.tree.export_graphviz(
        clf, out_file=out, feature_names=features, class_names=class_names)
    graph = pydot.graph_from_dot_data(out.getvalue())
    pydot.graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)
    print("Accuracy", 1 - np.sum(abs(clf.predict(X) - y)) / y.size)

    # TODO implement and evaluate parts c-h
    # Part e
    print('\n\nBaggedTrees on ' + dataset)
    bt = BaggedTrees(params, n=N)
    bt.fit(X, y)
    evaluate(bt)
    print("Accuracy", 1 - np.sum(abs(bt.predict(X) - y)) / y.size)

    # Part g
    print('\n\nRandomForest on ' + dataset)
    rf = RandomForest(params, n=N, m=np.int(np.sqrt(X.shape[1])))
    rf.fit(X, y)
    evaluate(rf)
    print("Accuracy", 1 - np.sum(abs(rf.predict(X) - y)) / y.size)

    # Part g
    print('\n\nBoostedRandomForest on ' + dataset)
    brf = BoostedRandomForest(params, n=N, m=np.int(np.sqrt(X.shape[1])))
    brf.fit(X, y)
    evaluate(brf)
    print("Accuracy", 1 - np.sum(abs(brf.predict(X) - y)) / y.size)


