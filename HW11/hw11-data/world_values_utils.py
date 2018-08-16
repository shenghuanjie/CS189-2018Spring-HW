from __future__ import print_function
import pandas as pd
from math import sqrt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


# This must be the first statement before other statements.
# You may only put a quoted or triple quoted string,
# Python comments, other future statements, or blank lines before the __future__ line.

import builtins as __builtin__

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
            if '_' in arg:
                arg.replace('_', r'\_')
            if '\\' in arg:
                arg.replace('\\', r' \textbackslash ')
            tempargs[iarg] = arg
        else:
            tempargs[iarg] = str(arg).replace('\\', r' \textbackslash ').replace('_', r'\_')
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
    rv =[r'\[']
    rv += [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']
    rv += [r'\]']
    return '\n'.join(rv)


def btabu(a):
    nCol = len(a.columns)
    rv = [r'\begin{tabu} to 1.0\textwidth {  '+ '|X[c] ' * (nCol + 1) + '| }']
    rv += [r'\hline']
    currentRow = ' '
    for idx, column in enumerate(a.columns):
        currentRow += ' & ' + column
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


def import_world_values_data():
    """
    Reads the world values data into data frames.

    Returns:
        values_train: world_values responses on the training set
        hdi_train: HDI (human development index) on the training set
        values_test: world_values responses on the testing set
    """
    values_train = pd.read_csv('world-values-train2.csv')
    values_train = values_train.drop(['Country'], axis=1)
    values_test = pd.read_csv('world-values-test.csv')
    values_test = values_test.drop(['Country'], axis=1)
    hdi_train = pd.read_csv('world-values-hdi-train2.csv')
    hdi_train = hdi_train.drop(['Country'], axis=1)
    return values_train, hdi_train, values_test


def plot_hdi_vs_feature(training_features, training_labels, feature, color, title):
    """
    Input:
    training_features: world_values responses on the training set
    training_labels: HDI (human development index) on the training set
    feature: name of one selected feature from training_features
    color: color to plot selected feature
    title: title of plot to display

    Output:
    Displays plot of HDI vs one selected feature.
    """
    plt.scatter(training_features[feature],
    training_labels['2015'],
    c=color)
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel('HDI')
    #plt.show()
    plt.savefig('Figure_3c-'+ title.split('(')[0] + '.png')
    plt.close()


def calculate_correlations(training_features,
                           training_labels):
    """
    Input:
        training_features: world_values responses on the training set
        training_labels: HDI (human development index) on the training set

    Output:
        Prints correlations between HDI and each feature, separately.
        Displays plot of HDI vs one selected feature.
    """
    # Calculate correlations between HDI and each feature
    correlations = []
    for column in training_features.columns:
        print(column, training_features[column].corr(training_labels['2015']))
        correlations.append(round(training_features[column].corr(training_labels['2015']), 4))
    print('Correlation matrix:')
    print(correlations)
    print()

    # Identify three features
    idxCorr = np.argmax(correlations)
    print('The feature that is most positively corrected with HDI: '+training_features.columns[idxCorr])
    plot_hdi_vs_feature(training_features, training_labels, training_features.columns[idxCorr],
                        'green', 'The most positively correlated feature('+str(correlations[idxCorr])+')')

    idxCorr = np.argmin(correlations)
    print('The feature that is most negatively corrected with HDI: '+training_features.columns[idxCorr])
    plot_hdi_vs_feature(training_features, training_labels, training_features.columns[idxCorr],
                        'green', 'The most negatively correlated feature('+str(correlations[idxCorr])+')')

    idxCorr = np.argmin(np.abs(correlations))
    print('The feature that is least corrected with HDI: '+training_features.columns[idxCorr])
    plot_hdi_vs_feature(training_features, training_labels, training_features.columns[idxCorr],
                        'green', 'The least correlated feature('+str(correlations[idxCorr])+')')

    #plot_hdi_vs_feature(training_features, training_labels, 'Action taken on climate change',
    #                    'green', 'HDI versus ActionTakenClimateChange')

def plot_pca(training_features,
             training_labels,
             training_classes):
    """
    Input:
        training_features: world_values responses on the training set
        training_labels: HDI (human development index) on the training set
        training_classes: HDI class, determined by hdi_classification(), on the training set

    Output:
        Displays plot of first two PCA dimensions vs HDI
        Displays plot of first two PCA dimensions vs HDI, colored by class
    """
    # Run PCA on training_features
    pca = PCA()
    transformed_features = pca.fit_transform(training_features)

    print(training_labels)

    # Plot countries by first two PCA dimensions
    plt.scatter(transformed_features[:, 0],     # Select first column
                transformed_features[:, 1],     # Select second column
                c=training_labels['2015'])
    plt.colorbar(label='Human Development Index')
    plt.title('Countries by World Values Responses after PCA')
    #plt.show()
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.savefig('Figure_3d-PCA_heatmap.png')
    plt.close()

    # Plot countries by first two PCA dimensions, color by class
    training_colors = training_classes.apply(lambda x: 'green' if x else 'red')
    plt.scatter(transformed_features[:, 0],     # Select first column
                transformed_features[:, 1],     # Select second column
                c=training_colors)
    plt.title('Countries by World Values Responses after PCA')
    #plt.show()
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.savefig('Figure_3d-PCA_multicolor.png')
    plt.close()

def hdi_classification(hdi):
    """
    Input:
        hdi: HDI (human development index) value

    Output:
        high HDI vs low HDI class identification
    """
    if 1.0 > hdi >= 0.7:
        return 1.0
    elif 0.7 > hdi >= 0.30:
        return 0.0
    else:
        raise ValueError('Invalid HDI')
