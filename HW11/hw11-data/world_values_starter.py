"""
The world_values data set is available online at http://54.227.246.164/dataset/. In the data,
    residents of almost all countries were asked to rank their top 6 'priorities'. Specifically,
    they were asked "Which of these are most important for you and your family?"

This code and world-values.tex guides the student through the process of training several models
    to predict the HDI (Human Development Index) rating of a country from the responses of its
    citizens to the world values data. The new model they will try is k Nearest Neighbors (kNN).
    The students should also try to understand *why* the kNN works well.
"""

from math import sqrt
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from world_values_utils import import_world_values_data
from world_values_utils import hdi_classification
from world_values_utils import calculate_correlations
from world_values_utils import plot_pca
from world_values_utils import print

from world_values_pipelines import ridge_regression_pipeline
from world_values_pipelines import lasso_regression_pipeline
from world_values_pipelines import k_nearest_neighbors_regression_pipeline
from world_values_pipelines import svm_classification_pipeline
from world_values_pipelines import k_nearest_neighbors_classification_pipeline
from world_values_pipelines import tree_classification_pipeline

from world_values_parameters import regression_ridge_parameters
from world_values_parameters import regression_lasso_parameters
from world_values_parameters import regression_knn_parameters
from world_values_parameters import classification_svm_parameters
from world_values_parameters import classification_knn_parameters
from world_values_parameters import classification_tree_parameters

import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from world_values_parameters import regression_knn_parameters_weighted
from world_values_pipelines import k_nearest_neighbors_regression_pipeline_scaled
from world_values_pipelines import k_nearest_neighbors_regression_pipeline_minmax
from world_values_pipelines import k_nearest_neighbors_regression_pipeline_binary
from world_values_pipelines import svm_classification_pipeline_pca_scaled
from world_values_parameters import classification_svm_parameters_pca_scaled
from world_values_parameters import classification_svm_parameters_rbf
from world_values_pipelines import k_nearest_neighbors_classification_pipeline_scaled

def main():
    #print("Predicting HDI from World Values Survey")
    #print()

    # Import Data #
    #print("Importing Training and Testing Data")
    values_train, hdi_train, values_test = import_world_values_data()

    # Center the HDI Values #
    hdi_scaler = StandardScaler(with_std=False)
    hdi_shifted_train = hdi_scaler.fit_transform(hdi_train)

    # Classification Data #
    hdi_class_train = hdi_train['2015'].apply(hdi_classification)

    # Data Information #
    #print('Training Data Count:', values_train.shape[0])
    #print('Test Data Count:', values_test.shape[0])
    #print()

    # Part b and c: Calculate Correlations #
    #calculate_correlations(values_train, hdi_train)

    # Part d, r: PCA #
    #plot_pca(values_train, hdi_train, hdi_class_train)

    # Part e,f,and g: Regression Grid Searches #
    #regression_grid_searches(training_features=values_train,training_labels=hdi_train, title='Figure_3j-RMSEvsk.png')

    # Part i, m, : Nearest neighbors to the US. n_neighbors is 8 because the first entry is the US.
    '''
    nbrs = NearestNeighbors(n_neighbors=8).fit(values_train)
    distances, indices = nbrs.kneighbors([values_train.iloc[45]])
    countries_train = pd.read_csv('world-values-train2.csv')
    countries_train = countries_train['Country']
    print('The nearest countries to the USA: ')
    for i, (dist, idx) in enumerate(zip(distances.T, indices.T)):
        print(str(i) + ': ' + str(countries_train[idx].to_string(index=False)) + '(idx=' + str(idx) + ') at a distance of ' + str(dist))
    print()
    '''

    # Part l
    '''
    print("k Nearest Neighbors Regression")
    _rmse_grid_search(training_features=values_train, training_labels=hdi_train,
                      pipeline=k_nearest_neighbors_regression_pipeline,
                      parameters=regression_knn_parameters_weighted, technique='knn', title='Figure_3l-RMSEvsk.png')
    '''

    # Part m
    '''
    stdscaler = StandardScaler()
    stdscaler.fit(values_train)
    values_train = stdscaler.transform(values_train)
    nbrs = NearestNeighbors(n_neighbors=8).fit(values_train)
    distances, indices = nbrs.kneighbors([values_train[45, :]])
    countries_train = pd.read_csv('world-values-train2.csv')
    countries_train = countries_train['Country']
    print('The nearest countries to the USA (after scaling): ')
    for i, (dist, idx) in enumerate(zip(distances.T, indices.T)):
        print(str(i) + ': ' + str(countries_train[idx].to_string(index=False))
              + '(idx=' + str(idx) + ') at a normalized distance of ' + str(dist))
    print()
    '''

    # Part n
    '''
    print("k Nearest Neighbors Regression")
    _rmse_grid_search(training_features=values_train, training_labels=hdi_train,
                      pipeline=k_nearest_neighbors_regression_pipeline_scaled,
                      parameters=regression_knn_parameters_weighted, technique='knn',
                      title='Figure_3n-RMSEvsk.png')
    '''

    # Part o
    '''
    print("k Nearest Neighbors Regression (MinMaxScaler)")
    _rmse_grid_search(training_features=values_train, training_labels=hdi_train,
                      pipeline=k_nearest_neighbors_regression_pipeline_minmax,
                      parameters=regression_knn_parameters_weighted, technique='knn',
                      title='Figure_3o-RMSEvsk_minmax.png')
    print("k Nearest Neighbors Regression (Binarizer)")
    _rmse_grid_search(training_features=values_train, training_labels=hdi_train,
                      pipeline=k_nearest_neighbors_regression_pipeline_binary,
                      parameters=regression_knn_parameters_weighted, technique='knn',
                      title='Figure_3o-RMSEvsk_binary.png')
    '''
    # Part p
    '''
    print('The current best model is: ')
    grid = _rmse_grid_search(training_features=values_train, training_labels=hdi_train,
                      pipeline=k_nearest_neighbors_regression_pipeline_scaled,
                      parameters=regression_knn_parameters_weighted, technique='knn',
                      title='Figure_3p-RMSEvsk.png')
    pred_values = grid.predict(values_test)
    countries_test = pd.read_csv('world-values-test.csv')
    countries_test = countries_test[['Country']]
    pred_values = pred_values.flatten()
    countries_test['2015'] = pred_values
    countries_test.to_csv('world-values-hdi-test-predictions.csv')
    print(countries_test)
    '''

    # Part t
    '''
    print("SVM Classification")
    _accuracy_grid_search(values_train, hdi_class_train,
                          svm_classification_pipeline,
                          classification_svm_parameters)
    '''

    # Part u
    '''
    print("SVM Classification (scaled PCA)")
    _accuracy_grid_search(values_train, hdi_class_train,
                          svm_classification_pipeline_pca_scaled,
                          classification_svm_parameters_pca_scaled)
    '''

    # Part v
    '''
    print("SVM Classification (RBF kernel)")
    _accuracy_grid_search(values_train, hdi_class_train,
                          svm_classification_pipeline,
                          classification_svm_parameters_rbf)
    '''

    # Part w
    '''
    print("k Nearest Neighbors Classification")
    _accuracy_grid_search(values_train, hdi_class_train,
                          k_nearest_neighbors_classification_pipeline,
                          classification_knn_parameters)
    print('Here is the result after scaling: ')
    print("k Nearest Neighbors Classification (scaled)")
    _accuracy_grid_search(values_train, hdi_class_train,
                          k_nearest_neighbors_classification_pipeline_scaled,
                          classification_knn_parameters)
    '''

    # Part x
    '''
    values_test = pd.read_csv('world-values-test-berkeley.csv')
    values_test = values_test.drop(['Country'], axis=1)
    print('The current best model is: ')
    grid = _rmse_grid_search(training_features=values_train, training_labels=hdi_train,
                             pipeline=k_nearest_neighbors_regression_pipeline_scaled,
                             parameters=regression_knn_parameters_weighted, technique='knn',
                             title='Figure_3x-RMSEvsk.png')
    pred_values = grid.predict(values_test)
    countries_test = pd.read_csv('world-values-test-berkeley.csv')
    countries_test = countries_test[['Country']]
    pred_values = pred_values.flatten()
    countries_test['2015'] = pred_values
    countries_test.to_csv('world-values-test-berkeley-predictions.csv')
    print(countries_test)
    '''
    # Part j, l, n, t, u, v, w,  Classification Grid Searches #
    classification_grid_searches(training_features=values_train,training_classes=hdi_class_train)


def _rmse_grid_search(training_features, training_labels, pipeline, parameters, technique, title='Figure.png'):
    """
    Input:
        training_features: world_values responses on the training set
        training_labels: HDI (human development index) on the training set
        pipeline: regression model specific pipeline
        parameters: regression model specific parameters
        technique: regression model's name

    Output:
        Prints best RMSE and best estimator
        Prints feature weights for Ridge and Lasso Regression
        Plots RMSE vs k for k Nearest Neighbors Regression
    """
    grid = GridSearchCV(estimator=pipeline,
                        param_grid=parameters,
                        scoring='neg_mean_squared_error')
    grid.fit(training_features,
             training_labels)
    print("RMSE:", sqrt(-grid.best_score_))
    print(grid.best_estimator_)

    # Check Ridge or Lasso Regression
    if hasattr(grid.best_estimator_.named_steps[technique], 'coef_'):
        print("Coefficients")
        print(grid.best_estimator_.named_steps[technique].coef_)
    else:
        idx = np.argmin(-grid.cv_results_['mean_test_score'])
        print('The best k is '+ str(grid.cv_results_['param_knn__n_neighbors'][idx]))
        # Plot RMSE vs k for k Nearest Neighbors Regression
        plt.plot(grid.cv_results_['param_knn__n_neighbors'],
                 (-grid.cv_results_['mean_test_score'])**0.5)
        plt.xlabel('k')
        plt.ylabel('RMSE')
        plt.title('RMSE versus k in kNN')
        # plt.show()
        plt.savefig(title)
        plt.close()

    print()
    return grid


def regression_grid_searches(training_features, training_labels, testing_features=None, title='Figure.png'):
    """
    Input:
        training_features: world_values responses on the training set
        training_labels: HDI (human development index) on the training set

    Output:
        Prints best RMSE, best estimator, feature weights for Ridge and Lasso Regression
        Prints best RMSE, best estimator, and plots RMSE vs k for k Nearest Neighbors Regression
    """

    print("Ridge Regression")
    _rmse_grid_search(training_features, training_labels,
                ridge_regression_pipeline, regression_ridge_parameters, 'ridge', title)

    print("Lasso Regression")
    _rmse_grid_search(training_features, training_labels,
                lasso_regression_pipeline, regression_lasso_parameters, 'lasso', title)

    print("k Nearest Neighbors Regression")
    grid = _rmse_grid_search(training_features, training_labels,
                k_nearest_neighbors_regression_pipeline,
                regression_knn_parameters, 'knn', title)

    if testing_features is not None:
        print(grid.predict(testing_features))


def _accuracy_grid_search(training_features, training_classes, pipeline, parameters, title='Figure.png'):
    """
    Input:
        training_features: world_values responses on the training set
        training_labels: HDI (human development index) on the training set
        pipeline: classification model specific pipeline
        parameters: classification model specific parameters

    Output:
        Prints best accuracy and best estimator of classification model
    """
    grid = GridSearchCV(estimator=pipeline,
                        param_grid=parameters,
                        scoring='accuracy')
    grid.fit(training_features, training_classes)
    print("Accuracy:", grid.best_score_)
    print(grid.best_estimator_)
    print()

    return grid


def classification_grid_searches(training_features, training_classes):
    """
    Input:
        training_features: world_values responses on the training set
        training_labels: HDI (human development index) on the training set

    Output:
        Prints best accuracy and best estimator for SVM and k Nearest Neighbors Classification
    """
    print("SVM Classification")
    _accuracy_grid_search(training_features, training_classes,
                        svm_classification_pipeline,
                        classification_svm_parameters)

    print("k Nearest Neighbors Classification")
    _accuracy_grid_search(training_features, training_classes,
                        k_nearest_neighbors_classification_pipeline,
                        classification_knn_parameters)

    # print("Decision Tree Classification")
    # decision_grid = _accuracy_grid_search(training_features, training_classes,
    #                                       tree_classification_pipeline,
    #                                       classification_tree_parameters)
    # print("Classes", decision_grid.best_estimator_.named_steps['tree'].classes_)
    # print("Feature Importances", decision_grid.best_estimator_.named_steps['tree'].feature_importances_)
    # estimator = decision_grid.best_estimator_.named_steps['tree']

    # n_nodes = estimator.tree_.node_count
    # print("Node Count", n_nodes)
    # children_left = estimator.tree_.children_left
    # print("Left Children", children_left)
    # children_right = estimator.tree_.children_right
    # print("Right Children", children_right)
    # feature = estimator.tree_.feature
    # print("Feature", feature)
    # threshold = estimator.tree_.threshold
    # print("Threshold", threshold)


if __name__ == '__main__':
    main()
