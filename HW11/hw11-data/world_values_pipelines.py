from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


tree_classification_pipeline = Pipeline(
    [
        ('tree', DecisionTreeClassifier()),

        # Forest instead of Trees
        # ('forest', RandomForestClassifier())
    ]
)


ridge_regression_pipeline = Pipeline(
        [
            # Apply scaling to Ridge Regression
            # ('scale', StandardScaler()),

            ('ridge', Ridge())
        ]
    )

lasso_regression_pipeline = Pipeline(
        [
            # Apply scaling to Lasso Regression
            # ('scale', StandardScaler()),

            ('lasso', Lasso())
        ]
    )

k_nearest_neighbors_regression_pipeline = Pipeline(
        [
            # Apply PCA to k Nearest Neighbors Regression
            # ('pca', PCA()),

            # Apply scaling to k Nearest Neighbors Regression
            # ('scale', StandardScaler()),

            ('knn', KNeighborsRegressor())
        ]
    )


k_nearest_neighbors_regression_pipeline_scaled = Pipeline(
        [
            # Apply PCA to k Nearest Neighbors Regression
            # ('pca', PCA()),

            # Apply scaling to k Nearest Neighbors Regression
            ('scale', StandardScaler()),

            ('knn', KNeighborsRegressor())
        ]
    )

k_nearest_neighbors_regression_pipeline_minmax = Pipeline(
        [
            # Apply PCA to k Nearest Neighbors Regression
            # ('pca', PCA()),

            # Apply scaling to k Nearest Neighbors Regression
            ('scale', MinMaxScaler()),

            ('knn', KNeighborsRegressor())
        ]
    )

k_nearest_neighbors_regression_pipeline_binary = Pipeline(
        [
            # Apply PCA to k Nearest Neighbors Regression
            # ('pca', PCA()),

            # Apply scaling to k Nearest Neighbors Regression
            ('scale', Binarizer()),

            ('knn', KNeighborsRegressor())
        ]
    )

svm_classification_pipeline = Pipeline(
        [
            # Apply PCA to SVM Classification
            # ('pca', PCA()),

            # Apply scaling to SVM Classification
            #('scale', StandardScaler()),

            ('svm', SVC())
        ]
    )

svm_classification_pipeline_pca_scaled = Pipeline(
        [
            # Apply PCA to SVM Classification
            ('pca', PCA()),

            # Apply scaling to SVM Classification
            ('scale', StandardScaler()),

            ('svm', SVC())
        ]
    )

k_nearest_neighbors_classification_pipeline = Pipeline(
        [
            # Apply scaling to k Nearest Neighbors Classification
            # ('scale', StandardScaler()),

            ('knn', KNeighborsClassifier())
        ]
    )


k_nearest_neighbors_classification_pipeline_scaled = Pipeline(
        [
            # Apply scaling to k Nearest Neighbors Classification
            ('scale', StandardScaler()),

            ('knn', KNeighborsClassifier())
        ]
    )
