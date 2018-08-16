import numpy as np

regression_ridge_parameters = {
    'ridge__alpha': np.arange(0.01, 1.0, 0.01)
}

regression_lasso_parameters = {
    'lasso__alpha': np.arange(0.0001, 0.01, 0.0001)
}

regression_knn_parameters = {
    # 'pca__n_components': np.arange(1, 17),

    'knn__n_neighbors': np.arange(1, 50),

    # Apply uniform weighting vs k for k Nearest Neighbors Regression
    'knn__weights': ['uniform']

    # Apply distance weighting vs k for k Nearest Neighbors Regression
    # 'knn__weights': ['distance']
}

regression_knn_parameters_weighted = {
    # 'pca__n_components': np.arange(1, 17),

    'knn__n_neighbors': np.arange(1, 50),

    # Apply uniform weighting vs k for k Nearest Neighbors Regression
    # 'knn__weights': ['uniform']

    # Apply distance weighting vs k for k Nearest Neighbors Regression
    'knn__weights': ['distance']
}

classification_svm_parameters = {
    # Use linear kernel for SVM Classification
    'svm__kernel': ['linear'],

    # Use rbf kernel for SVM Classification
    # 'svm__kernel': ['rbf'],

    # Original hyperparameters
    'svm__C': np.arange(1.0, 100.0, 1.0),

    # Original hyperparameters scaled by 1/100
    # 'svm__C': np.arange(0.01, 1.0, 0.01),

    # Hyperparameter search over all possible dimensions for PCA reduction
    # 'pca__n_components': np.arange(1, 17),

    # 'svm__gamma': np.arange(0.001, 0.1, 0.001)
}

classification_svm_parameters_pca_scaled = {
    # Use linear kernel for SVM Classification
    'svm__kernel': ['linear'],

    # Use rbf kernel for SVM Classification
    # 'svm__kernel': ['rbf'],

    # Original hyperparameters
    # 'svm__C': np.arange(1.0, 100.0, 1.0),

    # Original hyperparameters scaled by 1/100
    'svm__C': np.arange(0.01, 1.0, 0.01),

    # Hyperparameter search over all possible dimensions for PCA reduction
    'pca__n_components': np.arange(1, 17),

    # 'svm__gamma': np.arange(0.001, 0.1, 0.001)
}

classification_svm_parameters_rbf = {
    # Use linear kernel for SVM Classification
    # 'svm__kernel': ['linear'],

    # Use rbf kernel for SVM Classification
    'svm__kernel': ['rbf'],

    # Original hyperparameters
    'svm__C': np.arange(1.0, 100.0, 1.0),

    # Original hyperparameters scaled by 1/100
    # 'svm__C': np.arange(0.01, 1.0, 0.01),

    # Hyperparameter search over all possible dimensions for PCA reduction
    # 'pca__n_components': np.arange(1, 17),

    'svm__gamma': np.arange(0.001, 0.1, 0.001)
}

classification_knn_parameters = {
    'knn__n_neighbors': np.arange(1, 50),

    # Apply distance weighting vs k for k Nearest Neighbors Classification
    'knn__weights': ['distance']
}

classification_tree_parameters = {
    #'tree__min_samples_split': [20],
    'tree__max_depth': [2]
}
