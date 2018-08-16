import numpy as np
import scipy.spatial
from starter import *


#####################################################################
## Models used for predictions.
#####################################################################
def compute_update(single_obj_loc, sensor_loc, single_distance):
    """
    Compute the gradient of the log-likelihood function for part a.

    Input:
    single_obj_loc: 1 * d numpy array.
    Location of the single object.

    sensor_loc: k * d numpy array.
    Location of sensor.

    single_distance: k dimensional numpy array.
    Observed distance of the object.

    Output:
    grad: d-dimensional numpy array.

    """
    loc_difference = single_obj_loc - sensor_loc  # k * d.
    phi = np.linalg.norm(loc_difference, axis=1)  # k.
    grad = loc_difference / np.expand_dims(phi, 1)  # k * 2.
    update = np.linalg.solve(grad.T.dot(grad), grad.T.dot(single_distance - phi))

    return update


def get_object_location(sensor_loc, single_distance, num_iters=20, num_repeats=10):
    """
    Compute the gradient of the log-likelihood function for part a.

    Input:

    sensor_loc: k * d numpy array. Location of sensor.

    single_distance: k dimensional numpy array.
    Observed distance of the object.

    Output:
    obj_loc: 1 * d numpy array. The mle for the location of the object.

    """
    obj_locs = np.zeros((num_repeats, 1, 2))
    distances = np.zeros(num_repeats)
    for i in range(num_repeats):
        obj_loc = np.random.randn(1, 2) * 100
        for t in range(num_iters):
            obj_loc += compute_update(obj_loc, sensor_loc, single_distance)

        distances[i] = np.sum((single_distance - np.linalg.norm(obj_loc - sensor_loc, axis=1))**2)
        obj_locs[i] = obj_loc

    obj_loc = obj_locs[np.argmin(distances)]

    return obj_loc[0]


def generative_model(X, Y, Xs_test, Ys_test):
    """
    This function implements the generative model.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    initial_sensor_loc = np.random.randn(7, 2) * 100
    estimated_sensor_loc = find_mle_by_grad_descent_part_e(
        initial_sensor_loc, Y, X, lr=0.001, num_iters=1000)

    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = np.array(
            [get_object_location(estimated_sensor_loc, X_test_single) for X_test_single in X_test])
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses


def oracle_model(X, Y, Xs_test, Ys_test, sensor_loc):
    """
    This function implements the generative model.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    sensor_loc: location of the sensors.
    Output:
    mse: Mean square error on test data.
    """
    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = np.array([
            get_object_location(sensor_loc, X_test_single)
            for X_test_single in X_test
        ])
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses


def construct_second_order_data(X):
    """
    This function computes second order variables
    for polynomial regression.
    Input:
    X: Independent variables.
    Output:
    A data matrix composed of both first and second order terms.
    """
    X_second_order = []
    m = X.shape[1]
    for i in range(m):
        for j in range(m):
            if j <= i:
                X_second_order.append(X[:, i] * X[:, j])
    X_second_order = np.array(X_second_order).T
    return np.concatenate((X, X_second_order), axis=1)


def linear_regression(X, Y, Xs_test, Ys_test):
    """
    This function performs linear regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """

    ## YOUR CODE HERE
    #################
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    #w = np.linalg.solve(X.T @ X, X.T @ Y)
    w = np.linalg.lstsq(X, Y)[0]
    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        Y_pred = np.array(X_test @ w)
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test) ** 2, axis=1)))
        mses.append(mse)
    return mses


def linear_regression_no_bias(X, Y, Xs_test, Ys_test):
    """
    This function performs linear regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """

    ## YOUR CODE HERE
    #################
    w = np.linalg.lstsq(X, Y)[0]
    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = np.array(X_test @ w)
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test) ** 2, axis=1)))
        mses.append(mse)
    return mses


def polynomial(x, D):
    n_feature = x.shape[1]
    Q = [(np.ones(x.shape[0]), 0, 0)]
    i = 0
    while Q[i][1] < D:
        cx, degree, last_index = Q[i]
        for j in range(last_index, n_feature):
            Q.append((cx * x[:, j], degree + 1, j))
        i += 1
    return np.column_stack([q[0] for q in Q])


def poly_regression_second(X, Y, Xs_test, Ys_test):
    """
    This function performs second order polynomial regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    Xs_test_poly = []
    for i, X_test in enumerate(Xs_test):
        Xs_test_poly.append(polynomial(X_test, 2))

    return linear_regression_no_bias(polynomial(X, 2), Y, Xs_test_poly, Ys_test)


def poly_regression_cubic(X, Y, Xs_test, Ys_test):
    """
    This function performs third order polynomial regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    X = np.array(X)
    Y = np.array(Y)
    Xs_test_poly = []
    for i, X_test in enumerate(Xs_test):
        Xs_test_poly.append(polynomial(X_test, 3))
    return linear_regression_no_bias(polynomial(X, 3), Y, Xs_test_poly, Ys_test)


def neural_network(X, Y, Xs_test, Ys_test):
    """
    This function performs neural network prediction.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    # Copied from backprop_sol
    meanX = np.mean(X, axis=0)
    stdX = np.std(X, axis=0)
    meanY = np.mean(Y, axis=0)
    stdY = np.std(Y, axis=0)
    X = (X - meanX) / stdX
    Y = (Y - meanY) / stdY
    activations = dict(ReLU=ReLUActivation,
                       tanh=TanhActivation,
                       linear=LinearActivation)
    lr = dict(ReLU=0.1, tanh=0.02, linear=0.005)
    iterations = 2000
    names = ['ReLU', 'linear', 'tanh']
    key = names[0]
    #### PART G ####
    activation = activations[key]
    model = Model(X.shape[1])
    model.addLayer(DenseLayer(100, activation()))
    model.addLayer(DenseLayer(100, activation()))
    model.addLayer(DenseLayer(Y.shape[1], LinearActivation()))
    model.initialize(QuadraticCost())

    # Train the model and display the results
    model.train(X, Y, iterations, GDOptimizer(eta=lr[key]))
    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = model.predict((X_test - meanX) / stdX) * stdY + meanY
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test) ** 2, axis=1)))
        mses.append(mse)
    return mses
