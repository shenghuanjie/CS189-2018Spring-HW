import numpy as np
import matplotlib.pyplot as plt

from starter import *
from plot1 import *


def neural_network(X, Y, Xs_test, Ys_test,
                         lr=0.1, iterations=2000, num_neurons=1000, num_layers=1, activation='ReLU'):
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

    #### PART G ####
    activation_func = activations[activation]
    model = Model(X.shape[1])
    for iLayer in range(num_layers):
        model.addLayer(DenseLayer(num_neurons, activation_func()))
    model.addLayer(DenseLayer(Y.shape[1], LinearActivation()))
    model.initialize(QuadraticCost())

    # Train the model and display the results
    # model.trainBatch(X, Y, int(iterations / 10), iterations, GDOptimizer(eta=lr))
    model.train(X, Y, iterations, GDOptimizer(eta=lr))
    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = model.predict((X_test - meanX) / stdX) * stdY + meanY
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test) ** 2, axis=1)))
        mses.append(mse)
    return mses


def main():
    #############################################################################
    #######################PLOT PART 1###########################################
    #############################################################################
    np.random.seed(0)

    ns = np.arange(2, 7, 0.5)
    ns = np.array(np.power(3, ns), dtype=int)
    replicates = 5
    num_methods = 6
    num_sets = 3
    mses = np.zeros((len(ns), replicates, num_methods, num_sets))

    def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
        return generate_dataset(
            sensor_loc,
            num_sensors=k,
            spatial_dim=d,
            num_data=n,
            original_dist=original_dist,
            noise=noise)

    for s in range(replicates):
        sensor_loc = generate_sensors()
        X_test, Y_test = generate_data(sensor_loc, n=1000)
        X_test2, Y_test2 = generate_data(
            sensor_loc, n=1000, original_dist=False)
        for t, n in enumerate(ns):
            X, Y = generate_data(sensor_loc, n=n)  # X [n * 7] Y [n * 2]
            Xs_test, Ys_test = [X, X_test, X_test2], [Y, Y_test, Y_test2]
            ### Linear regression:
            mse = linear_regression(X, Y, Xs_test, Ys_test)
            mses[t, s, 0] = mse

            ### Second-order Polynomial regression:
            mse = poly_regression_second(X, Y, Xs_test, Ys_test)
            mses[t, s, 1] = mse

            ### 3rd-order Polynomial regression:
            mse = poly_regression_cubic(X, Y, Xs_test, Ys_test)
            mses[t, s, 2] = mse

            ### Neural Network:
            mse = neural_network(X, Y, Xs_test, Ys_test)
            mses[t, s, 3] = mse

            ### Generative model:
            mse = generative_model(X, Y, Xs_test, Ys_test)
            mses[t, s, 4] = mse

            ### Oracle model:
            mse = oracle_model(X, Y, Xs_test, Ys_test, sensor_loc)
            mses[t, s, 5] = mse

            print('{}th Experiment with {} samples done...'.format(s, n))

    ### Plot MSE for each model.
    plt.figure()
    regressors = [
        'Linear Regression', '2nd-order Polynomial Regression',
        '3rd-order Polynomial Regression', 'Neural Network',
        'Generative Model', 'Oracle Model'
    ]
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 0], axis=1), label=regressors[a])

    plt.title('Error on training data for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('Figure_4f-train_mse.png')
    #plt.show()
    plt.close()

    plt.figure()
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 1], axis=1), label=regressors[a])

    plt.title(
        'Error on test data from the same distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('Figure_4f-val_same_mse.png')
    #plt.show()
    plt.close()

    plt.figure()
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 2], axis=1), label=regressors[a])

    plt.title(
        'Error on test data from a different distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('Figure_4f-val_different_mse.png')
    #plt.show()
    plt.close()


if __name__ == '__main__':
    main()
