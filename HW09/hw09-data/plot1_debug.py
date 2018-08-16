import matplotlib.pyplot as plt
import numpy as np

from models import *
from starter import *


def neural_network_debug(X, Y, Xs_test, Ys_test,
                         lr=0.0001, iterations=5000, num_neurons=100, num_layers=2, activation='ReLU'):
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
    Y_predicts = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = model.predict((X_test - meanX) / stdX) * stdY + meanY
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test) ** 2, axis=1)))
        mses.append(mse)
        Y_predicts.append(Y_pred)
    return mses, Y_predicts


def main():
    np.random.seed(0)
    def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
        return generate_dataset(
            sensor_loc,
            num_sensors=k,
            spatial_dim=d,
            num_data=n,
            original_dist=original_dist,
            noise=noise)

    sensor_loc = generate_sensors()
    X_test, Y_test = generate_data(sensor_loc, n=1000)
    X_test2, Y_test2 = generate_data(
        sensor_loc, n=1000, original_dist=False)
    X, Y = generate_data(sensor_loc, n=200)
    Xs_test, Ys_test = [X, X_test, X_test2], [Y, Y_test, Y_test2]
   # int(np.sqrt(10000 / 40))
    mses, Y_predictions = \
        neural_network_debug(X, Y, Xs_test, Ys_test, lr=0.05, iterations=2000,
                             num_layers=1, num_neurons=1000, activation='ReLU')

    plt.scatter(sensor_loc[:, 0], sensor_loc[:, 1], label="sensors")
    plt.scatter(Y[:, 0], Y[:, 1], label="Y")
    plt.scatter(Y_test[:, 0], Y_test[:, 1], label="Y_test")
    plt.scatter(Y_test2[:, 0], Y_test2[:, 1], label="Y_test2")

    #Y_predictions = [Y_predictions[0]]
    pred_labels = ['Y_pred', 'Y_test_pred', 'Y_test2_pred']

    plt.title(''.join(label + ":{:.2f} | ".format(mses[i]) for i, label in enumerate(pred_labels)))
    #print(Y_predictions[0])
    for i, y_predict in enumerate(Y_predictions):
        plt.scatter(y_predict[:, 0], y_predict[:, 1], label=pred_labels[i])

    plt.legend()
    plt.savefig("Figure_4b-debug_test2.png")
    plt.show()


if __name__ == "__main__":
    main()
