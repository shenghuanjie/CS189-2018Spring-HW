import matplotlib.pyplot as plt
import numpy as np

from starter import *


def neural_network(X, Y, X_test, Y_test, num_layers, activation):
    """
    This function performs neural network prediction.
    Input:
        X: independent variables in training data.
        Y: dependent variables in training data.
        X_test: independent variables in test data.
        Y_test: dependent variables in test data.
        num_layers: number of layers in neural network
        activation: type of activation, ReLU or tanh
    Output:
        mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    meanX = np.mean(X, axis=0)
    stdX = np.std(X, axis=0)
    meanY = np.mean(Y, axis=0)
    stdY = np.std(Y, axis=0)
    X = (X - meanX) / stdX
    Y = (Y - meanY) / stdY
    activations = dict(ReLU=ReLUActivation,
                       tanh=TanhActivation,
                       linear=LinearActivation)
    lr = dict(ReLU=0.1, tanh=0.1, linear=0.005)
    iterations = dict(ReLU=2000, tanh=2000, linear=2000)
    roots = np.sort(np.roots([(num_layers - 1), (num_layers + Y.shape[1] + X.shape[1]), Y.shape[1]-10000]))
    num_neurons = int(roots[-1])
    names = ['ReLU', 'linear', 'tanh']
    #### PART G ####
    activation_func = activations[activation]
    model = Model(X.shape[1])
    for iLayer in range(num_layers):
        model.addLayer(DenseLayer(num_neurons, activation_func()))
    model.addLayer(DenseLayer(Y.shape[1], LinearActivation()))
    model.initialize(QuadraticCost())

    # Train the model and display the results
    model.train(X, Y, iterations[activation], GDOptimizer(eta=lr[activation]))
    Y_pred = model.predict((X_test - meanX) / stdX) * stdY + meanY
    mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test) ** 2, axis=1)))
    return mse


#############################################################################
#######################PLOT PART 2###########################################
#############################################################################
def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
    return generate_dataset(
        sensor_loc,
        num_sensors=k,
        spatial_dim=d,
        num_data=n,
        original_dist=original_dist,
        noise=noise)


np.random.seed(0)
n = 200
num_layerss = [1, 2, 3, 4]
mses = np.zeros((len(num_layerss), 2))

# for s in range(replicates):
sensor_loc = generate_sensors()
X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
X_test, Y_test = generate_data(sensor_loc, n=1000)
for t, num_layers in enumerate(num_layerss):
    ### Neural Network:
    mse = neural_network(X, Y, X_test, Y_test, num_layers, "ReLU")
    mses[t, 0] = mse

    mse = neural_network(X, Y, X_test, Y_test, num_layers, "tanh")
    mses[t, 1] = mse

    print('Experiment with {} layers done...'.format(num_layers))

### Plot MSE for each model.
plt.figure()
activation_names = ['ReLU', 'Tanh']
for a in range(2):
    plt.plot(num_layerss, mses[:, a], label=activation_names[a])

plt.title('Error on validation data verses number of neurons')
plt.xlabel('Number of layers')
plt.ylabel('Average Error')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('Figure_4e-num_layers.png')
plt.close()

