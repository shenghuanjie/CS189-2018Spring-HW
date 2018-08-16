import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
# Actual data
n_train = 6000
n_test = 1000
n_dim = 50

w_true = np.random.uniform(low=-2.0, high=2.0, size=[n_dim, 1])

x_true = np.random.uniform(low=-10.0, high=10.0, size=[n_train, n_dim])
x_ob = torch.Tensor(x_true + np.random.randn(n_train, n_dim))
y_ob = torch.Tensor(x_true @ w_true + np.random.randn(n_train, 1))

trainset = TensorDataset(x_ob, y_ob)  # PyTorch's class for datasets from raw data

learning_rate = 0.01
training_epochs = 100
batch_size = 100


def main():
    weight_tensor = torch.zeros(n_dim, 1)   # there's no handy shortcut for uniform, so we start with
    weight_tensor.uniform_(-2, 2)           # a zero tensor and then fill it with random values
    weight = Variable(weight_tensor, requires_grad=True)  # we need requires_grad=True for parameters

    ## YOUR CODE HERE (Same notes as previous question, minus softmax of course)
    def pred_fn(x):
        return None

    def loss_fn(y, y_pred):
        return None

    ###############################

    # Adam is a fancier version of SGD, which is insensitive to the learning
    # rate.  Try replace this with GradientDescentOptimizer and tune the
    # parameters!
    optimizer = torch.optim.Adam([weight], lr=learning_rate)
    for epoch in range(training_epochs):
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        avg_cost = Variable(torch.zeros(1))
        num_batches = len(trainset) / batch_size
        for i, (input, target) in enumerate(trainloader):
            input, target = Variable(input), Variable(target)
            optimizer.zero_grad()          # zero the gradient buffers
            output = pred_fn(input)
            loss = loss_fn(target, output)
            loss.backward()                # backpropagate from the loss to fill the gradient buffers
            optimizer.step()               # take a step according to the update rule (here Adam)
            if not loss.data.numpy() == loss.data.numpy():
                print("loss is NaN")       # less chance of NaN here since there's no softmax,
                return output, input, loss # but we keep this here just in case.
            avg_cost += loss / num_batches
        w_trained = weight.data.numpy()
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost.data.numpy()[0]),
              "|w-w_true|^2 = {:.9f}".format(np.sum((w_trained - w_true)**2)))

    # Total least square: SVD
    X = x_true
    y = y_ob.numpy()
    stacked_mat = np.hstack((X, y))
    u, s, vh = np.linalg.svd(stacked_mat)
    w_tls = -vh[-1, :-1] / vh[-1, -1]

    error = np.sum(np.square(w_tls - w_true.flatten()))
    print("TLS through SVD error: {}".format(error))


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    main()
