import torch
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt

transform = torchvision.transforms.ToTensor()

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True,
    transform=transform)  # torchvision has nice easy downloading
testset = torchvision.datasets.MNIST(
    root='./data',
    train=False,  # of datasets for CV tasks like MNIST
    download=True,
    transform=transform)


def optimize(prediction_fn, loss_fn, optim, epochs, batch_size):
    acc = []
    for epoch in range(epochs):
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True)  # iterable of batches
        avg_loss = Variable(torch.zeros(1))
        num_batches = len(trainset) / batch_size
        for i, (input, target_ind) in enumerate(trainloader):
            input = Variable(input.view(batch_size, -1))  # the input comes as a batch of 2d images which we flatten;
            target = Variable(torch.zeros(batch_size, 10)) # view(-1) tells pytorch to fill in a dimension; here it's 784
            for k in range(batch_size):                   # we have to make our one-hot vectors ourselves unfortunately
                target[k, target_ind[k]] = 1
            optim.zero_grad()                             # zero the gradient buffers
            output = prediction_fn(input)                 # compute the output
            loss = loss_fn(target, output)                # compute the loss
            loss.backward()                               # backpropagate from the loss to fill the gradient buffers
            optim.step()                                  # do a gradient descent step
            if not loss.data.numpy() == loss.data.numpy(): # Some errors make the loss NaN. this is a problem.
                print("loss is NaN")                       # This is helpful: it'll catch that when it happens,
                return output, input, loss                 # and give the output and input that made the loss NaN
            avg_loss += loss/num_batches                  # update the overall average loss with this batch's loss
        correct = 0
        for input, target in testset:                     # compute the testing accuracy
            input = Variable(input.view(1, -1))
            output = prediction_fn(input)
            pred_ind = torch.max(output, 1)[1]
            if pred_ind.data[0] == target:                # true/false Variables don't actually have a boolean value,
                correct += 1                              # so we have to unwrap it to see if it was correct
        accuracy = correct / len(testset)
        print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss.data[0]),
              "accuracy={:.9f}".format(accuracy))
        acc.append(accuracy)
    return acc


def train_linear(learning_rate=0.01, epochs=50, batch_size=100):
    weights = Variable(torch.randn(784, 10), requires_grad=True)
    bias = Variable(torch.randn(10), requires_grad=True)

    def linear(x):
        return torch.matmul(x, weights) + bias

    # We create functions to compute things and pass them to the training function.
    # This sort of thing is common in pytorch (although usually it's callable objects
    # instead of the functions themselves)
    def loss_fn(y, y_pred):
        return torch.sum((y - y_pred)**2) / batch_size

    optimizer = torch.optim.SGD([weights, bias], lr=learning_rate)
    return optimize(linear, loss_fn, optimizer, epochs, batch_size)


def train_logistic(learning_rate=0.01, epochs=50, batch_size=100):
    weights = Variable(torch.randn(784, 10), requires_grad=True)
    bias = Variable(torch.randn(10), requires_grad=True)

    ## a few notes:
    ##   1. you may want to make a softmax helper function
    ##   2. exp(x) is liable to cause NaNs for very large x, you might try fixing that with torch.clamp
    ##   3. if you use log(x), make sure x is never zero
    ##   4. shapes are difficult. Look up expand_as in the pytorch docs.
    ## other handy things: torch.sum(), torch.Tensor.view()
    ## IMPORTANT: Don't unwrap Variables! They hold the graph structure. Tensors and Variables have the same methods.
    ##
    ## YOUR CODE HERE
    def logistic(x):
        return None

    def loss_fn(y, y_pred):
        return None

    ######################################

    optimizer = torch.optim.SGD([weights, bias], lr=learning_rate)
    return optimize(logistic, loss_fn, optimizer, epochs, batch_size)


def train_nn(learning_rate=0.05, epochs=50, batch_size=50, n_hidden=64):
    weights1 = Variable(torch.randn(784, n_hidden), requires_grad=True)
    bias1 = Variable(torch.randn(n_hidden), requires_grad=True)
    weights2 = Variable(torch.randn(n_hidden, 10), requires_grad=True)
    bias2 = Variable(torch.randn(10), requires_grad=True)

    ## Same notes as above. Remember: backward() won't work if you unwrap Variables.
    ## YOUR CODE HERE
    def nn(x):
        return None

    def loss_fn(y, y_pred):
        return None

    ######################################
    optimizer = torch.optim.SGD([weights1, bias1, weights2, bias2], lr=learning_rate)
    return optimize(nn, loss_fn, optimizer, epochs, batch_size)


def main():
    for batch_size in [50, 100, 200]:
        acc_linear = train_linear(batch_size=batch_size)
        plt.plot(acc_linear, label="linear bs=%d" % batch_size)
        plt.legend()
        plt.show()

    acc_logistic = train_logistic()
    plt.plot(acc_logistic, label="logistic")
    plt.legend()
    plt.show()

    acc_nn = train_nn()
    plt.plot(acc_nn, label="neural network")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
