import time

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


def optimize(x, y, pred, loss, optimizer, training_epochs, batch_size):
    acc = []
    with tf.Session() as sess:  # start training
        sess.run(tf.global_variables_initializer())  # Run the initializer
        for epoch in range(training_epochs):  # Training cycle
            avg_loss = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, loss], feed_dict={x: batch_xs, y: batch_ys})
                avg_loss += c / total_batch

            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy = accuracy_.eval({x: mnist.test.images, y: mnist.test.labels})
            acc.append(accuracy)
            #print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss),
            #      "accuracy={:.9f}".format(accuracy))
    return acc


def train_linear(learning_rate=0.01, training_epochs=50, batch_size=100):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    pred = tf.matmul(x, W) + b
    loss = tf.reduce_mean((y - pred)**2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimize(x, y, pred, loss, optimizer, training_epochs, batch_size)


def train_logistic(learning_rate=0.01, training_epochs=50, batch_size=100):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # YOUR CODE HERE
    # copied from tf.nn.softmax
    logits = (tf.matmul(x, W) + b)
    pred = tf.exp(logits - tf.reduce_max(logits)) / tf.reduce_sum(tf.exp(logits - tf.reduce_max(logits)))
    # copied from https://github.com/tensorflow/tensorflow/issues/2462
    # loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
    # modified it to be the stable version
    loss = tf.reduce_mean(-tf.log(tf.reduce_sum(y * pred, reduction_indices=[1])))
    ################

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimize(x, y, pred, loss, optimizer, training_epochs, batch_size)


def train_nn(learning_rate=0.01, training_epochs=50, batch_size=50, n_hidden=64):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W1 = tf.Variable(tf.random_normal([784, n_hidden]))
    W2 = tf.Variable(tf.random_normal([n_hidden, 10]))
    b1 = tf.Variable(tf.random_normal([n_hidden]))
    b2 = tf.Variable(tf.random_normal([10]))

    # YOUR CODE HERE
    logits = tf.matmul(tf.tanh(tf.matmul(x, W1) + b1), W2) + b2
    # copied from tf.nn.softmax
    pred = tf.exp(logits - tf.reduce_max(logits)) / tf.reduce_sum(tf.exp(logits - tf.reduce_max(logits)))
    # copied from https://github.com/tensorflow/tensorflow/issues/2462
    # loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
    # modified it to be the stable version
    loss = tf.reduce_mean(-tf.log(tf.reduce_sum(y * pred, reduction_indices=[1])))
    ################

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimize(x, y, pred, loss, optimizer, training_epochs, batch_size)


def main():

    plt.figure()
    for batch_size in [50, 100, 200]:
        time_start = time.time()
        acc_linear = train_linear(batch_size=batch_size)
        print("train_linear finishes in %.3fs" % (time.time() - time_start))

        plt.plot(acc_linear, label="linear bs=%d" % batch_size)
        plt.legend()
        plt.ylabel('accuracy')
        plt.xlabel('epoch (start=0)')
    plt.savefig('Figure_3a-linear.png')
    plt.close()

    plt.figure()
    acc_logistic = train_logistic()
    plt.plot(acc_logistic, label="logistic regression")
    plt.legend()
    #plt.show()
    plt.ylabel('accuracy')
    plt.xlabel('epoch (start=0)')
    plt.savefig('Figure_3a-logistic.png')
    plt.close()

    plt.figure()
    acc_nn = train_nn()
    plt.plot(acc_nn, label="neural network")
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('epoch (start=0)')
    #plt.show()
    plt.savefig('Figure_3a-neural.png')
    plt.close()

if __name__ == "__main__":
    tf.set_random_seed(0)
    main()
