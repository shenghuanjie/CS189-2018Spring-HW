import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

n_data = 6000
n_dim = 50

np.random.seed(0)

w_true = np.random.uniform(low=-2.0, high=2.0, size=[n_dim])

x_true = np.random.uniform(low=-10.0, high=10.0, size=[n_data, n_dim])
x_ob = x_true + np.random.randn(n_data, n_dim)
y_ob = x_true @ w_true + np.random.randn(n_data)

learning_rate = 0.01
training_epochs = 100
batch_size = 100


def main():
    x = tf.placeholder(tf.float32, [None, n_dim])
    y = tf.placeholder(tf.float32, [None, 1])

    w = tf.Variable(tf.random_normal([n_dim, 1]))

    # YOUR CODE HERE
    cost = tf.log(tf.norm(w) + 1) / 2 + \
        1 / (2 * (tf.norm(w) + 1)) * tf.reduce_mean((tf.matmul(x, w) - y) ** 2)
    ################

    errors_sgd = np.zeros(training_epochs)

    # Adam is a fancier version of SGD, which is insensitive to the learning
    # rate.  Try replace this with GradientDescentOptimizer and tune the
    # parameters!
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        w_sgd = sess.run(w).flatten()

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_data / batch_size)
            for i in range(total_batch):
                start, end = i * batch_size, (i + 1) * batch_size
                _, c = sess.run(
                    [optimizer, cost],
                    feed_dict={
                        x: x_ob[start:end, :],
                        y: y_ob[start:end, np.newaxis]
                    })
                avg_cost += c / total_batch
            w_sgd = sess.run(w).flatten()
            errors_sgd[epoch] = np.sum((w_sgd - w_true) ** 2)
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost),
                  "$\|w-w_{{true}}\|^2 = {:.9f}".format(np.sum((w_sgd - w_true) ** 2)) + '$ \par')

    #print("Epoch:", '%04d' % (training_epochs), "cost=", "{:.9f}".format(avg_cost),
    #      "|w-w_true|^2 = {:.9f}".format(np.sum((w_sgd - w_true) ** 2)))

    # Total least squares: SVD
    X = x_true
    y = y_ob
    stacked_mat = np.hstack((X, y[:, np.newaxis])).astype(np.float32)
    u, s, vh = np.linalg.svd(stacked_mat)
    w_tls = -vh[-1, :-1] / vh[-1, -1]

    error = np.sum(np.square(w_tls - w_true))
    print("TLS through SVD error: |w-w_true|^2 = {}".format(error))

    plt.figure()
    plt.semilogy(errors_sgd, label="SGD")
    plt.semilogy([0, len(errors_sgd)], [errors_sgd[-1], errors_sgd[-1]], label="last error (SGD)", linewidth=2)
    plt.text(len(errors_sgd) * 0.6, errors_sgd[-1] * 2, str(errors_sgd[-1]), fontsize=12)
    plt.semilogy([0, len(errors_sgd)], [error, error], label="SVD", linewidth=2)
    plt.text(len(errors_sgd) * 0.6, error * 2, str(error), fontsize=12)
    plt.legend()
    plt.ylabel('|w-w_true|^2')
    plt.xlabel('run (start=0)')
    # plt.show()
    plt.savefig('Figure_3e-lr'+str(learning_rate) + '-batch' + str(batch_size) + '.png')
    plt.close()


if __name__ == "__main__":
    tf.set_random_seed(0)
    np.random.seed(0)
    main()
