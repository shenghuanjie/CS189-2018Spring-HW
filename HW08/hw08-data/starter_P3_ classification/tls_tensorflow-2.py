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


def tabu(sizes, widths, error):
    print(r'\begin{tabu} to 1.0\textwidth {  ' + ''.join('|X[c] ' * (len(sizes)+1)) + ' | }')
    print(r'\hline')
    header = '{:^8}'
    for _ in range(len(sizes)):
        header += ' {:^8}'
    header += ' {:^8}'
    headerText = [' '] + ['& ' + str(s) for s in sizes] + [' \\\\']
    print(header.format(*headerText))
    for width, row in zip(widths, error):
        text = '{:>8}'
        for _ in range(len(row)):
            text += ' {:<8}'
        text += ' {:<8}'
        #'{0:.1f}'.format(r)
        rowText = [str(width)] + ['& ' + str(r) for r in row] + [' \\\\']
        print(r'\hline')
        print(text.format(*rowText))
    print(r'\hline')
    print(r'\end{tabu}\par')
    print(r'\hfill \par')


def main():
    lrs = [0.001, 0.01, 1]
    batches = [10, 100, 1000]
    lrs_lines = ['-', '-.', ':']
    batches_colors = ['r', 'k', 'b']
    plt.figure()
    last_erros_sgd = np.zeros((len(lrs), len(batches)))

    for ilr, learning_rate in enumerate(lrs):
        for ibatch, batch_size in enumerate(batches):

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
            plt.semilogy(errors_sgd, batches_colors[ibatch]+lrs_lines[ilr], label='SGD-lr' + str(learning_rate) + '-batch' + str(batch_size))
            last_erros_sgd[ilr, ibatch] = errors_sgd[-1]
                    #plt.semilogy([0, len(errors_sgd)], [errors_sgd[-1], errors_sgd[-1]], label="last error (SGD)",
                    #             linewidth=2)
                    #plt.text(len(errors_sgd) * 0.6, errors_sgd[-1] * 2, str(errors_sgd[-1]), fontsize=12)
                    # print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost),
                    #      "|w-w_true|^2 = {:.9f}".format(np.sum((w_sgd - w_true) ** 2)))

                    # print("Epoch:", '%04d' % (training_epochs), "cost=", "{:.9f}".format(avg_cost),
                    #     "|w-w_true|^2 = {:.9f}".format(np.sum((w_sgd - w_true) ** 2)))

        # Total least squares: SVD
    X = x_true
    y = y_ob
    stacked_mat = np.hstack((X, y[:, np.newaxis])).astype(np.float32)
    u, s, vh = np.linalg.svd(stacked_mat)
    w_tls = -vh[-1, :-1] / vh[-1, -1]

    error = np.sum(np.square(w_tls - w_true))
    print("TLS through SVD error: |w-w_true|^2 = {}".format(error))

    plt.semilogy([0, len(errors_sgd)], [error, error], 'g', label="SVD", linewidth=2)
    # plt.text(len(errors_sgd) * 0.6, error * 2, str(error), fontsize=12)
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('|w-w_true|^2')
    plt.xlabel('run (start=0)')
    # plt.show()
    plt.savefig('Figure_3e-2.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.xlim([0, training_epochs * 1.5])
    plt.close()

    print('SGD errors with different learning rates and batch sizes:')
    tabu(lrs, batches, last_erros_sgd)


if __name__ == "__main__":
    tf.set_random_seed(0)
    np.random.seed(0)
    main()
