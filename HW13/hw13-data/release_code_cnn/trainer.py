import argparse
import datetime
import os
import sys

import tensorflow as tf

slim = tf.contrib.slim


class Solver(object):
    def __init__(self, net, data):

        self.net = net
        self.data = data

        # Number of iterations to train for
        self.max_iter = 5000
        # Every 200 iterations please record the trest and train loss
        self.summary_iter = 200

        '''
        Tensorflow is told to use a gradient descent optimizer 
        In the function optimize you will iteratively apply this on batches of data
        '''
        self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train = self.train_step.minimize(self.net.class_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def optimize(self):

        # Append the current train and test accuracy every 200 iterations
        self.train_accuracy = []
        self.test_accuracy = []

        '''
        Performs the training of the network. 
        Implement SGD using the data manager to compute the batches
        Make sure to record the training and test accuracy through out the process
        '''
        for i in range(self.max_iter + 1):
            images, labels = self.data.get_train_batch()
            self.sess.run(self.train, feed_dict={self.net.images: images, self.net.labels: labels})

            if i % self.summary_iter == 0:
                print('i: ' + str(i) + ' is recored')
                accuracy = self.sess.run(self.net.accurracy,
                                         feed_dict={self.net.images: images, self.net.labels: labels})
                self.train_accuracy.append(accuracy)
                images_val, labels_val = self.data.get_validation_batch()
                val_accuracy = self.sess.run(self.net.accurracy,
                                             feed_dict={self.net.images: images_val, self.net.labels: labels_val})
                self.test_accuracy.append(val_accuracy)
