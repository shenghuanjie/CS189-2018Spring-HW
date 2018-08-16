
import numpy as np
import tensorflow as tf
#import yolo.config_card as cfg

import IPython

slim = tf.contrib.slim


class CNN(object):

    def __init__(self,classes,image_size):

        self.classes = classes
        self.num_class = len(self.classes)
        self.image_size = image_size
    
        self.output_size = self.num_class
        self.batch_size = 40

        self.images = tf.placeholder(tf.float32, [None, self.image_size,self.image_size,3], name='images')
     

        self.logits = self.build_network(self.images, num_outputs=self.output_size)
            
        self.labels = tf.placeholder(tf.float32, [None, self.num_class])

        self.loss_layer(self.logits, self.labels)
        self.total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,
                      images,
                      num_outputs,
                      scope='yolo'):
       
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):

                ###SLIM BY DEFAULT ADDS A RELU AT THE END OF conv2d and fully_connected

                ###SLIM SPECIFYING A CONV LAYER WITH 80 FILters as SIZE 3 by 3
                net = slim.conv2d(images, 80, [3, 3], scope='conv_0')

                ### SLIM USING POOLING ON THE NETWORK. THE POOLING REGION CONSIDERED IS 5 by 5
                net = slim.max_pool2d(net, [5, 5], scope='pool')

                ### TO GO FROM A CONVOLUTIONAL LAYER TO A FULLY CONNECTED YOU NEED TO FLATTEN THE ARRAY
                net = slim.flatten(net, scope='flat')

                ###SLIM SPECIFYING A FULLY CONNECTED LAYER WHOSE OUT IS 10
                net = slim.fully_connected(net, 10, scope='fc_2')

                
        return net



    def loss_layer(self, predicts, classes, scope='loss_layer'):
        with tf.variable_scope(scope):
   
            #IMPLEMENTATION OF A SOFTMAX CROSS ENTROPY LOSS FUNCTION
            self.class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = classes,logits = predicts))

       
          
