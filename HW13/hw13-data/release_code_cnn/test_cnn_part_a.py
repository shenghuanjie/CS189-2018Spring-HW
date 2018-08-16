from data_manager import data_manager
from cnn import CNN
from trainer import Solver
import tensorflow as tf
import random

from confusion_mat import Confusion_Matrix
import numpy as np

random.seed(0)

CLASS_LABELS = ['apple','banana','nectarine','plum','peach','watermelon','pear','mango','grape','orange','strawberry','pineapple',
    'radish','carrot','potato','tomato','bellpepper','broccoli','cabbage','cauliflower','celery','eggplant','garlic','spinach','ginger']

image_size = 90
classes = CLASS_LABELS
dm = data_manager(classes, image_size)

cnn = CNN(classes,image_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

val_data = dm.val_data
train_data = dm.train_data

cm = Confusion_Matrix(val_data, train_data, CLASS_LABELS, sess)

plt = cm.test_net(cnn)
plt.savefig('Figure_3a.png')

