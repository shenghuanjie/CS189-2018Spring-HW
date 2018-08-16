from release_code_cnn.data_manager import data_manager
from release_code_cnn.cnn import CNN
from release_code_cnn.trainer import Solver
import tensorflow as tf
import random

from release_code_cnn.confusion_mat import Confusion_Matrix
import numpy as np

random.seed(0)

CLASS_LABELS = ['apple','banana','nectarine','plum','peach','watermelon','pear','mango','grape','orange','strawberry','pineapple',
    'radish','carrot','potato','tomato','bellpepper','broccoli','cabbage','cauliflower','celery','eggplant','garlic','spinach','ginger']

image_size = 90
classes = CLASS_LABELS
dm = data_manager(classes, image_size)

#cnn = CNN(classes,image_size)

#sess = tf.Session()
#sess.run(tf.global_variables_initializer())

val_data = dm.val_data
train_data = dm.train_data

#cm = Confusion_Matrix(val_data, train_data, CLASS_LABELS, sess)

images, labels = dm.get_train_batch()

print(images.shape)
print(labels.shape)

#plt = cm.test_net(cnn)
#plt.savefig('Figure_3a.png')

