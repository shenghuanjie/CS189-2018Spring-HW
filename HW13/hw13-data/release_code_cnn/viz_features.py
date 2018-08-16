import random

import IPython
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


class Viz_Feat(object):
    def __init__(self, val_data, train_data, class_labels, sess):

        self.val_data = val_data
        self.train_data = train_data
        self.CLASS_LABELS = class_labels
        self.sess = sess

    def vizualize_features(self, net):

        images = [0, 10, 100]
        '''
        Compute the response map for the index images
        '''
        for i in images:

            # validation data
            curr_img = self.val_data[i]['features']
            curr_img = np.reshape(curr_img, (1,) + curr_img.shape)
            curr_label = np.reshape(self.val_data[i]['label'], (1, -1))
            response_map = self.sess.run(net.response_map,
                                         feed_dict={net.images: curr_img, net.labels: curr_label})

            class_label = str(self.CLASS_LABELS[np.nonzero(curr_label[0])[0][0]])
            cv2.imwrite('val_' + str(i) + '_' + class_label  + '_raw.png', self.val_data[i]['c_img'])

            for ifilter in range(response_map.shape[-1]):
                cv2.imwrite('val_'  + str(i) + '_' + class_label  + '_filter-' + str(ifilter) + '.png',
                           self.revert_image(response_map[0, :, :, ifilter]))

    def revert_image(self, img):
        '''
        Used to revert images back to a form that can be easily visualized
        '''

        img = (img + 1.0) / 2.0 * 255.0

        img = np.array(img, dtype=int)

        blank_img = np.zeros([img.shape[0], img.shape[1], 3])

        blank_img[:, :, 0] = img
        blank_img[:, :, 1] = img
        blank_img[:, :, 2] = img

        img = blank_img.astype("uint8")

        return img
