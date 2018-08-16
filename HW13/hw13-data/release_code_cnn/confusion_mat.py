from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import cv2
import IPython
import numpy as np



class Confusion_Matrix(object):


    def __init__(self,val_data,train_data, class_labels,sess):

        self.val_data = val_data
        self.train_data = train_data
        self.CLASS_LABELS = class_labels
        self.sess = sess


    def test_net(self, net):

        true_labels = []
        predicted_labels = []
        error = []
        for datum in self.val_data:

            batch_eval = np.zeros([1,datum['features'].shape[0],datum['features'].shape[1],datum['features'].shape[2]])
            batch_eval[0,:,:,:] = datum['features']

            batch_label = np.zeros([1,len(self.CLASS_LABELS)]) 

            batch_label[0,:] = datum['label']

            prediction = self.sess.run(net.logits,
                                   feed_dict={net.images: batch_eval})

            softmax_error = self.sess.run(net.class_loss,
                                   feed_dict={net.images: batch_eval, net.labels: batch_label})

            error.append(softmax_error)

            class_pred = np.argmax(prediction)
            class_truth = np.argmax(datum['label'])

            true_labels.append(class_truth)
            predicted_labels.append(class_pred)


        return self.getConfusionMatrixPlot(true_labels,predicted_labels,self.CLASS_LABELS)


    def vizualize_features(self,net):

        for datum in self.val_data:

            batch_eval = np.zeros([1,datum['features'].shape[0],datum['features'].shape[1],datum['features'].shape[2]])
            batch_eval[0,:,:,:] = datum['features']


            batch_label = np.zeros([1,len(self.CLASS_LABELS)]) 


            batch_label[0,:] = datum['label']

            response_map = self.sess.run(net.response_map,
                                   feed_dict={net.images: batch_eval, net.labels: batch_label})

            for i in range(5):
                img = self.revert_image(response_map[0,:,:,i])
                cv2.imshow('debug',img)
                cv2.waitKey(300)



    def revert_image(self,img):
        img = (img+1.0)/2.0*255.0

        img = np.array(img,dtype=int)

        blank_img = np.zeros([img.shape[0],img.shape[1],3])

        blank_img[:,:,0] = img
        blank_img[:,:,1] = img
        blank_img[:,:,2] = img

        img = blank_img.astype("uint8")

        return img




    def getConfusionMatrix(self,true_labels, predicted_labels):
        """
        Input
        true_labels: actual labels
        predicted_labels: model's predicted labels

        Output
        cm: confusion matrix (true labels vs. predicted labels)
        """

        # Generate confusion matrix using sklearn.metrics
        cm = confusion_matrix(true_labels, predicted_labels)
        return cm


    def plotConfusionMatrix(self,cm, alphabet):
        """
        Input
        cm: confusion matrix (true labels vs. predicted labels)
        alphabet: names of class labels

        Output
        Plot confusion matrix (true labels vs. predicted labels)
        """
        fig = plt.figure()
        plt.clf()                       # Clear plot
        ax = fig.add_subplot(111)       # Add 1x1 grid, first subplot
        ax.set_aspect(1)
        res = ax.imshow(cm, cmap=plt.cm.binary,
                        interpolation='nearest', vmin=0, vmax=80)

        plt.colorbar(res)               # Add color bar

        width = len(cm)                 # Width of confusion matrix
        height = len(cm[0])             # Height of confusion matrix

        # Annotate confusion entry with numeric value

        for x in range(width):
            for y in range(height):

                ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                            verticalalignment='center', color=self.getFontColor(cm[x][y]))


        # Plot confusion matrix (true labels vs. predicted labels)
        plt.xticks(range(width), alphabet[:width], rotation=90)
        plt.yticks(range(height), alphabet[:height])
        # plt.show()
        return plt


    def getConfusionMatrixPlot(self,true_labels, predicted_labels, alphabet):
        """
        Input
        true_labels: actual labels
        predicted_labels: model's predicted labels
        alphabet: names of class labels

        Output
        Plot confusion matrix (true labels vs. predicted labels)
        """
        # Generate confusion matrix using sklearn.metrics
        cm = confusion_matrix(true_labels, predicted_labels)


        # Plot confusion matrix (true labels vs. predicted labels)
        return self.plotConfusionMatrix(cm, alphabet)


    def getFontColor(self,value):
        """
        Input
        value: confusion entry value

        Output
        font color for confusion entry
        """
        if value < -1:
            return "black"
        else:
            return "white"