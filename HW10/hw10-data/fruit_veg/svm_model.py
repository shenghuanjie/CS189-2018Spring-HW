from numpy.random import uniform
import random
import time

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys

from sklearn.svm import LinearSVC
from projection import Project2D, Projections


class SVM_Model(): 

	def __init__(self,class_labels,projection=None):

		self.C = 1.0
		



	def train_model(self,X,Y): 

		# a seed is needed to get repeatable results
		self.svm = LinearSVC(C=self.C, random_state=10)

		self.svm.fit(X,Y)
		
		

	def eval(self,x):

		prediction = self.svm.predict(np.array([x]))
	
		return prediction[0]

	def scores(self, x):
		return self.svm.decision_function(np.array(x))


