from numpy.random import uniform
import random
import time

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys

from sklearn.linear_model import LogisticRegression
from projection import Project2D, Projections


class Logistic_Model(): 

	def __init__(self,class_labels,projection=None):

		self.C = 1.0
		



	def train_model(self,X,Y): 


		self.lr = LogisticRegression(C=self.C)

		self.lr.fit(X,Y)
		
		

	def eval(self,x):

		prediction = self.lr.predict(np.array([x]))
	
		return prediction[0]

	def scores(self, x):
		return self.lr.decision_function(np.array(x))


	