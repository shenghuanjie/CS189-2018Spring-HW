import random
import time


import glob
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys
from numpy.linalg import inv
from numpy.linalg import det
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import LinearSVC
from projection import Project2D, Projections


class LDA_Model(): 

	def __init__(self,class_labels):

		self.lmbda = 0.001
		self.NUM_CLASSES = len(class_labels)



	def compute_class_means(self,X,Y): 

		class_data = {}

		for i in range(len(X)):

			x = X[i]
			y = Y[i]

			if str(y) in class_data:
				class_data[str(y)].append(x)
			else:
				class_data[str(y)] = [x]

		self.class_means = {}

		for cls in class_data.keys():

			data = class_data[cls]

			data = np.array(data)
			mu = np.mean(data,0)

			self.class_means[cls] = mu

			



	def compute_class_cov(self,X,Y):

		
		N = len(X)

		x_dim = np.max(X[0].shape)
		C = np.zeros([x_dim,x_dim], dtype=complex)


		for i in range(N):

			x = X[i]
			y = Y[i]

			mu = self.class_means[str(y)]

			C += np.outer((x-mu).T,(x-mu))
			#print((x-mu).T.shape)
			#print(x.shape)
			#print((np.dot((x-mu).T,(x-mu))).shape)

			

		reg = 0.001*np.eye(C.shape[0])

		C = C/float(N-1)
		C = C + reg
		print(C)
		self.inv_cov =  inv(C)


	def train_model(self,X,Y): 

		self.compute_class_means(X,Y)
		self.compute_class_cov(X,Y)

	def compute_likelihood(self,clss,x):

		inv_cov = self.inv_cov
		mean = self.class_means[clss]
	

		prod_term = -1*np.matmul((x-mean).T,np.matmul(inv_cov,(x-mean)))

		return prod_term 



	def eval(self,x):

		prediction = np.zeros(self.NUM_CLASSES)
		i = 0
		for i in range(self.NUM_CLASSES):
			prediction[i] = self.compute_likelihood(str(i),x)
			i = i+1
		

		return np.argmax(prediction)
