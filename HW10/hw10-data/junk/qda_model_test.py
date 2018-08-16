
import random
import time


import numpy as np
import numpy.linalg as LA


from numpy.linalg import inv
from numpy.linalg import det

from projection import Project2D, Projections


class QDA_Model(): 

	def __init__(self,class_labels):

		self.lmbda = 0.001
		self.NUM_CLASSES = len(class_labels)



	def compute_class_means(self,X,Y): 

		self.class_data = {}

		for i in range(len(X)):

			x = X[i]
			y = Y[i]

			if str(y) in self.class_data:
				self.class_data[str(y)].append(x)
			else:
				self.class_data[str(y)] = [x]

		self.class_means = {}

		for cls in self.class_data.keys():

			data = self.class_data[cls]

			data_n = np.array(data)
			mu = np.mean(data_n,0)

			self.class_means[cls] = mu

			



	def compute_class_cov(self,X,Y):

		
		N = len(X)

		x_dim = np.max(X[0].shape)
		
		self.class_inv_cov = {}
		self.class_cov = {}

		for cls in self.class_data.keys():

			C = np.zeros([x_dim,x_dim], dtype=complex)
			data = self.class_data[cls]
			mu = self.class_means[cls]

			N = len(data)

			for i in range(N):

				x = data[i]
				C += np.outer((x-mu).T,(x-mu))

			reg = 0.01*np.eye(C.shape[0])

			C = C/float(N-1)
			C = C + reg

			self.class_inv_cov[cls] =  inv(C)
			self.class_cov[cls] = np.copy(C)
		

			

		


	def train_model(self,X,Y): 

		self.compute_class_means(X,Y)
		self.compute_class_cov(X,Y)

	def compute_likelihood(self,clss,x):

		cov = self.class_cov[clss]
		inv_cov = self.class_inv_cov[clss]
		mean = self.class_means[clss]
	

		prod_term =  -np.log(det(cov)) - np.matmul((x-mean).T,np.matmul(inv_cov,(x-mean)))
		# prod_term = -np.linalg.slogdet(cov)[1] - np.matmul((x - mean).T, np.matmul(inv_cov, (x - mean)))

		return prod_term 



	def eval(self,x):

		prediction = np.zeros(self.NUM_CLASSES)
		i = 0
		for i in range(self.NUM_CLASSES):
			prediction[i] = self.compute_likelihood(str(i),x)
			i = i+1
		

		return np.argmax(prediction)

	
