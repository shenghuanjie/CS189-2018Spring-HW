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
from sklearn.svm import LinearSVC
from projection import Project2D, Projections


class LDA_Model(): 

	def __init__(self,class_labels):

		###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
		self.reg_cov = 0.001
		self.NUM_CLASSES = len(class_labels)



	def train_model(self,X,Y): 
		''''
		FILL IN CODE TO TRAIN MODEL
		MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

		'''
		X = np.array(X,dtype=float)
		Y = np.array(Y,dtype=float)
		X0 = X[Y == 0]
		self.mean0 = np.mean(X0, axis=0)
		X0n = X0-self.mean0
		self.cov0 = X0n.T @ X0n/X0.shape[0]
		X1 = X[Y == 1]
		self.mean1 = np.mean(X1, axis=0)
		X1n = X1-self.mean1
		self.cov1 = X1n.T @ X1n/X1.shape[0]
		X2 = X[Y == 2]
		self.mean2 = np.mean(X2, axis=0)
		X2n = X2-self.mean2
		self.cov2 = X2n.T @ X2n/X2.shape[0]
		self.cov = (X0.shape[0]*self.cov0+X1.shape[0]*self.cov1+X2.shape[0]*self.cov2)/X.shape[0] \
		            + self.reg_cov*np.eye(self.cov0.shape[0])

		
		

	def eval(self,x):
		''''
		Fill in code to evaluate model and return a prediction
		Prediction should be an integer specifying a class
		'''
		x = np.array(x,dtype=float)
		x0n = x - self.mean0
		l0 = x0n.T @ inv(self.cov) @x0n
		minclass=0
		minl=l0
		x1n = x - self.mean1
		l1 = x1n.T @ inv(self.cov) @x1n
		if l1<minl:
			minclass=1
			minl = l1
		x2n = x - self.mean2
		l2 = x2n.T @ inv(self.cov) @x2n
		if l2<minl:
			minclass=2
			minl = l2
		return minclass

