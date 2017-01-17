# @file MLP_Tool_Box.py
# @brief Contains MLP class
# @author Simon Corbeil-letourneau (simon@heddoko.com)
# @date January 2017
# Copyright Heddoko(TM) 2017, all rights reserved
# -*- coding: utf-8 -*-


from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class MLP:	
	"""
	Allow to create Multi-layers Perceptron
	with:  
	- training method
	- prediction method 
	- performance mesurement
	- training set creation
	""" 
	#the constructor of the MLP class
	#arguments need : 
	# 1)So : type of solver
	# 2)Al : regularisation term
	# 3)Hi : tuple of lengh of the number of hidden neurones in the hidden layer of MLP
	# 4)Ra : Seed of random state   
	def __init__(self,So,Al,Hi,Ra):					
		self.clf  = MLPClassifier(solver=So, alpha=Al, hidden_layer_sizes=Hi, random_state=Ra)
		self.Xpop = np.array(self.Q) 
		self.frac = 2.0/3.0		
		self.Ypred  = []		
		self.Xva    = []
		self.Xtr    = []	
		self.ysolva = []
		self.ysoltr = []
		self.Ys     = []
		self.Yns    = []
	# Produce a training set and a validation set of the prescribled size (by frac: fraction of the exemple reserved for training)
	# usable by sklearn.neural_network import MLPClassifier
	def CreationOfExampleSet(self,iWrong,iGood):	
		Ng  = len(iGood); Nw  = len(iWrong);	
		iOnes       = []; iZeros       = [];		
		for i in range(Ng/2):iOnes += range( int(iGood[2*i] ), int(iGood[2*i+1]) ) 
		for i in range(Nw/2):iZeros+= range( int(iWrong[2*i]), int(iWrong[2*i+1]))
		
		self.Ysol  = [ 1    for i in  iOnes] + [   0  for i in  iZeros];
		Exmpl = [ self.Q[i] for i in  iOnes] + [ self.Q[i] for i in  iZeros];
		NYsol = len(self.Ysol)
		r     = np.random.permutation(len(self.Ysol))		
		itend = int(self.frac*NYsol) 
		ivinit= int(self.frac*NYsol)+1     		
		
		self.ysol   = np.array(self.Ysol)  
		self.ysolr  = self.ysol[r]		
		self.Xex    = np.array(Exmpl) 
		self.Xtr    = self.Xex[r[:itend]]  
		self.Xva    = self.Xex[r[ivinit:]]
		self.ysoltr = self.ysol[r[:itend]] 
		self.ysolva = self.ysol[r[ivinit:]]
	# Train the Neural-Network with the good and wrong exemple (T-pose or not T-pose cases)
	def Training(self,iWrong,iGood):		
		self.CreationOfExampleSet(iWrong,iGood)
		Nbtr  = len(self.ysoltr)
		Nbva  = len(self.ysolva)
		self.clf.fit(self.Xtr, self.ysoltr)  
		self.Ypred = self.clf.predict(self.Xva) 
	# Show the performance on the validation test and on the rest of the none classified recoding 
	def PerfoMesure(self):
		Nbtr  = len(self.ysoltr)
		Nbva  = len(self.ysolva)
		Error = np.sum(np.abs(self.Ypred - self.ysolva));
		Fn    = [(self.Ypred[i] == 0) and (self.ysolva[i] == 1) for i in range(len(self.Ypred))].count(True)
		Fp    = [(self.Ypred[i] == 1) and (self.ysolva[i] == 0) for i in range(len(self.Ypred))].count(True)
		Tn    = [(self.Ypred[i] == 0) and (self.ysolva[i] == 0) for i in range(len(self.Ypred))].count(True)
		Tp    = [(self.Ypred[i] == 1) and (self.ysolva[i] == 1) for i in range(len(self.Ypred))].count(True)
		Tpr   = 1.0*Tp/(Tp + Fn)
		Fpr   = 1.0*Fp/(Fp+Tn)
		Perf  = 100.0*(Nbva - Error)/Nbva 
		self.Ypop = self.Predict(self.Xpop)

		print 'fraction of good answers :Perf:',Perf
		print 'Sensitivity  (Tpr = Tp/(Tp+Fn)):',Tpr
		print '1-Specificity(Fpr = Fp/(Tn+Fp)):',Fpr
		print 'Nb of training examples   : ', Nbtr 
		print 'Nb of validation examples : ', Nbva
		print 'Tp:',Tp,'  Fp:',Fp,'  Tn:',Tn,'  Fp:',Fp
			
		f2 = plt.figure(2)
		plt.subplot(111)
		plt.plot([0.0,1.0],[0.0,1.0],'r')
		plt.xlim([-0.1, 1.1])
		plt.ylim([ 0.0, 1.1])
		plt.plot(Fpr,Tpr,'*')
		plt.xlabel('1-specificity(Fp)')
		plt.ylabel('sensitivity(Tp)')				
		plt.show(f2)
		f3 = plt.figure(3)
		plt.subplot(211)
		for I  in range(9):
			plt.plot(self.FrIndex,self.q[I][0])
			plt.plot(self.FrIndex,self.q[I][1])
			plt.plot(self.FrIndex,self.q[I][2])
		plt.plot(self.FrIndex,self.Ypop,'*r')
		plt.plot(self.Ypop,'*r')
		plt.subplot(212)
		for I  in range(9): 
			plt.plot(self.FrIndex,self.dq[I][0])
			plt.plot(self.FrIndex,self.dq[I][1])
			plt.plot(self.FrIndex,self.dq[I][2])
		plt.plot(self.Ypop,'*r')
		plt.show(f3)
	# Methods used to classify new case not already classified
	def Predict(self,Xex):				
		return self.clf.predict(Xex) 
	
