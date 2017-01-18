# @file ErogScoreToolsRevamp.py
# @brief Contains Tools used to compute ErgoScore
# @author  Simon Corbeil-letourneau (simon@heddoko.com) and Pierre Giguere
# @date January 2017
# Copyright Heddoko(TM) 2017, all rights reserved
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import math
import sys

def ErgoPlot(Dt1,T,ListCol,df1,df,MinD,IMinD,MeanD,StdD):
		exception = set(['Frame Index','TPose Value','Timestamp'])
		DL    = ListCol
		###ErgoScore lines (Mean and StD)###
		xmean = [T[1],T[len(T)-1]]
		ymean = [MeanD,MeanD]	
		xstd  = [T[1],T[len(T)-1]]
		ystdn = [MeanD-StdD,MeanD-StdD]	
		ystdp = [MeanD+StdD,MeanD+StdD]
		plt.figure() 		
		for l in range(0,len(DL)):
			if not(DL[l] in exception):
				ax1 = plt.subplot(311)
				plt.plot(T,df[DL[l]][0:len(T)])
				plt.legend(loc = 6,fontsize = 8)
				plt.title("Distance from rest for each joint")				
				plt.xlim(T[0],T[len(T)-1])
				plt.subplot(312,sharex=ax1)
				plt.plot(T,df1[l],label=DL[l])
				plt.title("Weight of the contribution of each joint to the Global distance")
				plt.legend(loc = 6,fontsize = 8)
		plt.subplot(313,sharex=ax1)
		plt.title("Global risk score releated to how far a user is from the max extention (summation of each joint contribution)")
		plt.plot(T[IMinD],MinD,'*r',label="Min risk score: "+ str(round(MinD)))	
		plt.plot(xmean,ymean,'r',label="Mean score: "+ str(round(MeanD)))
		plt.plot(xstd,ystdn,'-b',label="Mean - Std: "+ str(round(MeanD - StdD)))
		plt.plot(xstd,ystdp,'-b',label="Mean + Std: "+ str(round(MeanD + StdD)))
		plt.plot(T,Dt1,'g',label="Risk score")	
		plt.legend()			
		plt.show()
		L   = list(df)
		l   = len(L)
		bdb    = []
		Moybdb = []
		for i in range(0,l-2): 
			bdb.append([])
		for i in range(0,len(df['Frame Index'])-2): 
			bdb[0].append(df.iat[i+1,0])
			Moybdb.append(0.0)
		return bdb
		
def Distance(df):
		exception = set(['Frame Index','TPose Value','Timestamp'])
		T     = []      ; 	Dt    = []           ; Dt1 = []    ;		
		L     = list(df); 	Ll    = len(df[L[0]]); Lc  = len(L);	
		MinD  = 100.0   ; 	IMinD = 0	          ; df1 = []    ;
		for i in range(Lc):df1.append([])		
		for i in range(0,Ll):				
			Dt.append(0.0);	T.append(0.0); Dt1.append(0.0); 				
			#print i,Ll
			for c in range(0,Lc):
				Col = L[c]	
				#print i,Col,Ll
				###Add absolute values of angles for each segment in df1.### 
				df1[c].append(math.fabs(df[Col][i]))
				if not(Col in exception):
					Dt[i] = Dt[i] + (df[Col][i])**2	
					###Add the distance in Dt1###
					Dt1[i]= Dt1[i] + (1 - math.fabs(df[Col][i])/180.0)
				else:
					T[i] = df[L[0]][i]
			###Average the distance values over the number of joints.###		
			Dt1[i] = 100*Dt1[i]/Lc				
			Dt[i] = math.sqrt(Dt[i])
			for Col in range(0,Lc): 
				if not(Col in exception): 
					df1[Col][i] = df1[Col][i]/Dt[i]			
			if MinD > Dt1[i]:
				MinD  = Dt1[i];	IMinD = i;
		###s to find the MeanD from the distance value average###	
		s  = []
		for r in Dt1: 			
			if not(math.isnan(r)): 		
				s.append(r)			
		MeanD = np.mean(s)		
		StdD  = np.std(s)		
		return Dt1,T,df1,MinD,IMinD,MeanD,StdD