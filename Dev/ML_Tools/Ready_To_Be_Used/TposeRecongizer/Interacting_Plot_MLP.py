# @file Interacting_Plot_MLP.py
# @brief Contains Interacting_Plot_MLP class
# @author Simon Corbeil-letourneau (simon@heddoko.com)
# @date January 2017
# Copyright Heddoko(TM) 2017, all rights reserved
# -*- coding: utf-8 -*-

from   Interacting_Tool import Point2d          as p2d
from   Interacting_Tool import Interacting_Plot as Ip 
from   MLP_Tool_Box     import MLP              as Mlp
import matplotlib.pyplot as plt
import pandas as pd

class Interacting_Plot_MLP(Ip,Mlp):	
	"""
	Class Enheriting from Interacting_Tool and MLP classes
	Allow:
	1) to open a file, look at all the signal of each sensor and its derivatives
	2) Select the training data to recognize the t-pose
	3) Evaluate the performance of the training 
	3) Allow to classifiy subsequent frame from the same recording with the resulting Multi-Layer perceptron 
	"""
	#Constructor
	def __init__(self):		
		#specification of the keys usable (r to remove last selected frame), (a to add a new end interval frame or begining frame)
		#(p to specify that an frame interval belong to t-pose class), (n to specify that a frame interval belong to the not a t-pose class)
		#(e to end the selection)
		#we need to press a or r before n or a once this is done you can add interval of the selected type by simply click in the plot
		#to end the selection you need to press twice e and click in the plot
		KL = ['r','a','p','n','e']
		#call of the Interacting plot constructor
		Ip.__init__(self,2,1,KL)		
		self.List_p_t = []
		self.List_n_t = []
		self.Get_Data()	
		#call of the Multilayer perceptron (MLP) constructor
		Mlp.__init__(self,'lbfgs',1e-3,(28,),1)		
		self.Axis()
	def Axis(self):
		self.q  = []
		self.dq = []
		for J in range(9):
			self.q.append([[],[],[]])
			self.dq.append([[],[],[]])			
			for I in range(self.Nl):
				self.q[J][0].append( self.Q[I][6*J + 0])
				self.q[J][1].append( self.Q[I][6*J + 2])
				self.q[J][2].append( self.Q[I][6*J + 4])
				self.dq[J][0].append(self.Q[I][6*J + 1])
				self.dq[J][1].append(self.Q[I][6*J + 3])
				self.dq[J][2].append(self.Q[I][6*J + 5])					 
			self.ax[0].plot( self.q[J][0])
			self.ax[0].plot( self.q[J][1])
			self.ax[0].plot( self.q[J][2])
			self.ax[1].plot(self.dq[J][0])
			self.ax[1].plot(self.dq[J][1])
			self.ax[1].plot(self.dq[J][2])
	def Get_Data(self):		
		# import os
		# import win32gui
		# from win32com.shell import shell, shellcon
		# mydocs_pidl = shell.SHGetFolderLocation (0, shellcon.CSIDL_PERSONAL, 0, 0)
		# pidl, display_name, image_list = shell.SHBrowseForFolder (win32gui.GetDesktopWindow ()    , mydocs_pidl, "Select a file or folder",
																  # shellcon.BIF_BROWSEINCLUDEFILES , None       , None )
		# if (pidl, display_name, image_list) == (None, None, None):  print "Nothing selected"
		# else: p2f = shell.SHGetPathFromIDList (pidl)
		# print "Opening", p2f
		p2f = self.Open_File()		
		df       = pd.read_csv(p2f)
		L        = list(df)
		self.FrIndex  = df[L[0]].values
		self.Nl  = len(df)
		self.Nc  = len(L)
		self.Q   = []; 
		self.DQ  = []; 
		self.Xpop= [];
		self.Ys  = [];
		self.Yns = [];
		
		self.q  = [];
		self.dq = [];
		for P in range(10):
			self.q.append( [[],[],[]])
			self.dq.append([[],[],[]])
		
		for s in range(self.Nl):
			self.Q.append([]) ; 			
			for i in range(1,10):
				Sx = str(i)+'x'; Sy = str(i)+'y'; Sz = str(i)+'z'	
				x = df[Sx].values[s]; y = df[Sy].values[s]; z = df[Sz].values[s]
				if  s==0      : 
					xm1= 0.0; xp1= df[Sx].values[s+1]; 
					ym1= 0.0; yp1= df[Sy].values[s+1]; 
					zm1= 0.0; zp1= df[Sz].values[s+1]; 
				elif s==self.Nl-1 : 
					xm1= df[Sx].values[s-1]; xp1= 0.0;
					ym1= df[Sy].values[s-1]; yp1= 0.0;
					zm1= df[Sz].values[s-1]; zp1= 0.0;			
				else: 
					xm1= df[Sx].values[s-1]; xp1= df[Sx].values[s+1];
					ym1= df[Sy].values[s-1]; yp1= df[Sy].values[s+1];
					zm1= df[Sz].values[s-1]; zp1= df[Sz].values[s+1];			
				dx = (xp1 - xm1)/2.0 ;
				dy = (yp1 - ym1)/2.0 ;
				dz = (zp1 - zm1)/2.0 ;
				self.Q[s].append(x)  ;
				self.Q[s].append(dx) ;
				self.Q[s].append(y)  ;
				self.Q[s].append(dy) ;
				self.Q[s].append(z)  ;
				self.Q[s].append(dz) ;	
				self.q[i-1][0].append(x)  ; self.dq[i-1][0].append(dx)  ; 
				self.q[i-1][1].append(y)  ; self.dq[i-1][1].append(dy)  ; 
				self.q[i-1][2].append(z)  ; self.dq[i-1][2].append(dz)  ;
	def Click_Mouse(self,event):
		Ip.Click_Mouse(self,event)
		self.Mouse_Function(event)
		self.Key_Function()	
	def Press_Key(self,event):
		Ip.Press_Key(self,event)
		print 'event.key: ',event.key,' key_1: ',self.first_last_key,' key_2: ',self.second_last_key
	def Key_Function(self): 
		Ip.Key_Function(self)
		print self.first_last_key , self.second_last_key		
		if self.second_last_key == 'a' and self.first_last_key == 'p':
			print 'add positive new point'			
			self.List_p_t.append( int(self.Last_Point.x) ) 
			print self.List_p_t
		if self.second_last_key == 'a' and self.first_last_key == 'n':
			print 'add negative new point'	
			self.List_n_t.append( int(self.Last_Point.x) ) 
			print self.List_n_t
		if (self.second_last_key == 'r') and (self.first_last_key == 'p') and (len(self.List_p_t) !=0) :
			print 'remove positive new point'
			self.List_p_t.pop()
			print 'p: ',self.List_p_t
		if (self.second_last_key == 'r') and (self.first_last_key == 'n') and (len(self.List_n_t) !=0) :
			print 'remove negative new point'
			self.List_n_t.pop()
			print 'n: ',self.List_n_t
		if (self.second_last_key == 'e') and (self.first_last_key == 'e') :
			print '---------------------'					
			print 'p: ',self.List_p_t, ' n: ',self.List_n_t
			print 'EnD'	
			self.CreationOfExampleSet(self.List_n_t,self.List_p_t)
			self.Training(self.List_n_t,self.List_p_t)
			self.PerfoMesure()
			plt.close(self.fig)	
	def Mouse_Function(self,event): 
		Ip.Mouse_Function(self,event)
		x = event.xdata; 
		y = event.ydata;
		self.Last_Point = p2d(x,y);	
		print '(',self.Last_Point.x,',',self.Last_Point.y,')'			
		

