import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import math
import sys

class Graph:
	nfig = 0
	#Default constructor
	def __init__(self):
		"""
		A Graph object can read, and plot the data contain on in a .csv or .xlsx file.
		"""
	#Method who enable the user to select through the menu the file containing the data	
	def Open_File(self):
		import os
		import win32gui
		from win32com.shell import shell, shellcon
		mydocs_pidl = shell.SHGetFolderLocation (0, shellcon.CSIDL_PERSONAL, 0, 0)
		pidl, display_name, image_list = shell.SHBrowseForFolder (win32gui.GetDesktopWindow (), mydocs_pidl, "Select a file or folder",shellcon.BIF_BROWSEINCLUDEFILES , None       , None )
		if (pidl, display_name, image_list) == (None, None, None):  print "Nothing selected"
		else: p2f = shell.SHGetPathFromIDList (pidl)
		print "Opening", p2f
		return p2f		

	#Method who allow to the user to specified the way he(she) wants to plot the data
	def Plot(self):
		#openning of the file 
		p2f = self.Open_File()
		try: 
			self.df = pd.read_excel(p2f)      #try for a excel file 
		except: 
			try : 
				self.df = pd.read_csv(p2f)    #otherwise try for a csv file 
			except:                           #error message if not xslx or csv file 
				print "An exception has occured"
				print "make sure the file exist"
				print "and make sure it has one of the following extention:" 
				print "1) csv "
				print "2) xslx"
		print " "
		print "this is the list of variables contained in the file:"
		print " "		
		print list(self.df)
		print " "
		print "In the following, the format required is :  "
		print "; is the separator of liste of variables on differents figures"
		print ", is the separator of liste variable on differents subplots"
		print ". is the separator of variable on the same subplot"
		print "the Y must be always specified "
		print "the X must be specified for all figures and subplots"
		print "the following lines will produce 3 figures with :"
		print "2 subplots (2 curves on each)"
		print "3 subplots and one subplot (1 , 2 and 3 curves)"
		print "1 subplot 3 curves "
		print "x: t,t;t,t,t;t"
		print "y: y1.y2,y3.y4;y5,y6.y7,y1.y2.y4;y1.y2.y4"
		print " "
		# specification by the user of the configuration of all the plot needed
		self.x = raw_input("Chose the list of variables x ")
		self.y = raw_input("Chose the list of variables y ") 
		# show the plots 
		self.Show()
	
	#method who show the figure needed 
	def Show(self):
		x = self.x.strip()   # strip the whitespaces
		y = self.y.strip()	 # strip the whitespaces
		# figure
		Xf  = str.split(x,';') # spit the x variables string in figure 
		Yf  = str.split(y,';') # spit the y variables string in figure
		nf = len(Yf) #number of figures
		self.ax = [] #array of axis for all the subplots
		for I in range(nf):
			self.ax.append([])				
			Xsbp = str.split(Xf[I],',') # spit the x variables string of one figure in subplots
			Ysbp = str.split(Yf[I],',') # spit the y variables string of one figure in subplots
			nsbp = len(Ysbp)            # number of subplot for the actual figure
			Graph.nfig +=1              
			self.fig    = plt.figure(Graph.nfig)  #created a new figure
			for J in range(nsbp):                 #loop of all the subplots of the actual figure							
				Ysbpv = str.split(Ysbp[J],'.')
				nsbpv = len(Ysbpv)	              #number of y variable for J subplot of the I figure : nota bene the x variable is always the same for a given subplot
				self.ax[I].append([])             
				self.ax[I][J] = plt.subplot(nsbp,1,J+1) #creation of creation of the J subplot of the I figure
				ylabel = ''                                 
				xlabel = Xsbp[J]                        #x label of J subplot 
				#loop on multiple curve for a given subplot 
				for K in range(nsbpv):					
					self.ax[I][J].plot(self.df[Xsbp[J]],self.df[Ysbpv[K]],'.')			
					ylabel+= Ysbpv[K] + ' '                                     # creation of the y label of the J subplot
				self.ax[I][J].set_xlabel(xlabel) 
				self.ax[I][J].set_ylabel(ylabel)
		plt.show()	
