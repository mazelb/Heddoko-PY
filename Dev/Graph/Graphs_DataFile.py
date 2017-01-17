import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import math
import sys

class Graph:
	nfig = 0
	def __init__(self,lx = " ",ly = " ",title = " ",xlim = " ",ylim = " ",xname = " ",yname = " "):
		self.x = lx
		self.y = ly
		self.title = title
		self.xlim  = xlim
		self.ylim  = ylim
		self.xname = xname
		self.xname = yname
		
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

	def Plot(self):
		p2f = self.Open_File()
		try: 
			self.df = pd.read_excel(p2f)
		except: 
			try : 
				self.df = pd.read_csv(p2f)
			except: 
				print "An exception has occured"
				print "make sure the file exist"
				print "and make sure it has one of the following extention:" 
				print "1) csv "
				print "2) xslx"
		print "this is the list of varialbe contained in the file:"
		print list(self.df)
		self.x = raw_input("Chose the list of variables x to plot in the following format x1, x2, x3; x4, x5; x6, x7;")
		self.y = raw_input("Chose the list of variable  y to plot in the following format y1, y2, y3; y4, y5; y6, y7;") 
		self.Show()
	
	def Show(self):
		X  = str.split(';',self.x)
		Y  = str.split(';',self.y)
		ng = len(X)
		self.ax = []
		for I in range(ng):
			self.ax.append([])
			Xg = str.split(',',X[I])
			Yg = str.split(',',Y[I])
			nv = len(X)
			Graph.nfig +=1
			self.fig    = plt.figure(Graph.nfig)
			for J in range(nv):
				self.ax[I].append([])
				self.ax[I][J] = plt.subplot(nv,1,J+1)
				self.ax[I][J].plot(self.df[Xg[J]],self.df[Yg[J]],'.')
			plt.show(self.fig)	
		
