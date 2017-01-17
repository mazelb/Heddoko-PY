# @file Interacting_Tool.py
# @brief Contains Point2d class and Interacting_plot class
# @author Simon Corbeil-letourneau (simon@heddoko.com)
# @date January 2017
# Copyright Heddoko(TM) 2017, all rights reserved
# -*- coding: utf-8 -*-

#Definition of point class in 2d 
class Point2d:
	"""
	Point in 2d: P(x,y)
	"""
	#constructor
	def __init__(self,x,y):
		self.x = x
		self.y = y
		self.coor = [self.x,self.y]
	#setter of x 
	def setx(self,x):
		self.x = x
		self.coor[0] = x
	#setter of y 
	def sety(self,y):
		self.y = y
		self.coor[1] = y
	#setter of x and y 
	def setc(self,x,y):
		self.x = x
		self.y = y
		self.coor[0] = y

#Definition of partial class of plot allowing interaction through mouse, and key board 
class Interacting_Plot:	
	"""
	Allow to create plot in which the user can interact
	with:  
	- mouse click
	- key board 
	- Click_Mouse, Key_Function, Mouse_Function methods need to be 
	- Implement in the class which will henerit from Interacting_Plot
	""" 
	#constructor
	#need as arguments the number of lines and the number of colomn to organize the all the subplot in the page 
	#need also the list of all the key that will be use
	def __init__(self,nLine,nCol,Keys_List):
		import matplotlib.pyplot as plt
		self.first_last_key  = '*'
		self.second_last_key = '*'
		self.Last_Point      = []
		self.Interval_Even   = [[],[]]	
		self.set_of_key      = set(Keys_List)
		self.fig, self.ax    = plt.subplots(nLine,nCol)		
		self.fig.canvas.mpl_connect('key_press_event',self.Press_Key)
		self.fig.canvas.mpl_connect('button_press_event',self.Click_Mouse)		
		self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)	
	#Allow to trigger an action when a key is pressed
	def Press_Key(self,event):
		self.second_last_key = self.first_last_key
		self.first_last_key  = event.key		
	#Allow to trigger an action when a click (any button) from the mouse is done  
	def Click_Mouse(self,event):
		print '----------------------'
	#Allow to define the key usable and the action resulting when the key is pressed
	def Key_Function(self): 
		print '----------------------'
	#Allow to define the action resulting when a click is done on the mouse
	def Mouse_Function(self,event): 
		print '----------------------'
	#Allow to open a file in the somewhere on Document  
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




