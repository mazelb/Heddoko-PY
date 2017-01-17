import matplotlib.pyplot as plt
import ErgoScoreToolsRevamp as Er
import numpy.random as npr
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib
import math
import sys
import os


print 'Which Excel/CSV file do you want me to Analyse in the actual Directory?'
print '----------------------------------------'
dirlist = os.listdir("./")
pprint(dirlist)
print '----------------------------------------'
Which_File  = raw_input(': ')
Which_DataS = raw_input('Which Data sheet: ')
if len(Which_DataS)==0:
	try:
		df  = pd.read_csv(Which_File)
	except: 
		df  = pd.read_excel(Which_File)
else:
	try:
		df  = pd.read_csv(Which_File,Which_DataS)
	except: 
		df  = pd.read_excel(Which_File,Which_DataS)
ListCol = list(df)
Dt1,T,df1,MinD,IMinD,MeanD,StdD  = Er.Distance(df)
Er.ErgoPlot(Dt1,T,ListCol,df1,df,MinD,IMinD,MeanD,StdD)