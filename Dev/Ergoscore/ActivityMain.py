# @file ActivityMain.py
# @brief Contains main app used to compute the Activity Score
# @author  Simon Corbeil-letourneau (simon@heddoko.com) and Pierre Giguere
# @date January 2017
# Copyright Heddoko(TM) 2017, all rights reserved
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import ActivityTools as At
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
Which_File1  = raw_input(': ')
Which_DataS1 = raw_input('Which Data sheet: ')
if len(Which_DataS1)==0:
	try:
		df  = pd.read_csv(Which_File1)
	except: 
		df  = pd.read_excel(Which_File1)
else:
	try:
		df  = pd.read_csv(Which_File1,Which_DataS1)
	except: 
		df  = pd.read_excel(Which_File1,Which_DataS1)
  
print('Do you want to add a parameter sheet? Yes/No')
Input = raw_input(':')
if(Input)=='Yes':
    print 'Which file in the actual directory?'
    print '-----------------------------------'
    pprint(dirlist)
    print '-----------------------------------'
    Which_File2 = raw_input(': ')
    Which_DataS2 = raw_input('Which Data sheet: ')
    if len(Which_DataS1)==0:
        try:
		dfA  = pd.read_csv(Which_File2)
        except: 
		dfA  = pd.read_excel(Which_File2)
    else:
        try:
		dfA  = pd.read_csv(Which_File2,Which_DataS2)
        except: 
		dfA  = pd.read_excel(Which_File2,Which_DataS2)
else:
    dfA = pd.read_excel('AnthropoSample.xlsx')

T, ThetaCapteur, DeltaCapteur, DeltaCapteur2, Velocity, Proportion, MaxV, IMaxV, MeanD, StdD, Ki  = At.Angles(df, dfA)
At.AnglesPlot(df, T, ThetaCapteur, DeltaCapteur, DeltaCapteur2, Velocity, Proportion, MaxV, IMaxV, MeanD, StdD, Ki)