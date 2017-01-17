#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 10:21:30 2016

@author: Pierre-Luc
"""
"""Ce programme sert a convertir les graphiques des quaternions en angles d'Euler. Le fichier utilise au depart doit donc contenir des quaternions. Il affiche egalement la norme. Les fonctions utilisees dans ce programme sont definies dans le fichier QuatDisplayFunctions"""
import QuatDisplayFunctions as QD

a = True
while a: 
    filename = QD.GetFile()  #Getting the file  with Tkinter
    data = QD.PdConvert(filename)  #Creating a pd Data Frame from the xls file
    G = QD.AngleGraph(data)  #Creating the graph ??
    Exit1= QD.Exit()    # Asking the user if he wants to continue or exit
    if Exit1 == False:
        a = False       # Exiting if answer is 'y'
    elif Exit1 == True:  
        a = True       # Asking user which graph he wants to see (line 13)