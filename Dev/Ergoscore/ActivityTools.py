# @file ActivityTools.py
# @brief Contains Tools used to compute the Activity Score
# @author  Simon Corbeil-letourneau (simon@heddoko.com) and Pierre Giguere
# @date January 2017
# Copyright Heddoko(TM) 2017, all rights reserved
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy.random as npr
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib
import math
import sys
import os

def Angles(df, dfA):
    ThetaCapteur = [] ; DeltaCapteur = [] ; DeltaCapteur2 = []   ;
    Velocity = []     ; Ll    = len(df)   ; Proportion = []      ;
    MaxV = 0          ; IMaxV = 0         ; T = []               ; 
    Poids = []        ; Size = []         ; InertiaDistance = [] ;
    Inertia = []      ; Ki = []           ;

    
    for c in range(1,10): 
        ###Append lists for each captor.
        ThetaCapteur.append([]) ; Proportion.append([])    ;
        DeltaCapteur.append([]) ; DeltaCapteur2.append([]) ;
        Ki.append([])
        ###Append Theta with values of each angles from df. Match Delta to same size as Theta. Repeat for x-y-z.
        ColX = str(c) + 'x'
        ThetaCapteur[c-1].append(df[ColX].values)
        DeltaCapteur[c-1].append([])
        ColY = str(c) + 'y'
        ThetaCapteur[c-1].append(df[ColY].values)
        DeltaCapteur[c-1].append([])
        ColZ = str(c) + 'z'
        ThetaCapteur[c-1].append(df[ColZ].values)
        DeltaCapteur[c-1].append([])
    for k in range(0,Ll-1):
        for i in range(1,10):
            for j in range(0,3):
                ###Approximate instant angular speed (Delta) through (Theta+1 - Theta-1)/2. Exceptions on end and beggining of list.
                if (k==0):
                    DeltaCapteur[i-1][j].append(1.0*(ThetaCapteur[i-1][j][k+1] - ThetaCapteur[i-1][j][k]))
                elif (k==(Ll-1)):
                    DeltaCapteur[i-1][j].append(1.0*(ThetaCapteur[i-1][j][k] - ThetaCapteur[i-1][j][k-1]))
                else:
                    DeltaCapteur[i-1][j].append(1.0*((ThetaCapteur[i-1][j][k+1] - ThetaCapteur[i-1][j][k-1])/2.0))
            ###Add squares of each components to Delta2 for each captor.
            DeltaCapteur2[i-1].append(((DeltaCapteur[i-1][0][k])**2 + (DeltaCapteur[i-1][1][k])**2 + (DeltaCapteur[i-1][2][k])**2))
        ###Velocity is the sum of Delta2 for all 9 captors. Represents an idea of the activity at each period in time.
        Velocity.append((DeltaCapteur2[0][k]) + (DeltaCapteur2[1][k]) + (DeltaCapteur2[2][k]) + (DeltaCapteur2[3][k]) + (DeltaCapteur2[4][k]) + (DeltaCapteur2[5][k]) + (DeltaCapteur2[6][k]) + (DeltaCapteur2[7][k]) + (DeltaCapteur2[8][k]))
        
        ###Comparison in time between Delta2 for a single captor and Velocity.
        for p in range(1,10):
            Proportion[p-1].append(((DeltaCapteur2[p-1][k])/(Velocity[k]))*100.0)
           
        if (Velocity[k] > MaxV):
            MaxV =  Velocity[k]
            IMaxV = k
        T.append(k)
        
    for m in range(0,Ll-1):
        for n in range(1,10):
            if ((Velocity[m]) < (MaxV*0.10)):
                Proportion[n-1][m] = 0
                
    listA=list(dfA)
    DL = listA
    lcA = len(DL)
    MeanD = np.mean(Velocity)
    StdD = np.std(Velocity)
    
    ###Inertia calculations
    ###Weight repartitions for each body segments.
    Poids.append(dfA[DL[0]].values)
    Poids.append((Poids[0])*0.022)
    Poids.append((Poids[0])*0.028)
    Poids.append((Poids[0])*0.061)
    Poids.append((Poids[0])*0.100)
    Poids.append((Poids[0])*0.497)
    Poids.append((Poids[0])*0.081)
    
    ###Size of each segments. Calculate distances from rotation point to center of mass.
    for o in range(1,lcA):
        Size.append(dfA[DL[o]].values)    
        
    InertiaDistance.append((Size[0])*0.682)
    InertiaDistance.append((Size[1])*0.436)
    InertiaDistance.append((Size[2])*0.606)
    InertiaDistance.append((Size[3])*0.433)    
    InertiaDistance.append((Size[4])*0.500)
    InertiaDistance.append((Size[5])*1.0)
    
    ###Calculating the inertia for each segment based on weight and distance to center of mass.
    for p in range(0,6):
        Inertia.append(Poids[p+1]*InertiaDistance[p]**2)
    
    ###Calculate rotational kinetic energy from segment inertia and angular speed squared.
    for q in range(0, Ll-1):
        Ki[0].append((Inertia[4]*DeltaCapteur2[0][q])*0.5)
        Ki[1].append((Inertia[1]*DeltaCapteur2[1][q])*0.5)
        Ki[2].append((Inertia[0]*DeltaCapteur2[2][q])*0.5)
        Ki[3].append((Inertia[1]*DeltaCapteur2[3][q])*0.5)
        Ki[4].append((Inertia[0]*DeltaCapteur2[4][q])*0.5)
        Ki[5].append((Inertia[3]*DeltaCapteur2[5][q])*0.5)
        Ki[6].append((Inertia[2]*DeltaCapteur2[6][q])*0.5)
        Ki[7].append((Inertia[3]*DeltaCapteur2[7][q])*0.5)
        Ki[8].append((Inertia[2]*DeltaCapteur2[8][q])*0.5)
 
    return T, ThetaCapteur, DeltaCapteur, DeltaCapteur2, Velocity, Proportion, MaxV, IMaxV, MeanD, StdD, Ki
    
def AnglesPlot(df, T, ThetaCapteur, DeltaCapteur, DeltaCapteur2, Velocity, Proportion, MaxV, IMaxV, MeanD, StdD, Ki):
    nl = len(df)
#    xmean = [T[0], T[-1]]
#    ymean = [MeanD, MeanD]
#    ystdd = [(MeanD-StdD), (MeanD-StdD)]
#    ystdu = [(MeanD+StdD), (MeanD+StdD)]
    #plt.figure(1)
    for c in range (1, 10):
        plt.figure(c)        
        ax1 = plt.subplot(221)
        plt.plot(df['Frame Index'].values[0:len(df)-1],Ki[c-1][0:nl-1], label="Capteur " + str(c))
        plt.plot(df['Frame Index'].values[0:len(df)-1],Ki[c-1][0:nl-1], '.', label="Capteur " + str(c))
        plt.legend(loc = 6,fontsize = 8)
        plt.title("Aproximate kinetic energy of rotation for each captor")				
        plt.xlim(T[0],T[len(T)-1])
        
        plt.subplot(222, sharex=ax1)
        plt.plot(df['Frame Index'].values[0:len(df)-1],DeltaCapteur2[c-1][0:nl-1], label="Capteur " + str(c))
        plt.plot(df['Frame Index'].values[0:len(df)-1],DeltaCapteur2[c-1][0:nl-1], '.', label="Capteur " + str(c))
        plt.legend(loc = 6,fontsize = 8)
        plt.title("Summs of squared angle for each captor")
        plt.xlim(T[0],T[len(T)-1])
        
        plt.subplot(223, sharex=ax1)
        plt.plot(df['Frame Index'].values[0:len(df)-1],Proportion[c-1][0:nl-1])
        plt.title("Weight of the contribution of each captor to the global activity")
        plt.legend(loc = 6,fontsize = 8)
        plt.xlim(T[0],T[len(T)-1])
        
        plt.subplot(224, sharex=ax1)
        for k in range(1,4):
            plt.plot(df['Frame Index'].values[0:len(df)-1],ThetaCapteur[c-1][k-1][0:nl-1])
        plt.title("Theta Values")
        plt.legend(loc = 6, fontsize = 8)
        plt.xlim(T[0],T[len(T)-1])
#    plt.subplot(313,sharex=ax1)
#    plt.title("Global activity score related to each joint angular speed (summation of each joint)")
#    plt.plot(IMaxV,MaxV,'*r',label="Max activity: "+ str(round(MaxV)))	
#    plt.plot(xmean,ymean,'r',label="Average activity: "+ str(MeanD))
#    plt.plot(xmean,ystdd,'-b',label="Mean - Std: "+ str(MeanD - StdD))
#    plt.plot(xmean,ystdu,'-b',label="Mean + Std: "+ str(MeanD + StdD))
#    plt.plot(df['Frame Index'].values[0:len(df)-1],Velocity,'g',label="Activity")
    plt.legend()
    
    plt.show()
    