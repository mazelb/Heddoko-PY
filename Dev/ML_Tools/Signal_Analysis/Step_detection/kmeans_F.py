#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Programme permettant de faire du clustering dans le but de separer les sauts normaux/anormaux sur UN capteur a la fois. Ce programme cree 3 figures (le signal, le clustering en 2 dimensions et le clustering en 3 dimensions. Une bonne amelioration au programme serait d'ajouter une demande a l'utilisateur pour savoir quel capteur il souhaite analyser. L'interface TKinter pourrait aussi etre utile pour selectionner le fichier. 
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
##Seaborn pour avoir des meilleurs graphiques
import seaborn
from mpl_toolkits.mplot3d import Axes3D

#Importer un fichier Raw Data
data = pd.read_excel('/Users/Pierre-Luc/Documents/Cegep/Rosemont/Heddoko/day6/Data/ElbowLRawData.xlsx')

 
# Calculer les sauts sur les differents axes. Pour l'instant, le programme accepte seulement un "capteur" (ici, c'est le 3)
serx = np.abs(data['3x'].diff(1)).values
sery = np.abs(data['3y'].diff(1)).values
serz = np.abs(data['3z'].diff(1)).values
# Change la taille de l'array des sauts
sery = sery[1:]
serx = serx[1:]
serz = serz[1:]
#Liste s que nous allons soumettre a KMeans 
s=[]
#Creation d'une liste de listes avec les sauts [x, y, z]
for i in range(len(serx)): 
    s.append([serx[i], sery[i], serz[i]])
#Kmeans avec parametres k=2 groupes				
y_pred = KMeans(n_clusters=2).fit_predict(s)
#Liste ayant pour but de changer la couleur des points pour ameliorer la visibilite
yp1=[]
for i in range(len(y_pred)): 
	if y_pred[i] == 0: 
		yp1.append('blue')
	else:
		yp1.append('yellow')
		
		
	
# Creation des graphiques de clustering en 2 dimensions
fig = plt.figure(figsize= (30,10))
#Graphique de Y en fonction de X 
ax1 = fig.add_subplot(311)
ax1.scatter(serx, sery, c=yp1)
plt.title('Y en fonction de X')
#Graphique de Y en fonction de Z 
ax2 = fig.add_subplot(312, sharex= ax1)
ax2.scatter(serz, sery, c=yp1)
plt.title('Y en fonction de Z')
#Graphique de Z en fonction de X
ax3 = fig.add_subplot(313, sharex= ax1)
ax3.scatter(serx, serz, c=yp1,)
plt.title('Z en fonction de X')
plt.show(fig)


		
#Creation de graphique du signal avec les points identifies a leur cluster respectif
fig2= plt.figure(figsize= (30,10))
#Graphique du signal sur l'axe X
ax4 =fig2.add_subplot(311)
plt.scatter(data['Frame Index'][1:],data['3x'][1:], c=yp1)
plt.title('Graphique des quaternions x')
#Graphique du signal sur l'axe Y
ax5 = fig2.add_subplot(312, sharex= ax4)
plt.scatter(data['Frame Index'][1:],data['3y'][1:], c=yp1)
plt.title('Graphique des quaternions y')
#Graphique du signal sur l'axe Z
ax6 = fig2.add_subplot(313, sharex = ax4)
plt.scatter(data['Frame Index'][1:],data['3z'][1:], c=yp1)
plt.title('Graphique des quaternions z')
plt.show(fig2)

#Creation d'un graphique en 3 dimensions ou les points sont identifies a leur cluster respectif
fig3 = plt.figure(figsize= (30,10))
ax7 = fig3.add_subplot(111, projection='3d')
ax7.scatter(serx, sery, serz , c=yp1)
plt.title('Clustering des sauts en x,y,z en 3 dimensions')
plt.show(fig3)
