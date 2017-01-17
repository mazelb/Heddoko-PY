"""Ce fichier definit les fonctions utiles pour le programme GraphData1."""

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter as scl

import Tkinter

from tkFileDialog import askopenfilename
##Classe de base permettant de creer les graphiques
class Graph:
    nfig = 0

    def __init__(self, data, L, Selected):
		
        Graph.nfig+=1
        
        #mov_average = data[L[3]].rolling(window=8,center=False).mean()
	#Definit x comme etant Time(ms) de facon automatique
        x = data[L[0]]
	
        y = data[L[Selected]]
        
	
        self.title = 'title'
	
        self.xlim  = [0  ,10]
	
        self.ylim  = [-10,10]
	#Definit les valeurs par defaut de xlabel et ylabel 
        self.xlabel = str(L[0])
	
        self.ylabel = str(L[Selected])
	
        self.fig   = plt.figure(Graph.nfig, figsize= (10,7))
	
        self.ax    = self.fig.add_subplot(211)
	
        self.ax.plot(x,y)
	  #Ajoute une moyenne mobile sur le graphique 
        self.ax.plot(x, data[L[Selected]].rolling(window=100,center=False).mean())
	#Boucle pour definir le titre et le nom des axes (etablit des valeurs par defaut)
        for i in range(3):
            if  i==0:
                t = raw_input("title ?:")	
                
                if len(t)!=0:
                    self.ax.set_title(t)
                else:
                    self.ax.set_title(str(L[Selected]) + ' en fonction du temps')
                    
            elif i==1:
                t = raw_input("xlabel ?:")
                if len(t)!=0:
                    self.ax.set_xlabel(t)
                else:
                    self.ax.set_xlabel(self.xlabel)
                     
            elif i==2:
                t = raw_input("ylabel ?:")
                
                if len(t)!=0:
                    self.ax.set_ylabel(t)
                else:
                    self.ax.set_ylabel(self.ylabel)
             
																	
																				# Desactive le 'offset' dans les axes
        self.ax.ticklabel_format(useOffset=False)    
        #Creation d'un histogramme de frequence 
        self.bx = self.fig.add_subplot(212) 
        self.bx.hist(data[L[Selected]], bins=len(data[L[Selected]].value_counts()), 
        histtype='bar')         
        self.bx.set_xlabel('Value')
        self.bx.set_ylabel('Frequency')
       
        
        def show(self):
            plt.show(self)


def GetFile():

    """fonction servant a aller chercher le fichier

    """

    print 'Please select the file that you want to analyse'



    root = Tkinter.Tk() ; root.withdraw()

    filename = askopenfilename(parent=root)

    return filename


def PdConvert (filename): 
#Converting the xls file to a Pandas DataFrame
    xls_file = pd.ExcelFile(filename)

    xls_file

    xls_file.sheet_names

    data = xls_file.parse(xls_file.sheet_names[0])

    return data


def ListGraph(L):
#Creates a list of the possible graph

    st = ''

    for x in range (2, len(L)):

        st = st + '\n' + str(x) + ':' + L[x]

    st= str(st)

    return st


def SelectGraph(st):
#Demande quel graphique afficher
    GNumber= raw_input ('Quels graphiques desirez-vous afficher? '+ st + '    ' )

            

    if GNumber   in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:

        GNumber= int(GNumber)
    return int(GNumber)

#Fonction a retravailler permettant d'afficher des parametres du signal analyse
#def Mean(data, L, Selected):
#    print 'Mean:' + str(data[L[Selected]].mean())
#
#
#    
#def STD(data, L, Selected):
#    print 'Standard deviation:' + str(data[L[Selected]].std()) 
#    

#Fonction pour sortir du programme ou afficher un autre graphique 
def Exit():

    Exit= raw_input('Do you want to exit? y/n')

    if Exit == 'y':

        Exit1 = False

        return False

        

    else:
        Exit1 = True



                

                           

    



    