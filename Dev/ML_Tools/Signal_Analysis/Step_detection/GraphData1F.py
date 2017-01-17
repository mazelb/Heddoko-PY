"""Programme a utiliser avec Graphs2. Permet d'afficher des graphiques d'une des composantes (M, Q, R, A) d'un des trois axes (XYZ). Il affiche egalement une moyenne mobile sur le graphique. Le deuxieme graphique est un histogramme de frequence qui permet de se faire une idee de la distribution"""
import pandas as pd
import matplotlib.pyplot as plt
import Graphs2 as gp


filename = gp.GetFile()  #Getting the file  with Tkinter
data = gp.PdConvert(filename)  #Creating a pd Data Frame from the xls file
L=list(data)  #Converts data to a list
st = gp.ListGraph(L) # Creating the list of available Graphs
a = True
while a:
    Selected = gp.SelectGraph(st) #Letting the user selected which graph he wants to see
    Selected = int (Selected) 
    G = gp.Graph(data,L, Selected,)  #Creating the graph
    plt.show(G)      #Showing the graph
#Il semble y avoir un probleme avec les fonctions Mean (moyenne) et STD	(ecart-type)		
#   Mean(data, L, Selected)
#   STD(data, L, Selected)
    Exit1= gp.Exit()    # Asking the user if he wants to continue or exit
    if Exit1 == False:
        a = False       # Exiting if answer is 'y'
    elif Exit1 == True:  
        a = True       # Asking user which graph he wants to see (line 13)
        

