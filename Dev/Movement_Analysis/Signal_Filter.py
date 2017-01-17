import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import Signal_Tools_Box  as stb

path         = r'C:\Users\Simon\Desktop\TposeToCorrect'
namefile     = '\MT2'
Locationcsv  = path + namefile + 'RawData.csv'

df  = pd.read_csv(Locationcsv)
Fdf = df.copy(deep=True)
Thmin  = np.pi/4.
C = True;
Opt = 'Plateaux'
while(C==True):	
	#df,Th   = stb.kmeanClustering(Fdf,Thmin)
	S       = stb.SignalCorrector(Fdf,Thmin,Opt)
	TempFdf = S.Get_Cleaned_Signal()
	if raw_input("Do you want to keep this Filtered Signal ? : (y / n) ") == 'y': 
		Fdf = TempFdf.copy(deep = True)
	if  raw_input("Do want to continue the cleaning of the Signals ? : (y / n) ") == 'y':
		ans = raw_input("Do you want to change the min threshold values" + "(which was :" + str(Thmin) +") ? : (no or/ new value) ") 
		if ans != 'no': Thmin = float(ans)		
		C = True
	else : C = False
		