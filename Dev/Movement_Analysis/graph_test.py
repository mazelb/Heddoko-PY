from AM_Functions import * 
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
path = r'C:\Users\Simon\Desktop\TposeToCorrect\MT3RawData.xlsx'

df  = pd.read_excel(path)
S   = SignalCorrector(df,)
dfEu= S.Clean_signal()
plt.show()
# for i in range(1,10):
	# f   = plt.figure(i)
	# cx1 = f.add_subplot(311)
	# cx2 = f.add_subplot(312,sharex = cx1)
	# cx3 = f.add_subplot(313,sharex = cx1)
	# sx  = str(i)+'x';sy = str(i)+'y';sz = str(i)+'z'
	# cx1.plot(df['Frame Index'].values,df[sx].values)
	# cx1.plot(dfEu['Frame Index'].values,dfEu[sx].values,'.')
	# cx2.plot(df['Frame Index'].values,df[sy].values)
	# cx2.plot(dfEu['Frame Index'].values,dfEu[sy].values,'.')
	# cx3.plot(df['Frame Index'].values,df[sz].values)
	# cx3.plot(dfEu['Frame Index'].values,dfEu[sz].values,'.')
# plt.show()