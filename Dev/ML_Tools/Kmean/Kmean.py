import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_excel('C:\\Users\\Simon\\Desktop\\danRawData.xlsx')
Nl = len(df)
from sklearn.cluster import KMeans	
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
Nl = len(df)
for i in range(1,10): 
	dEu = [];
	Sx  = str(i)+'x';
	Sy  = str(i)+'y';
	Sz  = str(i)+'z';
	Eux = df[Sx].values[1:];
	Euy = df[Sy].values[1:];
	Euz = df[Sz].values[1:];
	Dx  = np.abs(df[Sx].diff()[1:]);
	Dy  = np.abs(df[Sy].diff()[1:]);
	Dz  = np.abs(df[Sz].diff()[1:]);	
	dx  = []; dy = []; dz = []; G  = [];
	for j in range(Nl-1):		
		dx.append([Dx.values[j]]);dy.append([Dy.values[j]]);dz.append([Dz.values[j]]);
		G.append([Dx.values[j],Dy.values[j],Dz.values[j]])
	YG = KMeans(n_clusters=3).fit_predict(G)
	dG = np.array(G)
	YmG0= np.mean(dG[YG==0]);	YmG1= np.mean(dG[YG==1]);	YmG2= np.mean(dG[YG==2]);
	cGmx = 2; cGmm = 1; cGmn = 0;
	if YmG2 > YmG1 and YmG2 > YmG0 : cGmx = 2; cGmm = 1; cGmn = 0;
	if YmG1 > YmG2 and YmG1 > YmG0 : cGmx = 1; cGmm = 0; cGmn = 2;
	if YmG0 > YmG2 and YmG0 > YmG1 : cGmx = 0; cGmm = 1; cGmn = 2;	
	it = np.array(range(len(YG)))
	plt.figure(i)
	plt.subplot(321)
	plt.plot(it[YG==cGmx],Eux[YG==cGmx],'*r',markersize=12)
	plt.plot(it[YG==cGmm],Eux[YG==cGmm],'.g')
	plt.plot(it[YG==cGmn],Eux[YG==cGmn],'.y')
	plt.plot(it,Eux,'b')
	plt.subplot(323)
	plt.plot(it[YG==cGmx],Euy[YG==cGmx],'*r',markersize=12)
	plt.plot(it[YG==cGmm],Euy[YG==cGmm],'.g')
	plt.plot(it[YG==cGmn],Euy[YG==cGmn],'.y')
	plt.plot(it,Euy,'b')
	plt.subplot(325)
	plt.plot(it[YG==cGmx],Euz[YG==cGmx],'*r',markersize=12)
	plt.plot(it[YG==cGmm],Euz[YG==cGmm],'.g')
	plt.plot(it[YG==cGmn],Euz[YG==cGmn],'.y')
	plt.plot(it,Euz,'b')
	plt.subplot(322)
	plt.plot(Dx[YG==cGmx],Dy[YG==cGmx],'*r',markersize=12)
	plt.plot(Dx[YG==cGmm],Dy[YG==cGmm],'.g')
	plt.plot(Dx[YG==cGmn],Dy[YG==cGmn],'.y')	
	plt.xlabel('x')
	plt.ylabel('y')
	plt.subplot(324)
	plt.plot(Dx[YG==cGmx],Dz[YG==cGmx],'*r',markersize=12)
	plt.plot(Dx[YG==cGmm],Dz[YG==cGmm],'.g')
	plt.plot(Dx[YG==cGmn],Dz[YG==cGmn],'.y')
	plt.xlabel('x')
	plt.ylabel('z')
	plt.subplot(326)
	plt.plot(Dy[YG==cGmx],Dz[YG==cGmx],'*r',markersize=12)
	plt.plot(Dy[YG==cGmm],Dz[YG==cGmm],'.g')
	plt.plot(Dy[YG==cGmn],Dz[YG==cGmn],'.y')
	plt.xlabel('y')
	plt.ylabel('z')
plt.show()

# dx = np.array(dx)
# dy = np.array(dy)
# dz = np.array(dz)	
# yx = KMeans(n_clusters=2).fit_predict(dx)
# yy = KMeans(n_clusters=2).fit_predict(dy)
# yz = KMeans(n_clusters=2).fit_predict(dz)
# ym0x = np.mean(dx[yx==0]);
# ym1x = np.mean(dx[yx==1]);	
# ym0y = np.mean(dy[yy==0]);
# ym1y = np.mean(dy[yy==1]);	
# ym0z = np.mean(dz[yz==0]);
# ym1z = np.mean(dz[yz==1]);	
# cxmx = 1; cxmn = 0
# cymx = 1; cymn = 0
# czmx = 1; czmn = 0
# if ym0x > ym1x : cxmx=0;cxmn=1;
# if ym0y > ym1y : cymx=0;cymn=1;
# if ym0z > ym1z : czmx=0;czmn=1;	


