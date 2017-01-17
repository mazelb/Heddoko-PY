from AM_Functions import * 
import time
import matplotlib.pyplot as plt
Etx  = []; Ety  = []; Etz  = []; 
Location = r'C:\Users\Simon\Desktop\danRawDataQuat.xlsx'
df  = pd.read_excel(Location)
L   = list(df)
N   = len(df[L[0]])
f   = [];
ax1 = []; ax2 = []; ax3 = []; 
ax4 = []; ax5 = []; ax6 = []; 
for i in range(1,8):
	Etx = []; Ety = []; Etz = [];
	Sx = str(i)+'x';Sy = str(i)+'y';Sz = str(i)+'z';Sw = str(i)+'w'	
	Cump = [I/(1.0*N) for I in range(0,N-1)]
	f.append(plt.figure(i-1))
	ax1.append(f[i-1].add_subplot(321))
	ax2.append(f[i-1].add_subplot(322))
	ax3.append(f[i-1].add_subplot(323))
	ax4.append(f[i-1].add_subplot(324))
	ax5.append(f[i-1].add_subplot(325))
	ax6.append(f[i-1].add_subplot(326))	
	
	ax1[i-1].plot(df[Sx].values);
	ax1[i-1].plot(df[Sy].values);
	ax1[i-1].plot(df[Sz].values);
	ax1[i-1].plot(df[Sw].values);
	
	"""
	serx = np.log(np.array(np.abs(df[Sx].diff(1))))
	sery = np.log(np.array(np.abs(df[Sy].diff(1))))
	serz = np.log(np.array(np.abs(df[Sz].diff(1))))
	serw = np.log(np.array(np.abs(df[Sw].diff(1))))
	"""
	serx = (np.array(np.abs(df[Sx].diff(1))))
	sery = (np.array(np.abs(df[Sy].diff(1))))
	serz = (np.array(np.abs(df[Sz].diff(1))))
	serw = (np.array(np.abs(df[Sw].diff(1))))
	
	
	sserx= np.sort(serx[1:])
	ssery= np.sort(sery[1:])
	sserz= np.sort(serz[1:])
	sserw= np.sort(serw[1:])
	
	ax3[i-1].plot(serx)
	ax3[i-1].plot(sery)
	ax3[i-1].plot(serz)
	ax3[i-1].plot(serw)	
		
	ax5[i-1].plot(sserx,Cump,'.');
	ax5[i-1].plot(ssery,Cump,'.');
	ax5[i-1].plot(sserz,Cump,'.');	
	ax5[i-1].plot(sserw,Cump,'.');	
			
	E = [[],[],[]]
	for IFrame in range(N):		
		e = QuatToEuler([df[Sw].values[IFrame],df[Sx].values[IFrame],df[Sy].values[IFrame],df[Sz].values[IFrame]],'zyx')
		E[0].append(e[0])
		E[1].append(e[1])
		E[2].append(e[2])
	
	Ae0 = np.array(E[0])
	Ae1 = np.array(E[1])
	Ae2 = np.array(E[2])		
	ax2[i-1].plot(Ae0);
	ax2[i-1].plot(Ae1);
	ax2[i-1].plot(Ae2);
	
	"""
	dae0 = np.log(np.abs(np.diff(Ae0)))
	dae1 = np.log(np.abs(np.diff(Ae1)))
	dae2 = np.log(np.abs(np.diff(Ae2)))	
	"""
	dae0 = (np.abs(np.diff(Ae0)))
	dae1 = (np.abs(np.diff(Ae1)))
	dae2 = (np.abs(np.diff(Ae2)))	
	
	ax4[i-1].plot(dae0);
	ax4[i-1].plot(dae1);
	ax4[i-1].plot(dae2);
	
	serex = np.array(dae0)
	serey = np.array(dae1)
	serez = np.array(dae2)		
	sserex= np.sort(serex)
	sserey= np.sort(serey)
	sserez= np.sort(serez)	
	
	ax6[i-1].plot(sserex,Cump,'.');
	ax6[i-1].plot(sserey,Cump,'.');
	ax6[i-1].plot(sserez,Cump,'.');	
plt.show()