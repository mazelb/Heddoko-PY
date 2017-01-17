import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def Norm(vx,vy,vz):
	s = []
	for i in range (len(vx)):
		s.append(np.sqrt(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]));
	return s
	
def Pow(vx,vy,vz,n):
	s = []
	for i in range (len(vx)):
		s.append(vx[i]**n + vy[i]**n + vz[i]**n);
	return s	

def Log(v):
	s = []
	for i in range (len(v)):
		s.append(np.log(v[i]));
	return s	
	
def Sign(v):
	s = []
	for i in range (len(v)):
		s.append(-v[i]);
	return s	
	
def Plus(v,p):
	s = []
	for i in range (len(v)):
		s.append(p+v[i]);
	return s	
	
def Normq(vx,vy,vz,vw):
	s = []
	for i in range (len(vx)):
		s.append(np.sqrt(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i] + vw[i]*vw[i]));
	return s

def Normalize(v,n):
	vn = []
	for i in range (len(v)):
		if(n!=0):vn.append(v[i]/n[i])
		else: vn.append(0)
	return vn

def MultiQ(q1x,q1y,q1z,q1w,q2x,q2y,q2z,q2w):
	qw = q1w*q2w -(q1x*q2x  + q1y*q2y + q1z*q2z)
	qx = q1w*q2x + q2w*q1x  +(q1y*q2z - q1z*q2y)
	qy = q1w*q2y + q2w*q1y  -(q1x*q2z - q2x*q1z)
	qz = q1w*q2z + q2w*q1z  +(q1x*q2y - q1y*q2x)
	return qx,qy,qz,qw

def ResetQ(qx,qy,qz,qw):
	qx0 =-qx[0]
	qy0 =-qy[0]
	qz0 =-qz[0]
	qw0 = qw[0]
	for i in range(0,len(qx)):
		qx[i], qy[i], qz[i], qw[i] = MultiQ(qx0,qy0,qz0,qw0,qx[i],qy[i],qz[i],qw[i])
	return qx,qy,qz,qw
	
def Eangle(qx,qy,qz,qw):	
	H = []
	P = []
	R = []
	h = 0
	p = 0
	r = 0
	rad2deg = 180.0/np.pi
	for i in range(0,len(qx)):		
		
		a = qx[i]*qx[i]-qy[i]*qy[i]-qz[i]*qz[i]+qw[i]*qw[i]
		b = 2.0*(qx[i]*qy[i] + qz[i]*qw[i])
		h = np.arctan2(a,b)
		a =-2.0*(qx[i]*qz[i]-qy[i]*qw[i])		
		p = np.arcsin(a)
		a = -qx[i]*qx[i]-qy[i]*qy[i]+qz[i]*qz[i]+qw[i]*qw[i]
		b = 2.0*(qx[i]*qw[i] + qy[i]*qz[i])
		r = np.arctan2(a,b)
		H.append(h*rad2deg)
		P.append(p*rad2deg)
		R.append(r*rad2deg)
	return H,P,R

def Hanning(Lw,i):
	w =  0.5 - 0.5*np.cos(2.0*math.pi*i/Lw)
	return w 

def BlackmanHarris(Lw,i):
	a0=0.35875
	a1=0.48829
	a2=0.14128
	a3=0.01168	
	w = a0 - a1*np.cos(2.0*math.pi*i/(Lw-1.0)) + a2*np.cos(4.0*math.pi*i/(Lw-1.0))- a3*np.cos(6.0*math.pi*i/(Lw-1.0))
	return w

def WindowsFFT(s,fs):	
	ls   = len(s); ds   = int(2.0*fs) ; Lw  = int(10.0*fs); d    = ls - Lw	
	freq = np.fft.fftfreq(Lw,d=1.0/fs);	ff  = freq[2:int(len(freq)/2.0 - 1.0)]
	wffs = [] 
	sw   = []
	Tf   = []
	td   = Lw/2. 
	Tf.append(td)
	I    =  0 
	p    = []	
	while d > 0: 
		ST  = []
		WF  = [] 				
		for i in range(0,Lw): 
			ST.append(s[I+i]*BlackmanHarris(Lw,i))		
		sw.append(ST)
		ffs = np.fft.fft(ST)		
		WF  = np.abs(ffs[int(len(ffs)/2.0 - 1.0):2:-1]) 	
		WS  = sum(WF)
		WF  = np.log(WF)	
		wffs.append(WF) 
		d     = d - ds
		I     = I + ds
		Tf.append(1.*(td + I)/fs) 
	return wffs, sw, Tf, freq

def GraphNWfft1(namep1,N,namep3):
	Lw = 128	
	for i in range(0,N):	
		name = namep1 + str(i) + namep3;
		time  ,qx  ,qy  ,qz  ,qw  ,mx  ,my  ,mz  , ax  ,ay  ,az  ,wx  ,wy  ,wz   = readN(name)
		fs = len(time)/(time[-1]-time[0])		
		wffqx, sw, Tf, freq = WindowsFFT(qx,fs)
		wffqy, sw, Tf, freq = WindowsFFT(qy,fs)
		wffqz, sw, Tf, freq = WindowsFFT(qz,fs)
		
		wffmx, sw, Tf, freq = WindowsFFT(mx,fs)
		wffmy, sw, Tf, freq = WindowsFFT(my,fs)
		wffmz, sw, Tf, freq = WindowsFFT(mz,fs)
		
		wffax, sw, Tf, freq = WindowsFFT(ax,fs)
		wffay, sw, Tf, freq = WindowsFFT(ay,fs)
		wffaz, sw, Tf, freq = WindowsFFT(az,fs)
		
		wffwx, sw, Tf, freq = WindowsFFT(wx,fs)
		wffwy, sw, Tf, freq = WindowsFFT(wy,fs)
		wffwz, sw, Tf, freq = WindowsFFT(wz,fs)
		
		
		awffqx = np.array(wffqx)	
		awffqy = np.array(wffqy)
		awffqz = np.array(wffqz)
		
		awffmx = np.array(wffmx)	
		awffmy = np.array(wffmy)
		awffmz = np.array(wffmz)
		
		awffax = np.array(wffax)	
		awffay = np.array(wffay)
		awffaz = np.array(wffaz)
		
		awffwx = np.array(wffwx)	
		awffwy = np.array(wffwy)
		awffwz = np.array(wffwz)
		
		plt.figure(1 + i*4)	
		
		fig1, ax1  = plt.subplot(231)
		cax1 = plt.imshow(np.transpose(awffqx), extent=[0,Tf[len(awffqx)-3],0,fs/2], aspect='auto', interpolation='nearest', cmap='hot')		
		cbar1 = fig.colorbar(cax1, ticks=[-1, 0, 1])		
		ax2 = plt.subplot(234, sharex=ax1)			
		plt.plot(time,qx)	
		
		fig3, ax3 = plt.subplot(232)
		cax3 = plt.imshow(np.transpose(awffqy), extent=[0,Tf[len(awffqy)-3],0,fs/2], aspect='auto', interpolation='nearest', cmap='hot')		
		cbar3 = fig.colorbar(cax3, ticks=[-1, 0, 1])	
		ax4 = plt.subplot(235, sharex=ax3)			
		plt.plot(time,qy)		
				
		fig5, ax5 = plt.subplot(233)
		cax5 = plt.imshow(np.transpose(awffqz), extent=[0,Tf[len(awffqz)-3],0,fs/2], aspect='auto', interpolation='nearest', cmap='hot')		
		cbar5 = fig.colorbar(cax5, ticks=[-1, 0, 1])	
		ax6 = plt.subplot(236, sharex=ax5)			
		plt.plot(time,qz)
		print 'q'
		
		# plt.figure(2 + i*4)
		# ax1 = plt.subplot(231)
		# plt.imshow(np.transpose(awffmx), extent=[0,Tf[len(awffmx)-3],0,fs/2], aspect='auto', interpolation='nearest')		
		# ax2 = plt.subplot(234, sharex=ax1)			
		# plt.plot(time,mx)		
		# ax3 = plt.subplot(232)
		# plt.imshow(np.transpose(awffmy), extent=[0,Tf[len(awffmy)-3],0,fs/2], aspect='auto', interpolation='nearest')		
		# ax4 = plt.subplot(235, sharex=ax3)			
		# plt.plot(time,my)		
		# ax5 = plt.subplot(233)
		# plt.imshow(np.transpose(awffmz), extent=[0,Tf[len(awffmz)-3],0,fs/2], aspect='auto', interpolation='nearest')		
		# ax6 = plt.subplot(236, sharex=ax5)			
		# plt.plot(time,mz)
		# print 'm'		
		
		# plt.figure(3 + i*4)
		# ax1 = plt.subplot(231)
		# plt.imshow(np.transpose(awffax), extent=[0,Tf[len(awffax)-3],0,fs/2], aspect='auto', interpolation='nearest')		
		# ax2 = plt.subplot(234, sharex=ax1)			
		# plt.plot(time,ax)		
		# ax3 = plt.subplot(232)
		# plt.imshow(np.transpose(awffay), extent=[0,Tf[len(awffay)-3],0,fs/2], aspect='auto', interpolation='nearest')		
		# ax4 = plt.subplot(235, sharex=ax3)			
		# plt.plot(time,ay)			
		# ax5 = plt.subplot(233)
		# plt.imshow(np.transpose(awffaz), extent=[0,Tf[len(awffaz)-3],0,fs/2], aspect='auto', interpolation='nearest')		
		# ax6 = plt.subplot(236, sharex=ax5)			
		# plt.plot(time,az)
		# print 'a'		
		
		# plt.figure(4 + i*4)
		# ax1 = plt.subplot(231)
		# plt.imshow(np.transpose(awffwx), extent=[0,Tf[len(awffwx)-3],0,fs/2], aspect='auto', interpolation='nearest')		
		# ax2 = plt.subplot(234, sharex=ax1)			
		# plt.plot(time,wx)		
		# ax3 = plt.subplot(232)
		# plt.imshow(np.transpose(awffwy), extent=[0,Tf[len(awffwy)-3],0,fs/2], aspect='auto', interpolation='nearest')		
		# ax4 = plt.subplot(235, sharex=ax3)			
		# plt.plot(time,wy)		
		# ax5 = plt.subplot(233)
		# plt.imshow(np.transpose(awffwz), extent=[0,Tf[len(awffwz)-3],0,fs/2], aspect='auto', interpolation='nearest')		
		# ax6 = plt.subplot(236, sharex=ax5)			
		# plt.plot(time,wz)	
		# print 'w'
	plt.show()			
		
def GraphNWfft2(namep1,N,namep3,variable):
	for i in range(0,N):				
		name = namep1 + str(i) + namep3;
		time  ,qx  ,qy  ,qz  ,qw  ,mx  ,my  ,mz  , ax  ,ay  ,az  ,wx  ,wy  ,wz   = readN(name)			
		print  i, variable
		fs = len(time)/(time[-1]-time[0])
		if variable=='q':
			wffqx, sw, Tf, freq = WindowsFFT(qx,fs)
			wffqy, sw, Tf, freq = WindowsFFT(qy,fs)
			wffqz, sw, Tf, freq = WindowsFFT(qz,fs)
			awffqx = np.array(wffqx)	
			awffqy = np.array(wffqy)
			awffqz = np.array(wffqz)
					
			plt.figure(1 + 3*i)			
			x = plt.subplot(211)
			plt.imshow(np.transpose(awffqx), extent=[0,Tf[len(awffqx)-1],0,fs/2.0], aspect='auto', interpolation='nearest', cmap='hot')		
			plt.colorbar(orientation="horizontal")
			plt.ylabel("f(Hz)")
			plt.subplot(212, sharex=x)						
			plt.plot(time,qx)
			plt.xlabel("Time (sec)")
			plt.ylabel("Qx")
			plt.clim(-6.5,0)
			
			plt.figure(2 + 3*i)			
			x = plt.subplot(211)
			plt.imshow(np.transpose(awffqy), extent=[0,Tf[len(awffqy)-1],0,fs/2], aspect='auto', interpolation='nearest', cmap='hot')		
			plt.colorbar(orientation="horizontal")
			plt.ylabel("f(Hz)")
			plt.subplot(212, sharex=x)						
			plt.plot(time,qy)	
			plt.xlabel("Time (sec)")
			plt.ylabel("Qy")
			plt.clim(-6.5,0)
			
			plt.figure(3 + 3*i)
			x = plt.subplot(211)
			plt.imshow(np.transpose(awffqz), extent=[0,Tf[len(awffqz)-1],0,fs/2], aspect='auto', interpolation='nearest', cmap='hot')		
			plt.colorbar(orientation="horizontal")			
			plt.ylabel("f(Hz)")
			plt.subplot(212, sharex=x)						
			plt.plot(time,qz)
			plt.xlabel("Time (sec)")
			plt.ylabel("Qz")
			plt.clim(-6.5,0)
			
		if variable=='m':
			wffmx, sw, Tf, freq = WindowsFFT(mx,fs)
			wffmy, sw, Tf, freq = WindowsFFT(my,fs)
			wffmz, sw, Tf, freq = WindowsFFT(mz,fs)
			awffmx = np.array(wffmx)	
			awffmy = np.array(wffmy)
			awffmz = np.array(wffmz)
			
			plt.figure(1 + 3*i)			
			x = plt.subplot(211)
			plt.imshow(np.transpose(awffmx), extent=[0,Tf[len(awffmx)-3],0,fs/2], aspect='auto', interpolation='nearest', cmap='hot')		
			plt.colorbar(orientation="horizontal")
			plt.ylabel("f(Hz)")
			plt.subplot(212, sharex=x)						
			plt.plot(time,mx)
			plt.xlabel("Time (sec)")
			plt.ylabel("Mx (Micro Tesla)")
			plt.clim(-1.5,4)
			
			plt.figure(2 + 3*i)			
			x = plt.subplot(211)
			plt.imshow(np.transpose(awffmy), extent=[0,Tf[len(awffmy)-3],0,fs/2], aspect='auto', interpolation='nearest', cmap='hot')		
			plt.colorbar(orientation="horizontal")
			plt.subplot(212, sharex=x)						
			plt.plot(time,my)
			plt.xlabel("Time (sec)")
			plt.ylabel("My (Micro Tesla)")
			plt.clim(-1.5,4)
			
			plt.figure(3 + 3*i)			
			x = plt.subplot(211)
			plt.imshow(np.transpose(awffmz), extent=[0,Tf[len(awffmz)-3],0,fs/2], aspect='auto', interpolation='nearest', cmap='hot')		
			plt.colorbar(orientation="horizontal")
			plt.ylabel("f(Hz)")
			plt.subplot(212, sharex=x)						
			plt.plot(time,mz)
			plt.xlabel("Time (sec)")
			plt.ylabel("Mz (Micro Tesla)")
			plt.clim(-1.5,4)
			
		if variable=='a':
			wffax, sw, Tf, freq = WindowsFFT(ax,fs)
			wffay, sw, Tf, freq = WindowsFFT(ay,fs)
			wffaz, sw, Tf, freq = WindowsFFT(az,fs)
			awffax = np.array(wffax)	
			awffay = np.array(wffay)
			awffaz = np.array(wffaz)
			
			plt.figure(1 + 3*i)
			x = plt.subplot(211)
			plt.imshow(np.transpose(awffax), extent=[0,Tf[len(awffax)-3],0,fs/2], aspect='auto', interpolation='nearest', cmap='hot')		
			plt.colorbar(orientation="horizontal")
			plt.ylabel("f(Hz)")
			plt.subplot(212, sharex=x)						
			plt.plot(time,ax)	
			plt.xlabel("Time (sec)")
			plt.ylabel("ax (m/(sec*sec))")
			plt.clim(-1.5,4)
			
			plt.figure(2 + 3*i)
			x = plt.subplot(211)
			plt.imshow(np.transpose(awffay), extent=[0,Tf[len(awffay)-3],0,fs/2], aspect='auto', interpolation='nearest', cmap='hot')		
			plt.colorbar(orientation="horizontal")
			plt.ylabel("f(Hz)")
			plt.subplot(212, sharex=x)						
			plt.plot(time,ay)	
			plt.xlabel("Time (sec)")
			plt.ylabel("ay (m/(sec*sec))")
			plt.clim(-1.5,4)
			
			plt.figure(3 + 3*i)
			x = plt.subplot(211)
			plt.imshow(np.transpose(awffaz), extent=[0,Tf[len(awffaz)-3],0,fs/2], aspect='auto', interpolation='nearest', cmap='hot')		
			plt.colorbar(orientation="horizontal")
			plt.ylabel("f(Hz)")
			plt.subplot(212, sharex=x)					
			plt.plot(time,az)
			plt.xlabel("Time (sec)")
			plt.ylabel("az (m/(sec*sec))")
			plt.clim(-1.5,4)
			
		if variable=='w':			
			wffwx, sw, Tf, freq = WindowsFFT(wx,fs)
			wffwy, sw, Tf, freq = WindowsFFT(wy,fs)
			wffwz, sw, Tf, freq = WindowsFFT(wz,fs)
			awffwx = np.array(wffwx)	
			awffwy = np.array(wffwy)
			awffwz = np.array(wffwz)
			
			plt.figure(1 + 3*i)
			x = plt.subplot(211)
			plt.imshow(np.transpose(awffwx), extent=[0,Tf[len(awffwx)-3],0,fs/2], aspect='auto', interpolation='nearest', cmap='hot')		
			plt.ylabel("f(Hz)")
			plt.colorbar(orientation="horizontal")
			plt.subplot(212, sharex=x)						
			plt.plot(time,wx)
			plt.xlabel("Time (sec)")
			plt.ylabel("wx (degrees/sec)")
			plt.clim(-2.,1.)
			
			plt.figure(2 + 3*i)
			x = plt.subplot(211)
			plt.imshow(np.transpose(awffwy), extent=[0,Tf[len(awffwy)-3],0,fs/2], aspect='auto', interpolation='nearest', cmap='hot')		
			plt.ylabel("f(Hz)")
			plt.colorbar(orientation="horizontal")
			plt.subplot(212, sharex=x)						
			plt.plot(time,wy)
			plt.xlabel("Time (sec)")
			plt.ylabel("wy (degrees/sec)")
			plt.clim(-2.,1.)
			
			plt.figure(3 + 3*i)
			x = plt.subplot(211)
			plt.imshow(np.transpose(awffwz), extent=[0,Tf[len(awffwz)-3],0,fs/2], aspect='auto', interpolation='nearest', cmap='hot')		
			plt.ylabel("f(Hz)")
			plt.colorbar(orientation="horizontal")
			plt.subplot(212, sharex=x)						
			plt.plot(time,wz)
			plt.xlabel("Time (sec)")
			plt.ylabel("wz (degrees/sec)")
			plt.clim(-2.,1.)
	plt.show()

def Noisedistribution(S,nbin):	
	L  = len(S)	
	ms = np.mean(S)
	std = np.std(S)
	Ns = []
	for i in range(0,L):
		Ns.append(S[i]-ms)
	plt.hist(Ns[0:-1], nbin, normed=True)
	return ms,std

def distribution(S,nTpoint,nbin,nPopulation):
	plt.figure(1)
	f, axarr = plt.subplots(nTpoint,sharey=True,sharex=True)
	L  = len(S)	
	nh = int(nPopulation/2)
	DL = int(L/nTpoint)
	plt.ylim([0,1])
	plt.xlim([0,0.5])
	for i in range(0,nTpoint):
		Ji = (i+1)*DL - nh
		Jf = (i+1)*DL + nh		
		axarr[i].hist(np.abs(S[Ji:Jf]), nbin, normed=True)

def CummulantProb(S,Lw,step):	
	Ls   = len(S)	
	Nw   = (len(S)-step*Lw)/(step*Lw)
	P    = [1.0*i/Lw for i in range(1,Lw+1)]
	for J in range(0,Nw):		
		s  = np.sort(np.array(S[step*J*Lw/2:(step*J*Lw/2)+Lw]))
		plt.plot(s,P)	

def ContinuousTime(t,Ti):
		Tau = []
		Tau.append(Ti)
		for I in range(1,len(t)):
			Tau.append(Tau[I-1] + math.fabs(t[I]-t[I-1]))
		return Tau

def MovingAverage(S,time,Lw):
	MA = []
	its = []
	Ls = len(S)	
	Nw = (len(S)-Lw)/Lw
	for i in range(0,Nw):		
		MA.append(np.mean(S[i*Lw/2:(i*Lw/2)+Lw]))
		its.append(time[(i+1)*Lw + Lw/2])
	plt.plot(its,MA)
	
def MovingMoments(S,time,Lw):	
	Mstd = []
	MA   = []
	MBoundariesP = []
	MBoundariesM = []
	its  = []
	Ls   = len(S)
	Lp   = int(Lw/10.0)	
	Nw   = (Ls-Lw)/Lp
	for i in range(0,Nw):		
		MA.append( np.mean(S[i*Lp:i*Lp + Lw]))
		Mstd.append(np.std(S[i*Lp:i*Lp + Lw]))
		#MBoundariesP.append(MA[i]+Mstd[i])
		#MBoundariesM.append(MA[i]-Mstd[i])
		its.append(time[(i)*Lp + Lw/2])
	#plt.plot(its,MBoundariesP)
	#plt.plot(its,MBoundariesM)
	plt.plot(its,MA)

def read(n):
	with open(n) as myfile:
		count = sum(1 for line in myfile)	
	f = open(n,'r')	
	time = [];
	qx   = [];
	qy   = [];
	qz   = [];
	qw   = [];
	ax   = [];
	ay   = [];
	az   = [];
	mx   = [];
	my   = [];
	mz   = [];
	wx   = [];
	wy   = [];
	wz   = [];
	alphaA =   1.0 # 16*9.8/32768.0;
	alphaM =   1.0 #1000.0/32768.0;
	alphaW =   1.0 #5000.0/32768.0;
	i = 0;	
	for l in f.readlines():
		T = []
		T = l.split(',')
		if i!=0 and i<count-2:			
			time.append(float(T[0 ])/1000.0);
			qx.append(float(T[2 ]));qy.append(float(T[3 ]));qz.append(float(T[4 ]));qw.append(float(T[5 ]));
			mx.append(float(T[6 ])*alphaM) ;my.append(float(T[7 ])*alphaM)  ;mz.append(float(T[8 ])*alphaM);
			ax.append(float(T[9 ])*alphaA) ;ay.append(float(T[10 ])*alphaA) ;az.append(float(T[11])*alphaA);
			wx.append(float(T[12])*alphaW) ;wy.append(float(T[13])*alphaW)  ;wz.append(float(T[14])*alphaW);			
		i=i+1
	f.close()
	return time,qx,qy,qz,qw,mx,my,mz,ax,ay,az,wx,wy,wz  

def readN(n):
	with open(n) as myfile:
		count = sum(1 for line in myfile)	
	f = open(n,'r')	
	time = [];
	#int  = [];
	qx   = [];
	qy   = [];
	qz   = [];
	qw   = [];
	ax   = [];
	ay   = [];
	az   = [];
	mx   = [];
	my   = [];
	mz   = [];
	wx   = [];
	wy   = [];
	wz   = [];
	alphaA =   16*9.8/32768.0;
	alphaM =   1000.0/32768.0;
	alphaW =   5000.0/32768.0;
	i = 0;	
	for l in f.readlines():
		T = []
		T = l.split(',')	
		if i!=0 and i<count-2:
			time.append(float(T[0 ])/1000.0)   ;qx.append(float(T[2 ]))       ;
			qy.append(float(T[3 ]))          ;qz.append(float(T[4 ]))         ;qw.append(float(T[5 ]));
			mx.append(float(T[6 ])*alphaM)   ;my.append(float(T[7 ])*alphaM)  ;mz.append(float(T[8 ])*alphaM);
			ax.append(float(T[9 ])*alphaA)   ;ay.append(float(T[10 ])*alphaA) ;az.append(float(T[11])*alphaA);
			wx.append(float(T[12])*alphaW)   ;wy.append(float(T[13])*alphaW)  ;wz.append(float(T[14])*alphaW);			
		i=i+1
	f.close()
	return time,qx,qy,qz,qw,mx,my,mz,ax,ay,az,wx,wy,wz    	
	
def GraphCap(name,namec,title):
	time  , qx   ,qy  ,qz  ,qw  ,mx  ,my  ,mz  , ax  ,ay  ,az  ,wx  ,wy  ,wz    = read(name );
	time_c, qx_c ,qy_c,qz_c,qw_c,mx_c,my_c,mz_c, ax_c,ay_c,az_c,wx_c,wy_c,wz_c  = read(namec);		
			
	H,P,R       = Eangle(qx,qy,qz,qw)
	H_c,P_c,R_c = Eangle(qx_c,qy_c,qz_c,qw_c)		
	qn    = Normq(qx,qy,qz,qw)
	qn_c  = Normq(qx_c,qy_c,qz_c,qw_c)	
	
	Q_c = []
	Q_c.append(qx_c)
	Q_c.append(qy_c)
	Q_c.append(qz_c)
	Q_c.append(qw_c)		
	Qn_c = Normalize(Q_c,qn)
	qxn_c = Qn_c[0]
	qyn_c = Qn_c[1]
	qzn_c = Qn_c[2]
	qwn_c = Qn_c[3]
	
	qn_c = Normq(qx_c,qy_c,qz_c,qw_c)
	an_c = Norm(ax_c,ay_c,az_c)
	mn_c = Norm(mx_c,my_c,mz_c)
	wn_c = Norm(wx_c,wy_c,wz_c)	
	
	plt.figure(1)
	plt.subplot(411)	
	plt.plot(time,qx,'.b')
	plt.plot(time_c,qx_c,'r')	
	plt.title('Quaternion x,y,z,w'  + title)	
	plt.ylabel('qx')
	plt.ylim([-1,1])
	plt.subplot(412)		
	plt.plot(time,qy,'.b')
	plt.plot(time_c,qyn_c,'r')
	plt.ylabel('qy')
	plt.ylim([-1,1])
	plt.subplot(413)	
	plt.plot(time,qz,'.b')
	plt.plot(time_c,qzn_c,'r')	
	plt.ylabel('qz')
	plt.ylim([-1,1])
	plt.subplot(414)
	plt.plot(time,qw,'.b')
	plt.plot(time_c,qwn_c,'r')	
	plt.ylabel('qw')
	plt.plot(time_c,qn_c,'.b')
	
	plt.figure(2)
	plt.subplot(311)
	plt.plot(time,ax,'.b')
	plt.plot(time_c,ax_c,'r')	
	plt.title('linear acceleration x,y,z')
	plt.subplot(312)
	plt.plot(time,ay,'.b')
	plt.plot(time_c,ay_c,'r')	
	plt.subplot(313)
	plt.plot(time,az,'.b')
	plt.plot(time_c,az_c,'r')	
	
	plt.figure(3)
	plt.subplot(311)
	plt.plot(time, mx,'.b')
	plt.plot(time_c,mx_c,'.r')
	plt.title('magnetic field x,y,z')
	plt.subplot(312)
	plt.plot(time, my,'.b')
	plt.plot(time_c,my_c,'.r')
	plt.subplot(313)
	plt.plot(time, mz,'.b')
	plt.plot(time_c,mz_c,'.r')
	
	plt.figure(4)
	plt.subplot(311)
	plt.plot(time, wx,'.b')
	plt.plot(time_c,wx_c,'r')
	plt.title('angular velocity x,y,z'+title)
	plt.ylim([-2,2])
	plt.subplot(312)
	plt.plot(time, wy,'.b')
	plt.plot(time_c,wy_c,'r')
	plt.ylim([-2,2])
	plt.subplot(313)
	plt.plot(time, wz,'.b')
	plt.plot(time_c,wz_c,'r')
	plt.ylim([-2,2])	
	
	plt.figure(5)
	plt.subplot(311)	
	plt.plot(time,H,'.b')
	plt.plot(time_c,H_c,'.r')	
	plt.title('Quaternion Euler angles')	
	plt.ylabel('Heading')
	plt.subplot(312)		
	plt.plot(time,P,'.b')
	plt.plot(time_c,P_c,'.r')	
	plt.ylabel('Pitch')
	plt.subplot(313)	
	plt.plot(time,R,'.b')
	plt.plot(time_c,R_c,'.r')
	plt.ylabel('Roll')	
	plt.show()	

def GraphNCap1(namep1,N,namep3):
	path = 'C:/Users/Simon/Desktop/Data_stationnary_Sensor/Long_term_stationnary_recording/'
	import seaborn as sns
	Lw = 800
	for i in range(N):			
		sensorpath = path + 'sensor' + str(i) + '/'
		name = namep1 + str(i) + namep3;
		time  ,qx  ,qy  ,qz  ,qw  ,mx  ,my  ,mz  , ax  ,ay  ,az  ,wx  ,wy  ,wz   = readN(name)			
		fig1 = plt.figure(1)
		plt.subplot(311)
		plt.plot(time,qx)
		plt.title('Quaternion x,y,z')	
		plt.ylabel('qx')
		plt.subplot(312)
		plt.ylabel('qy')
		plt.plot(time,qy)
		plt.subplot(313)		
		plt.ylabel('qz')
		plt.plot(time,qz)
		plt.xlabel('time')			
		fig1.savefig(sensorpath + 'rawQxQyQZQw' + '.png')
		print sensorpath + 'rawQxQyQZQw' + '.png'
		plt.close(fig1)
		
		fig2 = plt.figure(2)
		plt.subplot(311)
		plt.plot(time,wx)
		plt.title('Angular velocity x,y,z')	
		plt.ylabel('Wx')
		plt.subplot(312)
		plt.plot(time,wy)
		plt.ylabel('Wy')
		plt.subplot(313)
		plt.plot(time,wz)
		plt.ylabel('Wz')
		plt.xlabel('time')	
		fig2.savefig(sensorpath + 'rawWxWyWZ' + '.png')
		print sensorpath + 'rawWxWyWZ' + '.png'
		plt.close(fig2)
			
		fig3 = plt.figure(3)
		plt.subplot(311)
		MovingMoments(qx,time,Lw)
		plt.title('moving average and Std of Quaternion x,y,z')	
		plt.ylabel('qx')
		plt.subplot(312)
		plt.ylabel('qy')
		MovingMoments(qy,time,Lw)
		plt.subplot(313)
		MovingMoments(qz,time,Lw)
		plt.ylabel('qz')
		plt.xlabel('time')	
		fig3.savefig(sensorpath + 'MovingAvQxQyQZ' + '.png')
		print sensorpath + 'MovingAvQxQyQZ' + '.png'
		plt.close(fig3)
		
		fig4 = plt.figure(4)
		plt.subplot(311)
		MovingMoments(wx,time,Lw)
		plt.title('moving average and Std of Angular velocity x,y,z')	
		plt.ylabel('Wx')
		plt.subplot(312)
		plt.ylabel('Wy')
		MovingMoments(wy,time,Lw)
		plt.subplot(313)
		MovingMoments(wz,time,Lw)
		plt.ylabel('Wz')
		plt.xlabel('time')	
		fig4.savefig(sensorpath + 'MovingAvWxWyW' + '.png')
		print sensorpath + 'MovingAvWxWyW' + '.png'
		plt.close(fig4)
		
		fig5 = plt.figure(5)
		plt.subplot(311)
		Noisedistribution(qx,50)
		plt.title('Quaternion Noise distribution')	
		plt.subplot(312)
		Noisedistribution(qy,50)
		plt.subplot(313)
		Noisedistribution(qz,50)
		fig5.savefig(sensorpath + 'NoiseDistQxQyQZQw' + '.png')
		print sensorpath + 'NoiseDistQxQyQZQw' + '.png'
		plt.close(fig5)
		
		fig6 = plt.figure(6)
		plt.subplot(311)
		Noisedistribution(wx,50)
		plt.title('Angular Velocity Noise distribution')	
		plt.subplot(312)
		Noisedistribution(wy,50)
		plt.subplot(313)
		Noisedistribution(wz,50)
		fig6.savefig(sensorpath + 'NoiseDistWxWyWZ' + '.png')
		print sensorpath + 'NoiseDistWxWyWZ' + '.png'
		plt.close(fig6)
		
		fig7 = plt.figure(7)
		plt.subplot(311)
		Noisedistribution(ax,50)
		plt.title('Acceleration Noise distribution')	
		plt.subplot(312)
		Noisedistribution(ay,50)
		plt.subplot(313)
		Noisedistribution(az,50)
		fig7.savefig(sensorpath + 'NoiseDistAxAyAz' + '.png')
		print sensorpath + 'NoiseDistAxAyAz' + '.png'
		plt.close(fig7)
		
		fig8 = plt.figure(8)
		plt.subplot(311)
		Noisedistribution(mx,50)
		plt.title('Magnetic Noise distribution')	
		plt.subplot(312)
		Noisedistribution(my,50)
		plt.subplot(313)
		Noisedistribution(mz,50)
		fig8.savefig(sensorpath + 'NoiseDistMxMyMz' + '.png')
		print sensorpath + 'NoiseDistMxMyMz' + '.png'
		plt.close(fig8)
		
		with sns.color_palette("RdBu_r", 10):
			fig9 = plt.figure(9)
			plt.subplot(311)
			CummulantProb(qx,Lw,5)
			plt.title('qx Cummulative Prob (time goes blue to red)')	
			plt.subplot(312)
			CummulantProb(qy,Lw,5)
			plt.title('qy Cummulative Prob')	
			plt.subplot(313)
			CummulantProb(qz,Lw,5)
			plt.title('qz Cummulative Prob')	
			fig9.savefig(sensorpath + 'CummuProbQxQyQz' + '.png')
			print sensorpath + 'CummuProbQxQyQz' + '.png'
			plt.close(fig9)
		with sns.color_palette("RdBu_r", 10):
			fig10 = plt.figure(10)
			plt.subplot(311)
			CummulantProb(wx,Lw,5)
			plt.title('Wx Cummulative Prob (time goes blue to red)')	
			plt.subplot(312)
			CummulantProb(wy,Lw,5)
			plt.title('Wy Cummulative Prob')	
			plt.subplot(313)
			CummulantProb(wz,Lw,5)
			plt.title('Wz Cummulative Prob')	
			fig10.savefig(sensorpath + 'CummuProbWxWyWz' + '.png')
			print sensorpath + 'CummuProbWxWyWz' + '.png'
			plt.close(fig10)
		
def GraphNCap2(namep1,N,namep3):	
	path = 'C:/Users/Simon/Desktop/Data_stationnary_Sensor/Long_term_stationnary_recording/Batch/'
	Lw = 240
	for i in range(N):	
		name = namep1 + str(i) + namep3;
		time  ,qx  ,qy  ,qz  ,qw  ,mx  ,my  ,mz  , ax  ,ay  ,az  ,wx  ,wy  ,wz   = readN(name)			
		fig1 = plt.figure(1)
		plt.subplot(311)
		plt.plot(time,qx)
		plt.title('Quaternion x,y,z')	
		plt.ylabel('qx')
		plt.subplot(312)
		plt.ylabel('qy')
		plt.plot(time,qy)
		plt.subplot(313)		
		plt.ylabel('qz')
		plt.plot(time,qz)
		plt.xlabel('time')				
		
		fig2 = plt.figure(2)
		plt.subplot(311)
		plt.plot(time,wx)
		plt.title('Angular velocity x,y,z')	
		plt.ylabel('Wx')
		plt.subplot(312)
		plt.plot(time,wy)
		plt.ylabel('Wy')
		plt.subplot(313)
		plt.plot(time,wz)
		plt.ylabel('Wz')
		plt.xlabel('time')			
			
		fig3 = plt.figure(3)
		plt.subplot(311)
		MovingMoments(qx,time,Lw)
		plt.title('moving average and Std of Quaternion x,y,z')	
		plt.ylabel('qx')
		plt.subplot(312)
		plt.ylabel('qy')
		MovingMoments(qy,time,Lw)
		plt.subplot(313)
		MovingMoments(qz,time,Lw)
		plt.ylabel('qz')
		plt.xlabel('time')			
		
		fig4 = plt.figure(4)
		plt.subplot(311)
		MovingMoments(wx,time,Lw)
		plt.title('moving average and Std of Angular velocity x,y,z')	
		plt.ylabel('Wx')
		plt.subplot(312)
		plt.ylabel('Wy')
		MovingMoments(wy,time,Lw)
		plt.subplot(313)
		MovingMoments(wz,time,Lw)
		plt.ylabel('Wz')
		plt.xlabel('time')			
	fig1.savefig(path + 'rawQxQyQZQw'       + '.png')		
	plt.close(fig1)
	fig2.savefig(path + 'rawWxWyWZ'         + '.png')
	plt.close(fig2)
	fig3.savefig(path + 'MovingAvQxQyQZ'   + '.png')
	plt.close(fig3)
	fig4.savefig(path + 'MovingAvWxWyW'    + '.png')
	plt.close(fig4)

def GraphNCap3(namep1,N,namep3):
	Lw = 240
	for i in range(N):	
		name = namep1 + str(i) + namep3;
		time  ,qx  ,qy  ,qz  ,qw  ,mx  ,my  ,mz  , ax  ,ay  ,az  ,wx  ,wy  ,wz   = readN(name)
		print i
		# plt.figure(1)
		# x = plt.subplot(211)
		# plt.plot(time,qx)		
		# plt.title('Qx component 9 sensors')	
		# plt.ylabel('qx')
		# plt.subplot(212, sharex=x)	
		# MovingMoments(qx,time,Lw)
		# plt.ylabel('moving average and Std of Qx ')
		# plt.xlabel('time (sec)')
		
		# plt.figure(2)
		# x =plt.subplot(211)
		# plt.plot(time,qy)		
		# plt.title('Qy component 9 sensors')	
		# plt.ylabel('qy')
		# plt.subplot(212, sharex=x)	
		# MovingMoments(qy,time,Lw)
		# plt.ylabel('moving average and Std of Qy ')
		# plt.xlabel('time (sec)')
		
		# plt.figure(3)
		# x = plt.subplot(211)
		# plt.plot(time,qz)		
		# plt.title('Qz component 9 sensors')	
		# plt.ylabel('qz')
		# plt.subplot(212, sharex=x)	
		# MovingMoments(qz,time,Lw)
		# plt.ylabel('moving average and Std of Qz ')
		# plt.xlabel('time (sec)')
				
				
		# H,P,R = Eangle(qx,qy,qz,qw)	
		# print H[0], P[0], R[0]
		# H = Plus(H,-H[0])	
		# P = Plus(P,-P[0])
		# R = Plus(R,-R[0])	
				
		# plt.figure(4)
		# x = plt.subplot(211)
		# plt.plot(time,H)		
		# plt.title('H Euler angle for 9 sensors')	
		# plt.ylabel('H')
		# plt.subplot(212, sharex=x)	
		# MovingMoments(H,time,Lw)
		# plt.ylabel('moving average of H ')
		# plt.xlabel('time (sec)')		
				
		# plt.figure(5)
		# x = plt.subplot(211)
		# plt.plot(time,P)		
		# plt.title('P Euler angles for 9 sensors')	
		# plt.ylabel('P')
		# plt.subplot(212, sharex=x)	
		# MovingMoments(P,time,Lw)
		# plt.ylabel('moving average of P ')
		# plt.xlabel('time (sec)')	
		
		# plt.figure(6)
		# x = plt.subplot(211)
		# plt.plot(time,R)		
		# plt.title('R Euler angles for 9 sensors')	
		# plt.ylabel('R')
		# plt.subplot(212, sharex=x)	
		# MovingMoments(R,time,Lw)
		# plt.ylabel('moving average of R ')
		# plt.xlabel('time (sec)')		
				
				
		# an = Norm(ax,ay,az)
		# plt.figure(7)
		# plt.subplot(411, sharex=x)
		# plt.plot(time,ax)
		# plt.ylabel('ax (degrees/sec*sec)')		
		# plt.title('linear acceleration x,y,z')
		# plt.subplot(412, sharex=x)
		# plt.plot(time,ay)
		# plt.ylabel('ay (degrees/sec*sec)')		
		# plt.subplot(413, sharex=x)
		# plt.plot(time,az)		
		# plt.ylabel('az (degrees/sec*sec)')		
		# plt.subplot(414, sharex=x)
		# plt.plot(time,an)
		# plt.ylabel('an (degrees/sec*sec)')			
		# plt.xlabel('time (sec)')	
		
		mn = Norm(mx,my,mz)
		zero = []
		Vm   = []
		for i in mx: zero.append(0.0)
		Hm = Norm(mx,my,zero)	
		dm = []	
		for i in range(0,len(Hm)): dm.append(np.arctan(-Hm[i]/mz[i])*180.0/(np.pi) )
		
		plt.figure(8)
		plt.subplot(231)
		plt.plot(time, mx)
		plt.ylabel('mx (micro Tesla)')	
		plt.xlabel('time (sec)')
		plt.subplot(232)
		plt.plot(time, my)
		plt.ylabel('my ((micro Tesla))')
		plt.xlabel('time (sec)')
		plt.subplot(233)
		plt.plot(time, mz,)
		plt.ylabel('mz ((micro Tesla))')
		plt.xlabel('time (sec)')
		plt.subplot(234)
		plt.plot(time, mn,)
		plt.ylabel('mn ((micro Tesla))')		
		plt.xlabel('time (sec)')	
		plt.subplot(235)
		plt.plot(time, dm)
		plt.ylabel('inclination (degrees)')
		plt.xlabel('time (sec)')		
		
		
		# plt.figure(4)
		# plt.subplot(311)
		# plt.plot(time, wx)
		# plt.ylabel('wx (degrees/sec)')
		# plt.title('angular velocity x,y,z')
		# plt.subplot(312)
		# plt.plot(time, wy)
		# plt.ylabel('wx (degrees/sec)')
		# plt.subplot(313)
		# plt.plot(time, wz)
		# plt.ylabel('wx (degrees/sec)')		
		# plt.xlabel('time (sec)')	
		
		# plt.figure(5)
		# plt.subplot(311)
		# MovingMoments(wx,time,Lw)
		# plt.title('moving average and Std of Angular velocity x,y,z')	
		# plt.ylabel('wx (degrees/sec)')
		# plt.subplot(312)
		# plt.ylabel('wy (degrees/sec)')
		# MovingMoments(wy,time,Lw)
		# plt.subplot(313)
		# MovingMoments(wz,time,Lw)
		# plt.ylabel('wz (degrees/sec)')
		# plt.xlabel('time (sec)')	
		
		# plt.figure(6)
		# plt.subplot(311)
		# MovingMoments(mx,time,Lw)
		# plt.title('moving average and Std of magnetic field x,y,z')	
		# plt.ylabel('mx (micro Tesla)')
		# plt.subplot(312)
		# plt.ylabel('my (micro Tesla)')
		# MovingMoments(my,time,Lw)
		# plt.subplot(313)
		# MovingMoments(mz,time,Lw)
		# plt.ylabel('mz (micro Tesla)')
		# plt.xlabel('time (sec)')
		
		# plt.figure(7)
		# plt.subplot(311)
		# MovingMoments(ax,time,Lw)
		# plt.title('moving average and Std of accelaration x,y,z')	
		# plt.ylabel('ax (m/sec*sec)')
		# plt.subplot(312)
		# plt.ylabel('ay (m/sec*sec)')
		# MovingMoments(ay,time,Lw)
		# plt.subplot(313)
		# MovingMoments(az,time,Lw)
		# plt.ylabel('az (m/sec*sec)')
		# plt.xlabel('time (sec)')	
	plt.show()

def GraphArm(SensorFileName,CorrectedFileName,RefFileName,RefTimeRefFileName):
	time  , qx   ,qy  ,qz  ,qw  ,mx  ,my  ,mz  , ax  ,ay  ,az  ,wx  ,wy  ,wz    = read(SensorFileName);
	time_c, qx_c ,qy_c,qz_c,qw_c,mx_c,my_c,mz_c, ax_c,ay_c,az_c,wx_c,wy_c,wz_c  = read(CorrectedFileName);		
	Theta1,Theta2,Theta3,time_r = DataTrajectory(RefFileName,RefTimeRefFileName)
	
	H,P,R       		= Eangle(qx,qy,qz,qw)
	H_c,P_c,R_c 		= Eangle(qx_c,qy_c,qz_c,qw_c)
	H = Plus(H,Theta1[70])
	H_c = Plus(H_c,Theta1[70])
	
	#qw_r,qx_r,qy_r,qz_r = Quat(Theta1,Theta2,Theta3)
	#time_r = ContinuousTime(time_r,time[0])	
	time_r = range(1900,63000,102)	
	plt.figure(1)
	plt.subplot(411)	
	plt.plot(time,qx,'.b')
	plt.plot(time_c,qx_c,'r')	
	plt.title('Quaternion x,y,z,w')	
	plt.ylabel('qx')
	plt.ylim([-1,1])
	plt.subplot(412)		
	plt.plot(time,qy,'.b')
	plt.plot(time_c,qy_c,'r')
	plt.ylabel('qy')
	plt.ylim([-1,1])
	plt.subplot(413)	
	plt.plot(time,qz,'.b')
	plt.plot(time_c,qz_c,'r')	
	plt.ylabel('qz')
	plt.ylim([-1,1])
	plt.subplot(414)
	plt.plot(time,qw,'.b')
	plt.plot(time_c,qw_c,'r')	
	plt.ylabel('qw')
	
	plt.figure(2)
	plt.subplot(311)	
	plt.plot(H,'.b')
	plt.plot(H_c,'.r')
	plt.plot(time_r,Theta1,'.g')
	plt.title('Quaternion Euler angles')	
	plt.ylabel('Heading')
	plt.ylim(-200,200)
	plt.subplot(312)		
	plt.plot(P,'.b')
	plt.plot(P_c,'.r')		
	#plt.plot(Theta2,'.g')
	plt.ylabel('Pitch')
	plt.ylim(-200,200)
	plt.subplot(313)	
	plt.plot(R,'.b')
	plt.plot(R_c,'.r')		
	#plt.plot(Theta3,'.g')
	plt.ylabel('Roll')	
	plt.ylim(-200,200)	
	plt.show()	

def ArmTrajectory(f,nStep,oi,of,ti,tf,zi,zf,wo,wt,wz):	
	file = open(f,'w')
	o = []
	t = []
	z = []
	for i in range(0,nStep):		
		if i<50: I = 0
		else: I = I+1		 
		o.append(oi*(1.0 - np.abs(math.sin(wo*I))) + of*np.abs(math.sin(wo*I))) 
		t.append(ti*(1.0 - np.abs(math.sin(wt*I))) + tf*np.abs(math.sin(wt*I)))
		z.append(zi*(1.0 - np.abs(math.sin(wz*I))) + zf*np.abs(math.sin(wz*I)))
		
	for l in range(nStep):	
		s = format(o[l],'.2f')+','+format(t[l],'.2f')+','+format(z[l],'.2f')+',\n' 
		file.writelines(s)
	file.close()

def DataTrajectory(f1,f2):	
	file1 = open(f1,'r')
	file2 = open(f2,'r')
	Tx = []
	Ty = []
	Tz = []
	t  = []
	for l in file1.readlines():
		Angle = []
		Angle = l.split(',')
		Tx.append(float(Angle[0]));Ty.append(float(Angle[1]));Tz.append(float(Angle[2]));	
	file1.close()
	for l in file2.readlines():
		t.append(int(l));
	file2.close()	
	return Tx,Ty,Tz,t
	










		# plt.plot(time,qx)
		# plt.title('Quaternion x,y,z,w')	{}
		# plt.ylabel('qx')
		# plt.subplot(212)		
		# plt.plot(time,qy)
		# plt.ylabel('qy')
		# plt.subplot(413)	
		# plt.plot(time,qz)
		# plt.ylabel('qz')
		# plt.subplot(414)
		# plt.plot(time,qw)
		# plt.ylabel('qw')
		
		# plt.figure(2)
		# plt.subplot(311)
		# plt.plot(time,ax)
		# plt.title('linear acceleration x,y,z')
		# plt.subplot(312)
		# plt.plot(time,ay)
		# plt.subplot(313)
		# plt.plot(time,az)
		
		# plt.figure(3)
		# plt.subplot(311)
		# plt.plot(time, mx)
		# plt.title('magnetic field x,y,z')
		# plt.subplot(312)
		# plt.plot(time, my)
		# plt.subplot(313)
		# plt.plot(time, mz,)
		
		# plt.figure(4)
		# plt.subplot(311)
		# plt.plot(time, wx)
		# plt.title('angular velocity x,y,z')
		# plt.subplot(312)
		# plt.plot(time, wy)
		# plt.subplot(313)
		# plt.plot(time, wz)
		
		# plt.figure(5)
		# plt.subplot(311)
		# MovingMoments(wx,time,Lw)
		# plt.title('moving average and Std of Angular velocity x,y,z')	
		# plt.ylabel('wx')
		# plt.subplot(312)
		# plt.ylabel('wy')
		# MovingMoments(wy,time,Lw)
		# plt.subplot(313)
		# MovingMoments(wz,time,Lw)
		# plt.ylabel('wz')
		# plt.xlabel('time')	
		
		# plt.figure(6)
		# plt.subplot(311)
		# MovingMoments(mx,time,Lw)
		# plt.title('moving average and Std of magnetic field x,y,z')	
		# plt.ylabel('mx')
		# plt.subplot(312)
		# plt.ylabel('my')
		# MovingMoments(my,time,Lw)
		# plt.subplot(313)
		# MovingMoments(mz,time,Lw)
		# plt.ylabel('mz')
		# plt.xlabel('time')
		
		# plt.figure(7)
		# plt.subplot(311)
		# MovingMoments(ax,time,Lw)
		# plt.title('moving average and Std of accelaration x,y,z')	
		# plt.ylabel('ax')
		# plt.subplot(312)
		# plt.ylabel('ay')
		# MovingMoments(ay,time,Lw)
		# plt.subplot(313)
		# MovingMoments(az,time,Lw)
		# plt.ylabel('az')
		# plt.xlabel('time')	
	#plt.show()











