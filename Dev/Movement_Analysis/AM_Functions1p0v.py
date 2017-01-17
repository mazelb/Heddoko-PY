from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib
	
import numpy.random as npr
import numpy as np
import pandas as pd
import math

from Tkinter import *	
import copy as cop
import sys


##########Class#############
class ListOfGraphPoints:
	def __init__(self, line):		
		self.line = line
		self.xs   = []
		self.ys   = []
		self.cid  = line.figure.canvas.mpl_connect('button_press_event', self)

	def __call__(self, event):
		self.xs.append(event.xdata)
		self.ys.append(event.ydata)
class rectangle: 
	def __init__(self,px,py,dx,dy,ax):		
		self.xc  = np.mean(px)
		self.yc  = np.mean(py)	
		self.dx  = dx
		self.dy  = dy
		self.ax  = ax
	def draw(self):
		self.ax.add_patch(Rectangle((self.xc - self.dx/2.0, self.yc - self.dy/2.0), self.dx, self.dy, alpha=1))
class TposeGroup:
	def __init__(self,LabelG):			
		self.LG = LabelG
		self.PG	= []
		self.GFMatrix = []
		self.MFvec    = []
		self.StdFvec  = []
		self.T  = [] 
	def add(self,I,df):		
		self.PG.append(df.ix[I])				
	def NTpp(self):
		return len(self.PG)
	def NFeatures(self):
		return (len(list(self.PG[0]))-3)
	def Time(self):
		T = [tp[0] for tp  in self.PG]
		return T
	def MFeature(self):
		self.MFvec   = np.mean(self.GFMx(), axis=0)		
		return self.MFvec
	def StdFeature(self):		
		self.StdFvec   = np.std(self.GFMx(), axis=0)		
		return self.StdFvec
	def GFMx(self):
		N = self.NTpp()
		P = self.NFeatures()
		self.GFMatrix = np.zeros((N,P),float)		
		for n in range(N):
			for p in range(P):self.GFMatrix[n,p] = self.PG[n][p+3] 
		return self.GFMatrix
class SignalCorrector:
	import numpy as np
	def __init__(self,df,MinThres=np.pi/4.,Opt='Plateaux'):	
		 from sklearn.cluster import KMeans	
		 import matplotlib.pyplot as plt
		 import pandas as pd
		 import numpy as np	
		 self.Nl    = len(df)		
		 self.Istep = []
		 self.df = df.copy(deep=True) 
		 it = range(self.Nl)
		 for i in range(1,10):
			 G   = [];  
			 Sx  = str(i)+'x'; Sy  = str(i)+'y'; Sz  = str(i)+'z';
			 sdx = self.df[Sx].diff().values[1:]; adx = np.abs(sdx); 
			 sdy = self.df[Sy].diff().values[1:]; ady = np.abs(sdy);
			 sdz = self.df[Sz].diff().values[1:]; adz = np.abs(sdz);			 
			 for j in range(self.Nl-1):	G.append([adx[j],ady[j],adz[j]])
			 YG  = KMeans(n_clusters=3).fit_predict(G)
			 dG  = np.array(G)
			 YmG0= np.mean(dG[YG==0]); YmG1= np.mean(dG[YG==1]); YmG2= np.mean(dG[YG==2]);
			 Threshold = max([YmG0,YmG1,YmG2,MinThres])
			 print Threshold
			 ########################
			 self.last_key = ''			 
			 self.subplot_remove = set([])			 
			 self.fig, self.ax = plt.subplots(3,1)
			 self.ax[0].plot(df['Frame Index'].values,df[Sx].values)
			 self.ax[1].plot(df['Frame Index'].values,df[Sy].values)
			 self.ax[2].plot(df['Frame Index'].values,df[Sz].values)			 
			 self.fig.canvas.mpl_connect('key_press_event',self.Press_key)
			 self.fig.canvas.mpl_connect('button_press_event',self.Click_mouse)		
			 self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)			 
			 ########################					 
			 self.Istep.append([[],[],[]])
			 for j in range(1,self.Nl):
				 ux = self.SON(sdx[j-1],Threshold);
				 uy = self.SON(sdy[j-1],Threshold);
				 uz = self.SON(sdz[j-1],Threshold);					 
				 if (ux!=0):self.Istep[i-1][0].append(ux*j);
				 if (uy!=0):self.Istep[i-1][1].append(uy*j);
				 if (uz!=0):self.Istep[i-1][2].append(uz*j);
			 Nx = len(self.Istep[i-1][0]); 
			 Ny = len(self.Istep[i-1][1]); 
			 Nz = len(self.Istep[i-1][2]); 
			 for Ii in range(Nx-1):				 
				 Iac = self.Istep[i-1][0][Ii]; Inx = self.Istep[i-1][0][Ii+1];
				 Ia  = np.abs(Iac);            In  = np.abs(Inx);
				 if  (Iac > 0) and (Inx < 0):
					 meanStep = (sdx[Ia-1]-sdx[In-1])/2.0
					 self.ax[0].plot(self.df['Frame Index'].values[Ia:In],self.df[Sx].values[Ia:In],'.r')						
					 self.df[Sx].values[Ia:In]-= meanStep
					 self.Istep[i-1][0][Ii]    = 0
					 self.Istep[i-1][0][Ii+1]  = 0
				 elif (Iac < 0) and (Inx > 0):
					 meanStep = (sdx[Ia-1]-sdx[In-1])/2.0
					 self.df[Sx].values[Ia:In]-= meanStep
					 self.ax[0].plot(self.df['Frame Index'].values[Ia:In],self.df[Sx].values[Ia:In],'.r')
					 self.Istep[i-1][0][Ii]    = 0
					 self.Istep[i-1][0][Ii+1]  = 0
			 for Ij in range(Ny-1):				 
				Iac = self.Istep[i-1][1][Ij]; Iny = self.Istep[i-1][1][Ij+1];
				Ia  = np.abs(Iac); 		   In  = np.abs(Iny);				 
				if  (Iac > 0) and (Iny < 0):
					 meanStep = (sdy[Ia-1]-sdy[In-1])/2.0						 
					 self.df[Sy].values[Ia:In]-= meanStep
					 self.ax[1].plot(self.df['Frame Index'].values[Ia:In],self.df[Sy].values[Ia:In],'.r')
					 self.Istep[i-1][1][Ij]    = 0
					 self.Istep[i-1][1][Ij+1]  = 0
				elif (Iac < 0) and (Iny > 0):
					 meanStep = (sdy[Ia-1]-sdy[In-1])/2.0								 
					 self.df[Sy].values[Ia:In]-= meanStep						 
					 self.ax[1].plot(self.df['Frame Index'].values[Ia:In],self.df[Sy].values[Ia:In],'.r')
					 self.Istep[i-1][1][Ij]   = 0
					 self.Istep[i-1][1][Ij+1] = 0
			 for Ik in range(Nz-1):		 
				 Iac = self.Istep[i-1][2][Ik]; Inz = self.Istep[i-1][2][Ik+1];
				 Ia  = np.abs(Iac); 		   In  = np.abs(Inz);						 
				 if  (Iac > 0) and (Inz < 0):
					 meanStep = (sdz[Ia-1]-sdz[In-1])/2.0
					 self.df[Sz].values[Ia:In]-= meanStep
					 self.ax[2].plot(self.df['Frame Index'].values[Ia:In],self.df[Sz].values[Ia:In],'.r')
					 self.Istep[i-1][2][Ik]   = 0
					 self.Istep[i-1][2][Ik+1] = 0
				 elif (Iac < 0) and (Inz > 0):
					 meanStep = (sdz[Ia-1]-sdz[In-1])/2.0
					 self.df[Sz].values[Ia:In]-= meanStep
					 self.ax[2].plot(self.df['Frame Index'].values[Ia:In],self.df[Sz].values[Ia:In],'.r')
					 self.Istep[i-1][2][Ik]   = 0
					 self.Istep[i-1][2][Ik+1] = 0
			 plt.show()
	def Press_key(self,event):
		self.last_key = event.key
	def Click_mouse(self,event):
		if self.last_key == 'c':
			if self.ax[0] == event.inaxes: 
				self.subplot_remove.add(0)
			if self.ax[1] == event.inaxes:
				self.subplot_remove.add(1)
			if self.ax[2] == event.inaxes:
				self.subplot_remove.add(2)
		if self.last_key == 'a':
			if self.ax[0] == event.inaxes and not(self.subplot_remove.isdisjoint(set([0]))): 
				self.subplot_remove.remove(0) 
			if self.ax[1] == event.inaxes and not(self.subplot_remove.isdisjoint(set([1]))):
				self.subplot_remove.remove(1)  
			if self.ax[2] == event.inaxes and not(self.subplot_remove.isdisjoint(set([2]))):
				self.subplot_remove.remove(2)  
		print self.subplot_remove
	def SON(self,t,T): 
		s = 0
		if(t >  np.abs(T)):
			s =  1
		if(t < -np.abs(T)):
			s = -1
		return s				
	def Get_Cleaned_Signal(self):
		return self.df						

###Ideal Tpose joint angle values###
Exceptions1 = ['RShould F/E','LShould F/E']
Exceptions2 = ['LShould Add/Abd']
Exceptions3 = ['TPose Value','Timestamp','Frame Index']
TpI = [0.,0.,0.,0.,0.,0.,0.,0.,-90.,-90.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
####################################

###Visualisation functions### 
def PlotDistance(Dt,Dt2,T,ListCol,df2,df,MinD,IMinD,MeanD,StdD):
		#Allow to analyse the data and to obtain the Ergo score
		DL    = ListCol
		xmean = [T[1],T[len(T)-1]]
		ymean = [MeanD,MeanD]	
		xstd  = [T[1],T[len(T)-1]]
		ystdn = [MeanD-StdD,MeanD-StdD]	
		ystdp = [MeanD+StdD,MeanD+StdD]
		plt.figure() 		
		for l in range(0,len(DL)):
			if DL[l] != 'TPose Value':
				ax1 = plt.subplot(311)
				plt.plot(T,df[DL[l]][0:len(T)])
				plt.legend(loc = 6,fontsize = 8)
				plt.title("Distance from rest for each joint")				
				plt.xlim(T[0],T[len(T)-1])
				plt.subplot(312,sharex=ax1)
				plt.plot(T,df2[l],label=DL[l])
				plt.title("Weight of the contribution of each joint to the Global distance")
				plt.legend(loc = 6,fontsize = 8)
		plt.subplot(313,sharex=ax1)
		plt.title("Global risk score releated to how far a user is from the max extention (summation of each joint contribution)")
		plt.plot(T[IMinD],MinD,'*r',label="Min risk score: "+ str(round(MinD)))	
		plt.plot(xmean,ymean,'r',label="Mean score: "+ str(round(MeanD)))
		plt.plot(xstd,ystdn,'-b',label="Mean - Std: "+ str(round(MeanD - StdD)))
		plt.plot(xstd,ystdp,'-b',label="Mean + Std: "+ str(round(MeanD + StdD)))
		plt.plot(T,Dt2,'g',label="Risk score")	
		plt.legend()			
		plt.show()
def PlotMoveDiffMoveIndex(df,bdb,Moybdb,MoyTpd):
		choice = 'y'
		L   = list(df);	l   = len(L)		
		Arecx  =  [];Arecy  =  [];Srecx  =  [];Srecy  =  [];
		#Main loop to analyse the two features 	
		while(choice=='y'):	
			##############################Joint angles distance#####################################
			Angle = raw_input("Do you want to seclect Tpose, using proximity index graph (middle) (y / n)  :")
			if Angle =='y':	
				print '##A box will be define by click twice in the middle plot window##'
				print '##The first point is for the left upper point of the box       ##'
				print '##The second is for the right lower point of the box           ##'
				Arecx = []
				Arecy = []
				fig = plt.figure()				
				for ic in range(3,l):
					ax1 = plt.subplot(311)
					plt.title("Graphic Interface")
					plt.plot(df['Frame Index'],df[L[ic]]-TpI[ic])
					plt.ylabel("Angle distance(degrees)")
					plt.xlim(bdb[0][0],bdb[0][len(bdb[0])-2])
				ax2 = plt.subplot(312,sharex=ax1)			
				plt.plot(bdb[0][0:-2],MoyTpd[0:-2])		
				plt.ylabel("Angle distance index")
				ax3 = plt.subplot(313,sharex=ax1)			
				plt.plot(bdb[0][0:-2],Moybdb[0:-2])	
				plt.ylabel('Stationnarity index')
				plt.xlabel('time')
				line, = ax1.plot([0], [0])
				APointsList = ListOfGraphPoints(line)
				plt.show()
				#creation of a list of points clicked on during the selection
				for i in range(0,len(APointsList.xs)/2):
					Arecx.append([APointsList.xs[2*i],APointsList.xs[2*i+1]])
					Arecy.append([APointsList.ys[2*i],APointsList.ys[2*i+1]])
					print Arecx[i],Arecy[i]
			#############################################################################
			StatIndex = raw_input("Do you want to seclect a region in the Stationnarity index (y / n)  :")	
			if StatIndex =='y':	
				print '##A box will be define by click twice in the middle plot window##'
				print '##The first point is for the left upper point of the box       ##'
				print '##The second is for the right lower point of the box           ##'
				Srecx = []
				Srecy = []
				fig = plt.figure()
				for ic in range(3,l):
					ax1 = plt.subplot(311)
					plt.title("Graphic Interface")
					plt.plot(df['Frame Index'],df[L[ic]]-TpI[ic-3])	
					plt.ylabel("Angle distance(degrees)")
					plt.xlim(bdb[0][0],bdb[0][len(bdb[0])-2])
				ax2 = plt.subplot(312,sharex=ax1)			
				plt.plot(bdb[0][0:-2],MoyTpd[0:-2])		
				plt.ylabel("Angle distance index")
				ax3 = plt.subplot(313,sharex=ax1)			
				plt.plot(bdb[0][0:-2],Moybdb[0:-2])	
				plt.ylabel('Stationnarity index')
				plt.xlabel('time')
				line, = ax1.plot([0], [0])
				SPointsList = ListOfGraphPoints(line)			
				plt.show()
				#creation of a list of points clicked on during the selection
				for i in range(0,len(SPointsList.xs)/2):
					Srecx.append([SPointsList.xs[2*i],SPointsList.xs[2*i+1]])
					Srecy.append([SPointsList.ys[2*i],0.0])
					print Srecx[i],Srecy[i]
			fig = plt.figure()
			for ic in range(3,l):
				ax1 = plt.subplot(311)
				plt.title("Results of the box selection")
				plt.plot(df['Frame Index'],df[L[ic]])
				plt.ylabel("Angle distance(degrees)")
				plt.xlim(bdb[0][0],bdb[0][len(bdb[0])-2])
			ax2 = plt.subplot(312,sharex=ax1)
			plt.plot(bdb[0][0:-2],MoyTpd[0:-2])
			plt.ylabel("Angle distance index")
			ax3 = plt.subplot(313,sharex=ax1)	
			plt.plot(bdb[0][0:-2],Moybdb[0:-2])
			plt.ylabel('Stationnarity index')
			plt.xlabel('time')
			ra = []
			rs = []
			for i in range(0,len(APointsList.xs)/2):
				#print i
				#print Arecy
				#print Arecy[i][1]
				#print Arecy[i][0]
				dy = np.abs(Arecy[i][1]-Arecy[i][0])
				dx = np.abs(Arecx[i][1]-Arecx[i][0])
				ra.append(rectangle(Arecx[i],Arecy[i],dx,dy,ax2))
				ra[i].draw()
			for i in range(0,len(SPointsList.xs)/2):			
				dy = np.abs(Srecy[i][1]-Srecy[i][0])
				dx = np.abs(Srecx[i][1]-Srecx[i][0])
				rs.append(rectangle(Srecx[i],Srecy[i],dx,dy,ax3))
				rs[i].draw()				
			plt.show()			
			choice = raw_input("continue (y / n)  :")
		return ra,rs
def GiveMeAnIdea(t,T,nt,NT,df):
	listc = list(df)
	Nsubp      = len(T)
	diffd      = []
	Logdiff    = []
	MoyLogdiff = []
	C 	  = 0
	for J in range(0,len(df['Frame Index'])-1): MoyLogdiff.append(0)	
	plt.figure(0)
	plt.subplot(311)
	plt.plot(df['Frame Index'],df['TPose Value'],'.')
	plt.subplot(312)
	for c in listc:		
		if c!='Timestamp' and c!='TPose Value' and c!='Frame Index':
			diffd.append([])
			Logdiff.append([])
			for J in range(1,len(df[c])):
				diffd[C].append(np.abs(df[c][J]-df[c][J-1]))
				Logdiff[C].append(np.log(1.0 + np.abs(df[c][J]-df[c][J-1])))
				MoyLogdiff[J-1] = MoyLogdiff[J-1] + Logdiff[C][J-1]
				if c == listc[-1] : MoyLogdiff[J-1] = MoyLogdiff[J-1]/(len(listc)-3)			
			plt.plot(df['Frame Index'][0:-1],Logdiff[C])
			C+=1	
	plt.subplot(313)	
	plt.plot(df['Frame Index'][0:-1],MoyLogdiff,'.')
	plt.figure(1)		
	for l in range(1,Nsubp):		
		plt.subplot(4,4,l)
		plt.plot(t,T[l],'.')
		plt.plot(nt,NT[l],'.r')
	plt.show()
#############################################

###Statistical tools###
def IntraGroupStat(TposeGroups):
	for ig in range(len(TposeGroups)):
		MFvec   = TposeGroups[ig].MFeature()
		StdFvec = TposeGroups[ig].StdFeature()
		GMatrixFeatures = TposeGroups[ig].GFMatrix
		NormMlessGMatrix = (GMatrixFeatures - MFvec)/StdFvec
		plt.figure(ig)	
		plt.subplot(211)
		plt.plot(GMatrixFeatures.transpose(),'.')
		plt.plot(MFvec,'b')
		plt.plot(MFvec - StdFvec, 'r' )
		plt.plot(MFvec + StdFvec, 'r' )
		plt.xlabel('Group features')
		plt.subplot(212)
		plt.plot(NormMlessGMatrix.transpose(),'.')
		plt.xlabel('Group features')
	plt.show()	
	return TposeGroup
def InterGroupStat(TposeGroups):
	from mpl_toolkits.mplot3d import Axes3D
	tpg = TposeGroups
	MFvec   = []
	StdFvec = []
	MTimeG  = []
	for ig in range(len(tpg)):			
		MFvec.append(tpg[ig].MFeature())
		StdFvec.append(tpg[ig].StdFeature())
		MTimeG.append(np.mean(tpg[ig].Time()))	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for s in range(0,len(tpg)-1):	
		for t in range(s+1,len(tpg)):
			nf = np.arange(16)
			dt = MTimeG[t]-MTimeG[s]			 
			dm = MFvec[t]-MFvec[s]
			ax.bar(nf, np.abs(dm), zs=np.abs(dt), zdir='y', alpha=0.8)
			#ax.bar(nf, np.abs(dm), zs=np.abs(dt), zdir='y', alpha=0.8)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')	
	plt.show()	
	return TposeGroup
#######################

###Features Computation functions###
def DiffBB(df):
		L   = list(df)
		l   = len(L)
		bdb    = []
		Moybdb = []
		for i in range(0,l-2): 
			bdb.append([])
		for i in range(0,len(df['Frame Index'])-2): 
			bdb[0].append(df.iat[i+1,0])
			Moybdb.append(0.0)
		return bdb
def Distance(df,StartReadingDataLine):			
		L  = list(df)
		Le = len(df[L[0]])
		LL = len(L)
		DTotalMax = LL*180.0
		T  = []
		Dt = []	
		Dt1= []	
		Dt2= []		
		MinD  = 100.0
		IMinD = 0
		I     = 0	
		df2 = []
		for i in range(0,LL): df2.append([])		
		for i in range(0,Le):
			if(i>=StartReadingDataLine ):
				Dt.append(0.0)
				Dt1.append(0.0)
				Dt2.append(0.0)
				T.append(0.0)
				for c in range(0,LL):
					Col = L[c]						
					df2[c].append(math.fabs(df[Col][i]))
					if Col != L[0] :
						Dt[I] = Dt[I] + (df[Col][i])**2						
						Dt2[I]= Dt2[I] + (1.0 - math.fabs(df[Col][i])/180.0)
					else: 
						T[I] = df[L[0]][i]				
				Dt1[I] = 1.0 - math.sqrt(Dt[I])/DTotalMax
				Dt2[I] = 100*Dt2[I]/LL
				Dt[I]  = math.sqrt(Dt[I])
				for Col in range(0,LL): 
					if Col!= L[0]: 
						df2[Col][I] = df2[Col][I]/Dt[I]				
				if MinD > Dt2[I]: 
					MinD  = Dt2[I]
					IMinD = I
				I=I+1
		s  = []
		for r in Dt2: 
			if not(math.isnan(r)): s.append(r);
		MeanD = np.mean(s)
		StdD  = np.std(s)
		return Dt,Dt2,T,df2,MinD,IMinD,MeanD,StdD
def IndexComputation(df):
		#Preparing the container for the differences between consecutive values
		bdb = DiffBB(df)
		##########################
		Moybdb = [];		MoyTpd = [];
		L      = list(df);	l      = len(L);			
		for i in range(0,len(df[L[0]])-2): 
			Moybdb.append(0.0)		
			MoyTpd.append(0.0)
		for ic in range(3,l):
			for i in range(0,len(bdb[0])):				
				d = np.abs(df.iat[i,ic] - TpI[ic])				
				b = df.iat[i,ic]
				n = df.iat[i+1,ic]
				a = df.iat[i+2,ic]
				bdb[ic-2].append(np.abs(n-b)+np.abs(a-n)/2.0)
				Moybdb[i] += bdb[ic-2][i]
				MoyTpd[i] += d				
		for i in range(0,len(bdb[0])-2): 
			Moybdb[i] = np.log(1.0+Moybdb[i]/(l-3.0))
			MoyTpd[i] = np.log(1.0+MoyTpd[i]/(l-3.0))			
		return bdb, Moybdb, MoyTpd
def TposeToRawData(TposeGroups,RDdf):
	tg      	  = []
	RawDataTpG    = []
	GroupsMeMiMx  = []	
	L = list(RDdf)
	for (g,ig) in zip(TposeGroups,range(len(TposeGroups))):
		tg.append(0.0)
		RD    = []		
		umin  = []	
		umax  = []
		umean = []			
		for itp in range(len(g.PG)): 			
			u = RDdf[RDdf['Frame Index'] == g.PG[itp]['Frame Index']]
			tg[ig] = tg[ig] + u['Frame Index'].values
			v = [u.iat[0,c] for c in range(1,len(L))]
			RD.append(v)
			if   itp==0: 
				for l in range(len(v)):
					umean.append(v[l])
					umin.append(v[l])
					umax.append(v[l])
			elif itp==len(g.PG)-1:
				for l in range(len(v)):
					umean[l] += v[l] 
					umean[l] = umean[l]/len(g.PG)
					if umin[l]>v[l]: umin[l]=v[l]	
					if umax[l]<v[l]: umax[l]=v[l]
			else: 
				for l in range(len(v)):
					umean[l] += v[l]
					if umin[l]>v[l]: umin[l]=v[l]	
					if umax[l]<v[l]: umax[l]=v[l]
		tg[ig]=tg[ig]/len(g.PG)
		RawDataTpG.append(RD)	
		GroupsMeMiMx.append([umean,umin,umax])
	return tg,RawDataTpG,GroupsMeMiMx
def Euler_ForwardErrorCorrection(I,TposeGroups,Eu,Tp):
	#SigC = SignalCorrector(10,1000)
	#SigC.Create_Mem('Ecfx')            							 
	#SigC.Create_Mem('Ecfy')
	#SigC.Create_Mem('Ecfz')	
	Eu_Ref = Tpose_Ref(Eu,'Init_Ref')
	CEu = Eu.copy(deep=True)	
	NiF = len(Eu['Frame Index'])
	TgI = TposeGroups[I].Time() 
	Tge,RDpEu,EuGmean = AE_GrFeature_Computation(TposeGroups[I],Eu)	
	for i in range(1,10):                                                                                   
		Sx       = str(i)+'x';Sy = str(i)+'y';Sz = str(i)+'z';Sw = str(i)+'w'
		mEuI     = [EuGmean[4*i+0],EuGmean[4*i+1],EuGmean[4*i+2]]
		Ex = []; Ecfx = []; 															 
		Ey = []; Ecfy = []; 																   
		Ez = []; Ecfz = [];
		for IFrame in range(NiF):			
			E = [Eu[Sx].values[IFrame],Eu[Sy].values[IFrame],Eu[Sz].values[IFrame]]	
			Ex.append(E[0]);   Ey.append(E[1]);   Ez.append(E[2]); 
			Ecfx.append(E[0]); Ecfy.append(E[1]); Ecfz.append(E[2]); 			
			if Eu['Frame Index'].values[IFrame] >= TgI[0]:
				Ecfx[IFrame]+=(-mEuI[0]+Eu_Ref[i-1][0]);
				Ecfy[IFrame]+=(-mEuI[1]+Eu_Ref[i-1][1]); 
				Ecfz[IFrame]+=(-mEuI[2]+Eu_Ref[i-1][2]);				
			CEu[Sx].values[IFrame] = Ecfx[IFrame];
			CEu[Sy].values[IFrame] = Ecfy[IFrame];
			CEu[Sz].values[IFrame] = Ecfz[IFrame];	
		meanGIx = [ mEuI[0] for j in range(len(TgI))]	
		meanGIy = [ mEuI[1] for j in range(len(TgI))]
		meanGIz = [ mEuI[2] for j in range(len(TgI))]		
		plt.figure(i)	
		plt.subplot(311)
		plt.plot(Eu['Frame Index'],Eu[Sx].values,'b')		
		plt.plot(TgI,meanGIx,'*r')
		plt.plot(Eu['Frame Index'],CEu[Sx].values,'-r')
		plt.subplot(312)
		plt.plot(Eu['Frame Index'],Eu[Sy].values,'b')					
		plt.plot(TgI,meanGIy,'*r')
		plt.plot(Eu['Frame Index'],CEu[Sy].values,'-r')
		plt.subplot(313)
		plt.plot(Eu['Frame Index'],Eu[Sz].values,'b')
		plt.plot(TgI,meanGIz,'*r')
		plt.plot(Eu['Frame Index'],CEu[Sz].values,'-r')
		plt.figure(0)		
		plt.subplot(311)
		plt.plot(Eu['Frame Index'],Ex,'b')				
		plt.plot(Eu['Frame Index'],Ecfx,'-r')
		plt.subplot(312)
		plt.plot(Eu['Frame Index'],Ey,'b')				
		plt.plot(Eu['Frame Index'],Ecfy,'-r')
		plt.subplot(313)
		plt.plot(Eu['Frame Index'],Ez,'b')				
		plt.plot(Eu['Frame Index'],Ecfz,'-r')		
	plt.show()
	if raw_input("Do you want to keep this Tpose ?  (y / n)  :") == 'y':		
		Tp["F"].append(I)
		return CEu.copy(deep=True)
	else:
		return Eu
def Euler_BackwardErrorCorrection(I,TposeGroups,Eu,Tp):
	#SigC = SignalCorrector(10,1000)	
	#SigC.Create_Mem('Ecbx')            							 
	#SigC.Create_Mem('Ecby')
	#SigC.Create_Mem('Ecbz')
	Eu_Ref = Tpose_Ref(Eu,'Init_Ref')
	CEu = Eu.copy(deep=True)
	TgI = TposeGroups[I].Time() 	
	Tge,RDpEu,EuGmean = AE_GrFeature_Computation(TposeGroups[I],Eu)	
	CEu = Eu.copy(deep=True)	
	NiF = len(Eu['Frame Index'])
	Ngr = len(TposeGroups)	
	IG  = 0
	if I>0 : 
		TgeIm1,RDpEuIm1,EuGmeanIm1 = AE_GrFeature_Computation(TposeGroups[I-1],Eu)	
		IG = TgeIm1
	elif I==0: 
		IG = Eu['Frame Index'].values[0]	
	for i in range(1,10):
		Sx = str(i)+'x';
		Sy = str(i)+'y';
		Sz = str(i)+'z';
		mEuI = [EuGmean[4*i+0],EuGmean[4*i+1],EuGmean[4*i+2]]
		Ex = []; Ecbx = []; 															 
		Ey = []; Ecby = []; 																   
		Ez = []; Ecbz = [];	
		t  = [];
		for IFrame in range(NiF):			
			##############
			E = [Eu[Sx].values[IFrame],Eu[Sy].values[IFrame],Eu[Sz].values[IFrame]]	 
			Ex.append(E[0]);
			Ey.append(E[1]);
			Ez.append(E[2]); 
			Ecbx.append(E[0]);
			Ecby.append(E[1]);
			Ecbz.append(E[2]); 			
			if Eu['Frame Index'].values[IFrame] <= Tge and Eu['Frame Index'].values[IFrame] >= IG:
				DT = Tge - IG;
				tu = Eu['Frame Index'].values[IFrame];
				dt = tu-IG;
				w = 1.0*dt/DT;
				Ecbx[IFrame]+= w*(-mEuI[0]+Eu_Ref[i-1][0]);
				Ecby[IFrame]+= w*(-mEuI[1]+Eu_Ref[i-1][1]);
				Ecbz[IFrame]+= w*(-mEuI[2]+Eu_Ref[i-1][2]);	
				CEu[Sx].values[IFrame] = Ecbx[IFrame];	
				CEu[Sy].values[IFrame] = Ecby[IFrame];
				CEu[Sz].values[IFrame] = Ecbz[IFrame];	
				t.append(IFrame)
			#Ecbx[IFrame] = SigC.Check_This('Ecbx',Ecbx[IFrame]) 
			#Ecby[IFrame] = SigC.Check_This('Ecby',Ecby[IFrame])
			#Ecbz[IFrame] = SigC.Check_This('Ecbz',Ecbz[IFrame])			
		wx = [1.0*mEuI[0]*(T-IG)/(TgI[-1]-IG) for T in t]		
		wy = [1.0*mEuI[1]*(T-IG)/(TgI[-1]-IG) for T in t]	
		wz = [1.0*mEuI[2]*(T-IG)/(TgI[-1]-IG) for T in t]	
		plt.figure(i)	
		plt.subplot(311)
		plt.plot(Eu['Frame Index'],Eu[Sx].values,'b')
		plt.plot(Eu['Frame Index'],CEu[Sx].values,'-r')
		plt.plot(t,wx,'*g')
		plt.subplot(312)
		plt.plot(Eu['Frame Index'],Ey,'b')
		plt.plot(Eu['Frame Index'],CEu[Sy].values,'-r')
		plt.plot(t,wy,'*g')
		plt.subplot(313)
		plt.plot(Eu['Frame Index'],Ez,'b')
		plt.plot(Eu['Frame Index'],CEu[Sz].values,'-r')
		plt.plot(t,wz,'*g')
		plt.figure(0)		
		plt.subplot(311)
		plt.plot(Eu['Frame Index'],Ex,'b')				
		plt.plot(Eu['Frame Index'],CEu[Sx].values,'-r')
		plt.subplot(312)
		plt.plot(Eu['Frame Index'],Ey,'b')				
		plt.plot(Eu['Frame Index'],CEu[Sy].values,'-r')
		plt.subplot(313)
		plt.plot(Eu['Frame Index'],Ez,'b')				
		plt.plot(Eu['Frame Index'],CEu[Sz].values,'-r')		
	plt.show()
	if raw_input("Do you want to keep this Tpose ?  (y / n)  :") == 'y':
		Tp["B"].append(I)
		return CEu.copy(deep=True)	
	else:
		return Eu
	return CumEu
def AE_GrFeature_Computation(tp,dfEu):
	LF = list(dfEu); Lf  = len(LF);
	RawDataTpG = []; tpg = tp.PG  ; 
	Np = len(tpg)  ; MeanTimeG = 0; 
	umean = []     ;  
	for itp in range(Np):		
		u  = dfEu[dfEu['Frame Index'].values == tpg[itp]['Frame Index']]		
		v  = [u[LF[c]].values for c in range(Lf)]
		RawDataTpG.append(v)
		if  itp==0: 
			for l in range(Lf):
				umean.append(v[l])
		elif itp==Np-1:
			for l in range(Lf):
				umean[l] += v[l] 
				umean[l]  = umean[l]/Np
		else:
			for l in range(Lf):
				umean[l] += v[l]
		MeanTimeG+=tpg[itp]['Frame Index']
	MeanTimeG = umean[0]; 
	GroupsMe  = umean[1:]	
	return MeanTimeG,RawDataTpG,GroupsMe
def Step_Between_Tpose(IFGr,RawDGr,StatGr):
	diF_Bet_Tp = {}
	for ig in range(len(StatGr)-1):		
		for ih in range(ig+1,len(StatGr)):
			dSt = str(ig)+str(ih)+'d'
			dtS = str(ih)+str(ig)+'d'
			TSt = str(ig)+str(ih)+'t'
			TtS = str(ih)+str(ig)+'t'				
			dh  = np.abs(np.array(StatGr[ig][0]) - np.array(StatGr[ih][0]))
			th  = np.abs(IFGr[ig] - IFGr[ih])
			diF_Bet_Tp.update({dSt:dh})
			diF_Bet_Tp.update({dtS:dh})
			diF_Bet_Tp.update({TSt:th})
			diF_Bet_Tp.update({TtS:th})
	d = diF_Bet_Tp
	f = plt.figure()	
	ax1 = f.add_subplot(221)	
	ax2 = f.add_subplot(222)
	ax3 = f.add_subplot(223)
	for ig in range(len(StatGr)-1):		
		for ih in range(ig+1,len(StatGr)):
			dSt = str(ig)+str(ih)+'d'
			dtS = str(ih)+str(ig)+'d'
			TSt = str(ig)+str(ih)+'t'
			TtS = str(ih)+str(ig)+'t'
			for i in range(1,9):
				ax1.plot(d[TSt],d[dSt][4*i+0],'.')
				ax2.plot(d[TSt],d[dSt][4*i+1],'.')			
				ax3.plot(d[TSt],d[dSt][4*i+2],'.')
				plt.show(f)
	return d
#####################################

###Clustering functions###
def CandidatTposeSelection(nw,df,rs,ra,Moybdb,MoyTpd):
		T = []; L = list(df); l = len(L)		
		iA= 0 ;	iS= 0       ; I = []; Ione=[]; Izero=[];
		I.append(0)
		for i in range(1,len(df['TPose Value'])-1):
			T = df.iat[i,0]
			I.append(0)	
			while  T>rs[iS].xc+rs[iS].dx/2.0 and iS<len(rs)-1:iS=iS+1
			while  T>ra[iA].xc+ra[iA].dx/2.0 and iA<len(ra)-1:iA=iA+1				
			xcLs =  rs[iS].xc - rs[iS].dx/2.0;	xcRs =  rs[iS].xc + rs[iS].dx/2.0;
			ycDs =  rs[iS].yc - rs[iS].dy/2.0;	ycHs =  rs[iS].yc + rs[iS].dy/2.0;
			xcLa =  ra[iA].xc - ra[iA].dx/2.0;	xcRa =  ra[iA].xc + ra[iA].dx/2.0;
			ycDa =  ra[iA].yc - ra[iA].dy/2.0;  ycHa =  ra[iA].yc + ra[iA].dy/2.0;
			#####################################################################
			if ((xcLs < T and T < xcRs) and (ycDs < Moybdb[i-1] and Moybdb[i-1] < ycHs)) and \
			   ((xcLa < T and T < xcRa) and (ycDa < MoyTpd[i-1] and MoyTpd[i-1] < ycHa)):
				I[i]=1;Ione.append(i);
			else: 
				Izero.append(i);
			#####################################################################
			df['TPose Value'].values[i] = I[i]
		for ic in range(3,l):
			ax1 = plt.subplot(311)
			plt.title("Final Tposes selection")
			plt.plot(df[L[2]],df[L[ic]])
			plt.ylabel("Angle distance(degrees)")
		ax2 = plt.subplot(312,sharex=ax1)
		plt.plot([df.iat[i,2] for i in  Ione[0:-2]],[MoyTpd[i-1] for i in  Ione[0:-2]],'.r')
		plt.plot([df.iat[i,2] for i in Izero[0:-2]],[MoyTpd[i-1] for i in Izero[0:-2]],'.b')
		plt.ylabel("Angle distance index")
		ax3 = plt.subplot(313,sharex=ax1)		
		plt.plot([df.iat[i,2] for i in  Ione[0:-2]],[Moybdb[i-1] for i in  Ione[0:-2]],'.r')
		plt.plot([df.iat[i,2] for i in Izero[0:-2]],[Moybdb[i-1] for i in Izero[0:-2]],'.b')
		plt.ylabel('Stationnarity index')
		plt.xlabel('time')
		plt.show()
		moddf = df
		writer = pd.ExcelWriter(nw,engine='xlsxwriter')
		moddf.to_excel(writer,sheet_name='Sheet1')
		writer.save()
		return moddf
def TposeGroupsFormation(moddf,Lneighborhood):
		Lengh            = len(moddf['TPose Value'])-1
		Labelgroup       = 0
		LGroup           = [TposeGroup(Labelgroup)]
		TpA              = False	
		for i in range(len(moddf['TPose Value'])):				
			if moddf['TPose Value'].values[i] == 1:
				LGroup[Labelgroup].add(i,moddf)
				TpA=True
			else:
				if TpA == True:
					TpA = TposeAround(i,Lengh,Lneighborhood,moddf)
					if TpA == False: 					
						Labelgroup+=1
						LGroup.append(TposeGroup(Labelgroup))	
		lg = []
		for g in range(len(LGroup)):
			if LGroup[g].NTpp()!=0:
				lg.append(LGroup[g])
		LGroup = lg	
		return LGroup
def RawDataSignal(RDdf):
	T      = []
	Thetax = []
	Thetay = [] 
	Thetaz = [] 
	for i in range(0,len(RDdf['Frame Index'])): T.append(RDdf['Frame Index'].values[i])	
	for sensor in range(0,10):
		Thetax.append([])
		Thetay.append([])
		Thetaz.append([])				
		sx = str(sensor) + 'x'
		sy = str(sensor) + 'y'
		sz = str(sensor) + 'z'
		for i in range(0,len(RDdf[sx])): 
			Thetax[sensor].append(RDdf[sx].values[i])	
			Thetay[sensor].append(RDdf[sy].values[i])
			Thetaz[sensor].append(RDdf[sz].values[i])	
	return T,Thetax,Thetay,Thetaz
##########################

###Reading,Extracting and Saving functions###
def PDreadEx(Location):	
	df = pd.read_excel(Location)
	df = MODdf(df)	
	return df
def RDreadEx(Location):	
	return pd.read_excel(Location)
def MODdf(df):
	for e1 in Exceptions1:df[e1] = df[e1]-df[e1]
	for e2 in Exceptions2:df[e2] = -df[e2]	
	return df
def ReadTableEx(Location, DataSheet):
	return pd.read_excel(Location, sheetname = DataSheet) 		
def ExcelReadandSave(nr,nw):
	df = pd.read_excel(nr)
	df = MODdf(df)	
	writer = pd.ExcelWriter(nw, engine='xlsxwriter')
	df.to_excel(writer, sheet_name='Sheet1')
	writer.save()
def SaveDf(df,nf):
	writer = pd.ExcelWriter(nf, engine='xlsxwriter')
	df.to_excel(writer, sheet_name='Sheet1')
	writer.save()
def QuatRawToExcel(nr,nw):
	df = pd.read_excel(nr)			
	for IFrame in range(0,len(df['Frame Index'])):
		for i in range(1,10):
			Sx = str(i)+'x';Sy = str(i)+'y';Sz = str(i)+'z';Sw = str(i)+'w';	
			Q  = EulerToQuat(df[Sx].values[IFrame],df[Sy].values[IFrame],df[Sz].values[IFrame])				
			df.iat[IFrame,4*i+3]     = Q[0]	
			df.iat[IFrame,4*i+0]     = Q[1]
			df.iat[IFrame,4*i+1]     = Q[2]
			df.iat[IFrame,4*i+2]     = Q[3]
	writer = pd.ExcelWriter(nw, engine='xlsxwriter')
	df.to_excel(writer)
	writer.save()
	return df
def ExtractD(df,StL):
	#Extract the data of a dataframe structure associated to an excel file of joint angles 
	#And divide the data in Tpose event or non-Tpose
	#Typical joint angle for an ideal Tpose
	Tpose  = []; NTpose = []; tpose = []; ntpose = [];	
	L = list(df); Le = len(df[L[0]]);LL = len(L);
	#Formation of the container for the data
	for c in L:
		if not(c in Exceptions3):
			NT.append([])
			T.append([])
	#Extraction and separation of the data into the two class of event
	for i in range(0,Le):
		if(i>=StL):	
			#Non-Tpose
			if df['TPose Value'][i] == 0:
				#time
				nt.append(df['Frame Index'][i])
				#angle
				for c in range(3,LL):
					Col = L[c]								
					NT[c-3].append(df[Col][i])	
			#Tpose
			if df['TPose Value'][i] == 1 :
				#time
				t.append(df['Frame Index'][i])
				#angle
				for c in range(3,LL):
					Col = L[c]
					T[c-3].append(df[Col][i])
	return NT,nt,T,t
def App_Formated_CSV_Writer(df,name):
	import csv
	csvFile   = open(name, 'w')
	csvWriter = csv.writer(csvFile, delimiter=',',lineterminator='\n')	
	L = list(df)
	nl = len(df[L[0]].values)
	for i in range(nl):
		if i==0:
			csvWriter.writerow(['3c271eea-d4b7-4464-8c95-a0876e428c75'])	
		elif i==1:
			csvWriter.writerow(['654318e5-1701-4c46-bc0e-203805951a2c'])		
		elif i==2:	
			csvWriter.writerow(['d5052541-36df-4dc6-9c4f-63e76b3dea38'])				
		else:
			Line = []
			K    = 0
			Line.append(str(i*0.033333))
			for j in range(1,10):
				Line.append(str(j))
				strx = str(j)+'x';stry = str(j)+'y';strz = str(j)+'z'
				l = str(df[strx].values[i])+';'+str(df[stry].values[i])+';'+str(df[strz].values[i])
				Line.append(l)
			l = Line.append('10')
			Line.append('0.0'+';'+'0.0'+';'+'0.0'+';')
			Line.append('1234')	
			Line.append('BBBB')
			Line.append('CCCC')
			Line.append('DDDD')
			Line.append('EEEE')
			csvWriter.writerow(Line)
	csvFile.close()
#############################################

###Helping functions#########################
def Rx(t):
	r11 = 1.0; r12 =        0.0; r13 =         0.0;
	r21 = 0.0; r22 =  np.cos(t); r23 =  -np.sin(t);
	r31 = 0.0; r32 =  np.sin(t); r33 =   np.cos(t);
	rx  = np.array([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])
	print 'Rx :',rx
	return rx
def Ry(t):
	r11 =  np.cos(t); r12 = 0.0; r13 = np.sin(t);
	r21 =        0.0; r22 = 1.0; r23 =       0.0;
	r31 = -np.sin(t); r32 = 0.0; r33 = np.cos(t);
	ry  = np.array([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])
	print 'Ry :',ry
	return ry
def Rz(t):
	r11 = np.cos(t); r12 = -np.sin(t); r13 = 0.0;
	r21 = np.sin(t); r22 =  np.cos(t); r23 = 0.0;
	r31 =       0.0; r32 =        0.0; r33 = 1.0;
	rz  = np.array([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])
	print 'Rz :',rz
	return rz
def EulerToQuat(Th1,Th2,Th3):
	"""
	Compostion of the three elementaries rotation
	around three axes 'here 3,2,1 meaning zyx'
	"""
	q1 = [np.cos(0.5*Th1),np.sin(0.5*Th1),0.0,0.0]	
	nq1 = np.sqrt(np.dot(q1,q1))
	q1  = [q1[i]/nq1 for i in range(len(q1))]	
	q2 = [np.cos(0.5*Th2),0.0,np.sin(0.5*Th2),0.0]	
	nq2 = np.sqrt(np.dot(q2,q2))
	q2  = [q2[i]/nq2 for i in range(len(q2))]	
	q3 = [np.cos(0.5*Th3),0.0,0.0,np.sin(0.5*Th3)]	
	nq3 = np.sqrt(np.dot(q3,q3))
	q3  = [q3[i]/nq3 for i in range(len(q3))]	
	qt = QuatDotProd(q1,q2)	
	q  = QuatDotProd(qt,q3)	
	nq = np.sqrt(np.dot(q,q))
	q  = [q[i]/nq for i in range(len(q))]	
	return q
def QuatDotProd(q1,q2):
	q3 = [0.0,0.0,0.0,0.0] 
	q3[0] = q1[0]*q2[0] - (q1[1]*q2[1]+q1[2]*q2[2]+q1[3]*q2[3])
	r = np.cross(q1[1:],q2[1:])
	s = [q1[0]*q2[1],q1[0]*q2[2],q1[0]*q2[3]]
	t = [q2[0]*q1[1],q2[0]*q1[2],q2[0]*q1[3]]
	q3[1] = r[0]+s[0]+t[0]
	q3[2] = r[1]+s[1]+t[1]
	q3[3] = r[2]+s[2]+t[2]
	return q3
def ConjQuat(q):
	return [q[0],-q[1],-q[2],-q[3]]
def NormQuat(q):
	n = np.sqrt(q[0]**2  + q[1]**2  + q[2]**2 + q[3]**2)
	if n==0: q = [1,0,0,0]; n = 1.0;	
	return [q[0]/n , q[1]/n, q[2]/n, q[3]/n]
def QuatRes(q0,q):
	q1 = [q[0],q[1],0.0,0.0]	
	q2 = [q[0],0.0,q[2],0.0]	
	q3 = [q[0],0.0,0.0,q[3]]
	invq1 = [q0[0],-q0[1],0.0,0.0]	
	invq2 = [q0[0],0.0,-q0[2],0.0]	
	invq3 = [q0[0],0.0,0.0,-q0[3]]	
	qreset1 = QuatDotProd(invq1,q1) 
	qreset2 = QuatDotProd(invq2,q2) 
	qreset3 = QuatDotProd(invq3,q3) 
	qresetT = QuatDotProd(qreset1,qreset2) 
	qreset  = QuatDotProd(qresetT,qreset3) 
	nq      = np.sqrt(qreset[0]**2 + qreset[1]**2 + qreset[2]**2 +qreset[3]**2) 
	qreset  = [qreset[i]/nq for i in range(len(qreset))]	
	return qreset
def QuatInv(q):
	n = np.sqrt(q[0]**2  + q[1]**2  + q[2]**2 + q[3]**2)
	qc= ConjQuat(q)	
	return [qc[0]/n , qc[1]/n, qc[2]/n, qc[3]/n]
def QuatToEuler(q,order):
	if order == 'zyx':
		I1 =  2*(q[0]*q[1] + q[3]*q[2])
		I2 =  q[3]*q[3] + q[0]*q[0] - q[1]*q[1] - q[2]*q[2]
		I3 = -2*(q[0]*q[2] - q[3]*q[1])
		I4 =  2*(q[1]*q[2] + q[3]*q[0])
		I5 = q[3]*q[3] - q[0]*q[0] - q[1]*q[1] + q[2]*q[2]
		E1,E2,E3 = Q2E(I1,I2,I3,I4,I5)
	elif order == 'zxy':
		I1 = -2*(q[0]*q[1] - q[3]*q[2])
		I2 = q[3]*q[3] - q[0]*q[0] + q[1]*q[1] - q[2]*q[2]
		I3 =  2*(q[1]*q[2] + q[3]*q[0])
		I4 = -2*(q[0]*q[2] - q[3]*q[1])
		I5 = q[3]*q[3] - q[0]*q[0] - q[1]*q[1] + q[2]*q[2]
		E1,E2,E3 = Q2E(I1,I2,I3,I4,I5)
	elif order == 'yxz':
		I1 = 2*(q[0]*q[2] + q[3]*q[1])
		I2 = q[3]*q[3] - q[0]*q[0] - q[1]*q[1] + q[2]*q[2]
		I3 = -2*(q[1]*q[2] - q[3]*q[0])
		I4 = 2*(q[0]*q[1] + q[3]*q[2])
		I5 = q[3]*q[3] - q[0]*q[0] + q[1]*q[1] - q[2]*q[2]
		E1,E2,E3 = Q2E(I1,I2,I3,I4,I5)
	elif order == 'yzx':
		I1 = -2*(q[0]*q[2] - q[3]*q[1])
		I2 = q[3]*q[3] + q[0]*q[0] - q[1]*q[1] - q[2]*q[2]
		I3 = 2*(q[0]*q[1] + q[3]*q[2])
		I4 = -2*(q[1]*q[2] - q[3]*q[0])
		I5 = q[3]*q[3] - q[0]*q[0] + q[1]*q[1] - q[2]*q[2]
		E1,E2,E3 = Q2E(I1,I2,I3,I4,I5)
	elif order == 'xyz':
		I1 = -2*(q[1]*q[2] - q[3]*q[0])
		I2 = q[3]*q[3] - q[0]*q[0] - q[1]*q[1] + q[2]*q[2]
		I3 = 2*(q[0]*q[2] + q[3]*q[1])
		I4 = -2*(q[0]*q[1] - q[3]*q[2])
		I5 = q[3]*q[3] + q[0]*q[0] - q[1]*q[1] - q[2]*q[2]
		E1,E2,E3 = Q2E(I1,I2,I3,I4,I5)
	elif order == 'xzy':
		I1 = 2*(q[1]*q[2] + q[3]*q[0])
		I2 = q[3]*q[3] - q[0]*q[0] + q[1]*q[1] - q[2]*q[2]
		I3 = -2*(q[0]*q[1] - q[3]*q[2])
		I4 = 2*(q[0]*q[2] + q[3]*q[1])
		I5 = q[3]*q[3] + q[0]*q[0] - q[1]*q[1] - q[2]*q[2]
		E1,E2,E3 = Q2E(I1,I2,I3,I4,I5)	
	return E1,E2,E3
def Q2E(r11,r12,r21,r31,r32):	
	E1 = ATAN(r31,r32) 
	E2 = np.arcsin(r21)
	E3 = ATAN(r11,r12)
	return E1,E2,E3
def TposeAround(i,Lengh,neighborhood,moddf):
		if   (i < neighborhood):
			su  = np.sum(moddf['TPose Value'][0:2*neighborhood])
			TpA = (su>2)
		elif (i > Lengh - neighborhood):
			su  = np.sum(moddf['TPose Value'][Lengh - 2*neighborhood:Lengh])
			TpA = (su>2)
		else :
			su = np.sum(moddf['TPose Value'][i-neighborhood:i+neighborhood])
			TpA = (su>2)
		return TpA
def RawToQuat(RDdfEul):
	df = RDdfEul.copy(deep=True)
	for IFrame in range(0,len(df['Frame Index'])):
		for i in range(1,10):
			Sx = str(i)+'x';Sy = str(i)+'y';Sz = str(i)+'z';Sw = str(i)+'w';	
			Q  = EulerToQuat(df[Sx].values[IFrame],df[Sy].values[IFrame],df[Sz].values[IFrame])	
			df[Sw].values[IFrame]= Q[0]
			df[Sx].values[IFrame]= Q[1]
			df[Sy].values[IFrame]= Q[2]
			df[Sz].values[IFrame]= Q[3]			
	return df
def ATAN(Y,X,Test=False):
	x=X
	y=Y
	"""
	Homemain Arctan function.	
	"""	
	import math
	import numpy as np
	if Test == False: 
		theta = np.arctan2(y,x)	
	elif   np.abs(x)==0:
		theta = np.pi/2.
	elif np.abs(x)   >0:
		theta = math.atan(np.abs(y/x))	
		if   x>0 and y>0: theta = theta	           #first  part of the circle up and right
		elif x>0 and y<0: theta = 2*np.pi - theta  #second part of the circle up and left
		elif x<0 and y>0: theta =   np.pi - theta  #third  part of the circle down and left
		elif x<0 and y<0: theta =   np.pi + theta  #forth  part of the circle down and right
	return theta
def Tpose_Ref(Eu_ref,Op):
	Eu = []
	if Op == 'Init_Ref':
		for i in range(1,10): 
			Sx = str(i)+'x';
			Sy = str(i)+'y';
			Sz = str(i)+'z';
			Eu.append([Eu_ref[Sx].values[0],Eu_ref[Sy].values[0],Eu_ref[Sz].values[0]])
	#elif Op == 'Statistical_Ref':
	return Eu
#############################################

#########Filtering functions#################
def kmeanClustering(df):	
	from sklearn.cluster import KMeans	
	import matplotlib.pyplot as plt
	import pandas as pd
	import numpy  as np
	Nl = len(df)
	for i in range(1,10): 
		Sx  = str(i)+'x'; Eux = df[Sx].values[1:]; Dx  = np.abs(df[Sx].diff().values[1:]);	dx  = [];	
		Sy  = str(i)+'y'; Euy = df[Sy].values[1:]; Dy  = np.abs(df[Sy].diff().values[1:]); dy  = [];
		Sz  = str(i)+'z'; Euz = df[Sz].values[1:]; Dz  = np.abs(df[Sz].diff().values[1:]); dz  = [];		
		G  = []; dEu = [];	
		for j in range(Nl-1):		
			dx.append([Dx.values[j]]);dy.append([Dy.values[j]]);dz.append([Dz.values[j]]);
			G.append([Dx.values[j],Dy.values[j],Dz.values[j]])
		YG = KMeans(n_clusters=3).fit_predict(G)
		dG = np.array(G)
		YmG0= np.mean(dG[YG==0]);YmG1= np.mean(dG[YG==1]);YmG2= np.mean(dG[YG==2]);
		cGmx = 2; cGmm = 1; cGmn = 0;
		if YmG2 > YmG1 and YmG2 > YmG0 : cGmx = 2; cGmm = 1; cGmn = 0;
		if YmG1 > YmG2 and YmG1 > YmG0 : cGmx = 1; cGmm = 0; cGmn = 2;
		if YmG0 > YmG2 and YmG0 > YmG1 : cGmx = 0; cGmm = 1; cGmn = 2;	
		it = np.array(range(len(YG)))	
		#for IT in it:df['TPose Value'].values[IT] = 13		
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
	return df,it
#############################################
	
