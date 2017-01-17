class SignalCorrector:
	def __init__(self,DF,MinThres,Opt):	
		 from sklearn.cluster import KMeans	
		 import matplotlib.pyplot as plt
		 import pandas as pd
		 import numpy as np	
		 self.Nl    = len(DF)		
		 self.Istep = []
		 self.df    = DF.copy(deep=True) 
		 self.dff   = DF.copy(deep=True)
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
			 cGmx = 2; cGmm = 1; cGmn = 0;
			 if YmG2 > YmG1 and YmG2 > YmG0 : cGmx = 2; cGmm = 1; cGmn = 0;
			 if YmG1 > YmG2 and YmG1 > YmG0 : cGmx = 1; cGmm = 0; cGmn = 2;
			 if YmG0 > YmG2 and YmG0 > YmG1 : cGmx = 0; cGmm = 1; cGmn = 2;	
			 Threshold = max([YmG0,YmG1,YmG2,MinThres])	
			 print 'Th:',Threshold, 'Thmin:',MinThres		 		
			 ########################
			 self.last_key = ''			 
			 self.subplot_remove = set([])			 
			 self.fig, self.ax = plt.subplots(3,2)
			 self.ax[0][0].plot(self.df['Frame Index'].values,self.df[Sx].values)
			 self.ax[1][0].plot(self.df['Frame Index'].values,self.df[Sy].values)
			 self.ax[2][0].plot(self.df['Frame Index'].values,self.df[Sz].values)
			 ################			 
			 self.ax[0][1].plot(adx[YG==cGmx],ady[YG==cGmx],'*r',markersize=12)
			 self.ax[0][1].plot(adx[YG==cGmm],ady[YG==cGmm],'.g')
			 self.ax[0][1].plot(adx[YG==cGmn],ady[YG==cGmn],'.y')			 
			 self.ax[1][1].plot(adx[YG==cGmx],adz[YG==cGmx],'*r',markersize=12)
			 self.ax[1][1].plot(adx[YG==cGmm],adz[YG==cGmm],'.g')
			 self.ax[1][1].plot(adx[YG==cGmn],adz[YG==cGmn],'.y')
			 self.ax[2][1].plot(ady[YG==cGmx],adz[YG==cGmx],'*r',markersize=12)
			 self.ax[2][1].plot(ady[YG==cGmm],adz[YG==cGmm],'.g')
			 self.ax[2][1].plot(ady[YG==cGmn],adz[YG==cGmn],'.y')
			 ################
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
				 if (Iac > 0 or Iac < 0): 
					 self.ax[0][0].plot(self.df['Frame Index'].values[Ia],self.df[Sx].values[Ia],'*g')				 
				 if  (Iac > 0) and (Inx < 0):
					 meanStep = (sdx[Ia-1]-sdx[In-1])/2.0
					 self.ax[0][0].plot(self.df['Frame Index'].values[Ia:In],self.df[Sx].values[Ia:In],'.r')						
					 self.df[Sx].values[Ia:In]-= meanStep
					 self.Istep[i-1][0][Ii]    = 0
					 self.Istep[i-1][0][Ii+1]  = 0
				 elif (Iac < 0) and (Inx > 0):
					 meanStep = (sdx[Ia-1]-sdx[In-1])/2.0
					 self.df[Sx].values[Ia:In]-= meanStep
					 self.ax[0][0].plot(self.df['Frame Index'].values[Ia:In],self.df[Sx].values[Ia:In],'.r')
					 self.Istep[i-1][0][Ii]    = 0
					 self.Istep[i-1][0][Ii+1]  = 0
			 for Ij in range(Ny-1):				 
				Iac = self.Istep[i-1][1][Ij]; Iny = self.Istep[i-1][1][Ij+1];
				Ia  = np.abs(Iac); 		   In  = np.abs(Iny);
				if (Iac > 0 or Iac < 0):
					self.ax[1][0].plot(self.df['Frame Index'].values[Ia],self.df[Sy].values[Ia],'*g')				 
				if  (Iac > 0) and (Iny < 0):
					 meanStep = (sdy[Ia-1]-sdy[In-1])/2.0						 
					 self.df[Sy].values[Ia:In]-= meanStep
					 self.ax[1][0].plot(self.df['Frame Index'].values[Ia:In],self.df[Sy].values[Ia:In],'.r')
					 self.Istep[i-1][1][Ij]    = 0
					 self.Istep[i-1][1][Ij+1]  = 0
				elif (Iac < 0) and (Iny > 0):
					 meanStep = (sdy[Ia-1]-sdy[In-1])/2.0								 
					 self.df[Sy].values[Ia:In]-= meanStep						 
					 self.ax[1][0].plot(self.df['Frame Index'].values[Ia:In],self.df[Sy].values[Ia:In],'.r')
					 self.Istep[i-1][1][Ij]   = 0
					 self.Istep[i-1][1][Ij+1] = 0
			 for Ik in range(Nz-1):		 
				 Iac = self.Istep[i-1][2][Ik]; Inz = self.Istep[i-1][2][Ik+1];
				 Ia  = np.abs(Iac); 		   In  = np.abs(Inz);	
				 if (Iac > 0 or Iac < 0):
					 self.ax[2][0].plot(self.df['Frame Index'].values[Ia],self.df[Sz].values[Ia],'*g')					 
				 if  (Iac > 0) and (Inz < 0):
					 meanStep = (sdz[Ia-1]-sdz[In-1])/2.0
					 self.df[Sz].values[Ia:In]-= meanStep
					 self.ax[2][0].plot(self.df['Frame Index'].values[Ia:In],self.df[Sz].values[Ia:In],'.r')
					 self.Istep[i-1][2][Ik]   = 0
					 self.Istep[i-1][2][Ik+1] = 0
				 elif (Iac < 0) and (Inz > 0):
					 meanStep = (sdz[Ia-1]-sdz[In-1])/2.0
					 self.df[Sz].values[Ia:In]-= meanStep
					 self.ax[2][0].plot(self.df['Frame Index'].values[Ia:In],self.df[Sz].values[Ia:In],'.r')
					 self.Istep[i-1][2][Ik]   = 0
					 self.Istep[i-1][2][Ik+1] = 0
			 plt.show()
			 pr = self.subplot_remove
			 if len(pr)!=0: 
				 for R in range(3):
					if not(R in pr): 
						if   R == 0: S = str(i)+'x'  
						elif R == 1: S = str(i)+'y'
						elif R == 2: S = str(i)+'z'
						print S, self.dff[S] 
						self.dff[S] = self.df[S]
	def Press_key(self,event):
		self.last_key = event.key
	def Click_mouse(self,event):
		if self.last_key == 'c':
			if self.ax[0][0] == event.inaxes: 
				self.subplot_remove.add(0)
			if self.ax[1][0] == event.inaxes:
				self.subplot_remove.add(1)
			if self.ax[2][0] == event.inaxes:
				self.subplot_remove.add(2)
		if self.last_key == 'a':
			if self.ax[0][0] == event.inaxes and not(self.subplot_remove.isdisjoint(set([0]))): 
				self.subplot_remove.remove(0) 
			if self.ax[1][0] == event.inaxes and not(self.subplot_remove.isdisjoint(set([1]))):
				self.subplot_remove.remove(1)  
			if self.ax[2][0] == event.inaxes and not(self.subplot_remove.isdisjoint(set([2]))):
				self.subplot_remove.remove(2)  
		print self.subplot_remove
	def SON(self,t,T): 
		import numpy as np
		s = 0
		if(t >  np.abs(T)):
			s =  1
		if(t < -np.abs(T)):
			s = -1
		return s				
	def Get_Cleaned_Signal(self):
		return self.dff

def kmeanClustering(df,Thmin):	
	from sklearn.cluster import KMeans	
	import matplotlib.pyplot as plt
	import pandas as pd
	import numpy  as np
	Nl = len(df)
	for i in range(1,10): 
		Sx  = str(i)+'x'; Eux = df[Sx].values[1:]; Dx  = np.abs(df[Sx].diff().values[1:]); dx  = [];	
		Sy  = str(i)+'y'; Euy = df[Sy].values[1:]; Dy  = np.abs(df[Sy].diff().values[1:]); dy  = [];
		Sz  = str(i)+'z'; Euz = df[Sz].values[1:]; Dz  = np.abs(df[Sz].diff().values[1:]); dz  = [];
		G   = []; dEu = [];	
		for j in range(Nl-1):		
			dx.append([Dx[j]]);
			dy.append([Dy[j]]);
			dz.append([Dz[j]]);
			G.append([Dx[j],Dy[j],Dz[j]])
		YG = KMeans(n_clusters=3).fit_predict(G)
		dG = np.array(G)
		YmG0= np.mean(dG[YG==0]);YmG1= np.mean(dG[YG==1]);YmG2= np.mean(dG[YG==2]);
		cGmx = 2; cGmm = 1; cGmn = 0;
		if YmG2 > YmG1 and YmG2 > YmG0 : cGmx = 2; cGmm = 1; cGmn = 0;
		if YmG1 > YmG2 and YmG1 > YmG0 : cGmx = 1; cGmm = 0; cGmn = 2;
		if YmG0 > YmG2 and YmG0 > YmG1 : cGmx = 0; cGmm = 1; cGmn = 2;	
		it = np.array(range(len(YG)))	
		Threshold = max([YmG0,YmG1,YmG2,Thmin])		
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
	return df,Threshold
	
