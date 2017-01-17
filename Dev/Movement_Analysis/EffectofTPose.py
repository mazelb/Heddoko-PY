from AM_Functions import * 

a = ['\DAN0','\DAN0sec','\DAN1','\DAN2','\DAN123court']
for i in range(4,5):
	Location0      = r'C:\Users\Simon\Desktop'+ a[0] +'.xlsx'
	Location1      = r'C:\Users\Simon\Desktop'+ a[i] +'.xlsx'
	Df0  = ReadTableEx(Location0, a[0][1:])
	Df1  = ReadTableEx(Location1, a[i][1:])	
	Exceptions = ['Frame Index','TPose Value','Timestamp']
	L    = list(Df0)
	l    = len(L)	
	ti0 = Df0.iat[0,0]
	ti1 = Df1.iat[0,0]
	tf0 = Df0.iat[-1,0]
	tf1 = Df1.iat[-1,0]	
	timax = np.max([ti0,ti1])
	tfmin = np.min([tf0,tf1])
	df0 = Df0[Df0[L[0]] >= timax]
	df1 = Df1[Df1[L[0]] >= timax]
	df0 = df0[df0[L[0]] <= tfmin]
	df1 = df1[df1[L[0]] <= tfmin]	
	M0   = len(df0[L[0]])
	M1   = len(df1[L[0]]) 
	
	j0MinDiff = []
	j1MinDiff = []
	T01 = []
	D01 = []
	
	if M0<M1:	
		for T in range(M0): 
			T01.append(df0.iat[T,0])
			mindif = 10**6
			if T<3:
				for S in range(M1): 		
					d = np.abs(df1.iat[S,0]-df0.iat[T,0])
					if d<mindif: 
						mindif = d
						jD     = S
				j0MinDiff.append(T)			
				j1MinDiff.append(jD)
			else:
				S = j0MinDiff[T-1]+1
				dim=0
				dip=0
				while S<M1-1 and (dim>0 or dip<0):				
					dm = np.abs(df1.iat[S-1,0]-df0.iat[T,0])
					d  = np.abs(df1.iat[S  ,0]-df0.iat[T,0])
					dp = np.abs(df1.iat[S+1,0]-df0.iat[T,0])
					dim = d  - dm
					dip = dp - d
					S+=1
				j0MinDiff.append(T)
				j1MinDiff.append(S)
	elif M0>M1:
		for T in range(M1): 
			T01.append(df1.iat[T,0])
			mindif = 10**6
			if T<3:
				for S in range(M0): 		
					d = np.abs(df0.iat[S,0]-df1.iat[T,0])
					if d<mindif: 
						mindif = d
						jD     = S
				j1MinDiff.append(T)
				j0MinDiff.append(jD)
			else:
				S = j1MinDiff[T-1]+1
				dim=0
				dip=0
				while S<M0-1 and (dim>0 or dip<0):				
					dm = np.abs(df0.iat[S-1,0]-df1.iat[T,0])
					d  = np.abs(df0.iat[S  ,0]-df1.iat[T,0])
					dp = np.abs(df0.iat[S+1,0]-df1.iat[T,0])
					dim = d  - dm
					dip = dp - d
					S+=1
				j1MinDiff.append(T)
				j0MinDiff.append(S)
	plt.figure()
	for (c,i) in zip(L,range(len(L))):
		if not(c in Exceptions):
			D01 = []
			for H in range(len(j0MinDiff)):			
				D01.append(np.abs(df0.iat[j0MinDiff[H],i]-df1.iat[j1MinDiff[H],i]))		
			ax0 = plt.subplot(211)
			plt.plot(df0[L[0]],df0[c],'.r')
			plt.plot(df1[L[0]],df1[c],'b')
			plt.subplot(212,sharex=ax0)
			plt.plot(T01,D01,'.')	
	plt.show()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	