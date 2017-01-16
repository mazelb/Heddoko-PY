from sklearn.neural_network import MLPClassifier
import Interacting_Tool as Ints
import MPL_Tool_Box as mpltb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#name   =  '.\MT3RawData.csv'
name   =  '.\MT8RawData.csv'
df     = pd.read_csv(name)
FrIndex= df['Frame Index'].values
Ncol   = list(df)
Nlin   = len(df)

frac = 2./3.;
Q    = [];
DQ   = [];
Xpop = [];
Ys   = [];
Yns  = [];

for s in range(Nlin):
	Q.append([])
	q = []
	for i in range(1,10):
		Sx = str(i)+'x'
		Sy = str(i)+'y'
		Sz = str(i)+'z'	
		dx = df[Sx].values[s]; dy = df[Sy].values[s]; dz = df[Sz].values[s]
		if   s==0      : 
			ddxm1= 0.0; ddxp1= df[Sx].values[s+1]; 
			ddym1= 0.0; ddyp1= df[Sy].values[s+1]; 
			ddzm1= 0.0; ddzp1= df[Sz].values[s+1]; 
		elif s==Nlin-1 : 
			ddxm1= df[Sx].values[s-1]; ddxp1= 0.0;
			ddym1= df[Sy].values[s-1]; ddyp1= 0.0;
			ddzm1= df[Sz].values[s-1]; ddzp1= 0.0;			
		else: 
			ddxm1= df[Sx].values[s-1]; ddxp1= df[Sx].values[s+1];
			ddym1= df[Sy].values[s-1]; ddyp1= df[Sy].values[s+1];
			ddzm1= df[Sz].values[s-1]; ddzp1= df[Sz].values[s+1];			
		ddx = (ddxp1 - ddxm1)/2.0 ; ddy = (ddyp1 - ddym1)/2.0; ddz = (ddzp1 - ddzm1)/2.0;
		Q[s].append(dx);Q[s].append(ddx);Q[s].append(dy);Q[s].append(ddy);Q[s].append(dz);Q[s].append(ddz);	
Xpop = Q
Interface = Ints.Interacting_Plot(Q)
plt.show()

iGood  = Interface.Stat_I_List[0]
iWrong = Interface.NStat_I_List[0]

Xtr, ysoltr, Xva, ysolva = mpltb.Train_Example(iWrong,iGood,Xpop,frac)
Nbtr = len(ysoltr)
Nbva = len(ysolva)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(28,), random_state=1) ;
clf.fit(Xtr, ysoltr)     ;   
Ypred = clf.predict(Xva) ;
Ypop  = clf.predict(Xpop);
Error = np.sum(np.abs(Ypred - ysolva));

Fn    = [(Ypred[i] == 0) and (ysolva[i] == 1) for i in range(len(Ypred))].count(True)
Fp    = [(Ypred[i] == 1) and (ysolva[i] == 0) for i in range(len(Ypred))].count(True)
Tn    = [(Ypred[i] == 0) and (ysolva[i] == 0) for i in range(len(Ypred))].count(True)
Tp    = [(Ypred[i] == 1) and (ysolva[i] == 1) for i in range(len(Ypred))].count(True)
Tpr   = 1.0*Tp/(Tp + Fn)
Fpr   = 1.0*Fp/(Fp+Tn)
Perf  = 100.0*(Nbva - Error)/Nbva 

print 'fraction of good answers :Perf:',Perf
print 'Sensitivity  (Tpr = Tp/(Tp+Fn)):',Tpr
print '1-Specificity(Fpr = Fp/(Tn+Fp)):',Fpr
print 'Nb of training examples   : ', Nbtr 
print 'Nb of validation examples : ', Nbva
print 'Tp:',Tp,'  Fp:',Fp,'  Tn:',Tn,'  Fp:',Fp

plt.figure(1)
plt.subplot(211)
for I  in range(9):
	plt.plot(FrIndex,Interface.q[I][0])
	plt.plot(FrIndex,Interface.q[I][1])
	plt.plot(FrIndex,Interface.q[I][2])
plt.plot(FrIndex,Ypop,'*r')
plt.subplot(212)
for I  in range(9): 
	plt.plot(FrIndex,Interface.dq[I][0])
	plt.plot(FrIndex,Interface.dq[I][1])
	plt.plot(FrIndex,Interface.dq[I][2])
plt.plot(FrIndex,Ypop,'*r')
plt.figure(2)
plt.subplot(111)
plt.plot([0.0,1.0],[0.0,1.0],'r')
plt.plot(Fpr,Tpr,'*')
plt.xlabel('1-specificity(Fp)')
plt.ylabel('sensitivity(Tp)')
plt.show()










