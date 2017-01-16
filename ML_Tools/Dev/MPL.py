from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
clf  = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
name = '.\MT3RawData.csv' 
df   = pd.read_csv(name)
Na   = list(df)
L    = len(df)
plt.ion()
Q=[];M=[];A=[];R=[];Ys=[];Yns=[]
Xpop = []
for s in range(L):
	Q.append([  df[Na[2]].values[s]  , df[Na[3]].values[s]  ,  df[Na[4]].values[s] , df[Na[5]].values[s]])
	M.append([  df[Na[6]].values[s]  , df[Na[7]].values[s]  ,  df[Na[8]].values[s]])
	A.append([  df[Na[9]].values[s]  ,df[Na[10]].values[s]  , df[Na[11]].values[s]])
	R.append([ df[Na[12]].values[s]  ,df[Na[13]].values[s]  , df[Na[14]].values[s]])

for s in range(2,L-3):	
	dq = []
	for J in range(len(Q[0])):			
		dq.append(Q[s][J]-Q[s-2][J])
		dq.append(Q[s][J]-Q[s-1][J])
		dq.append(Q[s][J]-Q[s+1][J])
		dq.append(Q[s][J]-Q[s+2][J])
	Xpop.append(dq)
	
plt.figure()
ax1 = plt.subplot(411)
plt.plot(Q)
ax2 = plt.subplot(412,sharex=ax1)
plt.plot(M)
ax3 = plt.subplot(413,sharex=ax1)
plt.plot(A)
ax4 = plt.subplot(414,sharex=ax1)
plt.plot(R)
Ys.append( raw_input("    Stationnary regions: "))
Yns.append(raw_input("Non-Stationnary regions: "))
plt.show()

Xexemple    = [];
Xex         = [];
Xtr         = [];
Xva 		= [];
Ysol		= [];
ys   		= [];
ts  = [int(float(i)) for i in  Ys[0].split(',')]; Ts  = [[ts[2*i],ts[2*i+1]] for i in  range(len(ts)/2)]
tns = [int(float(i)) for i in Yns[0].split(',')]; Tns = [[tns[2*i],tns[2*i+1]] for i in  range(len(tns)/2)]

for Is in range(len(Ts)):	
	for i in range(Ts[Is][0],Ts[Is][1]):
		Ysol.append(1)
		dq = []		
		for J in range(len(Q[0])):			
			dq.append(Q[i][J]-Q[i-2][J])
			dq.append(Q[i][J]-Q[i-1][J])
			dq.append(Q[i][J]-Q[i+1][J])
			dq.append(Q[i][J]-Q[i+2][J])
		Xexemple.append(dq)
for Ins in range(len(Tns)):
	for i in range(Tns[Ins][0],Tns[Ins][1]):
		Ysol.append(0)
		dq = []		
		for J in range(len(Q[0])):			
			dq.append(Q[i][J]-Q[i-2][J])
			dq.append(Q[i][J]-Q[i-1][J])
			dq.append(Q[i][J]-Q[i+1][J])
			dq.append(Q[i][J]-Q[i+2][J])
		Xexemple.append(dq)	

r           = np.random.permutation(len(Ysol))
itend       = int(2.*len(r)/3.)
ivinit      = int(2.*len(r)/3.)+1
ysol        = np.array(Ysol)
Xex         = np.array(Xexemple)
print len(Xex), r[itend],r[-1]
Xtr         = Xex[r[:itend]]
Xva         = Xex[r[ivinit:]]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)
clf.fit(Xtr, ysol[r[:itend]])    
Ypred = clf.predict(Xva)
Ypop  = clf.predict(Xpop)
Error = np.sum(np.abs(Ypred - ysol[r[ivinit:]]))
ysolv = ysol[r[ivinit:]]

Fn    = [(Ypred[i] == 0) and (ysolv[i] == 1) for i in range(len(Ypred))].count(True)
Fp    = [(Ypred[i] == 1) and (ysolv[i] == 0) for i in range(len(Ypred))].count(True)
Tn    = [(Ypred[i] == 0) and (ysolv[i] == 0) for i in range(len(Ypred))].count(True)
Tp    = [(Ypred[i] == 1) and (ysolv[i] == 1) for i in range(len(Ypred))].count(True)
Tpr   = 1.0*Tp/(Tp + Fn)
Fpr   = 1.0*Fp/(Fp+Tn)
Nb    = len(r[ivinit:])
Perf  = 100.0*(Nb - Error)/Nb 

plt.figure()
plt.subplot(211)
plt.plot(Q)
plt.plot(Ypop,'.')
plt.subplot(212)
plt.plot([0.0,1.0],[0.0,1.0],'r')
plt.plot(Fpr,Tpr,'*')
plt.xlabel('1-specificity(Fp)')
plt.ylabel('sensitivity(Tp)')
plt.show()

print 'fraction of good answers :Perf:',Perf
print 'Sensitivity  (Tpr = Tp/(Tp+Fn)):',Tpr
print '1-Specificity(Fpr = Fp/(Tn+Fp)):',Fpr
print 'Tp:',Tp
print 'Fp:',Fp
print 'Tn:',Tp
print 'Fp:',Tp








