from Fonctions_Analyse import *
#0  - 180
#43 - 180 
#10 - 120
file  = 'C:/Users/Simon/Desktop/Robot_arm/testsensor2.txt'
nStep =   600
oi    =  0.00; of    =  120.00;
ti    =163.60; tf    = 163.60;
zi    = 10.00; zf    =  10.00;
wo    =  0.010; wt    =   0.00; wz  = 0.0
ArmTrajectory(file,nStep,oi,of,ti,tf,zi,zf,wo,wt,wz)
