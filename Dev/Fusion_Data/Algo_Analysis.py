from Fonctions_Analyse import *

nameN = 'C:/Users/Simon/Desktop/sensor2test0Normal.csv'
nameC = 'C:/Users/Simon/Desktop/sensor2test0Corrected.csv'
nameE = 'C:/Users/Simon/Desktop/sensor2test0Corrupted.csv'

GraphCap(nameN,nameE,'N-E')
GraphCap(nameE,nameC,'E-C')
GraphCap(nameN,nameC,'N-C')


# nameN = 'C:/Users/Simon/Desktop/sensor2test0Normal.csv'
# nameC = 'C:/Users/Simon/Desktop/sensor2test0Corrected.csv'
# nameCo = 'C:/Users/Simon/Desktop/sensor2test0Corrupted.csv'
# nameTR= 'C:/Users/Simon/Desktop/Robot_arm/TimeOfMove.txt'
# nameR = 'C:/Users/Simon/Desktop/Robot_arm/testsensor2.txt'
# GraphArm(nameN,nameC,nameR, nameTR)

# GraphCap(nameN,nameC,'N-C')
# GraphCap(nameN,nameC,'N-C')