from AM_Functions0p0v import * 
import time
import matplotlib.pyplot as plt
Lneighborhood = 20

#path = r'C:\Users\Simon\Desktop\TposeToCorrect\OthersT\Jordanna Petroff\Small Vertical Lower'
#a = '\SVL16'

path = r'C:\Users\Simon\Desktop\TposeToCorrect\OthersT\Jordanna Petroff\Small Vertical Lift'
a = '\SVL1'

#path = r'C:\Users\Simon\Desktop\TposeToCorrect\OthersT\Jordanna Petroff\Low Push'
#a = '\LP'

#path = r'C:\Users\Simon\Desktop\TposeToCorrect\OthersT\Jordanna Petroff\Calibration Trials'
#a = '\CL2'


Location                  = path + a +'.xlsx'
LocationMod               = path + a +'_Mod.xlsx'
LocationRD                = path + a +'RawData.xlsx'
LocationNewRDxlsx         = path + a +'NewRawData.xlsx'
LocationNewRDcsv          = path + a +'NewRawData.csv'
LocationRelabeledOldRDcsv = path + a +'RelabeledOldRawData.csv'

#path = r'C:\Users\Simon\Desktop\TposeToCorrect\MT'
#a = '8'

##########################################################
#Data frame created from the excel data file
df  = PDreadEx(Location)
RDdf= RDreadEx(LocationRD)
ExcelReadandSave(Location,LocationMod)
BDB,Moybdb,MoyTpd = IndexComputation(df)
CumEu = RDdf.copy(deep=True)
#Selection of regions of possible T-pose 
#candidates by using and interactive graphics interface
ra,rs = PlotMoveDiffMoveIndex(df,BDB,Moybdb,MoyTpd)
#On the two feathure (stationnarity and Tpose-Distance) 
#Tpose candidate are selected automatically
IFrame,Thx,Thy,Thz = RawDataSignal(RDdf)
moddf = CandidatTposeSelection(LocationMod,df,rs,ra,Moybdb,MoyTpd)
TposeGroups = TposeGroupsFormation(moddf,Lneighborhood)
dfEu = df.copy(deep=True)
IFGr,RawDGr,StatGr = TposeToRawData(TposeGroups,dfEu)
I   = []; C = True;
Tp  = {"F":[],"B":[] }
Ngr = len(IFGr);
lgf = range(0,Ngr)
lgb = range(1,Ngr)
Threshold = np.pi/4.
while(C==True):
	print '----------------------------'
	print '### Take a look at this: '
	S   = SignalCorrector(CumEu,Threshold)
	dfTp= S.Get_Cleaned_Signal()
	plt.show()
	print '### Do you want to clean the signal ?'
	print 'Meaning that the red dot will become the new signal part.'
	if raw_input("(y / n)  :") == 'y':
		CumEu = dfTp.copy(deep=True)
	else:
		CumEu =   df.copy(deep=True)
	lf = list(set(lgf)-set(Tp["F"]))
	lb = list(set(lgb)-set(Tp["B"]))
	print '----------------------------'
	print '### Forward Error Correction.'
	print '### Groups of Tposes for the Forward correction:  [', lf ,']'
	print '### Choose a set of Tposesto apply to the actual signal (The format of the answer is number separated by , ).'
	I = raw_input(": ")
	I = I.split(',')
	I = [int(i) for i in I]
	############################
	#IFGr,RawDGr,StatGr = TposeToRawData(TposeGroups,CumEu)
	#d = Step_Between_Tpose(IFGr,RawDGr,StatGr)
	############################
	if len(I)!=0:
		for i in I:
			CumEu = Euler_ForwardErrorCorrection(i,TposeGroups,CumEu,Tp)
	print '----------------------------'
	print '### Backward Error Correction.'
	print '### Groups of Tposes for the Backward correction: [', lb ,']'
	print '### Choose a set of Tposes to apply to the actual signal (The format of the answer is number separated by , ).'
	I = raw_input(": ")
	I = I.split(',')
	I = [int(i) for i in I]
	if len(I)!=0:
		for i in I:
			CumEu = Euler_BackwardErrorCorrection(i,TposeGroups,CumEu,Tp)
	if raw_input("Continue/End ? (y / n)  :") == 'n': 
		C = False		
		App_Formated_CSV_Writer(CumEu,LocationNewRDcsv)
	SaveDf(CumEu,LocationNewRDxlsx)