import matplotlib.pyplot as plt
from AM_Functions1p0v import * 
Lneighborhood = 20
path = r'C:\Users\Simon\Desktop\TposeToCorrect\MT'
a = '2'
LocationMod               = path + a +'_Mod.xlsx'
Location                  = path + a +'.xlsx'
LocationRD                = path + a +'RawData.xlsx'
LocationNewRDxlsx         = path + a +'NewRawData.xlsx'
LocationNewRDcsv          = path + a +'NewRawData.csv'

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
I   = []; 
Tp  = {"F":[],"B":[] }
Ngr = len(IFGr);
lgf = range(0,Ngr)
lgb = range(1,Ngr)
Threshold = np.pi/4.

C = True;
Opt = 'Plateaux'
while(C==True):
	print '----------------------------'
	print '### Take a look at this: '
	S   = SignalCorrector(CumEu,Threshold,Opt)
	dfTp= S.Get_Cleaned_Signal()
	#plt.show()
	print '### Do you want to clean off the signal of its Plateaux?'
	print 'Meaning that the red dots will become the new signal part.'
	if raw_input("(y / n)  :") == 'y': CumEu = dfTp.copy(deep=True)
	else: CumEu = df.copy(deep=True)
	if raw_input("Do you want to continue the cleaning of the Signals ? y/n :")=='n':C=False
C = True;
Opt = 'Dirac'
while(C==True):
	print '----------------------------'
	print '### Take a look at this: '
	S   = SignalCorrector(CumEu,Threshold,Opt)
	dfTp= S.Get_Cleaned_Signal()
	plt.show()
	print '### Do you want to clean off the signal of its Plateaux?'
	print 'Meaning that the red dots will become the new signal part.'
	if raw_input("(y / n)  :") == 'y': CumEu = dfTp.copy(deep=True)
	else: CumEu = df.copy(deep=True)
	if raw_input("Do you want to continue the cleaning of the Signals ? y/n :")=='n':C=False


"""
C = True
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

"""
	
