# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 19:05:56 2018

@author: bruno
"""

import numpy as np
import pandas as pd
from sklearn import model_selection, svm
from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,  accuracy_score
from sklearn.preprocessing import StandardScaler  
import PCA1

fixc=0.0#8
prop=0.00#1
Anticipo=1 #Anticipación de pasos a pronosticar
Vent_mes=2 #Ventana en Meses que incluyo en los Inputs
Thresh=0.005
freq="Day" #"Month" o "Day"
clasificador="SVM" #"SVM" o "MLP"

if freq == "Day":
    #Leo, elimino columnas que no necesito y completo las mensuales
    df = pd.read_csv('InputData96.csv',index_col='DATE')   #Falta NYAVOL
    #df['ret1d'] = np.log(df.SP_LAST) - np.log(df.SP_LAST.shift(1))
    df.drop(df.index[0:22], inplace=True)
    df.drop(['ICJ','MVOLNU'], 1, inplace=True)
    monthdf=df[['DG_Ord','DG_Ship','ShortInt','CAPE','CPI']].copy()
    monthdf= monthdf.fillna(method='pad')
    df.drop(['DG_Ord','DG_Ship','ShortInt','CAPE','CPI'], 1, inplace=True)
    df= pd.concat([df, monthdf], axis=1)
    unidad_mes=21
    
elif freq == "Month":
    df = pd.read_csv('InputMonthlyAve.csv',index_col='DATE')   #Falta NYAVOL
    unidad_mes=1

shift=Anticipo
Vent_periods=Vent_mes*unidad_mes
MAwind=2*unidad_mes

#elimino las últimas filas
df['retModel'] = np.log(df.SP_LAST.shift(-shift)) - np.log(df.SP_LAST)
df['label'] = np.sign(df.retModel)
df.drop(df.tail(shift).index, inplace=True) #Hacer que borre el ultimo


for a in df.index:
    if np.abs(df.loc[a,'retModel']) < Thresh:
        df.loc[a,'label'] = 0 

#Ajusto variables para que tengan mejor sentido económico
df['DP'] = np.log(df['SPYDPS']) - np.log(df['SPY'])
df.dropna(axis=0, inplace=True)

Xpca=df[['DP','PE','BM','CAPE']].copy()

proj=PCA1.Principal(Xpca)
df['PCAPrice']=proj*(-1)

#fechas=df[[]].copy() #Arreglar esto en algun momento asi no hay que calcular en Excel
#df['SIM']=PCA1.Simm(fechas)

ewma=df.USGG10YR.ewm(span=12,adjust=True).mean()
df['BY']= df['USGG10YR']/ewma
df['DEF'] = df['BAA_Yield'] - df['AAA_Yield']
df['TERM'] = df['USGG10YR'] - df['USGG3M']
df['VRP'] = df['VIX'] #- df['VESTIM']
df['NOS'] = np.log(df['DG_Ord']) - np.log(df['DG_Ship'])
df['PCR'] = np.log(df['SPY']) - np.log(df['GSCI'])
wind=MAwind*3
wind2=MAwind
df['MA'] = df['SP_LAST'].rolling(window=wind,min_periods=None).mean()
df['MA2'] = df['SP_LAST'].rolling(window=wind2,min_periods=None).mean()
df['OIL'] = df['CL1'] - df['CL4'] #podria ser CL12 tmb

""" Falta NYAVOL
df['SI'] = df['ShortInt']/df['NYAVOL'].rolling(window=21,min_periods=None).mean()
"""

df['SI'] = df['ShortInt'].rolling(window=unidad_mes,min_periods=None).mean()
df.drop(df.index[:wind-1], inplace=True)
#df[['NYAVOL','MAVol']].plot()

#falta calcular Cay, VarianceStimator y PCA_Tech.
#Revisar calculo de SI. Combina diario y mensual


#col=['PCAPrice','BY', 'DEF', 'TERM', 'SIM','BDI','VRP','NOS','CPI', 'PCR', 'MA', 
#     'OIL','SI','label']
col=['DP','PE','BM','CAPE','BY', 'DEF', 'TERM', 'SIM','VRP','NOS','CPI', 
    'PCR', 'MA','OIL','SI','label']

df2=df.copy()
df=df2[col].copy()
df.dropna(axis=0, inplace=True)

#df= df.fillna(-99999)

#df['time']=0
#df['time']=[np.log(i+1) for i in range(0,len(df['time']))]
#print(df)
df.to_csv('out.csv', sep=',')
#df=(df.drop(['label'], 1)) #df no label

df2=df.copy()
df2=(df2.drop(['label'], 1))
for i in range(0,Vent_periods):
    df2 = df2.shift(1)
    df2.rename(columns=lambda x: x+str(i), inplace=True)
    df= pd.concat([df, df2], axis=1)
    df2.columns=col[:-1]
    

df.dropna(axis=0, inplace=True)
#df=(df.drop(['label'], 1))

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
X_train, X_test, y_train, y_test =model_selection.train_test_split(
        X, y, test_size=0.3, shuffle = False, stratify = None)

scaler = StandardScaler()  

scaler.fit(X_train)  # Don't cheat - fit only on training data
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  # apply same transformation to test data

lab=pd.DataFrame(y_test) #para imprimir en csv los labels.
lab.to_csv('label.csv',sep=',')

if clasificador == "MLP":
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(4,), random_state=1,max_iter=500)
elif clasificador == "SVM":
    igamma= 0.0001
    iC=100
    clf = svm.SVC(kernel="rbf",gamma=igamma,C=iC,
                  decision_function_shape='ovo')    
    
clf.fit(X_train, y_train) 
prediction=clf.predict(X_test)
confidence = clf.score(X_test, y_test)
print(confidence)
print(confusion_matrix(y_test,prediction))
#print(classification_report(y_test,prediction))


 #Aca es donde hacia la busqueda de C y Gamma para el SVM
C_range = np.logspace(0, 8, 9)
g_range = np.logspace(-5, 2, 9)

dfOpt= pd.DataFrame(columns=['Gamma','C','Conf','Cant-1','Cant0'])
i=0
for igamma in g_range:
    for iC in C_range:

        clf = svm.SVC(kernel="rbf",gamma=igamma,C=iC,
                      decision_function_shape='ovo')
        clf.fit(X_train, y_train)

        confidence = clf.score(X_test, y_test)
        prediction=clf.predict(X_test)
        pred=pd.DataFrame(prediction,columns=['Pred'])
        #pred.to_csv('pred.csv',sep=',')

        predlist = pred.values.tolist()
        cantm1=predlist.count([-1.0])
        cant0=predlist.count([0.0])
        print("i= "+str(i)+"; gamma= "+str(igamma)+ "; C= "+str(iC))
        row = [igamma,iC,confidence,cantm1,cant0]
        print(confidence)
        print(confusion_matrix(y_test,prediction))
        
        dfOpt.loc[len(dfOpt)] = row
        
        i = i+1

dfOpt.to_csv('Opt_Gamma_C.csv',sep=',')


#Acá es donde empieza la evolución de la cartera. Primero terminar de ajustar
#después seguir esto


"""
MaxLong=1
MaxShort=1

evol = df2[['SP_LAST']][-len(pred):]
pred.set_index(evol.index.values,inplace=True)
evol = pd.concat([evol, pred], axis=1)
#evol['B&H']= np.log(evol.SP_LAST) - np.log(evol.SP_LAST.shift(1))

Money0=evol['SP_LAST'][0]*(1+prop)+fixc
evol['S&P']= evol.SP_LAST/Money0
evol['Money']=0.0
evol['Cant']=0.0

Money0=evol['SP_LAST'][0]*(1+prop)+fixc

n=1 #cantidad que compro en cada decisión
step=21
#paso inicial
if evol['Pred'][0]==1:
    evol['Cant'][0]=1
    evol['Money'][0]=Money0-n*evol['SP_LAST'][0]*(1+prop)-fixc
if evol['Pred'][0]==-1:
    evol['Cant'][0]=-1
    evol['Money'][0]=Money0+n*evol['SP_LAST'][0]*(1-prop)-fixc
  
for i in range(1,len(pred)):    
    if evol['Pred'][i]==1 and evol['Cant'][i-1]<=MaxLong-1 and i%step == 0 :
        evol['Cant'][i]=evol['Cant'][i-1]+1
        evol['Money'][i]=evol['Money'][i-1]-n*evol['SP_LAST'][i]*(1+prop)-fixc
    elif evol['Pred'][i]==-1 and evol['Cant'][i-1]>=MaxShort-1 and i%step == 0 :
        evol['Cant'][i]=evol['Cant'][i-1]-1
        evol['Money'][i]=evol['Money'][i-1]+n*evol['SP_LAST'][i]*(1-prop)-fixc   
    else:
        evol['Cant'][i]=evol['Cant'][i-1]
        evol['Money'][i]=evol['Money'][i-1]
        
evol['Port']=evol['Money']+evol['Cant']*evol['SP_LAST']
evol['SVM']=evol.Port/Money0
evol[['S&P','SVM']].plot()

    
#with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
#    print(df.label)
"""