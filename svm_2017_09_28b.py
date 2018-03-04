# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 19:41:44 2017
Modifiqué variables para que se parezcan a las del paper.
Corré bien con Kernel='rbf', con el resto no corre (sigmoid termina sin valor)
El resultado está en torno a 73%, que es casi lo mismo que se obtiene cuando 
el vector de inputs es al azar.

Sacando BDI pasa a 92%!!!
@author: bruno
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
import PCA1

#Leo, elimino columnas que no necesito y completo las mensuales
df = pd.read_csv('InputData.csv',index_col='DATE')
#df['ret1d'] = np.log(df.SP_LAST) - np.log(df.SP_LAST.shift(1))
df.drop(df.index[0:19], inplace=True)
df.drop(['ICJ','MVOLNU'], 1, inplace=True)
monthdf=df[['DG_Ord','DG_Ship','ShortInt','CAPE','CPI']].copy()
monthdf= monthdf.fillna(method='pad')
df.drop(['DG_Ord','DG_Ship','ShortInt','CAPE','CPI'], 1, inplace=True)
df= pd.concat([df, monthdf], axis=1)

#elijo cantidad de meses, defino el "Label" a pronosticar y elimino las últimas filas
Model=3 #Meses a pronosticar
shift=Model*21
df['retModel'] = np.log(df.SP_LAST.shift(-shift)) - np.log(df.SP_LAST)
df['label'] = np.sign(df.retModel)
df.drop(df.index[-shift:], inplace=True)

#Ajusto variables para que tengan mejor sentido económico
df['DP'] = np.log(df['SPYDPS']) - np.log(df['SPY'])

Xpca=df[['DP','PE','BM','CAPE']].copy()
proj=PCA1.Principal(Xpca)
df['PCAPrice']= proj*(-1)

#fechas=df[[]].copy() #Arreglar esto en algun momento asi no hay que calcular en Excel
#df['SIM']=PCA1.Simm(fechas)

ewma=df.USGG10YR.ewm(span=252,adjust=True).mean()
df['BY']= df['USGG10YR']/ewma
df['DEF'] = df['BAA_Yield'] - df['AAA_Yield']
df['TERM'] = df['USGG10YR'] - df['USGG3M']
df['VRP'] = df['VIX'] #- df['VESTIM']
df['NOS'] = np.log(df['DG_Ord']) - np.log(df['DG_Ship'])
df['PCR'] = np.log(df['SPY']) - np.log(df['GSCI'])
wind=210
df['MA'] = df['SP_LAST'].rolling(window=wind,min_periods=None).mean()
df['OIL'] = df['CL1'] - df['CL4'].shift(63)
df['SI'] = df['ShortInt']/df['NYAVOL'].rolling(window=21,min_periods=None).mean()

df.drop(df.index[:wind-1], inplace=True)
#df[['NYAVOL','MAVol']].plot()

#falta calcular Cay, VarianceStimator y PCA_Tech.
#Revisar calculo de SI. Combina diario y mensual


#df = df[['PCAPrice','BY', 'DEF', 'TERM', 'SIM','VRP','NOS', 
#        'CPI', 'PCR', 'MA', 'OIL','SI','label']]

#col=['PCAPrice','BY', 'DEF', 'TERM', 'SIM','BDI','VRP','NOS','CPI', 'PCR', 'MA', 'OIL','SI']
col=['PCAPrice','BY', 'DEF', 'TERM', 'SIM','VRP','NOS','CPI', 'PCR', 'MA','SI','label']

#col=['CPI', 'MA','label']

df2=df.copy()

for i in range(1):
    df=df2[col].copy()


    df= df.fillna(-99999)
    df.to_csv('out.csv', sep=',')
    
    X = np.array(df.drop(['label'], 1))
    #X = np.random.rand(2647,13)
    y = np.array(df['label'])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    
    clf = svm.SVC(kernel="rbf")
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(i," ",confidence)

"""
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)
"""


"""
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(df.label)
"""




"""
example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)
"""
