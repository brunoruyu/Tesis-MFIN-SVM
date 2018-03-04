# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 19:41:44 2017

Primer modelo. No hago nada sobre las variables, simplemente era para observar
el funcionamiento de sklearn
De todas maneras, tomando 3Meses y sin acomodar nada, SVM Linear logra un 76%.
A veces corre, a veces no. En general corre se hace "mucho" que no lo corro. 
En dicho caso, tarda unos pocos segundos.


@author: bruno
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm

df = pd.read_csv('InputData.csv',index_col='DATE')
df['ret1d'] = np.log(df.SP_LAST) - np.log(df.SP_LAST.shift(1))
df.drop(df.index[0:19], inplace=True)
df.drop(['ICJ'], 1, inplace=True)
#df.drop(['MVOLNU','SP_HIGH','SP_LOW','SP_LAST','SPY'], 1, inplace=True)
df.drop(['MVOLNU','SPY'], 1, inplace=True)
monthdf=df[['DG_Ord','DG_Ship','ShortInt']].copy()
monthdf= monthdf.fillna(method='pad')
df.drop(['DG_Ord','DG_Ship','ShortInt'], 1, inplace=True)
df= pd.concat([df, monthdf], axis=1)

meses=3 #Meses a pronosticar
shift=meses*21
df['retModel'] = np.log(df.SP_LAST.shift(-shift)) - np.log(df.SP_LAST)
df['signo'] = np.sign(df.retModel)
df.drop(df.index[-shift:], inplace=True)
#df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df= df.fillna(-99999)
df.to_csv('out.csv', sep=',')

X = np.array(df.drop(['signo'], 1))
y = np.array(df['signo'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC(kernel="linear")
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

"""
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)
"""









"""
example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)
"""
