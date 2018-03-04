# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:56:00 2017

@author: bruno
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA 

def Principal(Xpca):
    pca = PCA(n_components = 1)
    pca.fit(Xpca)
    projection = pca.transform(Xpca)
    return projection

#Arreglar esto en algun momento
"""
def Simm(f):
    f['year']= pd.DatetimeIndex(f.index,dayfirst=True).year
    f['month']= pd.DatetimeIndex(f.index,dayfirst=True).month
    f['day']= pd.DatetimeIndex(f.index,dayfirst=True).day

    conditions = [
    (f['month'] <5 ),
    (f['month'] >=5 ) & (pd.datetime(f['year'],f['month'],f['day']) < pd.datetime(f['year'],6,7)),
    (f['month'] == 12)]
    choices = [130 -(pd.datetime(f['year'],5,1) - pd.datetime(f['year'],f['month'],f['day'])), 
               130, 
               max(0,f.day-23)]
    f['SIMM'] = np.select(conditions, choices, default=pd.datetime(f['year'],10,15) - pd.datetime(f['year'],f['month'],f['day']))
    return f['SIMM']
"""
