# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 19:55:06 2017

@author: bruno
"""
import numpy as np
import pandas as pd

fixc=1.32 #0.08*16.5
prop=0.001

def buy(Act,C,M,price,n,i):
    #print('  antes buy',M,price,Act)
    C[Act]=C[Act]+n
    cost=fixc+prop*price*n
    M=M-n*price-cost    
    #print('  dps buy',M,Act)
    return (C,M)
    
def sell(Act,C,M,price,n,i):
    #print('  antes sell',M,price,Act)
    C[Act]=C[Act]-n
    cost=fixc+prop*price*n
    M=M+n*price-cost    
    #print('  dps sell',M,Act)
    return (C,M)   