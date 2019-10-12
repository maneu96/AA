# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:41:55 2019

@author: manue
"""

import numpy as np
import matplotlib as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

x= np.load("data3_x.npy")
y= np.load("data3_y.npy")
i= 0.001
j=0;
coefs_L=[]
coefs_R=[]
alphas=[]
alphas.append(i)
x_0=[]
x_1=[]
x_2=[]
while (i <= 10) : 
    L= sklearn.linear_model.Lasso(alpha=i,max_iter=10000)
    L.fit(x,y)
    aux=L.coef_
    coefs_L.append(L.coef_)
    R= sklearn.linear_model.Ridge(alpha=i,max_iter=10000)
    R.fit(x,y)
    aux=R.coef_
    coefs_R.append(R.coef_)
    i= i + 0.01
    alphas.append(i)
    j=j+1
    
alphas_1 = []
ax = plt.pyplot.gca()
for i in range(1000):
    x_0.append(coefs_L[i][0]);
    x_1.append(coefs_L[i][1]);
    x_2.append(coefs_L[i][2])
    alphas_1.append(alphas[i])

ax = plt.pyplot.gca()  
ax.plot(alphas_1, x_0)
ax.plot(alphas_1, x_1)
ax.plot(alphas_1, x_2)
ax.set_xscale('log')