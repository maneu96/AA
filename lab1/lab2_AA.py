# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 19:20:47 2019

@author: manuel e guilherme
"""
from scipy import stats
import matplotlib as plt
import numpy as np

p = 2 ;
x = np.load("data2a_x.npy") ;
y = np.load("data2a_y.npy") ;  
y_lss=np.zeros(len(x));
SSE = np.zeros(len(x));
y_lss=y_lss.reshape(len(x),1);
SSE = SSE.reshape(len(x),1);       


def n_exp(a,n):
  if n==1:
      return a
  elif n==0:
      return 1
  else :
      return a*n_exp(a,n-1)
  
def vec_n_exp(v,n):
    v_out= np.zeros(len(v));
    v_out= v_out.reshape(len(v),1)
    for i in range(len(v)):
       v_out[i]=n_exp(v[i],n);
    
    return v_out


M=np.zeros(shape=(p+1,p+1));
N=np.zeros(p+1);
for i in range (p+1):
    for j in range (p+1):
        M[i][j]=np.sum(vec_n_exp(x,j+i));
    N[i] = np.sum(np.multiply(y,vec_n_exp(x,i)))

M = np.linalg.inv(M);
B = np.matmul(M,N);


i_x = np.argsort(x, axis=0, kind="quicksort", order=None)
x = np.sort(x, axis=0, kind="quicksort", order=None)
y=y[i_x];
y=y.reshape(len(x),1)
plt.pyplot.plot(x,y);


for i in range(len(x)):
    for j in range(p+1):
        if j==0:    
            y_lss[i]= B[0]; 
        else:
            y_lss[i]= y_lss[i] + (n_exp(x[i],j)) * B[j]

SSE = y - y_lss
SSE = vec_n_exp(SSE,2)
print("O valor SSE é :")
print(np.sum(SSE))
print("Os coeficientes são : [B[0] ... B[p]]")
print(B)
plt.pyplot.plot(x,y_lss)


z = np.abs(stats.zscore(y))

for i in range(len(z)):
    if z[i] > 3 :
        y[i] =  y_lss[i];
            
for i in range(len(x)):
    for j in range(p+1) :
        if j==0 :    
            y_lss[i]= B[0] ; 
        else :
            y_lss[i]= y_lss[i] + (n_exp(x[i],j)) * B[j]

nSSE = y - y_lss
nSSE = vec_n_exp(nSSE,2)
print("O valor SSE sem o outlier é :")
print(np.sum(nSSE))





    
