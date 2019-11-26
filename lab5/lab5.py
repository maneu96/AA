# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:59:08 2019

@author: manue
"""

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
X=np.load("spiral_X.npy")
Y=np.load("spiral_Y.npy")

SVM_poly = SVC(kernel = 'poly',max_iter=100000, gamma= 'auto')

n=1
flag=True
acc=[0.0]
while(flag) :
    SVM_poly.degree=n;
    SVM_poly.fit(X,Y)
    Y_pred=SVM_poly.predict(X)
    acc.append(accuracy_score(Y,Y_pred))
    print('\n For p=',n)
    print('Support vectors are :',SVM_poly.support_vectors_,'\n \n accuracy is :', acc[n], '\n')
    if acc[n]-acc[n-1] < 0 :
       flag=False
    else :
       n+=1
       
print('\n Optimal value is for p= ',n-1)

