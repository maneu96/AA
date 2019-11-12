# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:22:37 2019

@author: manue
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score
#from sklearn import MultinomialNB

xtest=np.load("data1_xtest.npy");
xtrain=np.load("data1_xtrain.npy");
ytest=np.load("data1_ytest.npy");
ytrain=np.load("data1_ytrain.npy");



#plt.scatter(xtrain[:,0],xtrain[:,1],c=np.reshape(ytrain,[150,]));
#plt.scatter(xtest[:,0],xtest[:,1],c=np.reshape(ytest,[150,]),marker='s');
#plt.xlim(-5, 5)
#plt.ylim(-3, 7)

mean1=np.mean(xtrain[0:50,:],axis=0)
mean2=np.mean(xtrain[50:100,:],axis=0)
mean3=np.mean(xtrain[100:150,:],axis=0)

var1=np.var(xtrain[0:50,:],axis=0)
var2=np.var(xtrain[50:100,:],axis=0)
var3=np.var(xtrain[100:150,:],axis=0)


#aux=np.matmul((xtrain[0:50,:] - mean1),np.transpose((xtrain[0:50,:] - mean1)))

#cov1=np.mean(aux,axis=0)
covtrain1=np.diag(var1)
covtrain2=np.diag(var2)
covtrain3=np.diag(var3)
#covtrain1=np.cov(np.transpose(xtrain[0:50,:]))
#covtrain2=np.cov(np.transpose(xtrain[50:100,:]))
#covtrain3=np.cov(np.transpose(xtrain[100:150,:]))
#normal_if_1_train=multivariate_normal.pdf(xtrain[0:150,:],mean=mean1,cov=covtrain1)
P_X_if_1=multivariate_normal.pdf(xtest[0:150,:],mean=mean1,cov=covtrain1)
P_X_if_2=multivariate_normal.pdf(xtest[0:150,:],mean=mean2,cov=covtrain2)
P_X_if_3=multivariate_normal.pdf(xtest[0:150,:],mean=mean3,cov=covtrain3)

#P_X_if_1_test=multivariate_normal.pdf(xtest[0:150,:],mean=mean1,cov=covtrain1)
#P_X_if_2_test=multivariate_normal.pdf(xtest[0:150,:],mean=mean2,cov=covtrain2)
#P_X_if_3_test=multivariate_normal.pdf(xtest[0:150,:],mean=mean3,cov=covtrain3)
classe=[]
#classe_test=[]
P_X_if_Classe = np.transpose(np.array([P_X_if_1 ,P_X_if_2, P_X_if_3]))
#P_X_if_test = np.transpose(np.array([P_X_if_1_test ,P_X_if_2_test, P_X_if_3_test]))
for i in range(150) :
    classe.append(np.argmax(P_X_if_Classe[i,:],axis=0))
    #classe_test.append(np.argmax(P_X_if_test[i,:],axis=0))
P_1=1/3
P_2=P_1
P_3=P_1

P_1_if_X= P_X_if_1*P_1
P_2_if_X= P_X_if_2*P_2
P_3_if_X= P_X_if_3*P_3

P_Classe_if_X=P_X_if_Classe * 1/3
classe_estimate=[]
error=0
for i in range(150) :
    classe_estimate.append(np.argmax(P_Classe_if_X[i,:],axis=0) + 1)
    #aux_error= np.abs(classe_estimate-np.transpose(ytest))
    #aux_error= classe_estimate[i]-ytest[i]
    if classe_estimate[i]-ytest[i] != 0:
        error=error+1
plt.figure()
plt.plot(range(150),classe_estimate)

classe_estimate = np.asarray(classe_estimate)
error=error/len(classe_estimate)
print(error)

#error_2=1 - accuracy_score(ytest,classe_estimate)


