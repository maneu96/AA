# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:12:08 2019

@author: manue
"""

from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

data_PT=read_csv("pt_trigram_count.tsv",sep='\t',header=None,names=["index","trigram","count"],usecols=[1,2]) #skiprows=1
data_EN=read_csv("en_trigram_count.tsv",sep='\t',header=None,names=["index","trigram","count"],usecols=[1,2])
data_FR=read_csv("fr_trigram_count.tsv",sep='\t',header=None,names=["index","trigram","count"],usecols=[1,2])
data_ES=read_csv("es_trigram_count.tsv",sep='\t',header=None,names=["index","trigram","count"],usecols=[1,2])

#temp = np.vstack((data_EN["count"],data_PT["count"]))
X_train= np.zeros((4,len(data_PT))) #temos que incializar com 1 caso contrario as ocorrencias nulas, fazem com que a probabilidade seja nula
X_train[0,:] = data_PT["count"] #+ X_train[0,:] 
X_train[1,:] = data_ES["count"] #+ X_train[1,:] 
X_train[2,:] = data_FR["count"] #+ X_train[2,:] 
X_train[3,:] = data_EN["count"] #+ X_train[3,:] 

Y_train= range(4) # 0-PT, 1-ES , 2-FR , 3-EN
model=MultinomialNB()  #with laplace smoothing
model.fit(X_train,Y_train)
#accuracy_score(Y_train,model.predict(X_train))
sentences = ["El cine esta abierto.","Tu vais à escola hoje.","Tu vais à escola hoje pois já estás melhor.","English is easy to learn.","Tu vas au cinéma demain matin.","É fácil de entender."]
vectorizer=CountVectorizer(ngram_range = (3,3), vocabulary=data_PT["trigram"],analyzer = 'char_wb')
vec_counter=vectorizer.fit_transform(sentences)
X_test=vec_counter.toarray()
Y_test=[1,0,0,3,2,0]

Y_predict=model.predict(X_test)
print(accuracy_score(Y_test,Y_predict))
Y_prob= model.predict_proba(X_test)

Y_prob.sort()
margin = Y_prob[:,3]-Y_prob[:,2]
print(margin)
