#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:13:46 2019

@author: guilherme
"""
import random
import numpy as np
from matplotlib import plt
import math
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from keras.callbacks import EarlyStopping
from keras import predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# %% DATA
trainX = np.load("mnist_train_data.npy")
trainy = np.load("mnist_train_labels.npy")

testX = np.load("mnist_test_data.npy")
testy = np.load("mnist_test_labels.npy")

sizetrainX = len(trainX)
sizetrainy = len(trainy)

sizetestX = len(testX)
sizetesty = len(testy)

print("The size of your training data set is")
print(sizetrainX,sizetrainy)

print("The size of your test data set is")
print(sizetestX,sizetesty)

print("Random image from training data set")
plt.imshow(trainX[random.randrange(1,3001,1)].squeeze(), cmap = "gray")
plt.show()

print("Random image from test data set")
plt.imshow(testX[random.randrange(1,501,1)].squeeze(), cmap = "gray")
plt.show()

ntrainX = trainX/255
ntrainy = trainy/255

ntestX = testX/255
ntesty = testy/255

ohtrainy = np_utils.to_categorical(ntrainy,num_classes=10)
ohtesty = np_utils.to_categorical(ntesty,num_classes=10)

ntestX, nvalidationX, ntesty, nvalidationy = train_test_split(ntrainX, ntrainy, test_size = 0.3)
# %% MULTI LAYER PERCEPTRON

MLPmodel = Sequential()
MLPmodel.add(Dense(784, input_shape = (28, 28, 1))) 
MLPmodel.add(Flatten())
MLPmodel.add(Dense(64, activation = 'relu'))
MLPmodel.add(Dense(128, activation = 'relu'))
MLPmodel.add(Dense(10, activation = 'softmax'))

print(MLPmodel.summary)

es = EarlyStopping(monitor = 'loss', patience = 15, restore_best_weights = True)
MLPmodel.compile(loss = 'categorical_crossentropy', otimizer = 'adam')
history = MLPmodel.fit(ntrainX, ohtrainy, epochs = 400, batch_size = 300, callbacks = [es])

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')


# %% CONVOLUTIONAL NEURAL NETWORK
# %%COMMENTS