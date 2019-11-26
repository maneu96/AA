# -*- coding: utf-8 -*-
"""
Created on Thu ‎Nov 14 ‎‏‎16:31:58 2019

@author: manue
"""

import numpy as np
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
#from matplotlib.pyplot import show
import random
from keras.utils import to_categorical
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import Sequential
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
X_test = np.load("mnist_test_data.npy")
Y_test = np.load("mnist_test_labels.npy")
X_train = np.load("mnist_train_data.npy")
Y_train = np.load("mnist_train_labels.npy")

print("Random image from train data set")
plt.imshow(X_train[random.randrange(1,3001,1)].squeeze(), cmap = "gray")
plt.show()
print("Random image from test data set")
plt.figure()
plt.imshow(X_test[random.randrange(1,501,1)].squeeze(), cmap = "gray")
plt.show()


X_test=X_test/255
X_train=X_train/255
Y_train_1hot = to_categorical(Y_train)
Y_test_1hot = to_categorical(Y_test)

#dividir tudo em 0.3 para validacao e o resto para treino

MLP = Sequential()

MLP.add(Flatten(input_shape=(28,28,1)))
MLP.add(Dense(units=64,activation='relu'))
MLP.add(Dense(units=128,activation='relu'))
MLP.add(Dense(10, activation='softmax'))
#MLP.summary()
MLP_2= MLP

ES = EarlyStopping(patience=15,restore_best_weights=True)
MLP.compile(loss='categorical_crossentropy',optimizer='Adam')
MLPhistory=MLP.fit(x=X_train,y=Y_train_1hot ,batch_size=300,epochs=400,callbacks=[ES],validation_split=0.3)


plt.figure()
plt.plot(MLPhistory.history['loss'], label='train')
plt.plot(MLPhistory.history['val_loss'], label='test')


Y_predicted= MLP.predict(X_test)
score=accuracy_score(Y_test_1hot.argmax(axis=1),Y_predicted.argmax(axis=1))
cm = confusion_matrix(Y_test_1hot.argmax(axis=1),Y_predicted.argmax(axis=1))



# sem Early Stopping
MLP_2.compile(loss='categorical_crossentropy',optimizer='Adam')
MLPhistory_2=MLP_2.fit(x=X_train,y=Y_train_1hot ,batch_size=300,epochs=400,validation_split=0.3)


plt.figure()
plt.plot(MLPhistory_2.history['loss'], label='train')
plt.plot(MLPhistory_2.history['val_loss'], label='test')


Y_predicted_2= MLP_2.predict(X_test)
score_2=accuracy_score(Y_test_1hot.argmax(axis=1),Y_predicted_2.argmax(axis=1))
cm_2 = confusion_matrix(Y_test_1hot.argmax(axis=1),Y_predicted_2.argmax(axis=1))




# Convolutional Neural Network

CNN = Sequential()
CNN.add(Conv2D(filters=16,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
CNN.add(MaxPooling2D(pool_size=(2,2)))
CNN.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
CNN.add(MaxPooling2D(pool_size=(2,2)))
CNN.add(Flatten())
CNN.add(Dense(units=64,activation='relu'))
CNN.add(Dense(10, activation='softmax'))
CNN.summary()

CNN.compile(loss='categorical_crossentropy',optimizer='Adam')
CNNhistory=CNN.fit(x=X_train,y=Y_train_1hot ,batch_size=300,epochs=400,callbacks=[ES],validation_split=0.3)


plt.figure()
plt.plot(CNNhistory.history['loss'], label='train')
plt.plot(CNNhistory.history['val_loss'], label='test')

Y_predicted_CNN= CNN.predict(X_test)
score_CNN=accuracy_score(Y_test_1hot.argmax(axis=1),Y_predicted_CNN.argmax(axis=1))
cm_CNN = confusion_matrix(Y_test_1hot.argmax(axis=1),Y_predicted_CNN.argmax(axis=1))