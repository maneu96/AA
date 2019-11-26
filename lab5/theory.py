# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:39:46 2019

@author: manue
"""

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]] )
y = np.array([-1,-1, -1, 1 ])

clf = SVC(C = 1e5, kernel = 'linear')
clf.fit(X, y) 


w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

plt.plot(xx, yy, 'k-')
plt.scatter(X[:,0],X[:,1])
plt.axis('tight')