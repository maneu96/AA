# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:10:45 2019

@author: manue
"""

import numpy as np


X = np.array([[1,24],[1,30],[1,36]]);
Y = np.array([[13],[14],[15]]);
X_T= np.transpose(X);
B= np.linalg.inv(np.matmul(X,X_T));
aux = np.matmul(X_T,Y);
B = np.matmul(B,aux);

print(B);