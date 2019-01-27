#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:13:29 2018

@author: hamsika
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#feature scaling needed as not included in SVR class
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape((-1,1)))

from sklearn.svm import SVR
regressor = SVR(kernel='rbf') #gaussian
regressor.fit(X, y)

#y_pred = regressor.predict(6.5)
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
y_pred = sc_y.inverse_transform(y_pred)

plt.scatter (X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.title('Truth or bluff (SVR)')
plt.show()