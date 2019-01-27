#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 10:56:14 2018

@author: hamsika
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#X is feature "matrix" and y is output "vector"
#splitting into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting test set results
y_pred = regressor.predict(X_test)

#data visualization using ploting
plt.title('Salary v/s Experience (Training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()

plt.title('Salary v/s Experience (Test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
#plt.plot(X_test, y_pred, color = 'blue')
plt.show()