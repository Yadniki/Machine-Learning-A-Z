#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:10:07 2018

@author: hamsika
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''dont need this here as only 10 values... and need very accurate predictions, hence max
information needed'''
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

'''
we build two models linear and polynomial to compare the two
'''
#Fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
l_reg = LinearRegression()
l_reg.fit(X, y)

#Fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures #helps to include ploy terms in linear regression eq.
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
l_reg2 = LinearRegression()
l_reg2.fit(X_poly, y)

#visualize linear regression model
plt.scatter(X, y, color = 'red')
plt.plot(X, l_reg.predict(X), color = 'blue')
plt.title('Linear regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#visualize polynomial regression model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, l_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Polynomial regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#predict new result with linear regression model
l_y_pred = l_reg.predict(6.5)
#predict new result with polynomial regression model
poly_y_pred = l_reg2.predict(poly_reg.fit_transform(6.5))