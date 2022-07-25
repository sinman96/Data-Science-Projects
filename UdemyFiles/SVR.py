# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 08:47:14 2022

@author: User
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Importing Datasets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
#Need to reshape y to (10,1) for further analysis
y = y.reshape(len(y),1)
#Feature scaling as salary is much bigger than rank
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)
print(X)
print(y)
#Splitting into Test and Train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#Importing SUpport vector Regression and fitting the model
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#plotting the SVR
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))
#Visualise trained data results through a plot
plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Position rank vs Salary (Training set)')
plt.xlabel('Position rank')
plt.ylabel('Salary')
plt.show()
#Visualise trained data results through a plot
X_grid = np.arange(min(sc_x.inverse_transform(X)), max(sc_x.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid))), color = 'blue')
plt.title('Position rank vs Salary (Training set)')
plt.xlabel('Position rank')
plt.ylabel('Salary')
plt.show()