# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:20:08 2022

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
#DECISION TREE
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X,y)
#Predicting a specific value of salary
regressor.predict([[6.5]])
#Plotting the grid at high resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Position rank vs Salary Decision Tree')
plt.xlabel('Position rank')
plt.ylabel('Salary')
plt.show()
#RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
randomforest_regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
randomforest_regressor.fit(X,y)
#Predicting a specific value of salary
randomforest_regressor.predict([[6.5]])
#Plotting the grid at high resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, randomforest_regressor.predict(X_grid), color = 'green')
plt.title('Position rank vs Salary Random Forest')
plt.xlabel('Position rank')
plt.ylabel('Salary')
plt.show()



