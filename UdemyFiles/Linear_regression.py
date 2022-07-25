# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 15:24:41 2022

@author: User
"""
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Importing Datasets
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
#Splitting into Test and Train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#Importing Linear Regression and fitting the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#predicting test results
y_pred = regressor.predict(X_test)
#Visualise trained data results through a plot
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
#Visualise trained data results through a plot
plt.scatter(X_test, y_test, color = 'maroon')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()