# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 17:02:51 2022

@author: User
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


#Encode categorical data
#Turn into how many categories there are with unit vectors
#In this case 3 countries, California is represented by [1,0,0],
#Florida by [0,1,0] and New York by [0,0,1]
#Once in these formats, data will be represented by a numerical matrix
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
#X = np.array(ct.fit_transform(X))
#print(X)


#Encoding Dependent variable
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#y = le.fit_transform(y)
#print(y)


#Split dataset into test and train data
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

#Feature scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
#X_test[:, 3:] = sc.transform(X_test[:, 3:])
#print(X_train)
#print(X_test)
