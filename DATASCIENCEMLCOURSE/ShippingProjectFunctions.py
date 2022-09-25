# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:54:58 2022

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def dataprocessor(dataset):
    #Checking all data types are unique for each column
    unique_data_types_count = 0
    for i in range(0,len(dataset.columns)):
        if(len([dataset[dataset.columns[i]].dtype]) == 1):
                unique_data_types_count += 1
        else:
            print("The " + dataset.columns[i] +
                  " column has multiple data types.")
    if (unique_data_types_count == len(dataset.columns)):
        print("Each column of the dataset has a unique data type," +
              " so the data set is ready to be processed.")
        #If all data points in a column are the same drop the columns    
        constant_data_fields = []
    for i in range(0, len(dataset.columns)):
        if(len(dataset[dataset.columns[i]].unique()) == 1):
            constant_data_fields.append(dataset.columns[i])
            #Drop constant_data_fields
    for i in range(0, len(constant_data_fields)):
        dataset = dataset.drop(constant_data_fields[i], axis = 1)
        #Printing unique data columns and how many unique elements they have
    print(dataset.nunique())
# Draw Plot
def plot_df(df, x, y, Title, Ylabel, dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=Title, xlabel = "Date", ylabel=Ylabel)
    plt.show()
    
def regression(X,y): 
    X = X.reshape(-1,1)
    y = y.reshape(-1,1)
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
    plt.title('Gross_Weight vs Freight Cost (Training set)')
    plt.xlabel('Gross_Weight')
    plt.ylabel('Freight Cost')
    plt.show()
    #Visualise trained data results through a plot
    plt.scatter(X_test, y_test, color = 'maroon')
    plt.plot(X_test, y_pred, color = 'blue')
    plt.title('Gross_Weight vs Freight Cost (Test set)')
    plt.xlabel('Gross_Weight')
    plt.ylabel('Freight Cost')
    plt.show()
     
def modelaccuracy(data):
    #Default predictive data split
    X_train = data.iloc[:, 2:-1].values
    y_train = data.iloc[:, -1].values
    print(X_train)    
    """## Splitting the dataset into the Training set and Test set"""
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)

    """## Applying K-Fold Cross Validation##"""
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))