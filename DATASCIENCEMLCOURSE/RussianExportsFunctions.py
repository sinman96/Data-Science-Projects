# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:34:54 2022

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
              "so the data set is ready to be processed.")
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
def euexportinformation(data, datasubset, time_period):
    #Find the amount traded in exports with each country
    export_totals = []
    for i in range(0,len(datasubset)):
        export_totals.append(data[data.Partner == datasubset[i]]
    ['Trade Value (US$)'].sum())
    export_total = sum(export_totals)
    for i in range(0, len(export_totals)):
        export_totals[i] /= export_total
    #Weighted percentages of trade with each country over this time
    most_traded_countries = []
    for i in range(0, len(datasubset)):
        most_traded_countries.append([datasubset[i],export_totals[i]])
    print(most_traded_countries)
    df = pd.DataFrame({'Percentage of Russian EU exports': export_totals},
                  index= datasubset)
    plot = df.plot.pie(figsize=(20, 20), subplots = "True")
    plt.title('Russian EU export partners ' + time_period, fontsize = 20)
    plt.legend(loc ='lower left')
    return export_totals

def ModelAccuracy(data):
    #Default predictive data split
    X1 = data.iloc[:, :2].values
    X2 = data.iloc[:, 3:].values
    X = con = np.concatenate((X1, X2), axis=1)
    y = data.iloc[:, 2].values
    #Only non numeric fields are Partner and Commodity, so those will be encoded
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct1 = ColumnTransformer(transformers=[('encoder',
    OneHotEncoder(sparse = False), [2])], remainder='passthrough')
    X = np.array(ct1.fit_transform(X))
    print(X)

    """## Splitting the dataset into the Training set and Test set"""
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size = 0.25, random_state = 0)
 

    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    """## Making the Confusion Matrix"""

    from sklearn.metrics import confusion_matrix, accuracy_score
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))

    """## Applying K-Fold Cross Validation##"""
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
