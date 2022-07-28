# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:40:58 2022

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import apriori


"""## Data Preprocessing"""
dataset = pd.read_csv('RUStoWorldTrade.csv')
#Dataset is large, so I will drop all rows with NAN as this shouldn't affect 
#Data processing
dataset.dropna(inplace = True)
#Separate project into data entries with a trade (Qty > 0) and ones without a trade
dataset = dataset[dataset.Qty == 0]
#Checking all data types are unique for each column
unique_data_types_count = 0
for i in range(0,len(dataset.columns)):
    if(len([dataset[dataset.columns[i]].dtype]) == 1):
            unique_data_types_count += 1
    else:
        print("The " + dataset.columns[i] +" column has multiple data types")
if (unique_data_types_count == len(dataset.columns)):
    print("Each column of the dataset has a unique data type, so the data set is ready to be processed")
#If all data points in a column are the same drop the columns    
constant_data_fields = []
for i in range(0, len(dataset.columns)):
    if(len(dataset[dataset.columns[i]].unique()) == 1):
        constant_data_fields.append(dataset.columns[i])
#Drop constant_data_fields
for i in range(0, len(constant_data_fields)):
    dataset = dataset.drop(constant_data_fields[i], axis = 1)
#Dropping redundant data fields that don't need to be used in modelling
dataset = dataset.drop('index', axis = 1)
dataset = dataset.drop('Partner Code', axis = 1)
dataset = dataset.drop('Partner ISO', axis = 1)
dataset = dataset.drop('Qty Unit Code', axis = 1)
print(dataset.nunique())
print(dataset)
#apriori.MyAprioriModel(dataset,len(dataset.columns))


#dataset = dataset.sort_values(by = 'Year')
#amount_of_russian_partners = str(len(dataset['Partner'].unique()))
