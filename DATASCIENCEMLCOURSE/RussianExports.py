# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:40:58 2022

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from MyApriori import *
from RussianExportsFunctions import *
"""## Data Preprocessing"""
dataset = pd.read_csv('RUStoWorldTrade.csv')
#Dataset is large, so I will drop all rows with NAN as this shouldn't affect 
#Data processing
dataset.dropna(inplace = True)
dataprocessor(dataset)
print(dataset)
print(dataset.nunique())
#Dropping redundant data fields that don't need to be used in modelling
dataset = dataset.drop(['Classification','index','Partner Code',
'Partner ISO','Reporter Code','Reporter','Reporter ISO','Qty Unit Code'], axis = 1)
#Reduce project into data entries with a trade (Qty > 0) 
dataset = dataset[dataset.Qty != 0.0]
print(dataset)
print(dataset.nunique())
dataset_fields = len(dataset.columns)
#MyAprioriModel(dataset, dataset_fields)
#dataset = dataset.sort_values(by = 'Year')
