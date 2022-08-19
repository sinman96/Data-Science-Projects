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
original_dataset_length = len(dataset)
print(dataset)
print(dataset.nunique())
#Dataset is large, so I will drop all rows with NAN as this shouldn't affect 
#data processing
dataset.dropna(inplace = True)
dataprocessor(dataset)
#Dropping redundant data fields that don't need to be used in modelling
dataset = dataset.drop(['Classification','index','Partner Code',
'Partner ISO','Reporter Code','Reporter',
'Reporter ISO','Qty Unit Code', 'Qty Unit','Commodity'], axis = 1)
#Reduce project into data entries with a trade (Qty > 0) and EU countries 
dataset = dataset[dataset.Qty != 0.0]
EU_countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 
                'Cyprus', 'Czechia', 'Denmark', 'Estonia',
                'Finland', 'France', 'Germany', 'Greece',
                'Hungary', 'Ireland', 'Italy', 'Latvia',
                'Lithuania', 'Luxembourg', 'Malta', 'Netherlands',
                'Poland', 'Portugal', 'Romania', 'Slovakia', 
                'Slovenia', 'Spain', 'Sweden']
dataset = dataset[dataset['Partner'].isin(EU_countries)]
print(dataset)
print(dataset.nunique())
#Display a random value for context
print(dataset.values[1000])
#Find the amount traded in exports with each country
pre_georgia_invasion_dataset = dataset[dataset['Year'] == 2007]

post_georgia_invasion_dataset = dataset[(dataset['Year'] >= 2009) 
                                        & (dataset['Year'] <= 2013)]
post_crimea_invasion_dataset = dataset[dataset['Year'] >= 2015]

#Displaying findings from the data over these two time periods
euexportinformation(pre_georgia_invasion_dataset
                    , EU_countries, 'Pre Georgia Invasion')
euexportinformation(post_georgia_invasion_dataset
                    , EU_countries, 'Pre Crimea/post Georgia invasion')
euexportinformation(post_crimea_invasion_dataset
                    , EU_countries, 'Post Crimea invasion')

#To display the ordering of the fields
print("First data entry")
#From these we can see what indexes of categorical fields
print(dataset.iloc[0])
#
ModelAccuracy(pre_georgia_invasion_dataset)
ModelAccuracy(post_georgia_invasion_dataset)
ModelAccuracy(post_crimea_invasion_dataset)


