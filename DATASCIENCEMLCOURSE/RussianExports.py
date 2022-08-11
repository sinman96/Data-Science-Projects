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
'Reporter ISO','Qty Unit Code'], axis = 1)
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
#Find the amount traded in exports with each country
export_totals = []
for i in range(0,len(EU_countries)):
    export_totals.append(dataset[dataset.Partner == EU_countries[i]]
    ['Trade Value (US$)'].sum())
export_total = sum(export_totals)
for i in range(0, len(export_totals)):
    export_totals[i] /= export_total
print(export_totals)
#Display a random value for context
print(dataset.values[1000])
#Weighted percentages of trade with each country over this time
most_traded_countries = []
for i in range(0, len(EU_countries)):
    most_traded_countries.append([EU_countries[i],export_totals[i]])
print(most_traded_countries)

df = pd.DataFrame({'Percentage of Russian EU exports': export_totals},
                  index= EU_countries)
plot = df.plot.pie(figsize=(20, 20), subplots = "True")
plt.title('Russian EU export partners', fontsize = 20)
plt.legend(loc ='lower left')
