# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:40:58 2022

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Data Preprocessing"""

#Arranging data by the year to 
dataset = pd.read_csv('RUStoWorldTrade.csv')
dataset = dataset.sort_values(by = 'Year')
amount_of_russian_partners = str(len(dataset['Partner'].unique()))
#Dataset is large, so I will drop all rows with NAN as this shouldn't affect 
#Data processing
dataset.dropna(inplace = True)
dataset['Reporter Code'].unique()
dataset['Reporter'].unique()
dataset['Reporter ISO'].unique()
#These indicate that Russia is the exporter the Reporter code is 643 and ISO, is RUS
dataset.dropcol
print(amount_of_russian_partners + " independent countries are Russian trade partners.")
print("There are " +str(len(dataset['Classification'].unique())) + " different Classification(s)")