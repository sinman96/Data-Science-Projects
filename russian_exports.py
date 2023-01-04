# -*- coding: utf-8 -*-

"""
Data Preprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from russian_exports_functions import *

"""## Data Preprocessing"""
dataset = pd.read_csv('RUStoWorldTrade.csv')
print(dataset)
print(dataset.nunique())
#Dataset is large, so I will drop all rows with NAN as this shouldn't affect 
#data processing
dataset.dropna(inplace = True)
data_processor(dataset)
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
median_trade_value = dataset["Trade Value (US$)"].median()
x = dataset["Trade Value (US$)"]
dataset["Large trades"] = np.select([x < median_trade_value, x >= median_trade_value], [0,1])  
print(dataset)
print(dataset.nunique())
#Display a random value for context
print(dataset.values[1000])
#Find the amount traded in exports with each country
pre_georgia_dataset = dataset[dataset['Year'] == 2007]

post_georgia_dataset = dataset[(dataset['Year'] >= 2009) 
                                        & (dataset['Year'] <= 2013)]
post_crimea_dataset = dataset[dataset['Year'] >= 2015]

#Displaying findings from the data over these two time periods
phase1 = eu_export_information(pre_georgia_dataset
                    , EU_countries, 'Pre Georgia')

phase2 = eu_export_information(post_georgia_dataset
                    , EU_countries, 'Pre Crimea/post Georgia')

phase3 = eu_export_information(post_crimea_dataset
                    , EU_countries, 'Post Crimea')
eu_export_changes(phase3, phase2, phase1, EU_countries)


model_accuracy_trades(pre_georgia_dataset)
model_accuracy_trades(post_georgia_dataset)
model_accuracy_trades(post_crimea_dataset)
#Obtained 100% accuracy in predictions of the model accuracy after applying
#XGBoost and K cross fold validation
#model_accuracy_aggregates(pre_georgia_dataset)
#model_accuracy_aggregates(post_georgia_dataset)
#model_accuracy_aggregates(post_crimea_dataset)