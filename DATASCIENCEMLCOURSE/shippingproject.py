# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:15:14 2022

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shippingprojectfunctions import *

"""## Importing the dataset"""
#People have split this dataset but I'll merge it resplit it to
dataset = pd.read_csv('shipping_train.csv')
print(dataset)
print(len(dataset))
#Dataset is large, so I will drop all rows with NAN as this shouldn't affect 
#data processing
dataset.dropna(inplace = True)
dataprocessor(dataset)
#There are three constant fields, the pick_up_point, source_country and selected
print(dataset['pick_up_point'].unique())
print(dataset['drop_off_point'].unique())
print(dataset['source_country'].unique())
print(dataset['destination_country'].unique())
print(dataset['shipment_mode'].unique())
print(dataset['shipping_company'].unique())
print(dataset['selected'].unique())
dataset = dataset.drop(['pick_up_point','source_country',
                        'Unnamed: 0','selected',], axis = 1)
dataset = dataset.sort_values(by='send_timestamp', ascending=True)
print(len(dataset))
#See if the only two drop off points correspond to the two countries
print(len((dataset[(dataset['drop_off_point'] == "Y")
               & (dataset['destination_country'] == "BD")])))
print((len(dataset[(dataset['drop_off_point'] == "X")
               & (dataset['destination_country'] == "IN")])))  
#From this we can tell all drop+off points X are in Bangladesh and all drop off 
#points Y are in India
print(dataset)
print(dataset.columns)
regression(dataset['gross_weight'].values, dataset['freight_cost'].values)
dataset['shipping_cost_in($)'] = dataset['freight_cost'] + dataset['gross_weight']*dataset['shipment_charges']
print(dataset)
print(dataset.columns)
dataset = dataset.drop(['freight_cost','shipment_charges',], axis = 1)
print(dataset)
print(dataset.columns)

India = dataset[dataset.destination_country == 'IN']
India = India.drop(['drop_off_point','destination_country'], axis = 1)
print(India)
print(India.columns)

Bangladesh = dataset[dataset.destination_country == 'BD']
Bangladesh = Bangladesh.drop(['drop_off_point','destination_country'], axis = 1)
print(Bangladesh)
print(Bangladesh.columns)