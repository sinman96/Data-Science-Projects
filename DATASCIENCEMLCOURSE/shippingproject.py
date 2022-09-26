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
regression(dataset['gross_weight'].values, dataset['freight_cost'].values,
'Gross_weight', 'Freight_cost')
dataset['shipping_cost_in($)'] = dataset['freight_cost'] + dataset['gross_weight']*dataset['shipment_charges']
print(dataset)
print(dataset.columns)
dataset = dataset.drop(['freight_cost','shipment_charges'], axis = 1)
print(dataset)
print(dataset.columns)

India_Air = dataset[(dataset.destination_country == 'IN') &
(dataset.shipment_mode == 'Air')]
India_Air = India_Air.drop(['drop_off_point','destination_country',
'shipment_mode'], axis = 1)
print(India_Air.nunique())
regression(India_Air['shipping_time'].values, India_Air['shipping_cost_in($)'].values,
'Shipping_time', 'Shipping_cost_in($)')

India_Ocean = dataset[(dataset.destination_country == 'IN') &
(dataset.shipment_mode == 'Ocean')]
India_Ocean = India_Ocean.drop(['drop_off_point','destination_country',
'shipment_mode'], axis = 1)
print(India_Ocean.nunique())
India_Ocean = India_Ocean.drop(['shipping_company'], axis = 1)
regression(India_Ocean['shipping_time'].values, India_Ocean['shipping_cost_in($)'].values,
'Shipping_time', 'Shipping_cost_in($)')
print(India_Ocean.nunique())


Bangladesh = dataset[dataset.destination_country == 'BD']
Bangladesh = Bangladesh.drop(['drop_off_point','destination_country'], axis = 1)
print(Bangladesh.nunique())
Bangladesh_Ocean = Bangladesh
Bangladesh_Ocean = Bangladesh_Ocean.drop(['shipment_mode'], axis = 1)
print(Bangladesh_Ocean.nunique())

