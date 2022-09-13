# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:15:14 2022

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ShippingProjectFunctions import *

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
print((dataset[(dataset['drop_off_point'] == "Y")
               & (dataset['destination_country'] == "BD")]))
print((dataset[(dataset['drop_off_point'] == "X")
               & (dataset['destination_country'] == "IN")]))  
#From this we can tell all drop+off points X are in Bangladesh and all drop off 
#points Y are in India
dataset = dataset.drop(['drop_off_point'], axis = 1)
print(dataset)
#Separate dataset into datasets for the three different shipping companies

SC1 = dataset[(dataset['shipping_company'] == "SC1")]
SC1['shipment_mode'].unique()
#We can see that SC1 only has its shipping mode as Ocean we'll drop it,
SC1 = SC1.drop(['shipment_mode', 'shipping_company'], axis = 1)
SC1_India = SC1[(SC1['destination_country'] == "IN")]
SC1_India = SC1_India.drop(['destination_country'], axis = 1)
SC1_India = SC1_India.sort_values(by='send_timestamp', ascending = False)
print(SC1_India.nunique())
SC1_Bangladesh = SC1[(SC1['destination_country'] == "BD")]
SC1_Bangladesh = SC1_Bangladesh.drop(['destination_country'], axis = 1)
SC1_Bangladesh = SC1_India.sort_values(by='send_timestamp', ascending = False)
print(SC1_Bangladesh.nunique())

SC2 = dataset[(dataset['shipping_company'] == "SC2")]
SC2['shipment_mode'].unique()
#We won't drop shipment mode here as it has Air and Ocean entries.
SC2_India = SC2[(SC2['destination_country'] == "IN")]
SC2_India = SC2_India.drop(['destination_country'], axis = 1)
SC2_India = SC2_India.sort_values(by='send_timestamp', ascending = False)
print(SC2_India.nunique())
SC2_Bangladesh = SC2[(SC2['destination_country'] == "BD")]
SC2_Bangladesh = SC2_Bangladesh.drop(['destination_country'], axis = 1)
SC2_Bangladesh = SC2_India.sort_values(by='send_timestamp', ascending = False)
print(SC2_Bangladesh.nunique())

SC3 = dataset[(dataset['shipping_company'] == "SC3")]
SC3['shipment_mode'].unique()
#We can see that SC1 only has its shipping mode as Ocean we'll drop it,
SC3 = SC3.drop(['shipment_mode', 'shipping_company'], axis = 1)
SC3_India = SC3[(SC3['destination_country'] == "IN")]
SC3_India = SC3_India.drop(['destination_country'], axis = 1)
SC3_India = SC3_India.sort_values(by='send_timestamp', ascending = False)
print(SC3_India.nunique())
#After Inspection SC3_Bangladesh is empty, so we'll just focus on the others.




