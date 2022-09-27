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
dataset = dataset.drop(['drop_off_point'], axis = 1) 
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

SC1_India = dataset[(dataset.shipping_company == 'SC1') &
(dataset.destination_country == 'IN')]
SC1_India = SC1_India.drop(['shipping_company', 'destination_country'], axis = 1)
print(SC1_India.nunique())
print("SC1_India_min_time: %.2f" % (SC1_India['shipping_time'].min()))
print("SC1_India_mean_time: %.2f" % (SC1_India['shipping_time'].mean()))
print("SC1_India_max_time: %.2f" % (SC1_India['shipping_time'].max()))
print("SC1_India_std_time: %.2f" % (SC1_India['shipping_time'].std()))

SC2_India = dataset[(dataset.shipping_company == 'SC2') &
(dataset.destination_country == 'IN')]
SC2_India = SC2_India.drop(['shipping_company', 'destination_country'], axis = 1)
print(SC2_India.nunique())
print("SC2_India_min_time: %.2f" % (SC2_India['shipping_time'].min()))
print("SC2_India_mean_time: %.2f" % (SC2_India['shipping_time'].mean()))
print("SC2_India_max_time: %.2f" % (SC2_India['shipping_time'].max()))
print("SC2_India_std_time: %.2f" % (SC2_India['shipping_time'].std()))

SC3_India = dataset[(dataset.shipping_company == 'SC3') &
(dataset.destination_country == 'IN')]
SC3_India = SC3_India.drop(['shipping_company', 'destination_country'], axis = 1)
print(SC3_India.nunique())
print("SC3_India_min_time: %.2f" % (SC3_India['shipping_time'].min()))
print("SC3_India_mean_time: %.2f" % (SC3_India['shipping_time'].mean()))
print("SC3_India_max_time: %.2f" % (SC3_India['shipping_time'].max()))
print("SC3_India_std_time: %.2f" % (SC3_India['shipping_time'].std()))


SC1_Bangladesh = dataset[(dataset.shipping_company == 'SC1') &
(dataset.destination_country == 'BD')]
SC1_Bangladesh = SC1_Bangladesh.drop(['shipping_company', 'destination_country'], axis = 1)
print(SC1_Bangladesh.nunique())
print("SC1_Bangladesh_min_time: %.2f" % (SC1_Bangladesh['shipping_time'].min()))
print("SC1_Bangladesh_mean_time: %.2f" % (SC1_Bangladesh['shipping_time'].mean()))
print("SC1_Bangladesh_max_time: %.2f" % (SC1_Bangladesh['shipping_time'].max()))
print("SC1_Bangladesh_std_time: %.2f" % (SC1_Bangladesh['shipping_time'].std()))

SC2_Bangladesh = dataset[(dataset.shipping_company == 'SC2') &
(dataset.destination_country == 'BD')]
SC2_Bangladesh = SC2_Bangladesh.drop(['shipping_company', 'destination_country'], axis = 1)
print(SC2_Bangladesh.nunique())
print("SC2_Bangladesh_min_time: %.2f" % (SC2_Bangladesh['shipping_time'].min()))
print("SC2_Bangladesh_mean_time: %.2f" % (SC2_Bangladesh['shipping_time'].mean()))
print("SC2_Bangladesh_max_time: %.2f" % (SC2_Bangladesh['shipping_time'].max()))
print("SC2_Bangladesh_std_time: %.2f" % (SC2_Bangladesh['shipping_time'].std()))

from scipy.stats import ttest_ind
print(ttest_ind(SC2_India['shipping_time'],
SC3_India['shipping_time']))
print(ttest_ind(SC1_Bangladesh['shipping_time'],
SC2_Bangladesh['shipping_time']))