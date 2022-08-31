# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:15:14 2022

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""
#People have split this dataset but I'll merge it resplit it to
training_dataset = pd.read_csv('shipping_train.csv')
test_dataset = pd.read_csv('shipping_test.csv')
dataset = pd.concat([training_dataset, test_dataset],
ignore_index=True, sort = True)
print(dataset)

print(len(dataset))
print(len(dataset))
#Dataset is large, so I will drop all rows with NAN as this shouldn't affect 
#data processing
dataset.dropna(inplace = True)
dataset.dropna(inplace = True)
print(len(dataset))
print(len(dataset))
print(dataset.nunique())
#There are three constant fields, the pick_up_point, source_country and selected
print(dataset['pick_up_point'].unique())
print(dataset['drop_off_point'].unique())
print(dataset['source_country'].unique())
print(dataset['destination_country'].unique())
print(dataset['shipment_mode'].unique())
print(dataset['shipping_company'].unique())
print(dataset['selected'].unique())
dataset = dataset.drop(['pick_up_point','source_country','selected',]
, axis = 1)
print(dataset)