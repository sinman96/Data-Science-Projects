# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:10:39 2022

@author: User
"""

import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values