# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:29:21 2022

@author: User
"""

import numpy as np
import matplotlib as plt
import pandas as pd
from flight_project_functions import *
from datetime import datetime

df = pd.read_csv('itineraries.csv')
print(len(df))
df.head(10)
df = df.dropna()
print(len(df))
print(df.columns)
df = df.drop(columns = 
['legId','fareBasisCode', 'travelDuration', 'segmentsDistance'], axis = 1)
print(df.isnull().sum())
print(df.nunique())

df.head(10)

df = df.drop(columns =
['segmentsArrivalTimeEpochSeconds','segmentsDepartureTimeEpochSeconds'], axis = 1)
df['segmentsDurationInSeconds'] = df['segmentsDurationInSeconds'].apply(total_seconds)
df['segmentsDurationInHours'] = df['segmentsDurationInSeconds']/3600
df = df.drop(columns = ['segmentsDurationInSeconds'], axis = 1)
x = df['isBasicEconomy'].values
df['isBasicEconomy']=np.select([x == "FALSE", x == "TRUE"], [0,1])
x = df['isRefundable'].values
df['isRefundable']=np.select([x == "FALSE", x == "TRUE"], [0,1])
x = df['isNonStop'].values
df['isNonStop']=np.select([x == "FALSE", x == "TRUE"], [0,1])
df['searchDate'] = df['searchDate'].astype('datetime64[ns]')
df['flightDate'] = df['flightDate'].astype('datetime64[ns]')
df['daysUntilFlight'] = df['flightDate'] - df['searchDate']
df = df.drop(columns = ['flightDate','searchDate'], axis = 1)
df['daysUntilFlight'] = df['daysUntilFlight'].astype('str')
df['daysUntilFlight'] = df['daysUntilFlight'].apply(lambda x: int(x[0]))
df= df[df['seatsRemaining'] > 0]
df['segmentsCabinCode'] = df['segmentsCabinCode'].apply(total_changes)
df['numberOfChanges'] = df['segmentsCabinCode'] - 1
df = df.drop(columns =
['segmentsCabinCode', 'segmentsAirlineCode'], axis = 1)


df = df.drop(columns = ['segmentsDepartureAirportCode'], axis = 1)
df = df.drop(columns = ['segmentsArrivalAirportCode'], axis = 1)
df['arrivalTimeToNearestHour'] = df['segmentsDepartureTimeRaw'].apply(nearest_hour_of_day)
df['departureTimeToNearestHour'] = df['segmentsArrivalTimeRaw'].apply(nearest_hour_of_day)
df = df.drop(columns = ['segmentsDepartureTimeRaw', 'segmentsArrivalTimeRaw'], axis = 1)
df = df.rename(columns={'segmentsAirlineName': "airlineName", 'segmentsEquipmentDescription': "equipmentDescription",
'segmentsDurationInHours': 'durationInHours'})
print(df.columns)
print(df.head(10))

df['%IncreaseFromBaseFare'] = 100*(df['totalFare'] - df['baseFare'])/df['totalFare'] 
x = df['%IncreaseFromBaseFare']
df['%IncreaseFromBaseFare']=np.select([x < df['%IncreaseFromBaseFare'].mean(), x >= df['%IncreaseFromBaseFare'].mean()], [0,1])
print(df.head(10))