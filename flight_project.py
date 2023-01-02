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
df['isBasicEconomy']=np.select([x == False, x == True], [0,1])
x = df['isRefundable'].values
df['isRefundable']=np.select([x == False, x == True], [0,1])
x = df['isNonStop'].values
df['isNonStop']=np.select([x == False, x == True], [0,1])
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
df['%IncreaseFromBaseFare'] = np.select([x < 10, x >= 10], [0,1])
print(df.head(10))

df = df.drop(columns = ['startingAirport', 'destinationAirport', 'airlineName',
       'equipmentDescription'], axis = 1)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(X)

"""## Splitting the dataset into the Training set and Test set"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size = 0.25, random_state = 0)
 
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

"""## Making the Confusion Matrix"""
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

"""## Applying K-Fold Cross Validation##"""
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

"""## Applying Logistic Regression to separate direct and non direct flights##"""
df.columns.get_loc("isNonStop")
X = df.iloc[:, 6:8].values
y = df.iloc[:, 3].values
print(X)
print(y)
y = y.reshape(len(y),1)
#Need to reshape y to (10,1) for further analysis

#Splitting into Test and Train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X)
print(y)
#Importing Support vector Regression and fitting the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
#Predict the first entry
#classifier.predict(sc.transform([[30,87000]]))
#Predicting the test set results
y_pred = classifier.predict(X_test)
#Make confusion matrix to determine accuracy of matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
#Visualising the results
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic regression(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic regression(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

