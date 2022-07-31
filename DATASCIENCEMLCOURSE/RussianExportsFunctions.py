# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:34:54 2022

@author: User
"""        
def dataprocessor(dataset):
    #Checking all data types are unique for each column
    unique_data_types_count = 0
    for i in range(0,len(dataset.columns)):
        if(len([dataset[dataset.columns[i]].dtype]) == 1):
                unique_data_types_count += 1
        else:
            print("The " + dataset.columns[i] +
                  " column has multiple data types.")
    if (unique_data_types_count == len(dataset.columns)):
        print("Each column of the dataset has a unique data type," +
              "so the data set is ready to be processed.")
        #If all data points in a column are the same drop the columns    
        constant_data_fields = []
    for i in range(0, len(dataset.columns)):
        if(len(dataset[dataset.columns[i]].unique()) == 1):
            constant_data_fields.append(dataset.columns[i])
            #Drop constant_data_fields
    for i in range(0, len(constant_data_fields)):
        dataset = dataset.drop(constant_data_fields[i], axis = 1)
        #Printing unique data columns and how many unique elements they have
    print(dataset.nunique())