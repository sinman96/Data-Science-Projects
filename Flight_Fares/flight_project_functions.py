# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:34:02 2022

@author: User
"""

def nearest_hour_of_day(time):
    time = time[11:16]
    if int(time[3:]) > 30:
       return int(time[:2]) + 1
    elif int(time[3:]) < 30:
       return int(time[:2])
    else:
       return int(time[:2]) + 0.5


def total_changes(stops):
    stops = stops.split("|")
    while "" in stops:
        stops.remove("")
    return len(stops)

def total_seconds(epoch_seconds):
    epoch_seconds = epoch_seconds.split("|")
    while "" in epoch_seconds:
        epoch_seconds.remove("")
    for i in range(0, len(epoch_seconds)):
        epoch_seconds[i] = int(epoch_seconds[i])
    return sum(epoch_seconds)




        