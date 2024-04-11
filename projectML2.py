#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:53:52 2024

@author: ai-1
"""

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from lightgbm.sklearn import LGBMRanker
import matplotlib.pyplot as plot 
import random 


df=pd.read_csv('Customer_support_data.csv')
df=df.drop('connected_handling_time',axis=1)
df=df.dropna()
print(df.info())
print(df.head())
print(df.shape)


n=len(df)

N=list(range(n))
#print("jolly",N)
random.seed(2023)
random.shuffle(N)
df=df.iloc[N].reset_index(drop=True)
#print("jolly",N)
#print("Yolo",range(n))
print(df.head())

def labelencoder(df):
    for c in df.columns:
        if df[c].dtype=='object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df

df=labelencoder(df)

print(df.info())






