# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 11:41:36 2018

@author: uchih
"""

#importing the libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset.
dataset = pd.read_csv('Student data - Sheet1.csv')
dataset['index'] = [i for i in range(len(dataset))]
X = dataset.iloc[:,:].values
dict1 = dict({})

for i in range(len(X)):
    dict1[X[i,0]] = X[i,-1]

X = X[:,1:]
#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X= sc.fit_transform(X)

#Mapping the SOM.
from minisom import MiniSom
som = MiniSom(x = 2, y = 2, input_len = X.shape[1])

som.random_weights_init(X)
som.train_random(X, num_iteration = 100)

#Mapping 
mappings = som.win_map(X)
Cl_1 = mappings[0,0]
Cl_2 = mappings[0,1]
Cl_3 = mappings[1,0]
Cl_4 = mappings[1,1]



