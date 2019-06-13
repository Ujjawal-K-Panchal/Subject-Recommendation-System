# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 12:06:09 2018

@author: Ujjawal.K.Panchal
"""


#importing the libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Number of Clusters :
clusters = 3 #found using Dendrogram below.


#importing the dataset.
dataset = pd.read_csv('Student data - Sheet1.csv')
X = dataset.iloc[:,:25]
names = list(X.iloc[:,:].columns)
X = X.iloc[:,:].values
Y = dataset.iloc[:,X.shape[1]:X.shape[1]+1]
Subject = list(Y.columns)[0]
Y = dataset.iloc[:,X.shape[1]:X.shape[1]+1].values


#Using the Dendrogram to find optimal number of clusters.
"""
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X[:,1:], method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Students')
plt.ylabel('Euclidean Distances')
plt.show()
"""

#Modelling.
from sklearn.cluster import AgglomerativeClustering
ac= AgglomerativeClustering(n_clusters = clusters, affinity = 'euclidean', linkage = 'ward')
Y_hc = ac.fit_predict(X[:,1:])
Y_hc1 = Y_hc.reshape(Y_hc.shape[0], -1)
clustered_dataset = np.concatenate((X, Y_hc1), axis = 1)
clustered_dataset = np.concatenate((clustered_dataset , Y), axis = 1)

names = list(names)
names.append('Cluster')
names.append(Subject)
clustered_dataset = pd.DataFrame(clustered_dataset, columns = names)




"""For the subject, selecting candidates with A category, Recommendations."""

#Randomly shuffling clustered_dataset
clustered_dataset = clustered_dataset.sample(frac=1)


clus_data = clustered_dataset.iloc[:,:].values 

i=0 #Cluster number.
clustered_dataset = clustered_dataset.sample(frac=1)#
recommend_cluster = list() #list will contain the cluster to which recommendations will be made.
recommend_students = list()
#Getting Clusters to which recommend to.
for x in clus_data:
    if(i>=clusters):      
        break
    if(x[-2] == i):
        if(x[-1] in ['A','B']):
            recommend_cluster.append(i)
        i+=1

for x in clus_data:
    if(x[-2] in recommend_cluster):
        recommend_students.append(x[0])

print('For subject',Subject,'we recommend it to the following students : ')

for i in recommend_students:
    print(i)




